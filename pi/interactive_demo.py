"""Interactive PC plant demo with add/remove vehicle-event controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    from .contracts import ExecutionConfig
    from .detector import DetectorState, TurnDetector
    from .model import dummy_model, load_weights
    from .pipeline import EdgePipeline
    from .streaming import PlaybackStream, PlantStream, StreamingRunner
    from .transport.adapters import FileEventLogSink, FileSampleSink
    from .transport.tcp import TcpSampleSender
except ImportError:  # pragma: no cover
    from contracts import ExecutionConfig
    from detector import DetectorState, TurnDetector
    from model import dummy_model, load_weights
    from pipeline import EdgePipeline
    from streaming import PlaybackStream, PlantStream, StreamingRunner
    from transport.adapters import FileEventLogSink, FileSampleSink
    from transport.tcp import TcpSampleSender


class InteractiveDemo:
    def __init__(
        self,
        root: tk.Tk,
        *,
        tcp_host: str | None = None,
        tcp_port: int = 8765,
        capture: Path | None = None,
        realtime: bool = True,
    ):
        self.root = root
        self.root.title("Interactive Python Plant")
        self.samples: list = []
        self.sample_queue: queue.Queue = queue.Queue(maxsize=256)
        self.processed_samples = 0
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.capture_path = capture
        self.realtime = realtime
        self.sender = None
        self.capture = None
        self.event_log = None
        self._open_transport()
        self.trajectory_options = self._find_trajectories()
        self.selected_trajectory = "Python default"
        self.stream = self._build_stream(self.selected_trajectory)
        default_weights = Path(__file__).resolve().parents[1] / "models" / "export" / "weights.mat"
        model = load_weights(default_weights) if default_weights.exists() else dummy_model()
        self.detector = TurnDetector(model)
        self.pipeline = EdgePipeline(self.detector, ExecutionConfig(realtime=False))
        self._transport_finished = False
        self.path_states: list[str] = []
        self.classifier_state = DetectorState.IDLE.value
        self.classifier_class: str | None = None
        self.verdict_history: list[tuple[float, object]] = []
        self.runner = StreamingRunner(
            self.stream,
            self._on_sample,
            self._on_complete,
            realtime=realtime,
        )
        self._build_widgets()
        self._build_plot()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._poll_samples)

    def _find_trajectories(self) -> dict[str, Path | None]:
        options: dict[str, Path | None] = {"Python default": None}
        folder = Path(__file__).resolve().parents[1] / "data" / "generated_trajectories" / "matlab"
        for path in sorted(folder.glob("*.mat")):
            options[f"MATLAB: {path.stem}"] = path
        return options

    def _build_stream(self, selection: str):
        path = self.trajectory_options[selection]
        return PlantStream.from_seed() if path is None else PlaybackStream.from_mat(path)

    def _open_transport(self) -> None:
        self.sender = (
            TcpSampleSender(self.tcp_host, self.tcp_port)
            if self.tcp_host
            else None
        )
        self.capture = FileSampleSink(self.capture_path) if self.capture_path else None
        self.event_log = (
            FileEventLogSink(self.capture_path.with_suffix(".events.json"))
            if self.capture_path
            else None
        )

    def _build_widgets(self) -> None:
        trajectory_frame = ttk.Frame(self.root, padding=6)
        trajectory_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(trajectory_frame, text="Trajectory").pack(side=tk.LEFT)
        self.trajectory_value = tk.StringVar(value=self.selected_trajectory)
        self.trajectory_combo = ttk.Combobox(
            trajectory_frame,
            textvariable=self.trajectory_value,
            values=list(self.trajectory_options),
            state="readonly",
            width=48,
        )
        self.trajectory_combo.pack(side=tk.LEFT, padx=6)
        self.trajectory_combo.bind("<<ComboboxSelected>>", self._select_trajectory)

        controls = ttk.Frame(self.root, padding=6)
        controls.pack(side=tk.TOP, fill=tk.X)
        buttons = (
            ("Start", self.runner.start),
            ("Pause", self.runner.pause),
            ("Resume", self.runner.resume),
            ("Reset", self.reset),
            ("Stop", self.runner.stop),
            ("Add implement", lambda: self.runner.add_event("implement_attached")),
            ("Remove implement", lambda: self.runner.remove_event("implement_attached")),
            ("Flat tire", lambda: self.runner.add_event("tire_flat")),
            ("Repair tire", lambda: self.runner.repair_event("tire_flat")),
        )
        self.event_buttons: list[tk.Widget] = []
        for label, callback in buttons:
            button = ttk.Button(controls, text=label, command=callback)
            button.pack(
                side=tk.LEFT,
                padx=2,
            )
            if label in {
                "Add implement",
                "Remove implement",
                "Flat tire",
                "Repair tire",
            }:
                self.event_buttons.append(button)
        speed_frame = ttk.Frame(self.root, padding=(6, 0))
        speed_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(speed_frame, text="Playback speed").pack(side=tk.LEFT)
        self.speed_value = tk.DoubleVar(value=1.0)
        self.speed_value.trace_add("write", self._speed_variable_changed)
        ttk.Scale(
            speed_frame,
            from_=0.1,
            to=20.0,
            variable=self.speed_value,
            command=self._set_speed,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT)
        self.trajectory_info = ttk.Label(
            speed_frame,
            text=self._stream_description(),
            padding=(12, 0),
        )
        self.trajectory_info.pack(side=tk.LEFT)

        indicator_frame = ttk.Frame(self.root, padding=(6, 0))
        indicator_frame.pack(side=tk.TOP, fill=tk.X)
        self.implement_indicator = tk.Label(
            indicator_frame,
            text="IMPLEMENT: OFF",
            width=18,
            relief=tk.GROOVE,
            bg="lightgray",
        )
        self.implement_indicator.pack(side=tk.LEFT, padx=2)
        self.tire_indicator = tk.Label(
            indicator_frame,
            text="TIRE: NORMAL",
            width=18,
            relief=tk.GROOVE,
            bg="lightgray",
        )
        self.tire_indicator.pack(side=tk.LEFT, padx=2)
        self.class_indicators: dict[str, tk.Label] = {}
        class_frame = ttk.Frame(self.root, padding=(6, 0))
        class_frame.pack(side=tk.TOP, fill=tk.X)
        for class_name in ("nominal", "A", "B", "AB"):
            indicator = tk.Label(
                class_frame,
                text=f"CLASS {class_name.upper()}",
                width=16,
                relief=tk.GROOVE,
                bg="lightgray",
            )
            indicator.pack(side=tk.LEFT, padx=2)
            self.class_indicators[class_name] = indicator
        self.status = tk.StringVar(value="ready")
        ttk.Label(self.root, textvariable=self.status, padding=6).pack(
            side=tk.BOTTOM,
            fill=tk.X,
        )

    def _select_trajectory(self, _event=None) -> None:
        if self.runner.running:
            self.runner.stop()
        self._on_complete()
        self.selected_trajectory = self.trajectory_value.get()
        self.stream = self._build_stream(self.selected_trajectory)
        self.runner = StreamingRunner(
            self.stream,
            self._on_sample,
            self._on_complete,
            realtime=self.realtime,
        )
        self.runner.set_speed(float(self.speed_value.get()))
        self.detector.reset()
        self.pipeline.reset()
        self.samples.clear()
        self.path_states.clear()
        self.verdict_history.clear()
        self.classifier_class = None
        self._transport_finished = False
        self._open_transport()
        supports_events = self.stream.supports_events
        for button in self.event_buttons:
            button.configure(state=tk.NORMAL if supports_events else tk.DISABLED)
        self._update_class_indicators()
        self._update_indicators("")
        self.trajectory_info.configure(text=self._stream_description())
        self._redraw()
        self.status.set(f"selected {self.selected_trajectory}")

    def _stream_description(self) -> str:
        duration = (
            (len(self.stream.measured) - 1) * self.stream.period_s
            if hasattr(self.stream, "measured")
            else (len(self.stream.delta) - 1) * self.stream.period_s
        )
        source = "fixed MATLAB playback" if not self.stream.supports_events else "Python synthesized"
        return f"{source}, {duration:.1f}s"

    def _build_plot(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        self.figure, self.axes = plt.subplots(
            3,
            2,
            figsize=(11, 8),
            gridspec_kw={"height_ratios": (1.5, 1, 1)},
        )
        path_axis = self.axes[0, 0]
        edge_axis = self.axes[0, 1]
        dynamics_axis = self.axes[1, 0]
        confidence_axis = self.axes[1, 1]
        mass_axis = self.axes[2, 0]
        stiffness_axis = self.axes[2, 1]
        path_axis.set_title("Vehicle path")
        path_axis.set_xlabel("X [m]")
        path_axis.set_ylabel("Y [m]")
        edge_axis.set_title("Edge signals")
        edge_axis.set_ylabel("delta / Vx")
        dynamics_axis.set_title("Plant signals")
        dynamics_axis.set_xlabel("time [s]")
        dynamics_axis.set_ylabel("r / ay")
        confidence_axis.set_title("Classifier probabilities")
        confidence_axis.set_ylabel("probability")
        mass_axis.set_title("Added mass estimate")
        mass_axis.set_xlabel("time [s]")
        mass_axis.set_ylabel("delta_m [kg]")
        stiffness_axis.set_title("Front stiffness estimate")
        stiffness_axis.set_xlabel("time [s]")
        stiffness_axis.set_ylabel("k_f")
        self.path_collection = LineCollection([], linewidths=2.5)
        path_axis.add_collection(self.path_collection)
        path_axis.legend(handles=[
            Line2D([], [], color="tab:blue", label="idle"),
            Line2D([], [], color="tab:orange", label="turn detection"),
            Line2D([], [], color="tab:red", label="classifier window"),
        ], loc="upper right")
        self.delta_line, = edge_axis.plot([], [], label="delta")
        self.vx_line, = edge_axis.plot([], [], label="Vx")
        self.r_line, = dynamics_axis.plot([], [], label="r")
        self.ay_line, = dynamics_axis.plot([], [], label="ay")
        self.probability_lines = {
            name: confidence_axis.plot([], [], label=name.upper())[0]
            for name in ("nominal", "A", "B", "AB")
        }
        self.mass_line, = mass_axis.plot([], [], label="delta_m")
        self.stiffness_line, = stiffness_axis.plot([], [], label="k_f")
        for axis in self.axes.flat:
            if axis is not path_axis:
                axis.legend(loc="upper right")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_sample(self, sample) -> None:
        pipeline_event = self.pipeline.push(sample)
        detector_result = pipeline_event.consumer_result
        verdict = getattr(detector_result, "new_verdict", None)
        if verdict is not None:
            self.classifier_class = verdict.class_name
            self.verdict_history.append((sample.timestamp_s, verdict))
        self.classifier_state = self.detector.state.value
        if self.sender:
            self.sender.send(sample)
        if self.capture:
            self.capture.send(sample)
        self.processed_samples += 1
        display_stride = max(1, int(np.ceil(self.runner.speed)))
        should_display = (
            sample.sequence % display_stride == 0
            or verdict is not None
            or self.stream.finished
        )
        if should_display:
            try:
                self.sample_queue.put_nowait((sample, self.classifier_state))
            except queue.Full:
                try:
                    self.sample_queue.get_nowait()
                except queue.Empty:
                    pass
                self.sample_queue.put_nowait((sample, self.classifier_state))

    def _set_speed(self, value: str) -> None:
        speed = float(value)
        self.runner.set_speed(speed)
        if hasattr(self, "speed_label"):
            self.speed_label.configure(text=f"{speed:.1f}x")

    def _speed_variable_changed(self, *_args) -> None:
        self._set_speed(str(self.speed_value.get()))

    def _on_complete(self) -> None:
        if self._transport_finished:
            return
        if self.sender:
            self.sender.finish()
            self.sender.close()
        if self.capture:
            self.capture.close()
        if self.event_log:
            self.event_log.write(self.stream.schedule.history)
        self._transport_finished = True

    def _poll_samples(self) -> None:
        changed = False
        while True:
            try:
                sample, path_state = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            self.samples.append(sample)
            self.path_states.append(path_state)
            changed = True
        if changed:
            self._redraw()
        if self.runner.errors:
            self.status.set(f"error: {self.runner.errors[-1]}")
        else:
            active = self.samples[-1].diagnostics.get("active_events", "") if self.samples else ""
            transport = self.sender.statuses[-1].kind if self.sender and self.sender.statuses else "local"
            self._update_indicators(active)
            self._update_class_indicators()
            verdict_text = self.classifier_class or "waiting"
            self.status.set(
                f"t={self.stream.time_s:.2f}s samples={self.processed_samples} "
                f"speed={self.runner.speed:.1f}x classifier={self.classifier_state} "
                f"output={verdict_text} active={active or 'none'} "
                f"transport={transport}"
            )
        self.root.after(50, self._poll_samples)

    def _redraw(self) -> None:
        if not self.samples:
            return
        stride = max(1, len(self.samples) // 2000)
        indices = list(range(0, len(self.samples), stride))
        if indices[-1] != len(self.samples) - 1:
            indices.append(len(self.samples) - 1)
        display_samples = [self.samples[index] for index in indices]
        time_s = np.asarray([sample.timestamp_s for sample in display_samples])
        x = np.asarray([sample.diagnostics["X"] for sample in display_samples])
        y = np.asarray([sample.diagnostics["Y"] for sample in display_samples])
        delta = np.asarray([sample.delta for sample in display_samples])
        vx = np.asarray([sample.vx for sample in display_samples])
        yaw = np.asarray([sample.yaw_rate for sample in display_samples])
        ay = np.asarray([sample.diagnostics["ay"] for sample in display_samples])
        if len(x) > 1:
            points = np.column_stack((x, y))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            colors = [
                {
                    DetectorState.IDLE.value: "tab:blue",
                    DetectorState.IN_TURN.value: "tab:orange",
                    DetectorState.WAITING_WINDOW.value: "tab:red",
                }.get(state, "tab:gray")
                for state in [self.path_states[index] for index in indices[1:]]
            ]
            self.path_collection.set_segments(segments)
            self.path_collection.set_color(colors)
        else:
            self.path_collection.set_segments([])
        self.delta_line.set_data(time_s, delta)
        self.vx_line.set_data(time_s, vx)
        self.r_line.set_data(time_s, yaw)
        self.ay_line.set_data(time_s, ay)
        for axis in self.axes.flat:
            axis.relim()
            axis.autoscale_view()
        self.axes[0, 0].set_aspect("auto")
        if len(x) > 1:
            x_span = max(float(np.ptp(x)), 1.0)
            y_span = max(float(np.ptp(y)), 1.0)
            x_pad = 0.05 * x_span
            y_pad = 0.05 * y_span
            self.axes[0, 0].set_xlim(
                float(np.min(x) - x_pad),
                float(np.max(x) + x_pad),
            )
            self.axes[0, 0].set_ylim(
                float(np.min(y) - y_pad),
                float(np.max(y) + y_pad),
            )
        self._redraw_verdicts()
        self.canvas.draw_idle()

    def _redraw_verdicts(self) -> None:
        if not self.verdict_history:
            return
        times = np.asarray([item[0] for item in self.verdict_history])
        probabilities = np.asarray([
            item[1].class_probabilities for item in self.verdict_history
        ])
        for index, name in enumerate(("nominal", "A", "B", "AB")):
            self.probability_lines[name].set_data(times, probabilities[:, index])
        mass_points = [
            (time_s, verdict.delta_m_kg)
            for time_s, verdict in self.verdict_history
            if verdict.delta_m_kg is not None
        ]
        stiffness_points = [
            (time_s, verdict.k_f)
            for time_s, verdict in self.verdict_history
            if verdict.k_f is not None
        ]
        self.mass_line.set_data(
            [point[0] for point in mass_points],
            [point[1] for point in mass_points],
        )
        self.stiffness_line.set_data(
            [point[0] for point in stiffness_points],
            [point[1] for point in stiffness_points],
        )

    def _update_class_indicators(self) -> None:
        for name, indicator in self.class_indicators.items():
            active = name == self.classifier_class
            indicator.configure(
                text=f"CLASS {name.upper()}" + ("  ACTIVE" if active else ""),
                bg="lightgreen" if active else "lightgray",
            )

    def _update_indicators(self, active: str) -> None:
        active_names = set(filter(None, str(active).split(",")))
        implement_on = bool({"implement_attached", "A", "AB"} & active_names)
        tire_flat = bool({"tire_flat", "B", "AB"} & active_names)
        self.implement_indicator.configure(
            text=f"IMPLEMENT: {'ON' if implement_on else 'OFF'}",
            bg="lightgreen" if implement_on else "lightgray",
        )
        self.tire_indicator.configure(
            text=f"TIRE: {'FLAT' if tire_flat else 'NORMAL'}",
            bg="tomato" if tire_flat else "lightgray",
        )

    def reset(self) -> None:
        self.runner.pause()
        self._on_complete()
        self.runner.reset()
        self.pipeline.reset()
        self.samples.clear()
        self.processed_samples = 0
        self.path_states.clear()
        self.classifier_state = DetectorState.IDLE.value
        self.classifier_class = None
        self.verdict_history.clear()
        self._transport_finished = False
        self._open_transport()
        self._update_class_indicators()
        self._update_indicators("")
        self._redraw()
        self.status.set("reset")

    def close(self) -> None:
        self.runner.stop()
        self._on_complete()
        if self.sender:
            self.sender.close()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcp-host", help="send samples to a Pi")
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--capture", type=Path, help="save the generated stream")
    parser.add_argument("--accelerated", action="store_true")
    args = parser.parse_args(argv)
    root = tk.Tk()
    InteractiveDemo(
        root,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        capture=args.capture,
        realtime=not args.accelerated,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

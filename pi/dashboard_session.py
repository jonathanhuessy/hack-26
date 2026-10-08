"""Shared operator session for the Dash and Streamlit PC dashboards."""

from __future__ import annotations

from pathlib import Path
import queue
from typing import Any

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


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("nominal", "A", "B", "AB")
MAX_DISPLAY_SAMPLES = 5000
MAX_PLOT_POINTS = 600
MAX_UI_BATCH = 128
LIVE_UPDATE_SECONDS = 0.5
PATH_COLORS = {
    DetectorState.IDLE.value: "#1f77b4",
    DetectorState.IN_TURN.value: "#ff7f0e",
    DetectorState.WAITING_WINDOW.value: "#d62728",
}


def trajectory_options() -> dict[str, Path | None]:
    options: dict[str, Path | None] = {"Python default": None}
    folder = ROOT / "data" / "generated_trajectories" / "matlab"
    for path in sorted(folder.glob("*.mat")):
        demo_name = path.stem.lower()
        if demo_name in {"demo_a", "demo_b", "demo_ab"}:
            label = f"MATLAB demo {demo_name.removeprefix('demo_').upper()}"
            options[label] = path
    return options


def build_stream(selection: str, options: dict[str, Path | None]):
    path = options[selection]
    return PlantStream.from_seed() if path is None else PlaybackStream.from_mat(path)


def model_for_runtime():
    path = ROOT / "models" / "export" / "weights.mat"
    return load_weights(path) if path.exists() else dummy_model()


class DashboardSession:
    """Own one operator run and expose bounded data for a browser dashboard."""

    def __init__(self, selection: str, options: dict[str, Path | None]):
        self.options = options
        self.selection = selection
        self.stream = build_stream(selection, options)
        self.detector = TurnDetector(model_for_runtime())
        self.pipeline = EdgePipeline(self.detector, ExecutionConfig(realtime=False))
        self.samples: list = []
        self.path_states: list[str] = []
        self.verdicts: list[tuple[float, Any]] = []
        self.verdict_cursor = 0
        self.commands: list = []
        self.queue: queue.Queue = queue.Queue(maxsize=512)
        self.processed_samples = 0
        self.classifier_class: str | None = None
        self.classifier_state = DetectorState.IDLE.value
        self.speed = 1.0
        self.tcp_host = ""
        self.tcp_port = 8765
        self.capture_path: Path | None = None
        self.sender = None
        self.capture = None
        self.event_log = None
        self.transport_finished = False
        self.runner = self._new_runner()

    def _new_runner(self) -> StreamingRunner:
        return StreamingRunner(
            self.stream,
            self._on_sample,
            self._on_complete,
            realtime=True,
        )

    @property
    def supports_events(self) -> bool:
        return bool(self.stream.supports_events)

    @property
    def running(self) -> bool:
        return self.runner.running

    @property
    def finished(self) -> bool:
        return self.stream.finished and not self.runner.running

    @property
    def error_message(self) -> str | None:
        """Return a worker failure in a form the dashboard can display."""
        if not self.runner.errors:
            return None
        error = self.runner.errors[-1]
        return f"{type(error).__name__}: {error}"

    def configure_transport(self, host: str, port: int, capture: str) -> None:
        self.tcp_host = host.strip()
        self.tcp_port = int(port)
        self.capture_path = Path(capture) if capture.strip() else None

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

    def _on_sample(self, sample) -> None:
        result = self.pipeline.push(sample)
        detector_result = result.consumer_result
        verdict = getattr(detector_result, "new_verdict", None)
        if verdict is not None:
            self.classifier_class = verdict.class_name
            self.verdicts.append((sample.timestamp_s, verdict))
        self.classifier_state = self.detector.state.value
        if self.sender:
            self.sender.send(sample)
        if self.capture:
            self.capture.send(sample)
        self.processed_samples += 1
        stride = max(1, int(np.ceil(self.speed)))
        if sample.sequence % stride == 0 or verdict is not None or self.stream.finished:
            item = (sample, self.classifier_state)
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(item)

    def _on_complete(self) -> None:
        if self.transport_finished:
            return
        if self.sender:
            self.sender.finish()
            self.sender.close()
        if self.capture:
            self.capture.close()
        if self.event_log:
            self.event_log.write(self.stream.schedule.history)
        self.transport_finished = True

    def start(self) -> None:
        self._open_transport()
        self.transport_finished = False
        self.runner.start()

    def pause(self) -> None:
        self.runner.pause()

    def resume(self) -> None:
        self.runner.resume()

    def stop(self) -> None:
        self.runner.stop()
        self._on_complete()

    def reset(self) -> None:
        self.stop()
        self.stream.reset()
        self.detector.reset()
        self.pipeline.reset()
        self.samples.clear()
        self.path_states.clear()
        self.verdicts.clear()
        self.verdict_cursor = 0
        self.commands.clear()
        self.processed_samples = 0
        self.classifier_class = None
        self.classifier_state = DetectorState.IDLE.value
        self.queue = queue.Queue(maxsize=512)
        self.transport_finished = False
        self._open_transport()

    def switch(self, selection: str) -> None:
        self.stop()
        self.selection = selection
        self.stream = build_stream(selection, self.options)
        self.detector = TurnDetector(model_for_runtime())
        self.pipeline = EdgePipeline(self.detector, ExecutionConfig(realtime=False))
        self.samples.clear()
        self.path_states.clear()
        self.verdicts.clear()
        self.verdict_cursor = 0
        self.commands.clear()
        self.processed_samples = 0
        self.classifier_class = None
        self.classifier_state = DetectorState.IDLE.value
        self.queue = queue.Queue(maxsize=512)
        self.runner = self._new_runner()
        self.runner.set_speed(self.speed)
        self.transport_finished = False

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)
        self.runner.set_speed(self.speed)

    def event(self, action: str, name: str) -> None:
        if not self.supports_events:
            return
        method = {
            "add": self.runner.add_event,
            "remove": self.runner.remove_event,
            "repair": self.runner.repair_event,
        }[action]
        method(name)

    def drain(self, limit: int = MAX_UI_BATCH) -> list[tuple[Any, str]]:
        """Drain a bounded batch so one dashboard callback cannot monopolize the UI."""
        if limit <= 0:
            raise ValueError("drain limit must be positive")
        drained = []
        while len(drained) < limit:
            try:
                sample, state = self.queue.get_nowait()
            except queue.Empty:
                break
            self.samples.append(sample)
            self.path_states.append(state)
            drained.append((sample, state))
        if len(self.samples) > MAX_DISPLAY_SAMPLES:
            del self.samples[:-MAX_DISPLAY_SAMPLES]
            del self.path_states[:-MAX_DISPLAY_SAMPLES]
        self.commands = list(getattr(self.stream.schedule, "history", ()))
        return drained

    def display_samples(self) -> tuple[list, list]:
        samples = self.samples[-MAX_DISPLAY_SAMPLES:]
        states = self.path_states[-MAX_DISPLAY_SAMPLES:]
        stride = max(1, int(np.ceil(len(samples) / MAX_PLOT_POINTS)))
        return samples[::stride], states[::stride]

    def new_verdicts(self) -> list[tuple[float, Any]]:
        values = self.verdicts[self.verdict_cursor:]
        self.verdict_cursor = len(self.verdicts)
        return values

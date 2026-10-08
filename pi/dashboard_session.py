"""Shared operator session for the Streamlit PC UI."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
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
CLASS_LABELS = {
    "nominal": "No Change",
    "A": "Implement Attached",
    "B": "Flat Tire",
    "AB": "Implement + Flat Tire",
}
PERTURBATION_NAMES = {
    "A": "Implement attached",
    "B": "Flat tire",
    "AB": "Implement attached and flat tire",
    "IMPLEMENT_ATTACHED": "Implement attached",
    "TIRE_FLAT": "Tire fault",
}
MAX_DISPLAY_SAMPLES = 5000
MAX_PLOT_POINTS = 600
MAX_UI_BATCH = 128
LIVE_UPDATE_SECONDS = 0.5
TRANSPORT_QUEUE_SIZE = 512
PATH_COLORS = {
    DetectorState.IDLE.value: "#1f77b4",
    DetectorState.IN_TURN.value: "#ff7f0e",
    DetectorState.WAITING_WINDOW.value: "#d62728",
}
PATH_LABELS = {
    DetectorState.IDLE.value: "Straight / No Turn",
    DetectorState.IN_TURN.value: "Turning",
    DetectorState.WAITING_WINDOW.value: "Analyzing Turn",
}


class AsyncTcpTransport:
    """Send samples on a dedicated thread without blocking the producer."""

    def __init__(self, host: str, port: int, *, queue_size: int = TRANSPORT_QUEUE_SIZE):
        self.sender = TcpSampleSender(host, port)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._sentinel = object()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None
        self.state = "Not connected"
        self.message = f"Ready for {host}:{port}"
        self.enqueued_samples = 0
        self.sent_samples = 0
        self.last_sent_sequence: int | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.error = None
        self.state = "Connecting"
        self.message = f"Connecting to {self.sender.host}:{self.sender.port}"
        self._thread = threading.Thread(
            target=self._run,
            name="tcp-sample-sender",
            daemon=True,
        )
        self._thread.start()

    def send(self, sample) -> None:
        if self.error is not None:
            raise ConnectionError(self.message) from self.error
        if self._stop.is_set():
            raise ConnectionError("TCP transport is stopped")
        try:
            self._queue.put(sample, timeout=self.sender.timeout_s)
        except queue.Full as exc:
            self._fail("TCP transport queue is full; Pi is not keeping up", exc)
            raise ConnectionError(self.message) from exc
        self.enqueued_samples += 1

    def finish(self) -> None:
        if self._thread is None:
            self.close()
            return
        if self.error is None and not self._stop.is_set():
            try:
                self._queue.put(self._sentinel, timeout=self.sender.timeout_s)
            except queue.Full as exc:
                self._fail("TCP transport could not drain before completion", exc)
        self._thread.join(timeout=self.sender.timeout_s + 1.0)
        if self._thread.is_alive():
            self._fail("TCP sender thread did not stop", TimeoutError(self.message))
        self.close()

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            pass
        self.sender.close()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if self.error is None:
            self.state = "Disconnected"
            self.message = "TCP transport closed"

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is self._sentinel:
                    self.sender.finish()
                    self.state = "Complete"
                    self.message = (
                        f"Sent {self.sent_samples} samples to "
                        f"{self.sender.host}:{self.sender.port}"
                    )
                    return
                self.sender.send(item)
                self.sent_samples += 1
                self.last_sent_sequence = item.sequence
                self.state = "Connected"
                self.message = (
                    f"Sent {self.sent_samples}/{self.enqueued_samples} samples"
                )
        except Exception as exc:
            self._fail(f"TCP transport failed: {exc}", exc)
        finally:
            self.sender.close()

    def _fail(self, message: str, error: Exception) -> None:
        if self.error is None:
            self.error = error
            self.state = "Disconnected"
            self.message = message
        self._stop.set()
        self.sender.close()


def trajectory_options() -> dict[str, Path | None]:
    options: dict[str, Path | None] = {"Python default": None}
    folders = (
        ROOT / "data" / "generated_trajectories" / "matlab",
        ROOT / "data" / "generated_trajectories",
    )
    for folder in folders:
        for path in sorted(folder.glob("*.mat")):
            demo_name = path.stem.lower()
            if demo_name in {"demo_a", "demo_b", "demo_ab"}:
                label = f"MATLAB demo {demo_name.removeprefix('demo_').upper()}"
                options.setdefault(label, path)
    return options


def build_stream(selection: str, options: dict[str, Path | None]):
    path = options[selection]
    return PlantStream.from_seed() if path is None else PlaybackStream.from_mat(path)


def model_for_runtime():
    path = ROOT / "models" / "export" / "weights.mat"
    return load_weights(path) if path.exists() else dummy_model()


class DashboardSession:
    """Own one operator run and expose bounded data for the PC UI."""

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
        self.transport_state = "Local playback"
        self.transport_message = "Pi host is not configured"
        self.capture_path: Path | None = None
        self.transport: AsyncTcpTransport | None = None
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
        """Return a worker failure in a form the UI can display."""
        if not self.runner.errors:
            return None
        error = self.runner.errors[-1]
        return f"{type(error).__name__}: {error}"

    def configure_transport(self, host: str, port: int, capture: str) -> None:
        new_host = host.strip()
        new_port = int(port)
        changed = (new_host, new_port) != (self.tcp_host, self.tcp_port)
        self.tcp_host = new_host
        self.tcp_port = new_port
        self.capture_path = Path(capture) if capture.strip() else None
        if changed and self.tcp_host:
            self.transport_state = "Not connected"
            self.transport_message = f"Ready for {self.tcp_host}:{self.tcp_port}"
        elif changed or not self.tcp_host:
            self.transport_state = "Local playback"
            self.transport_message = "Pi host is not configured"

    def _open_transport(self) -> None:
        self.transport = (
            AsyncTcpTransport(self.tcp_host, self.tcp_port)
            if self.tcp_host
            else None
        )
        if self.transport:
            self.transport_state = self.transport.state
            self.transport_message = self.transport.message
        else:
            self.transport_state = "Local playback"
            self.transport_message = "Samples stay on the PC"
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
        if self.transport:
            self.transport.send(sample)
            self._sync_transport_status()
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
        try:
            if self.transport:
                try:
                    self.transport.finish()
                except Exception as exc:
                    self.runner.errors.append(exc)
                if self.transport.error is not None and not self.runner.errors:
                    self.runner.errors.append(
                        ConnectionError(self.transport.message)
                    )
                self._sync_transport_status()
            if self.capture:
                self.capture.close()
            if self.event_log:
                self.event_log.write(self.stream.schedule.history)
        finally:
            self.transport_finished = True
            if not self.transport:
                self.transport_state = "Complete"
                self.transport_message = "Local playback complete"

    def _sync_transport_status(self) -> None:
        if not self.transport:
            return
        self.transport_state = self.transport.state
        self.transport_message = self.transport.message

    def transport_stats(self) -> dict[str, object]:
        """Return transport counters for the live operator view."""
        if not self.transport:
            return {
                "enqueued": 0,
                "sent": 0,
                "last_sequence": None,
                "queued": 0,
            }
        return {
            "enqueued": self.transport.enqueued_samples,
            "sent": self.transport.sent_samples,
            "last_sequence": self.transport.last_sent_sequence,
            "queued": self.transport.queue_depth,
        }

    def start(self) -> None:
        self._open_transport()
        if self.transport:
            self.transport.start()
        self.transport_finished = False
        self.runner.start()

    def pause(self) -> None:
        self.runner.pause()

    def resume(self) -> None:
        self.runner.resume()

    def stop(self) -> None:
        self.runner.stop()
        if self.transport:
            self.transport.close()
        self._on_complete()

    def retry(self) -> None:
        """Restart a failed run from sample zero with a fresh transport."""
        self.reset()
        self.start()

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

    def perturbation_status(self) -> dict[str, object]:
        """Return read-only perturbation truth from the selected playback."""
        metadata = getattr(self.stream, "metadata", {})
        class_name = str(metadata.get("className", "nominal")).upper()
        change_time = metadata.get("changeTimeS")

        if class_name in PERTURBATION_NAMES and change_time is not None:
            change_time_s = float(change_time)
            active = self.stream.time_s >= change_time_s
            details = []
            added_mass = metadata.get("addedMassKg")
            kf = metadata.get("kf")
            if class_name in ("A", "AB") and added_mass is not None:
                details.append(f"+{float(added_mass):.0f} kg implement mass")
            if class_name in ("B", "AB") and kf is not None:
                details.append(f"front tire stiffness ×{float(kf):.2f}")
            return {
                "state": "PERTURBED" if active else "NOMINAL",
                "name": PERTURBATION_NAMES[class_name],
                "time_s": change_time_s,
                "details": ", ".join(details),
            }

        active_events = set()
        if self.samples:
            active_events = set(
                filter(
                    None,
                    self.samples[-1].diagnostics.get("active_events", "").split(","),
                )
            )
        names = [PERTURBATION_NAMES.get(name, name) for name in active_events]
        return {
            "state": "PERTURBED" if names else "NOMINAL",
            "name": ", ".join(names) if names else "None",
            "time_s": None,
            "details": "Runtime event status" if names else "No playback perturbation metadata",
        }

    def configuration_status(self) -> dict[str, bool]:
        """Return the current implement and tire states for the selected playback."""
        metadata = getattr(self.stream, "metadata", {})
        class_name = str(metadata.get("className", "nominal")).upper()
        change_time = metadata.get("changeTimeS")
        if class_name in {"A", "B", "AB"} and change_time is not None:
            active = self.stream.time_s >= float(change_time)
            return {
                "implement": active and class_name in {"A", "AB"},
                "tire": active and class_name in {"B", "AB"},
            }

        active_events = set()
        if self.samples:
            active_events = set(
                filter(
                    None,
                    self.samples[-1].diagnostics.get("active_events", "").split(","),
                )
            )
        return {
            "implement": bool(active_events & {"implement_attached", "A", "AB"}),
            "tire": bool(active_events & {"tire_flat", "B", "AB"}),
        }

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
        """Drain a bounded batch so one UI update cannot monopolize the process."""
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

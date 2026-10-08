"""Stateful PC-side plant streaming and runtime event control."""

from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Callable, Iterable

import numpy as np

try:
    from .contracts import MeasuredSample
    from .local_plant import (
        EVENT_ALIASES,
        NOMINAL_PARAMS,
        VehicleEvent,
        _changed_parameters,
        make_disturbance,
        make_maneuver,
    )
    from .plant import StreamingPlant
except ImportError:  # pragma: no cover
    from contracts import MeasuredSample
    from local_plant import (
        EVENT_ALIASES,
        NOMINAL_PARAMS,
        VehicleEvent,
        _changed_parameters,
        make_disturbance,
        make_maneuver,
    )
    from plant import StreamingPlant


EVENT_NAMES = ("implement_attached", "tire_flat", "A", "B", "AB")


@dataclass(frozen=True)
class EventCommand:
    action: str
    name: str
    timestamp_s: float
    duration_s: float = 0.0
    added_mass_kg: float = 1000.0
    kf: float = 0.8


@dataclass(frozen=True)
class EventTransition:
    event: VehicleEvent
    target_active: bool

    def progress(self, time_s: float) -> float:
        if self.target_active:
            return self.event.progress(time_s)
        reverse = VehicleEvent(
            name=self.event.name,
            start_s=self.event.start_s,
            end_s=self.event.end_s,
            reverse=True,
            added_mass_kg=self.event.added_mass_kg,
            mount_x_m=self.event.mount_x_m,
            kf=self.event.kf,
        )
        return reverse.progress(time_s)


class RuntimeEventSchedule:
    """Mutable event schedule whose commands take effect at sample boundaries."""

    def __init__(self, base_params: Iterable[float] = NOMINAL_PARAMS):
        self.base_params = tuple(float(value) for value in base_params)
        self._transitions: dict[str, EventTransition] = {}
        self.history: list[EventCommand] = []

    def reset(self) -> None:
        self._transitions.clear()
        self.history.clear()

    def command(
        self,
        action: str,
        name: str,
        timestamp_s: float,
        *,
        duration_s: float = 0.0,
        added_mass_kg: float = 1000.0,
        kf: float = 0.8,
    ) -> EventCommand:
        if action not in ("add", "remove", "repair"):
            raise ValueError("action must be add, remove, or repair")
        if name not in EVENT_NAMES:
            raise ValueError(f"unsupported event {name!r}")
        if timestamp_s < 0 or duration_s < 0:
            raise ValueError("event times must be non-negative")
        if action == "repair":
            action = "remove"
        event = VehicleEvent(
            name=name,
            start_s=timestamp_s,
            end_s=timestamp_s + duration_s if duration_s else None,
            added_mass_kg=added_mass_kg,
            kf=kf,
        )
        self._transitions[name] = EventTransition(
            event=event,
            target_active=action == "add",
        )
        command = EventCommand(
            action=action,
            name=name,
            timestamp_s=timestamp_s,
            duration_s=duration_s,
            added_mass_kg=added_mass_kg,
            kf=kf,
        )
        self.history.append(command)
        return command

    def parameter_at(self, time_s: float) -> list[float]:
        params = np.asarray(self.base_params, dtype=float)
        for transition in self._transitions.values():
            progress = transition.progress(time_s)
            if progress == 0:
                continue
            target = _changed_parameters(params, transition.event)
            if not transition.target_active:
                target = np.asarray(self.base_params, dtype=float)
            params = params + progress * (target - params)
        return params.tolist()

    def active_events(self, time_s: float) -> tuple[str, ...]:
        return tuple(
            name
            for name, transition in self._transitions.items()
            if transition.progress(time_s) > 0
        )


class PlantStream:
    """Generate one ``MeasuredSample`` per call from a deterministic maneuver."""

    def __init__(
        self,
        delta: Iterable[float],
        vx: Iterable[float],
        fyd: Iterable[float],
        mzd: Iterable[float],
        *,
        sample_rate_hz: float = 100.0,
        schedule: RuntimeEventSchedule | None = None,
    ):
        self.delta = np.asarray(tuple(delta), dtype=float)
        self.vx = np.asarray(tuple(vx), dtype=float)
        self.fyd = np.asarray(tuple(fyd), dtype=float)
        self.mzd = np.asarray(tuple(mzd), dtype=float)
        lengths = {len(self.delta), len(self.vx), len(self.fyd), len(self.mzd)}
        if len(lengths) != 1 or not lengths:
            raise ValueError("all plant input arrays must have equal non-zero lengths")
        self.sample_rate_hz = sample_rate_hz
        self.period_s = 1.0 / sample_rate_hz
        self.schedule = schedule or RuntimeEventSchedule()
        self.plant = StreamingPlant(
            parameter_schedule=self.schedule.parameter_at,
            h=self.period_s,
        )
        self.supports_events = True
        self.index = 0

    @classmethod
    def from_seed(
        cls,
        *,
        profile_seed: int = 1,
        disturbance_seed: int = 1,
        sample_rate_hz: float = 100.0,
        swaths: int = 3,
    ) -> "PlantStream":
        delta, vx = make_maneuver(
            seed=profile_seed,
            fs=sample_rate_hz,
            swaths=swaths,
        )
        fyd, mzd = make_disturbance(len(delta), sample_rate_hz, disturbance_seed)
        return cls(delta, vx, fyd, mzd, sample_rate_hz=sample_rate_hz)

    @property
    def finished(self) -> bool:
        return self.index >= len(self.delta)

    @property
    def time_s(self) -> float:
        return self.index * self.period_s

    def reset(self) -> None:
        self.index = 0
        self.plant.reset()
        self.schedule.reset()

    def step(self) -> MeasuredSample | None:
        if self.finished:
            return None
        index = self.index
        current = (self.delta[index], self.vx[index], self.fyd[index], self.mzd[index])
        next_index = min(index + 1, len(self.delta) - 1)
        following = (
            self.delta[next_index],
            self.vx[next_index],
            self.fyd[next_index],
            self.mzd[next_index],
        )
        output = self.plant.step(current, following)
        diagnostics = {
            "ay": float(output["ay"]),
            "params": output["params"].tolist(),
            "X": float(output["X"]),
            "Y": float(output["Y"]),
            "ydot": float(output["ydot"]),
            "alphaF": float(output["alphaF"]),
            "psi": float(output["psi"]),
            "active_events": ",".join(self.schedule.active_events(self.time_s)),
            "event_commands": len(getattr(self.schedule, "history", ())),
        }
        sample = MeasuredSample(
            delta=float(self.delta[index]),
            vx=float(self.vx[index]),
            yaw_rate=float(output["r"]),
            timestamp_s=self.time_s,
            sequence=index,
            diagnostics=diagnostics,
        )
        self.index += 1
        return sample


class PlaybackStream:
    """Replay a fixed measured trajectory without regenerating its plant output."""

    def __init__(
        self,
        measured: np.ndarray,
        *,
        sample_rate_hz: float,
        diagnostics: dict[str, Iterable] | None = None,
        metadata: dict[str, object] | None = None,
    ):
        measured = np.asarray(measured, dtype=float)
        if measured.ndim != 2 or measured.shape[1] < 3 or measured.shape[0] == 0:
            raise ValueError("measured playback must contain non-empty [delta, Vx, r] rows")
        self.measured = measured[:, :3]
        self.sample_rate_hz = float(sample_rate_hz)
        self.period_s = 1.0 / self.sample_rate_hz
        self.diagnostics = {
            name: list(values) for name, values in (diagnostics or {}).items()
        }
        self.metadata = dict(metadata or {})
        for name, values in self.diagnostics.items():
            if len(values) != len(self.measured):
                raise ValueError(f"diagnostic {name!r} has the wrong length")
        self.schedule = RuntimeEventSchedule()
        self.supports_events = False
        self.index = 0

    @classmethod
    def from_mat(cls, path) -> "PlaybackStream":
        from scipy.io import loadmat

        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        measured = np.asarray(data["meas"], dtype=float)
        plant = data.get("plant")
        diagnostics = {}
        if plant is not None:
            for name in ("r", "ay", "X", "Y", "ydot", "alphaF", "psi", "params"):
                value = getattr(plant, name, None)
                if value is not None:
                    diagnostics[name] = np.asarray(value)
        fs = float(np.asarray(data["fs"]).reshape(-1)[0])
        metadata = {}
        meta = data.get("meta")
        if meta is not None:
            for name in ("scenario", "className", "changeTurn", "changeTimeS",
                         "addedMassKg", "kf", "noiseSeed"):
                value = getattr(meta, name, None)
                if value is not None:
                    array = np.asarray(value).reshape(-1)
                    metadata[name] = array[0].item() if array.size else None
        return cls(
            measured,
            sample_rate_hz=fs,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    @property
    def finished(self) -> bool:
        return self.index >= len(self.measured)

    @property
    def time_s(self) -> float:
        return self.index * self.period_s

    def reset(self) -> None:
        self.index = 0
        self.schedule.reset()

    def step(self) -> MeasuredSample | None:
        if self.finished:
            return None
        index = self.index
        diagnostics = {
            name: values[index].tolist()
            if isinstance(values[index], np.ndarray)
            else float(values[index])
            for name, values in self.diagnostics.items()
        }
        diagnostics["active_events"] = ""
        diagnostics["event_commands"] = 0
        sample = MeasuredSample(
            delta=float(self.measured[index, 0]),
            vx=float(self.measured[index, 1]),
            yaw_rate=float(self.measured[index, 2]),
            timestamp_s=self.time_s,
            sequence=index,
            diagnostics=diagnostics,
        )
        self.index += 1
        return sample


class StreamingRunner:
    """Run a ``PlantStream`` on a worker thread with command/control methods."""

    def __init__(
        self,
        stream: PlantStream,
        on_sample: Callable[[MeasuredSample], None] | None = None,
        on_complete: Callable[[], None] | None = None,
        *,
        realtime: bool = True,
    ):
        self.stream = stream
        self.on_sample = on_sample
        self.on_complete = on_complete
        self.realtime = realtime
        self._commands: queue.Queue[tuple[str, tuple, dict]] = queue.Queue()
        self._running = threading.Event()
        self._paused = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[Exception] = []
        self.speed = 1.0

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def paused(self) -> bool:
        return not self._paused.is_set()

    def start(self) -> None:
        if self.running:
            return
        self.errors.clear()
        self._stop_requested.clear()
        self._paused.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def set_speed(self, speed: float) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("playback speed must be positive")
        self.speed = float(speed)

    def stop(self) -> None:
        self._stop_requested.set()
        self._paused.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def reset(self) -> None:
        self.pause()
        self.stream.reset()

    def add_event(self, name: str, duration_s: float = 0.0, **kwargs) -> None:
        self._commands.put(("add", (name, duration_s), kwargs))

    def remove_event(self, name: str, duration_s: float = 0.0, **kwargs) -> None:
        self._commands.put(("remove", (name, duration_s), kwargs))

    def repair_event(self, name: str, duration_s: float = 0.0, **kwargs) -> None:
        self._commands.put(("repair", (name, duration_s), kwargs))

    def _apply_commands(self) -> None:
        while True:
            try:
                action, args, kwargs = self._commands.get_nowait()
            except queue.Empty:
                return
            self.stream.schedule.command(
                action,
                args[0],
                self.stream.time_s,
                duration_s=args[1],
                **kwargs,
            )

    def _run(self) -> None:
        self._running.set()
        try:
            while not self._stop_requested.is_set() and not self.stream.finished:
                self._paused.wait()
                if self._stop_requested.is_set():
                    break
                self._apply_commands()
                started = time.monotonic()
                sample = self.stream.step()
                if sample is not None and self.on_sample:
                    self.on_sample(sample)
                if self.realtime:
                    target_period = self.stream.period_s / max(self.speed, 0.01)
                    remaining = target_period - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
                    else:
                        # At high playback rates the worker is compute-bound. Yield
                        # explicitly so the Dash server can service poll callbacks.
                        time.sleep(0)
        except Exception as exc:  # surfaced to the UI/test owner
            self.errors.append(exc)
        finally:
            if self.on_complete:
                self.on_complete()
            self._running.clear()

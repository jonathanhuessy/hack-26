"""Deterministic co-located plant source for the Phase 2 Pi demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from scipy.signal import butter, lfilter

try:
    from .plant import simulate
    from .sources import ArraySampleSource
except ImportError:
    from plant import simulate
    from sources import ArraySampleSource


NOMINAL_PARAMS = (
    10178.6,
    1.6562,
    1.1538,
    10178.6 * 1.6562 * 1.1538,
    356356.4449,
    270000.0,
    1.10369051,
)
WHEELBASE_M = 2.81
IMU_X_FROM_REAR_AXLE_M = 0.0


EVENT_ALIASES = {
    "implement_attached": "A",
    "tire_flat": "B",
}


@dataclass(frozen=True)
class VehicleEvent:
    """A deterministic transition to one of the trained parameter changes."""

    name: str
    start_s: float
    end_s: float | None = None
    reverse: bool = False
    added_mass_kg: float = 1000.0
    mount_x_m: float = -1.2
    kf: float = 0.8

    def __post_init__(self) -> None:
        if self.name not in ("A", "B", "AB", *EVENT_ALIASES):
            raise ValueError(f"unsupported vehicle event {self.name!r}")
        if self.start_s < 0:
            raise ValueError("event start_s must be non-negative")
        if self.end_s is not None and self.end_s < self.start_s:
            raise ValueError("event end_s must be at or after start_s")
        if self.added_mass_kg < 0:
            raise ValueError("added_mass_kg must be non-negative")
        if self.kf <= 0:
            raise ValueError("kf must be positive")

    @property
    def scenario_name(self) -> str:
        return EVENT_ALIASES.get(self.name, self.name)

    def progress(self, t_s: float) -> float:
        """Return the event activation fraction at a time in seconds."""
        end_s = self.end_s if self.end_s is not None else self.start_s
        if t_s < self.start_s:
            value = 1.0 if self.reverse else 0.0
        elif end_s <= self.start_s:
            value = 0.0 if self.reverse else 1.0
        elif t_s >= end_s:
            value = 0.0 if self.reverse else 1.0
        else:
            fraction = (t_s - self.start_s) / (end_s - self.start_s)
            value = 1.0 - fraction if self.reverse else fraction
        return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class EventSchedule:
    """Compose deterministic trained changes over a nominal parameter vector."""

    base_params: tuple[float, ...]
    events: tuple[VehicleEvent, ...] = ()

    def __post_init__(self) -> None:
        if len(self.base_params) != 7:
            raise ValueError("base_params must contain seven plant parameters")

    def parameter_at(self, t_s: float) -> list[float]:
        params = np.asarray(self.base_params, dtype=float)
        for event in self.events:
            progress = event.progress(t_s)
            if progress == 0.0:
                continue
            target = _changed_parameters(params, event)
            params = params + progress * (target - params)
        return params.tolist()

    def active_events(self, t_s: float) -> tuple[str, ...]:
        return tuple(
            event.name for event in self.events if event.progress(t_s) > 0.0
        )


def _changed_parameters(base: Sequence[float], event: VehicleEvent) -> np.ndarray:
    """Apply one trained A/B-style change to a parameter vector."""
    params = np.asarray(base, dtype=float).copy()
    scenario = event.scenario_name
    if scenario in ("A", "AB"):
        mass, _, lr, izz, _, _, _ = params
        new_lr = (mass * lr + event.added_mass_kg * event.mount_x_m) / (
            mass + event.added_mass_kg
        )
        params[0] = mass + event.added_mass_kg
        params[2] = new_lr
        params[1] = WHEELBASE_M - new_lr
        params[3] = izz + mass * (lr - new_lr) ** 2 + event.added_mass_kg * (
            event.mount_x_m - new_lr
        ) ** 2
    if scenario in ("B", "AB"):
        params[4] *= event.kf
    return params


def event_preset(
    name: str,
    *,
    start_s: float,
    end_s: float | None = None,
    reverse: bool = False,
    added_mass_kg: float = 1000.0,
    mount_x_m: float = -1.2,
    kf: float = 0.8,
) -> VehicleEvent:
    """Create a named event using the same dimensions as model training."""
    return VehicleEvent(
        name=name,
        start_s=start_s,
        end_s=end_s,
        reverse=reverse,
        added_mass_kg=added_mass_kg,
        mount_x_m=mount_x_m,
        kf=kf,
    )


@dataclass(frozen=True)
class LocalScenario:
    """Scenario controls shared by offline and realtime local demos."""

    name: str = "nominal"
    change_type: str = "constant"
    added_mass_kg: float = 1000.0
    mount_x_m: float = -1.2
    kf: float = 0.8
    t_start_s: float = 0.0
    t_end_s: float = 0.0
    reverse: bool = False
    profile_seed: int = 1
    disturbance_seed: int = 1
    sample_rate_hz: float = 100.0
    swaths: int = 3
    events: tuple[VehicleEvent, ...] | None = None


def scenario_parameters(
    scenario: LocalScenario,
) -> tuple[list[float], list[float]]:
    """Return MATLAB-compatible start and end parameter vectors."""
    base = np.asarray(NOMINAL_PARAMS, dtype=float)
    scenario_name = EVENT_ALIASES.get(scenario.name, scenario.name)
    if scenario_name not in ("nominal", "A", "B", "AB"):
        raise ValueError(f"unsupported scenario {scenario.name!r}")
    if scenario_name == "nominal":
        return base.tolist(), base.tolist()
    changed = _changed_parameters(
        base,
        VehicleEvent(
            name=scenario_name,
            start_s=0.0,
            added_mass_kg=scenario.added_mass_kg,
            mount_x_m=scenario.mount_x_m,
            kf=scenario.kf,
        ),
    )

    if scenario.change_type == "constant":
        return changed.tolist(), changed.tolist()
    if scenario.reverse:
        return changed.tolist(), base.tolist()
    return base.tolist(), changed.tolist()


def scenario_schedule(scenario: LocalScenario, duration_s: float) -> EventSchedule:
    """Build the parameter schedule used by a local scenario."""
    if scenario.events is not None:
        return EventSchedule(tuple(NOMINAL_PARAMS), scenario.events)
    scenario_name = EVENT_ALIASES.get(scenario.name, scenario.name)
    if scenario_name == "nominal":
        return EventSchedule(tuple(NOMINAL_PARAMS))
    if scenario.change_type not in ("constant", "step", "ramp"):
        raise ValueError(f"unsupported change_type {scenario.change_type!r}")
    if scenario.change_type == "constant":
        end_s = None
        start_s = 0.0
    elif scenario.change_type == "step":
        start_s = scenario.t_start_s
        end_s = start_s
    else:
        start_s = scenario.t_start_s
        end_s = scenario.t_end_s if scenario.t_end_s else duration_s
    event = event_preset(
        scenario_name,
        start_s=start_s,
        end_s=end_s,
        reverse=scenario.reverse and scenario.change_type != "constant",
        added_mass_kg=scenario.added_mass_kg,
        mount_x_m=scenario.mount_x_m,
        kf=scenario.kf,
    )
    return EventSchedule(tuple(NOMINAL_PARAMS), (event,))


def _raised_cosine_transition(start: float, end: float, duration_s: float,
                              fs: float) -> np.ndarray:
    n = max(2, round(duration_s * fs))
    phase = np.linspace(0.0, 1.0, n)
    return start + (end - start) * 0.5 * (1.0 - np.cos(np.pi * phase))


def make_maneuver(
    *,
    seed: int = 1,
    fs: float = 100.0,
    swaths: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic straight/deceleration/turn maneuver."""
    rng = np.random.default_rng(seed)
    straight_speed = rng.uniform(13.0, 16.0) / 3.6
    turn_speed = rng.uniform(4.0, 6.0) / 3.6
    radius = rng.uniform(8.5, 11.0)
    straight_length = rng.uniform(80.0, 120.0)
    kus = NOMINAL_PARAMS[0] / WHEELBASE_M * (
        NOMINAL_PARAMS[2] / NOMINAL_PARAMS[4]
        - NOMINAL_PARAMS[1] / NOMINAL_PARAMS[5]
    )
    plateau = (WHEELBASE_M + kus * turn_speed**2) / radius

    speeds: list[np.ndarray] = []
    steering: list[np.ndarray] = []
    for turn_index in range(swaths):
        n_straight = max(1, round(straight_length / straight_speed * fs))
        n_transition = max(1, round((straight_speed - turn_speed) / 0.5 * fs))
        n_turn = max(2, round(np.pi * radius / turn_speed * fs))
        n_accel = n_transition
        direction = 1.0 if turn_index % 2 == 0 else -1.0
        n_ramp = max(2, round(fs))
        n_plateau = max(0, n_turn - 2 * n_ramp)
        speeds.extend((
            np.full(n_straight, straight_speed),
            np.linspace(straight_speed, turn_speed, n_transition),
            np.full(n_turn, turn_speed),
            np.linspace(turn_speed, straight_speed, n_accel),
        ))
        steering.extend((
            np.zeros(n_straight),
            np.zeros(n_transition),
            _raised_cosine_transition(0.0, direction * plateau, n_ramp / fs, fs),
            np.full(n_plateau, direction * plateau),
            _raised_cosine_transition(direction * plateau, 0.0, n_ramp / fs, fs),
            np.zeros(n_accel),
        ))
    return np.concatenate(steering), np.concatenate(speeds)


def make_disturbance(length: int, fs: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate the same kind of band-limited force and yaw disturbance as MATLAB."""
    rng = np.random.default_rng(seed)
    warm = round(60 * fs)
    b, a = butter(2, np.asarray((0.05, 2.0)) / (fs / 2.0), btype="bandpass")
    noise = lfilter(b, a, rng.normal(size=(length + warm, 2)), axis=0)[warm:]
    noise /= np.maximum(noise.std(axis=0), 1e-12)
    return 800.0 * noise[:, 0], 400.0 * noise[:, 1]


def build_source(
    scenario: LocalScenario | None = None,
    *,
    realtime: bool = False,
) -> ArraySampleSource:
    """Build a source backed by the local plant and optional truth diagnostics."""
    scenario = scenario or LocalScenario()
    if scenario.change_type not in ("constant", "step", "ramp"):
        raise ValueError(f"unsupported change_type {scenario.change_type!r}")
    delta, vx = make_maneuver(
        seed=scenario.profile_seed,
        fs=scenario.sample_rate_hz,
        swaths=scenario.swaths,
    )
    fyd, mzd = make_disturbance(
        len(delta), scenario.sample_rate_hz, scenario.disturbance_seed
    )
    duration_s = (len(delta) - 1) / scenario.sample_rate_hz
    schedule = scenario_schedule(scenario, duration_s)
    output = simulate(
        delta,
        vx,
        fyd,
        mzd,
        list(NOMINAL_PARAMS),
        imu_x=IMU_X_FROM_REAR_AXLE_M,
        h=1.0 / scenario.sample_rate_hz,
        parameter_schedule=schedule.parameter_at,
    )
    diagnostics = {
        "ay": output["ay"],
        "params": output["params"],
        "X": output["X"],
        "Y": output["Y"],
        "ydot": output["ydot"],
        "alphaF": output["alphaF"],
        "psi": output["psi"],
        "scenario": np.full(len(delta), scenario.name, dtype=object),
        "active_events": np.asarray(
            [
                ",".join(schedule.active_events(i / scenario.sample_rate_hz))
                for i in range(len(delta))
            ],
            dtype=object,
        ),
    }
    return ArraySampleSource(
        delta,
        vx,
        output["r"],
        diagnostics=diagnostics,
        sample_period_s=1.0 / scenario.sample_rate_hz,
        realtime=realtime,
    )


def iter_local_samples(
    scenario: LocalScenario | None = None,
    *,
    realtime: bool = False,
) -> Iterator:
    """Convenience iterator used by callers that only need samples."""
    yield from build_source(scenario, realtime=realtime)

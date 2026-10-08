"""Deterministic co-located plant source for the Phase 2 Pi demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

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


def scenario_parameters(
    scenario: LocalScenario,
) -> tuple[list[float], list[float]]:
    """Return MATLAB-compatible start and end parameter vectors."""
    base = np.asarray(NOMINAL_PARAMS, dtype=float)
    changed = base.copy()
    if scenario.name in ("A", "AB"):
        dm = scenario.added_mass_kg
        old_lr = base[2]
        new_lr = (base[0] * old_lr + dm * scenario.mount_x_m) / (base[0] + dm)
        changed[0] = base[0] + dm
        changed[2] = new_lr
        changed[1] = WHEELBASE_M - new_lr
        changed[3] = (
            base[3]
            + base[0] * (old_lr - new_lr) ** 2
            + dm * (scenario.mount_x_m - new_lr) ** 2
        )
    if scenario.name in ("B", "AB"):
        changed[4] = base[4] * scenario.kf
    if scenario.name not in ("nominal", "A", "B", "AB"):
        raise ValueError(f"unsupported scenario {scenario.name!r}")

    if scenario.change_type == "constant":
        return changed.tolist(), changed.tolist()
    if scenario.reverse:
        return changed.tolist(), base.tolist()
    return base.tolist(), changed.tolist()


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
    p0, p1 = scenario_parameters(scenario)
    if scenario.change_type == "constant":
        t_start = t_end = 0.0
    elif scenario.change_type == "step":
        t_start = scenario.t_start_s
        t_end = t_start
    else:
        default_end = (len(delta) - 1) / scenario.sample_rate_hz
        t_start = scenario.t_start_s
        t_end = scenario.t_end_s if scenario.t_end_s else default_end
    output = simulate(
        delta,
        vx,
        fyd,
        mzd,
        p0,
        p1,
        t_start,
        t_end,
        IMU_X_FROM_REAR_AXLE_M,
        1.0 / scenario.sample_rate_hz,
    )
    diagnostics = {
        "ay": output["ay"],
        "params": output["params"],
        "scenario": np.full(len(delta), scenario.name, dtype=object),
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

"""Validate stateful plant streaming and detector replay against references."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

try:
    from .detector import TurnDetector
    from .model import load_weights
    from .pipeline import EdgePipeline
    from .sources import MatlabFeatureReplaySource
    from .streaming import PlantStream
    from .plant import params_at
except ImportError:  # pragma: no cover
    from detector import TurnDetector
    from model import load_weights
    from pipeline import EdgePipeline
    from sources import MatlabFeatureReplaySource
    from streaming import PlantStream
    from plant import params_at


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "test_vectors"
PLANT_FIXTURE = FIXTURE_DIR / "plant_run.mat"
WEIGHTS = ROOT / "models" / "export" / "weights.mat"
LIMIT = 1e-6


class ReferenceSchedule:
    """Adapter for the single MATLAB parameter schedule in plant_run.mat."""

    def __init__(self, p0, p1, start_s, end_s):
        self.p0 = list(np.asarray(p0, dtype=float).reshape(-1))
        self.p1 = list(np.asarray(p1, dtype=float).reshape(-1))
        self.start_s = float(start_s)
        self.end_s = float(end_s)

    def parameter_at(self, time_s: float) -> list[float]:
        return params_at(
            time_s,
            self.p0,
            self.p1,
            self.start_s,
            self.end_s,
        )

    def active_events(self, time_s: float) -> tuple[str, ...]:
        return ()

    def reset(self) -> None:
        pass


def _relative_error(actual, expected) -> float:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    scale = max(float(np.max(np.abs(expected))), 1e-12)
    return float(np.max(np.abs(actual - expected)) / scale)


def validate_stateful_plant() -> bool:
    reference = loadmat(PLANT_FIXTURE, squeeze_me=True)
    schedule = ReferenceSchedule(
        reference["p0"],
        reference["p1"],
        reference["tStart"],
        reference["tEnd"],
    )
    stream = PlantStream(
        reference["delta"],
        reference["Vx"],
        reference["Fyd"],
        reference["Mzd"],
        sample_rate_hz=100.0,
        schedule=schedule,
    )
    samples = []
    while not stream.finished:
        sample = stream.step()
        if sample is not None:
            samples.append(sample)
    checks = {
        "r": [sample.yaw_rate for sample in samples],
        "ay": [sample.diagnostics["ay"] for sample in samples],
        "ydot": [sample.diagnostics["ydot"] for sample in samples],
        "alphaF": [sample.diagnostics["alphaF"] for sample in samples],
        "X": [sample.diagnostics["X"] for sample in samples],
        "Y": [sample.diagnostics["Y"] for sample in samples],
        "psi": [sample.diagnostics["psi"] for sample in samples],
        "params": [sample.diagnostics["params"] for sample in samples],
    }
    ok = True
    for name, actual in checks.items():
        expected = reference[name]
        error = _relative_error(actual, expected)
        ok &= error < LIMIT
        print(f"plant {name:7s} max relative difference {error:.2e}")
    print("stateful plant:", "PASS" if ok else "FAIL")
    return ok


def validate_detector_replays() -> bool:
    if not WEIGHTS.exists():
        print(f"missing model artifact: {WEIGHTS}")
        return False
    model = load_weights(WEIGHTS)
    ok = True
    for name in ("nominal", "A", "B", "AB", "step", "ramp"):
        path = FIXTURE_DIR / f"detector_{name}.mat"
        if not path.exists():
            print(f"detector {name:7s} SKIP (missing {path.name})")
            continue
        reference = loadmat(path, squeeze_me=True)
        events = EdgePipeline(TurnDetector(model)).run(
            MatlabFeatureReplaySource(path)
        )
        verdicts = [
            event.consumer_result.new_verdict
            for event in events
            if event.consumer_result is not None
            and event.consumer_result.new_verdict is not None
        ]
        expected_classes = ("nominal", "A", "B", "AB")
        expected = [
            expected_classes[int(index) - 1]
            for index in np.asarray(reference["classIndex"]).reshape(-1)
        ]
        actual = [verdict.class_name for verdict in verdicts]
        case_ok = actual == expected
        ok &= case_ok
        print(
            f"detector {name:7s} {'PASS' if case_ok else 'FAIL'} "
            f"verdicts={len(verdicts)} classes={actual}"
        )
    print("detector replays:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if not PLANT_FIXTURE.exists():
        print(f"missing plant reference: {PLANT_FIXTURE}")
        return 1
    return 0 if validate_stateful_plant() and validate_detector_replays() else 1


if __name__ == "__main__":
    sys.exit(main())

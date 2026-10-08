import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from .detector import TurnDetector
    from .features import selected_features
    from .model import load_weights
    from .pipeline import EdgePipeline
    from .sources import MatlabFeatureReplaySource
except ImportError:
    from detector import TurnDetector
    from features import selected_features
    from model import load_weights
    from pipeline import EdgePipeline
    from sources import MatlabFeatureReplaySource


FIXTURE_DIR = Path(__file__).parent / "test_vectors"
WEIGHTS = FIXTURE_DIR.parent.parent / "models" / "export" / "weights.mat"
FEATURE_RTOL = 1e-6
MODEL_RTOL = 1e-5
ABS_TOL = 1e-8


def _fixture(name: str) -> tuple[dict, Path]:
    path = FIXTURE_DIR / f"detector_{name}.mat"
    return loadmat(path, squeeze_me=True), path


def _verdicts(path: Path):
    model = load_weights(WEIGHTS)
    events = EdgePipeline(TurnDetector(model)).run(MatlabFeatureReplaySource(path))
    return [
        event.consumer_result.new_verdict
        for event in events
        if event.consumer_result is not None
        and event.consumer_result.new_verdict is not None
    ]


def _optional_values(verdicts, field: str) -> np.ndarray:
    return np.asarray(
        [getattr(verdict, field) if getattr(verdict, field) is not None else np.nan
         for verdict in verdicts],
        dtype=float,
    )


class DetectorParityTests(unittest.TestCase):
    def test_features_and_model_match_matlab_references(self):
        model = load_weights(WEIGHTS)
        for name in ("nominal", "A", "B", "AB", "step", "ramp"):
            with self.subTest(case=name):
                data, _ = _fixture(name)
                measured = np.asarray(data["meas"], dtype=float)[:, :3]
                actual_features = []
                for i0, i1, s0, s1 in zip(
                    data["i0"], data["i1"], data["s0"], data["s1"]
                ):
                    window = measured[int(i0) - 1:int(i1), :]
                    straight = measured[int(s0) - 1:int(s1), :]
                    actual_features.append(selected_features(window, straight, data["fs"]))
                actual_features = np.asarray(actual_features)
                expected_features = np.asarray(data["X_selected"], dtype=float)
                np.testing.assert_allclose(
                    actual_features,
                    expected_features,
                    rtol=FEATURE_RTOL,
                    atol=ABS_TOL,
                    err_msg=f"feature parity failed for {name}",
                )

                actual_probabilities = np.asarray([
                    model.predict(row).class_probabilities
                    for row in expected_features
                ])
                np.testing.assert_allclose(
                    actual_probabilities,
                    np.asarray(data["per_turn_probabilities"], dtype=float),
                    rtol=MODEL_RTOL,
                    atol=ABS_TOL,
                    err_msg=f"classifier parity failed for {name}",
                )
                actual_regression = np.asarray([
                    [prediction.delta_m_kg or np.nan, prediction.k_f or np.nan]
                    for prediction in (model.predict(row) for row in expected_features)
                ])
                np.testing.assert_allclose(
                    actual_regression,
                    np.asarray(data["per_turn_regression"], dtype=float),
                    rtol=MODEL_RTOL,
                    atol=ABS_TOL,
                    equal_nan=True,
                    err_msg=f"regression parity failed for {name}",
                )

    def test_streaming_verdicts_match_matlab(self):
        for name in ("nominal", "A", "B", "AB", "step", "ramp"):
            with self.subTest(case=name):
                data, path = _fixture(name)
                verdicts = _verdicts(path)
                self.assertEqual(len(verdicts), len(np.asarray(data["turnIndex"])), name)
                np.testing.assert_array_equal(
                    [verdict.turn_index + 1 for verdict in verdicts],
                    np.asarray(data["turnIndex"], dtype=int),
                    err_msg=f"turn index parity failed for {name}",
                )
                np.testing.assert_allclose(
                    [verdict.timestamp_s for verdict in verdicts],
                    np.asarray(data["timestamp_s"], dtype=float),
                    rtol=0,
                    atol=ABS_TOL,
                    err_msg=f"verdict timing parity failed for {name}",
                )
                np.testing.assert_array_equal(
                    [("nominal", "A", "B", "AB").index(verdict.class_name) + 1
                     for verdict in verdicts],
                    np.asarray(data["classIndex"], dtype=int),
                    err_msg=f"class verdict parity failed for {name}",
                )
                np.testing.assert_allclose(
                    [verdict.class_probabilities for verdict in verdicts],
                    np.asarray(data["verdict_probabilities"], dtype=float),
                    rtol=MODEL_RTOL,
                    atol=ABS_TOL,
                    err_msg=f"aggregated probability parity failed for {name}",
                )
                np.testing.assert_allclose(
                    _optional_values(verdicts, "delta_m_kg"),
                    np.asarray(data["verdict_deltaM_kg"], dtype=float),
                    rtol=MODEL_RTOL,
                    atol=ABS_TOL,
                    equal_nan=True,
                    err_msg=f"aggregated mass parity failed for {name}",
                )
                np.testing.assert_allclose(
                    _optional_values(verdicts, "k_f"),
                    np.asarray(data["verdict_kF"], dtype=float),
                    rtol=MODEL_RTOL,
                    atol=ABS_TOL,
                    equal_nan=True,
                    err_msg=f"aggregated stiffness parity failed for {name}",
                )

    def test_step_and_ramp_replays_are_deterministic_offline(self):
        for name in ("step", "ramp"):
            with self.subTest(case=name):
                _, path = _fixture(name)
                first = _verdicts(path)
                second = _verdicts(path)
                self.assertEqual(
                    [(v.turn_index, v.class_name, v.timestamp_s) for v in first],
                    [(v.turn_index, v.class_name, v.timestamp_s) for v in second],
                )


if __name__ == "__main__":
    unittest.main()

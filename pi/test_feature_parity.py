import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from .features import FEATURE_NAMES, selected_features
except ImportError:
    from features import FEATURE_NAMES, selected_features


FIXTURE_DIR = Path(__file__).parent / "test_vectors"


class FeatureParityTests(unittest.TestCase):
    def test_selected_features_match_matlab_vectors(self):
        for name in ("nominal", "A", "B", "AB"):
            data = loadmat(FIXTURE_DIR / f"features_{name}.mat", squeeze_me=True)
            measured = np.asarray(data["meas"], dtype=float)[:, :3]
            actual = []
            for i0, i1, s0, s1 in zip(data["i0"], data["i1"], data["s0"], data["s1"]):
                window = measured[int(i0) - 1:int(i1), :]
                straight = (
                    measured[int(s0) - 1:int(s1), :]
                    if int(s0) > 0 else np.empty((0, 3))
                )
                actual.append(selected_features(window, straight, float(data["fs"])))
            expected = np.asarray(data["X"])[:, np.asarray(data["selectedIdx"]).astype(int) - 1]
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-8)
            self.assertEqual(len(FEATURE_NAMES), expected.shape[1])


if __name__ == "__main__":
    unittest.main()

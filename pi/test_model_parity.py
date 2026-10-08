import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

try:
    from .model import dummy_model, load_weights
except ImportError:
    from model import dummy_model, load_weights


class ModelParityTests(unittest.TestCase):
    def test_dummy_model_is_deterministic_and_gated(self):
        model = dummy_model()
        prediction = model.predict(np.arange(18, dtype=float))
        np.testing.assert_allclose(prediction.class_probabilities, np.full(4, 0.25))
        self.assertIsNone(prediction.delta_m_kg)
        self.assertIsNone(prediction.k_f)

    def test_matlab_weight_shapes_load_and_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.mat"
            savemat(path, {
                "mu": np.zeros(2),
                "sigma": np.ones(2),
                "W1": np.eye(2),
                "b1": np.zeros(2),
                "W2": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float),
                "b2": np.zeros(4),
                "featureNames": "a,b",
                "feature_version": 1,
            })
            prediction = load_weights(path).predict(np.array([1.0, 2.0]))
            np.testing.assert_allclose(prediction.class_probabilities.sum(), 1.0)
            self.assertEqual(prediction.class_probabilities.shape, (4,))

    def test_exported_weights_match_matlab_fixture_outputs(self):
        artifact = Path(__file__).parent.parent / "models" / "export" / "weights.mat"
        fixture = Path(__file__).parent / "test_vectors" / "features_nominal.mat"
        if not artifact.exists() or not fixture.exists():
            self.skipTest("MATLAB-exported artifact is not available")
        model = load_weights(artifact)
        data = loadmat(fixture, squeeze_me=True)
        selected = np.asarray(data["selectedIdx"]).astype(int) - 1
        expected = np.asarray(data["P"], dtype=float)
        actual = np.asarray([
            model.predict(row[selected]).class_probabilities
            for row in np.asarray(data["X"], dtype=float)
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

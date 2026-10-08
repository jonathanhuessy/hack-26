"""NumPy model inference and exported MATLAB weight loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat


CLASS_NAMES = ("nominal", "A", "B", "AB")
FEATURE_VERSION = 1


@dataclass(frozen=True)
class Prediction:
    class_probabilities: np.ndarray
    delta_m_kg: float | None
    k_f: float | None
    class_name: str


@dataclass
class ModelArtifact:
    mu: np.ndarray
    sigma: np.ndarray
    classifier: tuple[np.ndarray, ...]
    regressor: tuple[np.ndarray, ...] | None = None
    feature_names: tuple[str, ...] = ()
    feature_version: int = FEATURE_VERSION

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        if self.mu.size != self.sigma.size:
            raise ValueError("mu and sigma must have equal lengths")
        if np.any(~np.isfinite(self.mu)) or np.any(~np.isfinite(self.sigma)):
            raise ValueError("normalization values must be finite")
        if np.any(self.sigma == 0):
            raise ValueError("sigma must not contain zero values")
        if len(self.classifier) not in (2, 4, 6):
            raise ValueError("classifier must contain one to three weight/bias layer pairs")

    def predict(self, features: np.ndarray) -> Prediction:
        x = np.asarray(features, dtype=float).reshape(-1)
        if x.size != self.mu.size:
            raise ValueError(f"expected {self.mu.size} features, got {x.size}")
        normalized = (x - self.mu) / self.sigma
        logits = _forward_layers(normalized, self.classifier)
        probabilities = _softmax(logits)
        class_index = int(np.argmax(probabilities))
        regression = _forward_layers(normalized, self.regressor) if self.regressor else None
        delta_m = None
        k_f = None
        if regression is not None and regression.size >= 2:
            if class_index in (1, 3):
                delta_m = float(regression[0])
            if class_index in (2, 3):
                k_f = float(regression[1])
        return Prediction(probabilities, delta_m, k_f, CLASS_NAMES[class_index])


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _forward_layers(x: np.ndarray, layers: tuple[np.ndarray, ...] | None) -> np.ndarray:
    if not layers:
        raise ValueError("model layers are missing")
    value = np.asarray(x, dtype=float).reshape(-1)
    for index in range(0, len(layers), 2):
        weights = np.asarray(layers[index], dtype=float)
        bias = np.asarray(layers[index + 1], dtype=float).reshape(-1)
        if weights.shape[1] != value.size:
            raise ValueError(
                f"weight input width {weights.shape[1]} does not match {value.size}"
            )
        value = weights @ value + bias
        if index + 2 < len(layers):
            value = _relu(value)
    return value


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def dummy_model(feature_count: int = 18) -> ModelArtifact:
    """Return deterministic uniform classifier and zero regression outputs."""
    return ModelArtifact(
        mu=np.zeros(feature_count),
        sigma=np.ones(feature_count),
        classifier=(
            np.zeros((8, feature_count)),
            np.zeros(8),
            np.zeros((4, 8)),
            np.zeros(4),
        ),
        regressor=(
            np.zeros((2, feature_count)),
            np.zeros(2),
        ),
        feature_names=tuple(f"feature_{i}" for i in range(feature_count)),
    )


def _field(data: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = data.get(name, default)
    if value is None:
        return default
    return value


def load_weights(path: str | Path) -> ModelArtifact:
    """Load the Phase 0/Phase 1 MATLAB weight artifact."""
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    mu = _field(raw, "mu")
    sigma = _field(raw, "sigma")
    if mu is None or sigma is None:
        raise ValueError("weights artifact must contain mu and sigma")
    classifier_names = ("W1", "b1", "W2", "b2", "W3", "b3")
    classifier = tuple(np.asarray(raw[name], dtype=float) for name in classifier_names if name in raw)
    if len(classifier) not in (2, 4, 6):
        raise ValueError("weights artifact has incomplete classifier layers")
    regressor_names = ("V1", "c1", "V2", "c2", "V3", "c3")
    reg_values = tuple(np.asarray(raw[name], dtype=float) for name in regressor_names if name in raw)
    regressor = reg_values if len(reg_values) == 6 else None
    names_raw = _field(raw, "featureNames", "")
    if isinstance(names_raw, str):
        feature_names = tuple(name for name in names_raw.split(",") if name)
    else:
        feature_names = tuple(str(name) for name in np.asarray(names_raw).reshape(-1))
    version = int(np.asarray(_field(raw, "feature_version", FEATURE_VERSION)).reshape(-1)[0])
    return ModelArtifact(mu, sigma, classifier, regressor, feature_names, version)

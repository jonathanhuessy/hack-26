"""Deterministic sensor-noise adapter for local Phase 1 demonstrations."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Iterator

import numpy as np

try:
    from .contracts import MeasuredSample, SampleSource
except ImportError:
    from contracts import MeasuredSample, SampleSource


def add_noise(
    sample: MeasuredSample,
    rng: np.random.Generator,
    *,
    delta_std_rad: float = np.deg2rad(0.2),
    vx_std_mps: float = 0.01,
    yaw_rate_std_rad_s: float = 0.01,
) -> MeasuredSample:
    """Add noise only to required measured channels; preserve metadata."""
    return replace(
        sample,
        delta=sample.delta + float(rng.normal(0.0, delta_std_rad)),
        vx=sample.vx + float(rng.normal(0.0, vx_std_mps)),
        yaw_rate=sample.yaw_rate + float(rng.normal(0.0, yaw_rate_std_rad_s)),
    )


class NoisySource:
    def __init__(self, source: SampleSource, seed: int = 1, **noise_kwargs: float):
        self.source = source
        self.rng = np.random.default_rng(seed)
        self.noise_kwargs = noise_kwargs

    def __iter__(self) -> Iterator[MeasuredSample]:
        for sample in self.source:
            yield add_noise(sample, self.rng, **self.noise_kwargs)

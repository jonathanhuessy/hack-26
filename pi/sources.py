"""Local and replay sample sources for the Phase 0 edge contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

try:
    from .contracts import MeasuredSample, SAMPLE_PERIOD_S, SCHEMA_VERSION
except ImportError:  # Allows `python pi/...py` from the repository root.
    from contracts import MeasuredSample, SAMPLE_PERIOD_S, SCHEMA_VERSION


@dataclass
class ArraySampleSource:
    """Adapt sampled arrays from a local plant or MATLAB export."""

    delta: Sequence[float]
    vx: Sequence[float]
    yaw_rate: Sequence[float]
    diagnostics: Mapping[str, Sequence[Any]] | None = None
    sample_period_s: float = SAMPLE_PERIOD_S
    realtime: bool = False

    def __iter__(self) -> Iterator[MeasuredSample]:
        lengths = {len(self.delta), len(self.vx), len(self.yaw_rate)}
        if len(lengths) != 1:
            raise ValueError("delta, vx, and yaw_rate must have equal lengths")
        diagnostics = self.diagnostics or {}
        for sequence, (delta, vx, yaw_rate) in enumerate(
            zip(self.delta, self.vx, self.yaw_rate)
        ):
            if self.realtime and sequence:
                time.sleep(self.sample_period_s)
            item_diagnostics = {
                name: values[sequence]
                for name, values in diagnostics.items()
                if sequence < len(values)
            }
            yield MeasuredSample(
                delta=float(delta),
                vx=float(vx),
                yaw_rate=float(yaw_rate),
                timestamp_s=sequence * self.sample_period_s,
                sequence=sequence,
                diagnostics=item_diagnostics,
            )

    @classmethod
    def from_plant(
        cls,
        inputs: Mapping[str, Sequence[float]],
        output: Mapping[str, Sequence[float]],
        diagnostics: Mapping[str, Sequence[Any]] | None = None,
        **kwargs: Any,
    ) -> "ArraySampleSource":
        """Build a source from ``plant.simulate`` inputs and outputs."""
        optional = dict(diagnostics or {})
        for name in ("ay", "ydot", "alphaF", "X", "Y", "psi", "params"):
            if name in output:
                optional.setdefault(name, output[name])
        return cls(inputs["delta"], inputs["Vx"], output["r"], optional, **kwargs)


@dataclass
class ReplaySampleSource:
    """Replay a JSON Phase 0 fixture without MATLAB or networking."""

    path: str | Path
    realtime: bool = False

    def __iter__(self) -> Iterator[MeasuredSample]:
        envelope = json.loads(Path(self.path).read_text(encoding="utf-8"))
        if envelope.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported replay schema version")
        samples = envelope.get("samples")
        if not isinstance(samples, list):
            raise ValueError("replay fixture must contain a samples list")
        for item in samples:
            if not isinstance(item, dict):
                raise ValueError("each replay sample must be an object")
            diagnostics = item.get("diagnostics", {})
            if self.realtime and item.get("sequence", 0):
                time.sleep(float(envelope.get("sample_period_s", SAMPLE_PERIOD_S)))
            yield MeasuredSample(
                delta=float(item["delta"]),
                vx=float(item["Vx"]),
                yaw_rate=float(item["r"]),
                timestamp_s=float(item["timestamp_s"]),
                sequence=int(item["sequence"]),
                schema_version=int(item.get("schema_version", SCHEMA_VERSION)),
                diagnostics=diagnostics,
            )

"""Source-independent Phase 0 contracts for the Raspberry Pi edge pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Iterator, Mapping, Protocol


SCHEMA_VERSION = 1
UNITS_VERSION = 1
FEATURE_VERSION = 1
MODEL_VERSION = 0
SAMPLE_PERIOD_S = 0.01
TIMING_TOLERANCE_S = 0.002
CHANNEL_ORDER = ("delta", "Vx", "r")


class SampleStatus(str, Enum):
    ACCEPTED = "accepted"
    GAP = "gap"
    LATE = "late"
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    RESET = "reset"


class SampleValidationError(ValueError):
    """Raised when a sample violates the normative input contract."""

    def __init__(self, message: str, status: SampleStatus = SampleStatus.INVALID):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class MeasuredSample:
    """The only sample shape exposed to an edge consumer."""

    delta: float
    vx: float
    yaw_rate: float
    timestamp_s: float
    sequence: int
    schema_version: int = SCHEMA_VERSION
    diagnostics: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def channels(self) -> tuple[float, float, float]:
        return (self.delta, self.vx, self.yaw_rate)

    def edge_payload(self) -> tuple[float, float, float]:
        """Return required channels only; diagnostics cannot cross the seam."""
        return self.channels


@dataclass(frozen=True)
class ValidationPolicy:
    sample_period_s: float = SAMPLE_PERIOD_S
    timing_tolerance_s: float = TIMING_TOLERANCE_S
    require_zero_origin: bool = True


@dataclass(frozen=True)
class SampleResult:
    status: SampleStatus
    sample: MeasuredSample | None
    message: str = ""


@dataclass(frozen=True)
class ExecutionConfig:
    realtime: bool = False
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    deterministic_seed: int | None = None


class SampleSource(Protocol):
    def __iter__(self) -> Iterator[MeasuredSample]:
        ...


class SampleSink(Protocol):
    def start(self) -> None:
        ...

    def push(self, sample: MeasuredSample) -> Any:
        ...

    def flush(self) -> Any:
        ...

    def reset(self) -> None:
        ...

    def close(self) -> None:
        ...


def validate_sample(sample: MeasuredSample, policy: ValidationPolicy | None = None) -> None:
    """Validate fields that are intrinsic to one sample."""
    policy = policy or ValidationPolicy()
    if sample.schema_version != SCHEMA_VERSION:
        raise SampleValidationError(
            f"unsupported schema_version {sample.schema_version}; expected {SCHEMA_VERSION}"
        )
    if sample.sequence < 0:
        raise SampleValidationError("sequence must be non-negative")
    values = {"delta": sample.delta, "Vx": sample.vx, "r": sample.yaw_rate}
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value))
               for value in values.values()):
        raise SampleValidationError("delta, Vx, and r must be finite numeric values")
    if not math.isfinite(sample.timestamp_s):
        raise SampleValidationError("timestamp_s must be finite")
    if policy.require_zero_origin and sample.sequence == 0 and abs(sample.timestamp_s) > policy.timing_tolerance_s:
        raise SampleValidationError("the first sample must start at timestamp zero")


def compare_sample(
    previous: MeasuredSample | None,
    sample: MeasuredSample,
    policy: ValidationPolicy | None = None,
) -> SampleStatus:
    """Return the deterministic stream status for a sample."""
    policy = policy or ValidationPolicy()
    validate_sample(sample, policy)
    if previous is None:
        return SampleStatus.ACCEPTED
    if sample.sequence == previous.sequence or sample.timestamp_s == previous.timestamp_s:
        return SampleStatus.DUPLICATE
    if sample.sequence < previous.sequence or sample.timestamp_s < previous.timestamp_s:
        return SampleStatus.LATE
    expected_dt = policy.sample_period_s * (sample.sequence - previous.sequence)
    actual_dt = sample.timestamp_s - previous.timestamp_s
    if abs(actual_dt - expected_dt) > policy.timing_tolerance_s:
        return SampleStatus.GAP
    return SampleStatus.ACCEPTED


def validate_batch(
    samples: Iterable[MeasuredSample],
    policy: ValidationPolicy | None = None,
) -> list[SampleResult]:
    """Validate a stream while retaining a result for every input sample."""
    previous: MeasuredSample | None = None
    results: list[SampleResult] = []
    for sample in samples:
        try:
            status = compare_sample(previous, sample, policy)
        except SampleValidationError as exc:
            results.append(SampleResult(exc.status, None, str(exc)))
            continue
        if status in (SampleStatus.ACCEPTED, SampleStatus.GAP):
            previous = sample
            message = "timestamp/sequence gap" if status is SampleStatus.GAP else ""
            results.append(SampleResult(status, sample, message))
        else:
            results.append(SampleResult(status, None, f"sample rejected: {status.value}"))
    return results

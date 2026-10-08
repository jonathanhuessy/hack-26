"""Versioned, newline-delimited JSON wire format for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

try:
    from ..contracts import (
        CHANNEL_ORDER,
        FEATURE_VERSION,
        MODEL_VERSION,
        SCHEMA_VERSION,
        UNITS_VERSION,
        MeasuredSample,
    )
except ImportError:  # pragma: no cover - supports direct module execution
    from contracts import (  # type: ignore
        CHANNEL_ORDER,
        FEATURE_VERSION,
        MODEL_VERSION,
        SCHEMA_VERSION,
        UNITS_VERSION,
        MeasuredSample,
    )


PROTOCOL_VERSION = 1
MESSAGE_HELLO = "hello"
MESSAGE_HELLO_ACK = "hello_ack"
MESSAGE_SAMPLE = "sample"
MESSAGE_END = "end"


class WireFormatError(ValueError):
    """Raised when a wire record is malformed or unsafe to consume."""


class CompatibilityError(WireFormatError):
    """Raised when a producer and consumer cannot share a stream."""


def _reject_constant(value: str) -> None:
    raise WireFormatError(f"non-finite JSON constant {value!r} is not allowed")


def _loads(line: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(
            line.decode("utf-8") if isinstance(line, bytes) else line,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireFormatError(f"invalid JSON record: {exc}") from exc
    if not isinstance(value, dict):
        raise WireFormatError("wire record must be a JSON object")
    return value


def _dumps(value: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise WireFormatError(f"record is not JSON-serializable: {exc}") from exc


def _json_safe(value: Any) -> Any:
    """Convert array-scalar diagnostics without weakening JSON validation."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    return value


@dataclass(frozen=True)
class Handshake:
    """Compatibility declaration exchanged before sample records."""

    role: str
    protocol_version: int = PROTOCOL_VERSION
    schema_version: int = SCHEMA_VERSION
    units_version: int = UNITS_VERSION
    feature_version: int = FEATURE_VERSION
    model_version: int = MODEL_VERSION
    channel_order: tuple[str, ...] = CHANNEL_ORDER
    sample_period_s: float = 0.01
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self, message_type: str = MESSAGE_HELLO) -> dict[str, Any]:
        return {
            "type": message_type,
            "role": self.role,
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "units_version": self.units_version,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "channel_order": list(self.channel_order),
            "sample_period_s": self.sample_period_s,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Handshake":
        if record.get("type") not in (MESSAGE_HELLO, MESSAGE_HELLO_ACK):
            raise WireFormatError("expected hello or hello_ack record")
        try:
            order = tuple(record["channel_order"])
            handshake = cls(
                role=str(record["role"]),
                protocol_version=int(record["protocol_version"]),
                schema_version=int(record["schema_version"]),
                units_version=int(record["units_version"]),
                feature_version=int(record["feature_version"]),
                model_version=int(record["model_version"]),
                channel_order=order,
                sample_period_s=float(record["sample_period_s"]),
                metadata=record.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WireFormatError(f"invalid handshake: {exc}") from exc
        if not isinstance(handshake.metadata, Mapping):
            raise WireFormatError("handshake metadata must be an object")
        return handshake

    def validate_compatible(self, expected: "Handshake") -> None:
        fields = (
            "protocol_version",
            "schema_version",
            "units_version",
            "feature_version",
            "model_version",
            "channel_order",
        )
        for name in fields:
            actual = getattr(self, name)
            wanted = getattr(expected, name)
            if actual != wanted:
                raise CompatibilityError(
                    f"incompatible {name}: received {actual!r}, expected {wanted!r}"
                )


def serialize_handshake(handshake: Handshake, message_type: str = MESSAGE_HELLO) -> bytes:
    return _dumps(handshake.to_record(message_type))


def deserialize_handshake(line: bytes | str) -> Handshake:
    return Handshake.from_record(_loads(line))


def serialize_sample(sample: MeasuredSample) -> bytes:
    """Serialize one sample as one newline-delimited JSON record."""
    return _dumps(
        {
            "type": MESSAGE_SAMPLE,
            "schema_version": sample.schema_version,
            "timestamp_s": sample.timestamp_s,
            "sequence": sample.sequence,
            "delta": sample.delta,
            "Vx": sample.vx,
            "r": sample.yaw_rate,
            "diagnostics": _json_safe(dict(sample.diagnostics)),
        }
    )


def deserialize_sample(line: bytes | str) -> MeasuredSample:
    """Decode a sample and reject missing, non-finite, or extra-channel ambiguity."""
    record = _loads(line)
    if record.get("type") != MESSAGE_SAMPLE:
        raise WireFormatError("expected a sample record")
    required = ("schema_version", "timestamp_s", "sequence", "delta", "Vx", "r")
    missing = [name for name in required if name not in record]
    if missing:
        raise WireFormatError(f"sample is missing required fields: {', '.join(missing)}")
    diagnostics = record.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise WireFormatError("sample diagnostics must be an object")
    try:
        return MeasuredSample(
            delta=float(record["delta"]),
            vx=float(record["Vx"]),
            yaw_rate=float(record["r"]),
            timestamp_s=float(record["timestamp_s"]),
            sequence=int(record["sequence"]),
            schema_version=int(record["schema_version"]),
            diagnostics=dict(diagnostics),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise WireFormatError(f"invalid sample value: {exc}") from exc


def serialize_control(message_type: str) -> bytes:
    if message_type not in (MESSAGE_END,):
        raise ValueError(f"unsupported control message {message_type!r}")
    return _dumps({"type": message_type})


def record_type(line: bytes | str) -> str:
    """Return a record type without interpreting a sample payload."""
    value = _loads(line).get("type")
    if not isinstance(value, str):
        raise WireFormatError("wire record type must be a string")
    return value

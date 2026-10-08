"""Phase 4 transport adapters for the source-independent Pi pipeline."""

from .wire import (
    PROTOCOL_VERSION,
    CompatibilityError,
    Handshake,
    WireFormatError,
    deserialize_sample,
    serialize_handshake,
    serialize_sample,
)
from .adapters import (
    FaultPolicy,
    FileSampleSink,
    FileSampleSource,
    LoopbackSampleSource,
    LoopbackTransport,
)
from .tcp import TcpConfig, TcpSampleSender, TcpSampleSource

__all__ = [
    "PROTOCOL_VERSION",
    "CompatibilityError",
    "Handshake",
    "WireFormatError",
    "deserialize_sample",
    "serialize_handshake",
    "serialize_sample",
    "FaultPolicy",
    "FileSampleSink",
    "FileSampleSource",
    "LoopbackSampleSource",
    "LoopbackTransport",
    "TcpConfig",
    "TcpSampleSender",
    "TcpSampleSource",
]

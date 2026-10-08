"""Deterministic loopback and file transports used before live networking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import queue
import random
import time
from typing import Iterator

from .wire import (
    Handshake,
    MESSAGE_END,
    deserialize_handshake,
    deserialize_sample,
    record_type,
    serialize_control,
    serialize_handshake,
    serialize_sample,
)

try:
    from ..contracts import MeasuredSample
except ImportError:  # pragma: no cover
    from contracts import MeasuredSample  # type: ignore


@dataclass(frozen=True)
class FaultPolicy:
    """Repeatable transport faults for integration tests."""

    drop_every: int | None = None
    duplicate_every: int | None = None
    delay_s: float = 0.0
    reorder_window: int = 0
    seed: int = 1


@dataclass
class TransportStatus:
    kind: str
    message: str
    timestamp_s: float = field(default_factory=time.monotonic)


class LoopbackEndpoint:
    def __init__(self, maxsize: int = 256):
        self._queue: queue.Queue[MeasuredSample | None] = queue.Queue(maxsize=maxsize)
        self.statuses: list[TransportStatus] = []
        self.closed = False

    def put(self, sample: MeasuredSample, timeout_s: float | None = None) -> None:
        if self.closed:
            raise ConnectionError("loopback endpoint is closed")
        self._queue.put(sample, timeout=timeout_s)

    def get(self, timeout_s: float | None = None) -> MeasuredSample | None:
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError("loopback receive timed out") from exc

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._queue.put(None)


class LoopbackTransport:
    """A bounded source/sink pair with optional deterministic faults."""

    def __init__(
        self,
        endpoint: LoopbackEndpoint,
        *,
        fault_policy: FaultPolicy | None = None,
        put_timeout_s: float | None = None,
    ):
        self.endpoint = endpoint
        self.policy = fault_policy or FaultPolicy()
        self.put_timeout_s = put_timeout_s
        self._count = 0
        self._random = random.Random(self.policy.seed)
        self._reorder_buffer: list[MeasuredSample] = []

    @classmethod
    def pair(
        cls, maxsize: int = 256, fault_policy: FaultPolicy | None = None
    ) -> tuple["LoopbackTransport", "LoopbackSampleSource"]:
        endpoint = LoopbackEndpoint(maxsize)
        return cls(endpoint, fault_policy=fault_policy), LoopbackSampleSource(endpoint)

    def send(self, sample: MeasuredSample) -> None:
        self._count += 1
        if self.policy.delay_s:
            time.sleep(self.policy.delay_s)
        if self.policy.drop_every and self._count % self.policy.drop_every == 0:
            self.endpoint.statuses.append(
                TransportStatus("dropped", f"dropped sequence {sample.sequence}")
            )
            return
        self._reorder_buffer.append(sample)
        if len(self._reorder_buffer) <= max(0, self.policy.reorder_window):
            return
        outgoing = self._reorder_buffer.pop(0)
        if self.policy.reorder_window and self._random.random() < 0.5:
            outgoing = self._reorder_buffer.pop() if self._reorder_buffer else outgoing
        self.endpoint.put(outgoing, self.put_timeout_s)
        if self.policy.duplicate_every and self._count % self.policy.duplicate_every == 0:
            self.endpoint.put(outgoing, self.put_timeout_s)
            self.endpoint.statuses.append(
                TransportStatus("duplicated", f"duplicated sequence {outgoing.sequence}")
            )

    def close(self) -> None:
        while self._reorder_buffer:
            self.endpoint.put(self._reorder_buffer.pop(0), self.put_timeout_s)
        self.endpoint.close()


class LoopbackSampleSource:
    def __init__(self, endpoint: LoopbackEndpoint, timeout_s: float | None = None):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def __iter__(self) -> Iterator[MeasuredSample]:
        while True:
            sample = self.endpoint.get(self.timeout_s)
            if sample is None:
                return
            yield sample


class FileSampleSink:
    """Write a transport stream that can be replayed without a network."""

    def __init__(self, path: str | Path, handshake: Handshake | None = None):
        self.path = Path(path)
        self.handshake = handshake or Handshake(role="pc_sender")
        self._file = None

    def start(self) -> None:
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
        self._file.write(serialize_handshake(self.handshake).decode("utf-8"))

    def send(self, sample: MeasuredSample) -> None:
        if self._file is None:
            self.start()
        self._file.write(serialize_sample(sample).decode("utf-8"))
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.write(serialize_control(MESSAGE_END).decode("utf-8"))
            self._file.close()
            self._file = None


class FileEventLogSink:
    """Write runtime event commands beside a transport capture."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, commands) -> None:
        records = [asdict(command) for command in commands]
        self.path.write_text(
            json.dumps(records, indent=2) + "\n",
            encoding="utf-8",
        )


class FileSampleSource:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_handshake: Handshake | None = None,
        realtime: bool = False,
    ):
        self.path = Path(path)
        self.expected_handshake = expected_handshake
        self.realtime = realtime
        self.handshake: Handshake | None = None
        self.statuses: list[TransportStatus] = []

    def __iter__(self) -> Iterator[MeasuredSample]:
        previous_timestamp: float | None = None
        with self.path.open("rb") as stream:
            first = stream.readline()
            self.handshake = deserialize_handshake(first)
            if self.expected_handshake:
                self.handshake.validate_compatible(self.expected_handshake)
            for line in stream:
                if not line.strip():
                    continue
                kind = record_type(line)
                if kind == MESSAGE_END:
                    return
                sample = deserialize_sample(line)
                if self.realtime and previous_timestamp is not None:
                    time.sleep(max(0.0, sample.timestamp_s - previous_timestamp))
                previous_timestamp = sample.timestamp_s
                yield sample

"""TCP sender and receiver for the separated PC/Pi demo."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import time
from typing import Iterable, Iterator

from .adapters import TransportStatus
from .wire import (
    Handshake,
    MESSAGE_END,
    MESSAGE_HELLO_ACK,
    MESSAGE_SAMPLE,
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
class TcpConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    backlog: int = 1
    socket_timeout_s: float = 0.25
    stale_timeout_s: float = 2.0
    reconnect: bool = False


class TcpSampleSender:
    """PC-side sender; TCP provides ordered delivery and back-pressure."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        handshake: Handshake | None = None,
        timeout_s: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.handshake = handshake or Handshake(role="pc_sender")
        self.timeout_s = timeout_s
        self.socket: socket.socket | None = None
        self.statuses: list[TransportStatus] = []

    def connect(self) -> None:
        if self.socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), self.timeout_s)
        sock.settimeout(self.timeout_s)
        stream = sock.makefile("rwb")
        stream.write(serialize_handshake(self.handshake))
        stream.flush()
        response = stream.readline()
        received = deserialize_handshake(response)
        received.validate_compatible(self.handshake)
        if received.role != "pi_receiver":
            raise ConnectionError(f"unexpected handshake role {received.role!r}")
        self.socket = sock
        self._stream = stream
        self.statuses.append(TransportStatus("connected", f"{self.host}:{self.port}"))

    def send(self, sample: MeasuredSample) -> None:
        self.connect()
        assert self.socket is not None
        try:
            self._stream.write(serialize_sample(sample))
            self._stream.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self.statuses.append(TransportStatus("disconnected", str(exc)))
            self.close()
            raise ConnectionError("TCP receiver disconnected") from exc

    def send_samples(
        self,
        samples: Iterable[MeasuredSample],
        *,
        realtime: bool = False,
    ) -> None:
        try:
            previous_timestamp: float | None = None
            for sample in samples:
                if realtime and previous_timestamp is not None:
                    time.sleep(max(0.0, sample.timestamp_s - previous_timestamp))
                self.send(sample)
                previous_timestamp = sample.timestamp_s
            self.finish()
        finally:
            self.close()

    def finish(self) -> None:
        """Terminate a live stream cleanly before closing its socket."""
        if self.socket is not None:
            self._stream.write(serialize_control(MESSAGE_END))
            self._stream.flush()

    def close(self) -> None:
        if self.socket is not None:
            try:
                self._stream.close()
            finally:
                self.socket.close()
                self.socket = None


class TcpSampleSource:
    """Pi-side source that converts a TCP stream into ``MeasuredSample`` objects."""

    def __init__(
        self,
        config: TcpConfig | None = None,
        *,
        expected_handshake: Handshake | None = None,
    ):
        self.config = config or TcpConfig()
        self.expected_handshake = expected_handshake or Handshake(role="pi_receiver")
        self.handshake: Handshake | None = None
        self.statuses: list[TransportStatus] = []
        self._server: socket.socket | None = None
        self._stop = False

    def start(self) -> None:
        if self._server is not None:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.config.host, self.config.port))
        server.listen(self.config.backlog)
        server.settimeout(self.config.socket_timeout_s)
        self._server = server

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("TCP source has not started")
        return self._server.getsockname()

    def __iter__(self) -> Iterator[MeasuredSample]:
        self.start()
        assert self._server is not None
        while not self._stop:
            try:
                connection, address = self._server.accept()
            except socket.timeout:
                continue
            self.statuses.append(TransportStatus("connected", f"{address[0]}:{address[1]}"))
            try:
                yield from self._read_connection(connection)
                if not self.config.reconnect:
                    return
            except (ConnectionError, OSError) as exc:
                self.statuses.append(TransportStatus("disconnected", str(exc)))
                if not self.config.reconnect:
                    raise
            finally:
                connection.close()
        self.close()

    def _read_connection(self, connection: socket.socket) -> Iterator[MeasuredSample]:
        connection.settimeout(self.config.socket_timeout_s)
        stream = connection.makefile("rwb")
        try:
            line = self._readline(stream, connection)
            received = deserialize_handshake(line)
            if received.role != "pc_sender":
                raise ConnectionError(f"unexpected handshake role {received.role!r}")
            received.validate_compatible(self.expected_handshake)
            self.handshake = received
            stream.write(serialize_handshake(self.expected_handshake, MESSAGE_HELLO_ACK))
            stream.flush()
            self.statuses.append(TransportStatus("handshake", "compatible stream accepted"))
            last_sample_at = time.monotonic()
            while True:
                line = self._readline(stream, connection)
                kind = record_type(line)
                if kind == MESSAGE_END:
                    return
                if kind != MESSAGE_SAMPLE:
                    raise ConnectionError(f"unexpected record type {kind!r}")
                sample = deserialize_sample(line)
                last_sample_at = time.monotonic()
                yield sample
                if time.monotonic() - last_sample_at > self.config.stale_timeout_s:
                    self.statuses.append(
                        TransportStatus("stale", "sample age exceeded stale timeout")
                    )
        finally:
            stream.close()

    def _readline(self, stream, connection: socket.socket) -> bytes:
        deadline = time.monotonic() + self.config.stale_timeout_s
        while True:
            try:
                line = stream.readline()
                if not line:
                    raise ConnectionError("TCP peer closed the stream")
                return line
            except socket.timeout as exc:
                if time.monotonic() >= deadline:
                    self.statuses.append(
                        TransportStatus("stale", "no sample received before timeout")
                    )
                    raise ConnectionError("TCP stream became stale") from exc
                # A socket timeout is transient; the next read checks the peer again.

    def close(self) -> None:
        self._stop = True
        if self._server is not None:
            self._server.close()
            self._server = None

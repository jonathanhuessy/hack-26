import tempfile
import threading
import unittest
from pathlib import Path

try:
    from .contracts import MeasuredSample
    from .transport.adapters import (
        FaultPolicy,
        FileSampleSink,
        FileSampleSource,
        LoopbackTransport,
    )
    from .transport.tcp import TcpConfig, TcpSampleSender, TcpSampleSource
    from .transport.wire import (
        CompatibilityError,
        Handshake,
        WireFormatError,
        deserialize_sample,
        serialize_sample,
    )
except ImportError:
    from contracts import MeasuredSample
    from transport.adapters import FaultPolicy, FileSampleSink, FileSampleSource, LoopbackTransport
    from transport.tcp import TcpConfig, TcpSampleSender, TcpSampleSource
    from transport.wire import CompatibilityError, Handshake, WireFormatError, deserialize_sample, serialize_sample


def samples(count=5):
    return [
        MeasuredSample(
            delta=0.01 * index,
            vx=4.0,
            yaw_rate=0.02 * index,
            timestamp_s=0.01 * index,
            sequence=index,
            diagnostics={"source": "test"},
        )
        for index in range(count)
    ]


class TransportTests(unittest.TestCase):
    def test_json_round_trip_preserves_edge_channels_and_metadata(self):
        original = samples(1)[0]
        decoded = deserialize_sample(serialize_sample(original))
        self.assertEqual(decoded, original)
        self.assertEqual(decoded.edge_payload(), original.edge_payload())

    def test_wire_rejects_malformed_and_nonfinite_records(self):
        with self.assertRaises(WireFormatError):
            deserialize_sample(b'{"type":"sample","sequence":0}')
        with self.assertRaises(WireFormatError):
            deserialize_sample(
                b'{"type":"sample","schema_version":1,"timestamp_s":NaN,'
                b'"sequence":0,"delta":0,"Vx":1,"r":0}'
            )

    def test_handshake_rejects_incompatible_versions(self):
        expected = Handshake(role="pi_receiver")
        received = Handshake(role="pc_sender", units_version=99)
        with self.assertRaises(CompatibilityError):
            received.validate_compatible(expected)

    def test_loopback_is_bounded_and_can_inject_duplicate(self):
        sender, source = LoopbackTransport.pair(
            fault_policy=FaultPolicy(duplicate_every=2)
        )
        for sample in samples():
            sender.send(sample)
        sender.close()
        received = list(source)
        self.assertEqual([sample.sequence for sample in received], [0, 1, 1, 2, 3, 3, 4])

    def test_file_capture_replays_with_handshake(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.ndjson"
            sink = FileSampleSink(path)
            for sample in samples():
                sink.send(sample)
            sink.close()
            received = list(FileSampleSource(path))
        self.assertEqual(received, samples())

    def test_tcp_sender_and_receiver_round_trip(self):
        receiver = TcpSampleSource(
            TcpConfig(host="127.0.0.1", port=0, reconnect=False, stale_timeout_s=1.0)
        )
        receiver.start()
        host, port = receiver.address
        expected = samples()
        errors = []

        def send():
            try:
                TcpSampleSender(host, port).send_samples(expected)
            except Exception as exc:  # surfaced below rather than lost in a thread
                errors.append(exc)

        thread = threading.Thread(target=send)
        thread.start()
        received = list(receiver)
        thread.join(timeout=2.0)
        receiver.close()
        self.assertFalse(errors, errors)
        self.assertEqual(received, expected)
        self.assertTrue(any(status.kind == "handshake" for status in receiver.statuses))


if __name__ == "__main__":
    unittest.main()

import unittest
import threading
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from .streaming import PlaybackStream, PlantStream
    from .dashboard_session import AsyncTcpTransport
    from .contracts import MeasuredSample
    from .transport.tcp import TcpConfig, TcpSampleSource
    from .streamlit_app import build_stream, run_status_text, trajectory_options
except ImportError:
    from streaming import PlaybackStream, PlantStream
    from dashboard_session import AsyncTcpTransport
    from contracts import MeasuredSample
    from transport.tcp import TcpConfig, TcpSampleSource
    from streamlit_app import build_stream, run_status_text, trajectory_options


class StreamlitAppTests(unittest.TestCase):
    def test_trajectory_options_include_python_and_generated_matlab_files(self):
        options = trajectory_options()
        self.assertIn("Python default", options)
        self.assertEqual(
            set(options) - {"Python default"},
            {"MATLAB demo A", "MATLAB demo B", "MATLAB demo AB"},
        )

    def test_build_stream_selects_the_requested_source(self):
        options = trajectory_options()
        python_stream = build_stream("Python default", options)
        self.assertIsInstance(python_stream, PlantStream)
        matlab_name = "MATLAB demo A"
        matlab_stream = build_stream(matlab_name, options)
        self.assertIsInstance(matlab_stream, PlaybackStream)
        self.assertFalse(matlab_stream.supports_events)

    def test_demo_trajectories_have_clear_labels(self):
        options = trajectory_options()
        self.assertIn("MATLAB demo A", options)
        self.assertIn("MATLAB demo B", options)
        self.assertIn("MATLAB demo AB", options)

    def test_run_status_waits_before_start_and_after_finish(self):
        self.assertEqual(
            run_status_text(SimpleNamespace(running=False, samples=[])),
            "Waiting for run",
        )
        self.assertEqual(
            run_status_text(SimpleNamespace(running=False, samples=[object()])),
            "Waiting for run",
        )
        self.assertEqual(
            run_status_text(SimpleNamespace(running=True, samples=[object()])),
            "Running",
        )

    def test_listener_wrapper_targets_only_tcp_pi_listeners(self):
        script = (
            Path(__file__).parents[1] / "scripts" / "start_pi_listener.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('"pi.app"', script)
        self.assertIn('"--tcp-listen"', script)
        self.assertIn('exec "$PYTHON" -m pi.app "$@"', script)
        self.assertNotIn("pkill -f", script)


class AsyncTcpTransportTests(unittest.TestCase):
    class FakeSender:
        timeout_s = 0.2
        host = "fake"
        port = 8765

        def __init__(self, failure: Exception | None = None):
            self.failure = failure
            self.sent = []
            self.closed = threading.Event()

        def send(self, sample):
            if self.failure:
                raise self.failure
            self.sent.append(sample)

        def finish(self):
            pass

        def close(self):
            self.closed.set()

    @staticmethod
    def sample(sequence=0):
        return MeasuredSample(
            delta=0.0,
            vx=4.0,
            yaw_rate=0.0,
            timestamp_s=sequence * 0.01,
            sequence=sequence,
        )

    def test_enqueue_does_not_wait_for_socket_write(self):
        transport = AsyncTcpTransport("fake", 8765)
        sender = self.FakeSender()
        transport.sender = sender
        transport.start()
        transport.send(self.sample())
        deadline = time.monotonic() + 1.0
        while not sender.sent and time.monotonic() < deadline:
            time.sleep(0.005)
        transport.finish()
        self.assertEqual([sample.sequence for sample in sender.sent], [0])
        self.assertEqual(transport.sent_samples, 1)

    def test_sender_failure_is_observable_and_rejects_more_samples(self):
        transport = AsyncTcpTransport("fake", 8765)
        transport.sender = self.FakeSender(ConnectionError("receiver stopped"))
        transport.start()
        transport.send(self.sample())
        deadline = time.monotonic() + 1.0
        while transport.error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertIsNotNone(transport.error)
        self.assertEqual(transport.state, "Disconnected")
        with self.assertRaises(ConnectionError):
            transport.send(self.sample(1))
        transport.close()

    def test_close_releases_a_blocked_sender_thread(self):
        class BlockingSender(self.FakeSender):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()

            def send(self, sample):
                self.started.set()
                self.closed.wait(1.0)
                raise ConnectionError("sender closed")

        sender = BlockingSender()
        transport = AsyncTcpTransport("fake", 8765)
        transport.sender = sender
        transport.start()
        transport.send(self.sample())
        self.assertTrue(sender.started.wait(1.0))
        transport.close()
        self.assertFalse(transport.running)

    def test_transport_delivers_ordered_samples_end_to_end(self):
        receiver = TcpSampleSource(
            TcpConfig(host="127.0.0.1", port=0, reconnect=False, stale_timeout_s=1.0)
        )
        receiver.start()
        host, port = receiver.address
        received = []
        reader = threading.Thread(
            target=lambda: received.extend(list(receiver)),
            daemon=True,
        )
        reader.start()

        transport = AsyncTcpTransport(host, port)
        transport.start()
        for sequence in range(5):
            transport.send(self.sample(sequence))
        transport.finish()
        reader.join(timeout=2.0)
        receiver.close()

        self.assertFalse(reader.is_alive())
        self.assertEqual([sample.sequence for sample in received], list(range(5)))


if __name__ == "__main__":
    unittest.main()

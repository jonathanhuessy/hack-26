import time
import unittest
from pathlib import Path

import numpy as np

try:
    from .local_plant import NOMINAL_PARAMS
    from .plant import StreamingPlant, simulate
    from .model import load_weights
    from .detector import TurnDetector
    from .pipeline import EdgePipeline
    from .contracts import ExecutionConfig
    from .streaming import (
        PlaybackStream,
        PlantStream,
        RuntimeEventSchedule,
        StreamingRunner,
    )
    from .transport.tcp import TcpConfig, TcpSampleSender, TcpSampleSource
except ImportError:
    from local_plant import NOMINAL_PARAMS
    from plant import StreamingPlant, simulate
    from model import load_weights
    from detector import TurnDetector
    from pipeline import EdgePipeline
    from contracts import ExecutionConfig
    from streaming import PlaybackStream, PlantStream, RuntimeEventSchedule, StreamingRunner
    from transport.tcp import TcpConfig, TcpSampleSender, TcpSampleSource


class StreamingPlantTests(unittest.TestCase):
    def test_streaming_plant_matches_batch_simulation(self):
        n = 20
        delta = np.linspace(0.0, 0.1, n)
        vx = np.full(n, 4.0)
        fyd = np.linspace(0.0, 10.0, n)
        mzd = np.linspace(0.0, 5.0, n)
        expected = simulate(delta, vx, fyd, mzd, list(NOMINAL_PARAMS))
        plant = StreamingPlant(parameter_schedule=lambda _: list(NOMINAL_PARAMS))
        actual = [plant.step(row, next_row) for row, next_row in zip(
            np.column_stack((delta, vx, fyd, mzd)),
            np.column_stack((delta, vx, fyd, mzd))[1:].tolist() + [
                [delta[-1], vx[-1], fyd[-1], mzd[-1]]
            ],
        )]
        for name in ("r", "ay", "ydot", "alphaF", "X", "Y", "psi"):
            np.testing.assert_allclose(
                [item[name] for item in actual],
                expected[name],
                rtol=1e-12,
                atol=1e-12,
            )

    def test_streaming_plant_matches_batch_event_schedule(self):
        n = 20
        delta = np.linspace(0.0, 0.1, n)
        vx = np.full(n, 4.0)
        fyd = np.zeros(n)
        mzd = np.zeros(n)
        schedule = RuntimeEventSchedule()
        schedule.command("add", "implement_attached", 0.05, duration_s=0.04)
        expected = simulate(
            delta,
            vx,
            fyd,
            mzd,
            list(NOMINAL_PARAMS),
            h=0.01,
            parameter_schedule=schedule.parameter_at,
        )
        plant = StreamingPlant(parameter_schedule=schedule.parameter_at)
        rows = np.column_stack((delta, vx, fyd, mzd))
        actual = [plant.step(row, next_row) for row, next_row in zip(rows, rows[1:].tolist() + [rows[-1].tolist()])]
        np.testing.assert_allclose(
            [item["params"] for item in actual],
            expected["params"],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_runtime_commands_add_and_remove_at_sample_boundaries(self):
        schedule = RuntimeEventSchedule()
        schedule.command("add", "implement_attached", 0.02)
        self.assertEqual(schedule.active_events(0.01), ())
        self.assertEqual(schedule.active_events(0.02), ("implement_attached",))
        schedule.command("remove", "implement_attached", 0.04)
        self.assertEqual(schedule.active_events(0.04), ())
        self.assertEqual(schedule.active_events(0.05), ())
        self.assertEqual(len(schedule.history), 2)

    def test_plant_stream_emits_contract_and_event_metadata(self):
        stream = PlantStream.from_seed(profile_seed=2, disturbance_seed=3, swaths=1)
        stream.schedule.command("add", "tire_flat", 0.0)
        sample = stream.step()
        assert sample is not None
        self.assertEqual(len(sample.edge_payload()), 3)
        self.assertEqual(sample.diagnostics["active_events"], "tire_flat")
        self.assertEqual(sample.sequence, 0)

    def test_matlab_playback_stream_preserves_fixed_edge_channels(self):
        path = (
            Path(__file__).parents[1]
            / "data"
            / "generated_trajectories"
            / "matlab"
            / "trajectory_nominal_01.mat"
        )
        if not path.exists():
            self.skipTest("generated MATLAB trajectory is not present")
        stream = PlaybackStream.from_mat(path)
        first = stream.step()
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.sequence, 0)
        self.assertIn("X", first.diagnostics)
        self.assertFalse(stream.supports_events)

    def test_demo_playback_stream_preserves_measured_contract_and_metadata(self):
        path = (
            Path(__file__).parents[1]
            / "data"
            / "generated_trajectories"
            / "matlab"
            / "demo_ab.mat"
        )
        if not path.exists():
            self.skipTest("generated MATLAB demo trajectory is not present")
        stream = PlaybackStream.from_mat(path)
        first = stream.step()
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.edge_payload(), tuple(stream.measured[0, :3]))
        self.assertIn("X", first.diagnostics)
        self.assertIn("Y", first.diagnostics)
        self.assertEqual(stream.metadata["className"], "AB")
        self.assertFalse(stream.supports_events)

    def test_demo_playback_detector_replays_expected_delayed_classes(self):
        weights = Path(__file__).parents[1] / "models" / "export" / "weights.mat"
        if not weights.exists():
            self.skipTest("MATLAB-exported weights are not present")
        expected = {
            "demo_a": ["nominal", "nominal", "nominal", "A", "A", "A"],
            "demo_b": ["nominal", "nominal", "nominal", "B", "B", "B"],
            "demo_ab": ["nominal", "nominal", "nominal", "AB", "AB", "AB"],
        }
        for name, classes in expected.items():
            with self.subTest(name=name):
                path = Path(__file__).parents[1] / "data" / "generated_trajectories" / "matlab" / f"{name}.mat"
                if not path.exists():
                    self.skipTest("generated MATLAB demo trajectories are not present")
                stream = PlaybackStream.from_mat(path)
                detector = TurnDetector(load_weights(weights))
                pipeline = EdgePipeline(detector, ExecutionConfig(realtime=False))
                actual = []
                while (sample := stream.step()) is not None:
                    event = pipeline.push(sample)
                    verdict = getattr(event.consumer_result, "new_verdict", None)
                    if verdict is not None:
                        actual.append(verdict.class_name)
                flushed = pipeline.flush()
                verdict = getattr(flushed, "new_verdict", None)
                if verdict is not None:
                    actual.append(verdict.class_name)
                self.assertEqual(actual, classes)


class StreamingRunnerTests(unittest.TestCase):
    def test_runner_pause_resume_reset_and_stop(self):
        samples = []
        stream = PlantStream.from_seed(profile_seed=4, disturbance_seed=5, swaths=1)
        runner = StreamingRunner(stream, samples.append, realtime=False)
        runner.start()
        deadline = time.monotonic() + 2.0
        while runner.running and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(runner.errors)
        self.assertGreater(len(samples), 0)
        runner.pause()
        runner.reset()
        self.assertEqual(stream.index, 0)
        runner.resume()
        runner.stop()
        self.assertFalse(runner.running)

    def test_runner_applies_queued_command(self):
        received = []
        stream = PlantStream.from_seed(profile_seed=6, disturbance_seed=7, swaths=1)
        runner = StreamingRunner(stream, received.append, realtime=False)
        runner.add_event("implement_attached")
        runner.start()
        deadline = time.monotonic() + 2.0
        while runner.running and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(runner.errors)
        self.assertTrue(received)
        self.assertEqual(received[0].diagnostics["active_events"], "implement_attached")

    def test_runner_streams_samples_and_closes_tcp_stream(self):
        stream = PlantStream(
            delta=np.zeros(8),
            vx=np.full(8, 4.0),
            fyd=np.zeros(8),
            mzd=np.zeros(8),
        )
        receiver = TcpSampleSource(
            TcpConfig(host="127.0.0.1", port=0, reconnect=False, stale_timeout_s=1.0)
        )
        receiver.start()
        host, port = receiver.address
        sender = TcpSampleSender(host, port)
        runner = StreamingRunner(
            stream,
            sender.send,
            sender.finish,
            realtime=False,
        )
        runner.start()
        received = list(receiver)
        runner.stop()
        sender.close()
        receiver.close()
        self.assertFalse(runner.errors)
        self.assertEqual(len(received), 8)
        self.assertEqual(received[-1].sequence, 7)


if __name__ == "__main__":
    unittest.main()

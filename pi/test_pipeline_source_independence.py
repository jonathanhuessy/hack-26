import unittest
from pathlib import Path

try:
    from .pipeline import EdgePipeline, NullConsumer
    from .sources import ArraySampleSource, ReplaySampleSource
except ImportError:
    from pipeline import EdgePipeline, NullConsumer
    from sources import ArraySampleSource, ReplaySampleSource


FIXTURE = Path(__file__).parent / "test_vectors" / "phase0_three_input.json"


class PipelineSourceIndependenceTests(unittest.TestCase):
    def test_local_and_replay_sources_produce_same_edge_payload(self):
        local = ArraySampleSource(
            [0.0, 0.005, 0.01, 0.015, 0.02],
            [4.17, 4.17, 4.16, 4.16, 4.15],
            [0.0, 0.002, 0.004, 0.006, 0.008],
            diagnostics={"ay": [0.0, 0.01, 0.02, 0.03, 0.04], "label": ["A"] * 5},
        )
        local_consumer = NullConsumer()
        replay_consumer = NullConsumer()
        EdgePipeline(local_consumer).run(local)
        EdgePipeline(replay_consumer).run(ReplaySampleSource(FIXTURE))
        self.assertEqual(local_consumer.samples, replay_consumer.samples)

    def test_pipeline_consumer_cannot_receive_diagnostics(self):
        class RecordingConsumer(NullConsumer):
            def push(self, sample):
                self.seen = sample
                return super().push(sample)

        consumer = RecordingConsumer()
        EdgePipeline(consumer).run(ReplaySampleSource(FIXTURE))
        self.assertEqual(consumer.seen.edge_payload(), (0.02, 4.15, 0.008))
        self.assertIn("ay", consumer.seen.diagnostics)
        self.assertEqual(consumer.samples[-1], (0.02, 4.15, 0.008))

    def test_offline_replay_is_repeatable(self):
        first = NullConsumer()
        second = NullConsumer()
        EdgePipeline(first).run(ReplaySampleSource(FIXTURE))
        EdgePipeline(second).run(ReplaySampleSource(FIXTURE))
        self.assertEqual(first.samples, second.samples)


if __name__ == "__main__":
    unittest.main()

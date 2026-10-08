import json
from pathlib import Path
import unittest

try:
    from .contracts import (
        MeasuredSample,
        SampleStatus,
        SampleValidationError,
        validate_batch,
        validate_sample,
    )
    from .pipeline import EdgePipeline, NullConsumer
    from .sources import ReplaySampleSource
except ImportError:
    from contracts import (
        MeasuredSample,
        SampleStatus,
        SampleValidationError,
        validate_batch,
        validate_sample,
    )
    from pipeline import EdgePipeline, NullConsumer
    from sources import ReplaySampleSource


FIXTURE = Path(__file__).parent / "test_vectors" / "phase0_three_input.json"


class ContractTests(unittest.TestCase):
    def test_three_required_channels_are_valid_without_ay(self):
        sample = MeasuredSample(0.1, 4.0, 0.02, 0.0, 0)
        validate_sample(sample)
        self.assertEqual(sample.edge_payload(), (0.1, 4.0, 0.02))

    def test_optional_diagnostics_do_not_change_edge_payload(self):
        bare = MeasuredSample(0.1, 4.0, 0.02, 0.0, 0)
        diagnosed = MeasuredSample(
            0.1, 4.0, 0.02, 0.0, 0, diagnostics={"ay": 0.4, "label": "A"}
        )
        self.assertEqual(bare.edge_payload(), diagnosed.edge_payload())

    def test_schema_and_non_finite_values_are_rejected(self):
        with self.assertRaises(SampleValidationError):
            validate_sample(MeasuredSample(0.1, 4.0, 0.02, 0.0, 0, schema_version=99))
        with self.assertRaises(SampleValidationError):
            validate_sample(MeasuredSample(float("nan"), 4.0, 0.02, 0.0, 0))

    def test_gap_is_reported_and_forwarded(self):
        samples = [
            MeasuredSample(0.0, 4.0, 0.0, 0.0, 0),
            MeasuredSample(0.0, 4.0, 0.0, 0.01, 2),
        ]
        results = validate_batch(samples)
        self.assertEqual([result.status for result in results], [SampleStatus.ACCEPTED, SampleStatus.GAP])

    def test_late_and_duplicate_samples_are_rejected(self):
        samples = [
            MeasuredSample(0.0, 4.0, 0.0, 0.0, 0),
            MeasuredSample(0.0, 4.0, 0.0, 0.01, 1),
            MeasuredSample(0.0, 4.0, 0.0, 0.01, 1),
            MeasuredSample(0.0, 4.0, 0.0, 0.005, 2),
        ]
        results = validate_batch(samples)
        self.assertEqual(results[2].status, SampleStatus.DUPLICATE)
        self.assertEqual(results[3].status, SampleStatus.LATE)

    def test_replay_fixture_runs_without_labels_in_edge_path(self):
        consumer = NullConsumer()
        events = EdgePipeline(consumer).run(ReplaySampleSource(FIXTURE))
        self.assertEqual(len(events), 5)
        self.assertEqual(len(consumer.samples), 5)
        self.assertEqual(consumer.samples[0], (0.0, 4.17, 0.0))
        self.assertNotIn("nominal", json.dumps(consumer.samples))


if __name__ == "__main__":
    unittest.main()

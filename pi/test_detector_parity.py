import unittest
from pathlib import Path

try:
    from .detector import TurnDetector
    from .model import dummy_model
    from .pipeline import EdgePipeline
    from .sources import MatlabFeatureReplaySource
except ImportError:
    from detector import TurnDetector
    from model import dummy_model
    from pipeline import EdgePipeline
    from sources import MatlabFeatureReplaySource


FIXTURE_DIR = Path(__file__).parent / "test_vectors"


class DetectorParityTests(unittest.TestCase):
    def test_streaming_detector_emits_three_nominal_turn_verdicts(self):
        detector = TurnDetector(dummy_model())
        events = EdgePipeline(detector).run(
            MatlabFeatureReplaySource(FIXTURE_DIR / "features_nominal.mat")
        )
        verdicts = [
            event.consumer_result.new_verdict
            for event in events
            if event.consumer_result is not None and event.consumer_result.new_verdict is not None
        ]
        self.assertEqual(len(verdicts), 3)
        self.assertEqual([verdict.turn_index for verdict in verdicts], [0, 1, 2])
        self.assertTrue(all(verdict.class_name == "nominal" for verdict in verdicts))

    def test_detector_uses_same_path_for_all_classes(self):
        for name in ("nominal", "A", "B", "AB"):
            detector = TurnDetector(dummy_model())
            events = EdgePipeline(detector).run(
                MatlabFeatureReplaySource(FIXTURE_DIR / f"features_{name}.mat")
            )
            verdicts = [
                event.consumer_result.new_verdict
                for event in events
                if event.consumer_result is not None and event.consumer_result.new_verdict is not None
            ]
            self.assertEqual(len(verdicts), 3, name)


if __name__ == "__main__":
    unittest.main()

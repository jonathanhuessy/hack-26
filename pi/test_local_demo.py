import unittest

import numpy as np

from .detector import TurnDetector
from .local_plant import LocalScenario, NOMINAL_PARAMS, build_source, scenario_parameters
from .model import dummy_model
from .pipeline import EdgePipeline


class LocalScenarioTests(unittest.TestCase):
    def test_ballast_updates_coupled_parameters(self):
        nominal, changed = scenario_parameters(
            LocalScenario(name="A", change_type="step")
        )
        self.assertEqual(nominal, list(NOMINAL_PARAMS))
        self.assertGreater(changed[0], nominal[0])
        self.assertNotEqual(changed[1], nominal[1])
        self.assertNotEqual(changed[2], nominal[2])
        self.assertGreater(changed[3], nominal[3])
        self.assertEqual(changed[4], nominal[4])

    def test_front_stiffness_changes_only_front_stiffness(self):
        nominal, changed = scenario_parameters(
            LocalScenario(name="B", kf=0.8, change_type="step")
        )
        self.assertEqual(changed[4], nominal[4] * 0.8)
        np.testing.assert_allclose(changed[:4], nominal[:4])
        np.testing.assert_allclose(changed[5:], nominal[5:])

    def test_step_and_ramp_endpoints(self):
        step_start, step_end = scenario_parameters(
            LocalScenario(name="A", change_type="step")
        )
        ramp_start, ramp_end = scenario_parameters(
            LocalScenario(name="A", change_type="ramp")
        )
        self.assertEqual(step_start, list(NOMINAL_PARAMS))
        self.assertEqual(step_end[0], ramp_end[0])
        self.assertEqual(ramp_start, list(NOMINAL_PARAMS))

    def test_source_contract_and_seed_determinism(self):
        config = LocalScenario(name="AB", profile_seed=9, disturbance_seed=4)
        first = list(build_source(config))
        second = list(build_source(config))
        self.assertEqual(len(first), len(second))
        self.assertEqual(first[0].sequence, 0)
        self.assertEqual(first[0].timestamp_s, 0.0)
        self.assertAlmostEqual(first[1].timestamp_s - first[0].timestamp_s, 0.01)
        np.testing.assert_allclose(
            [sample.channels for sample in first[:100]],
            [sample.channels for sample in second[:100]],
        )
        self.assertIn("ay", first[0].diagnostics)
        self.assertEqual(len(first[0].edge_payload()), 3)

    def test_all_scenarios_reach_three_turn_verdicts(self):
        for name in ("nominal", "A", "B", "AB"):
            detector = TurnDetector(dummy_model())
            events = EdgePipeline(detector).run(
                build_source(LocalScenario(name=name))
            )
            verdicts = [
                event.consumer_result.new_verdict
                for event in events
                if event.consumer_result is not None
                and event.consumer_result.new_verdict is not None
            ]
            self.assertEqual(len(verdicts), 3, name)
            self.assertEqual([item.turn_index for item in verdicts], [0, 1, 2])
            self.assertLessEqual(len(detector._samples), detector.history_n)

    def test_reverse_ramp_uses_changed_to_nominal_endpoints(self):
        start, end = scenario_parameters(
            LocalScenario(name="A", change_type="ramp", reverse=True)
        )
        self.assertGreater(start[0], end[0])
        self.assertEqual(end, list(NOMINAL_PARAMS))

    def test_step_schedule_switches_at_requested_time(self):
        source = build_source(
            LocalScenario(name="A", change_type="step", t_start_s=30.0)
        )
        samples = list(source)
        before = samples[2999].diagnostics["params"]
        after = samples[3001].diagnostics["params"]
        self.assertAlmostEqual(before[0], NOMINAL_PARAMS[0])
        self.assertGreater(after[0], NOMINAL_PARAMS[0])


if __name__ == "__main__":
    unittest.main()

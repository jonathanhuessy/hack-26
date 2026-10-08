import unittest

import numpy as np

try:
    from .detector import TurnDetector
    from .local_plant import (
        EventSchedule,
        LocalScenario,
        NOMINAL_PARAMS,
        VehicleEvent,
        build_source,
        event_preset,
        scenario_parameters,
    )
    from .model import dummy_model
    from .pipeline import EdgePipeline
except ImportError:
    from detector import TurnDetector
    from local_plant import (
        EventSchedule,
        LocalScenario,
        NOMINAL_PARAMS,
        VehicleEvent,
        build_source,
        event_preset,
        scenario_parameters,
    )
    from model import dummy_model
    from pipeline import EdgePipeline


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

    def test_named_event_aliases_match_training_scenarios(self):
        alias_start, alias_end = scenario_parameters(
            LocalScenario(name="implement_attached", change_type="step")
        )
        scenario_start, scenario_end = scenario_parameters(
            LocalScenario(name="A", change_type="step")
        )
        self.assertEqual(alias_start, scenario_start)
        self.assertEqual(alias_end, scenario_end)

        alias_start, alias_end = scenario_parameters(
            LocalScenario(name="tire_flat", change_type="step")
        )
        scenario_start, scenario_end = scenario_parameters(
            LocalScenario(name="B", change_type="step")
        )
        self.assertEqual(alias_start, scenario_start)
        self.assertEqual(alias_end, scenario_end)

    def test_event_schedule_composes_implement_and_tire_change(self):
        schedule = EventSchedule(
            tuple(NOMINAL_PARAMS),
            (
                event_preset("implement_attached", start_s=10.0),
                event_preset("tire_flat", start_s=20.0),
            ),
        )
        before = schedule.parameter_at(9.99)
        after_implement = schedule.parameter_at(15.0)
        after_both = schedule.parameter_at(25.0)
        np.testing.assert_allclose(before, NOMINAL_PARAMS)
        self.assertGreater(after_implement[0], NOMINAL_PARAMS[0])
        self.assertEqual(after_implement[4], NOMINAL_PARAMS[4])
        self.assertGreater(after_both[0], NOMINAL_PARAMS[0])
        self.assertLess(after_both[4], NOMINAL_PARAMS[4])
        self.assertEqual(schedule.active_events(15.0), ("implement_attached",))
        self.assertEqual(
            schedule.active_events(25.0),
            ("implement_attached", "tire_flat"),
        )

    def test_event_schedule_ramp_and_reverse(self):
        event = VehicleEvent(name="A", start_s=10.0, end_s=20.0)
        schedule = EventSchedule(tuple(NOMINAL_PARAMS), (event,))
        self.assertEqual(schedule.parameter_at(9.0), list(NOMINAL_PARAMS))
        midpoint = schedule.parameter_at(15.0)
        endpoint = schedule.parameter_at(20.0)
        self.assertGreater(midpoint[0], NOMINAL_PARAMS[0])
        self.assertLess(midpoint[0], endpoint[0])
        reverse = EventSchedule(
            tuple(NOMINAL_PARAMS),
            (VehicleEvent(name="A", start_s=10.0, end_s=20.0, reverse=True),),
        )
        self.assertGreater(reverse.parameter_at(9.0)[0], NOMINAL_PARAMS[0])
        self.assertEqual(reverse.parameter_at(20.0), list(NOMINAL_PARAMS))

    def test_invalid_event_is_rejected(self):
        with self.assertRaises(ValueError):
            VehicleEvent(name="new_physics", start_s=0.0)
        with self.assertRaises(ValueError):
            VehicleEvent(name="A", start_s=20.0, end_s=10.0)


if __name__ == "__main__":
    unittest.main()

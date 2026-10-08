import unittest

try:
    from .local_plant import LocalScenario, build_source
    from .visualize_demo import collect_samples, parse_event_spec
except ImportError:
    from local_plant import LocalScenario, build_source
    from visualize_demo import collect_samples, parse_event_spec


class VisualizationDemoTests(unittest.TestCase):
    def test_parse_named_event_and_ramp(self):
        event = parse_event_spec(
            "tire_flat:10:12",
            added_mass_kg=1000.0,
            kf=0.8,
        )
        self.assertEqual(event.name, "tire_flat")
        self.assertEqual(event.start_s, 10.0)
        self.assertEqual(event.end_s, 12.0)
        reverse = parse_event_spec(
            "implement_attached:10:12",
            added_mass_kg=1000.0,
            kf=0.8,
            reverse=True,
        )
        self.assertTrue(reverse.reverse)

    def test_parse_event_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            parse_event_spec("engine_failure:10", added_mass_kg=1000.0, kf=0.8)

    def test_custom_event_source_contains_event_metadata(self):
        event = parse_event_spec(
            "implement_attached:1",
            added_mass_kg=1000.0,
            kf=0.8,
        )
        samples = collect_samples(
            build_source(
                LocalScenario(
                    events=(event,),
                    profile_seed=2,
                    disturbance_seed=3,
                )
            )
        )
        self.assertEqual(samples[0].diagnostics["active_events"], "")
        self.assertIn("implement_attached", samples[100].diagnostics["active_events"])
        self.assertEqual(len(samples[100].edge_payload()), 3)


if __name__ == "__main__":
    unittest.main()

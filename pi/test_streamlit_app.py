import unittest

try:
    from .streaming import PlaybackStream, PlantStream
    from .streamlit_app import build_stream, trajectory_options
except ImportError:
    from streaming import PlaybackStream, PlantStream
    from streamlit_app import build_stream, trajectory_options


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


if __name__ == "__main__":
    unittest.main()

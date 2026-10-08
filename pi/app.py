"""Offline Phase 1 classifier demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .contracts import ExecutionConfig
    from .detector import TurnDetector
    from .model import dummy_model, load_weights
    from .noise import NoisySource
    from .pipeline import EdgePipeline
    from .sources import MatlabFeatureReplaySource, ReplaySampleSource
except ImportError:
    from contracts import ExecutionConfig
    from detector import TurnDetector
    from model import dummy_model, load_weights
    from noise import NoisySource
    from pipeline import EdgePipeline
    from sources import MatlabFeatureReplaySource, ReplaySampleSource


def run(replay: str | Path, weights: str | Path | None = None, noise_seed: int | None = None) -> int:
    model = load_weights(weights) if weights else dummy_model()
    source = MatlabFeatureReplaySource(replay) if Path(replay).suffix == ".mat" else ReplaySampleSource(replay)
    if noise_seed is not None:
        source = NoisySource(source, seed=noise_seed)
    detector = TurnDetector(model)
    pipeline = EdgePipeline(detector, ExecutionConfig(realtime=False))
    events = pipeline.run(source)
    verdicts = [event.consumer_result.new_verdict for event in events
                if event.consumer_result is not None and event.consumer_result.new_verdict is not None]
    for verdict in verdicts:
        print(
            f"{verdict.window_id}: class={verdict.class_name} "
            f"confidence={max(verdict.class_probabilities):.3f} "
            f"delta_m_kg={verdict.delta_m_kg} k_f={verdict.k_f}"
        )
    print(f"processed {len(events)} samples, {len(verdicts)} verdicts")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="Phase 0/Phase 1 replay JSON fixture")
    parser.add_argument("--weights", type=Path, help="MATLAB weights.mat artifact")
    parser.add_argument("--noise-seed", type=int, help="deterministic sensor-noise seed")
    args = parser.parse_args(argv)
    return run(args.replay, args.weights, args.noise_seed)


if __name__ == "__main__":
    sys.exit(main())

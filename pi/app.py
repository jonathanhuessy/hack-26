"""Offline Phase 1 classifier demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .contracts import ExecutionConfig
    from .detector import TurnDetector
    from .local_plant import LocalScenario, build_source
    from .model import dummy_model, load_weights
    from .noise import NoisySource
    from .pipeline import EdgePipeline
    from .sources import MatlabFeatureReplaySource, ReplaySampleSource
except ImportError:
    from contracts import ExecutionConfig
    from detector import TurnDetector
    from local_plant import LocalScenario, build_source
    from model import dummy_model, load_weights
    from noise import NoisySource
    from pipeline import EdgePipeline
    from sources import MatlabFeatureReplaySource, ReplaySampleSource


def run(
    replay: str | Path | None = None,
    weights: str | Path | None = None,
    noise_seed: int | None = None,
    *,
    local: bool = False,
    scenario: str = "nominal",
    change_type: str = "constant",
    t_start: float = 0.0,
    t_end: float = 0.0,
    reverse: bool = False,
    profile_seed: int = 1,
    disturbance_seed: int = 1,
    realtime: bool = False,
) -> int:
    default_weights = Path(__file__).resolve().parents[1] / "models" / "export" / "weights.mat"
    weight_path = Path(weights) if weights else (default_weights if default_weights.exists() else None)
    model = load_weights(weight_path) if weight_path else dummy_model()
    if local or replay is None:
        local_config = LocalScenario(
            name=scenario,
            change_type=change_type,
            t_start_s=t_start,
            t_end_s=t_end,
            reverse=reverse,
            profile_seed=profile_seed,
            disturbance_seed=disturbance_seed,
        )
        source = build_source(local_config, realtime=realtime)
        source_name = f"local/{scenario}/{change_type}"
    else:
        source = MatlabFeatureReplaySource(replay, realtime=realtime) if Path(replay).suffix == ".mat" else ReplaySampleSource(replay, realtime=realtime)
        source_name = str(replay)
    if noise_seed is not None:
        source = NoisySource(source, seed=noise_seed)
    detector = TurnDetector(model)
    pipeline = EdgePipeline(detector, ExecutionConfig(realtime=realtime))
    events = pipeline.run(source)
    accepted = sum(event.result.status.value in ("accepted", "gap") for event in events)
    rejected = len(events) - accepted
    verdicts = [event.consumer_result.new_verdict for event in events
                if event.consumer_result is not None and event.consumer_result.new_verdict is not None]
    print(f"source={source_name} realtime={realtime} samples={accepted} rejected={rejected}")
    for verdict in verdicts:
        print(
            f"{verdict.window_id}: class={verdict.class_name} "
            f"confidence={max(verdict.class_probabilities):.3f} "
            f"delta_m_kg={verdict.delta_m_kg} k_f={verdict.k_f} "
            f"t={verdict.timestamp_s:.2f}s"
        )
    print(f"processed {len(events)} samples, {len(verdicts)} verdicts")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, nargs="?", help="Phase 0/Phase 1 replay fixture; omit for local mode")
    parser.add_argument("--local", action="store_true", help="run the co-located local plant demo")
    parser.add_argument("--scenario", choices=("nominal", "A", "B", "AB"), default="nominal")
    parser.add_argument("--change-type", choices=("constant", "step", "ramp"), default="constant")
    parser.add_argument("--t-start", type=float, default=0.0, help="change start time in seconds")
    parser.add_argument("--t-end", type=float, default=0.0, help="change end time in seconds; zero uses stream end")
    parser.add_argument("--reverse", action="store_true", help="run changed-to-nominal transition")
    parser.add_argument("--profile-seed", type=int, default=1)
    parser.add_argument("--disturbance-seed", type=int, default=1)
    parser.add_argument("--realtime", action="store_true", help="pace samples at the configured sample period")
    parser.add_argument("--weights", type=Path, help="MATLAB weights.mat artifact")
    parser.add_argument("--noise-seed", type=int, help="deterministic sensor-noise seed")
    args = parser.parse_args(argv)
    return run(
        args.replay,
        args.weights,
        args.noise_seed,
        local=args.local,
        scenario=args.scenario,
        change_type=args.change_type,
        t_start=args.t_start,
        t_end=args.t_end,
        reverse=args.reverse,
        profile_seed=args.profile_seed,
        disturbance_seed=args.disturbance_seed,
        realtime=args.realtime,
    )


if __name__ == "__main__":
    sys.exit(main())

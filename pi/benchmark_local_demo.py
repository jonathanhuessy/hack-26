"""Measure local Phase 2 demo throughput and bounded detector memory."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import tracemalloc

try:
    from .detector import TurnDetector
    from .local_plant import LocalScenario, build_source
    from .model import dummy_model, load_weights
    from .pipeline import EdgePipeline
except ImportError:
    from detector import TurnDetector
    from local_plant import LocalScenario, build_source
    from model import dummy_model, load_weights
    from pipeline import EdgePipeline


def run(
    scenario: str = "nominal",
    change_type: str = "constant",
    weights: str | Path | None = None,
) -> dict[str, float]:
    config = LocalScenario(name=scenario, change_type=change_type)
    source = build_source(config)
    duration_s = (len(source.delta) - 1) * source.sample_period_s
    model = load_weights(weights) if weights else dummy_model()
    detector = TurnDetector(model)
    tracemalloc.start()
    started = time.perf_counter()
    events = EdgePipeline(detector).run(source)
    elapsed_s = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    verdicts = [
        event.consumer_result.new_verdict
        for event in events
        if event.consumer_result is not None
        and event.consumer_result.new_verdict is not None
    ]
    return {
        "duration_s": duration_s,
        "elapsed_s": elapsed_s,
        "realtime_factor": elapsed_s / duration_s,
        "peak_memory_mb": peak_bytes / 1024**2,
        "samples": float(len(source.delta)),
        "verdicts": float(len(verdicts)),
        "last_verdict_s": verdicts[-1].timestamp_s if verdicts else -1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("nominal", "A", "B", "AB"), default="nominal")
    parser.add_argument("--change-type", choices=("constant", "step", "ramp"), default="constant")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    result = run(args.scenario, args.change_type, args.weights)
    for name, value in result.items():
        print(f"{name}={value:.6f}" if isinstance(value, float) else f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

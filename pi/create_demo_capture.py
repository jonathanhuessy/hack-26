"""Generate the deterministic Phase 5 fallback transport capture."""

from __future__ import annotations

from pathlib import Path

from .contracts import MeasuredSample
from .local_plant import LocalScenario, build_source
from .transport.adapters import FileSampleSink


def main() -> int:
    destination = Path(__file__).resolve().parent / "test_vectors" / "demo_capture.ndjson"
    source = build_source(
        LocalScenario(
            name="A",
            change_type="step",
            t_start_s=30.0,
            profile_seed=1,
            disturbance_seed=1,
        )
    )
    sink = FileSampleSink(destination)
    try:
        for sample in source:
            sink.send(
                MeasuredSample(
                    delta=sample.delta,
                    vx=sample.vx,
                    yaw_rate=sample.yaw_rate,
                    timestamp_s=sample.timestamp_s,
                    sequence=sample.sequence,
                )
            )
    finally:
        sink.close()
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

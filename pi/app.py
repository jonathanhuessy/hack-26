"""Offline Phase 1 classifier demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

try:
    from .contracts import ExecutionConfig
    from .detector import TurnDetector
    from .local_plant import LocalScenario, build_source
    from .model import dummy_model, load_weights
    from .noise import NoisySource
    from .pipeline import EdgePipeline
    from .sources import MatlabFeatureReplaySource, ReplaySampleSource
    from .transport.adapters import FileSampleSource
    from .transport.tcp import TcpConfig, TcpSampleSource
except ImportError:
    from contracts import ExecutionConfig
    from detector import TurnDetector
    from local_plant import LocalScenario, build_source
    from model import dummy_model, load_weights
    from noise import NoisySource
    from pipeline import EdgePipeline
    from sources import MatlabFeatureReplaySource, ReplaySampleSource
    from transport.adapters import FileSampleSource
    from transport.tcp import TcpConfig, TcpSampleSource


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
    tcp_listen: tuple[str, int] | None = None,
    transport_file: str | Path | None = None,
    status_interval_s: float = 0.5,
) -> int:
    default_weights = Path(__file__).resolve().parents[1] / "models" / "export" / "weights.mat"
    weight_path = Path(weights) if weights else (default_weights if default_weights.exists() else None)
    model = load_weights(weight_path) if weight_path else dummy_model()
    if tcp_listen:
        host, port = tcp_listen
        source = TcpSampleSource(TcpConfig(host=host, port=port, reconnect=True))
        source_name = f"tcp://{host}:{port}"
    elif transport_file:
        source = FileSampleSource(transport_file, realtime=realtime)
        source_name = str(transport_file)
    elif local or replay is None:
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
    events = []
    started_at = time.monotonic()
    last_status_at = started_at
    for sample in source:
        event = pipeline.push(sample)
        events.append(event)
        now = time.monotonic()
        if status_interval_s > 0 and now - last_status_at >= status_interval_s:
            age = max(0.0, now - started_at - sample.timestamp_s)
            print(
                f"status source={source_name} mode={_source_mode(source_name)} "
                f"detector={detector.state.value} samples={len(events)} "
                f"sample_age_s={age:.3f}"
            )
            last_status_at = now
    flushed = pipeline.flush()
    accepted = sum(event.result.status.value in ("accepted", "gap") for event in events)
    rejected = len(events) - accepted
    verdicts = [event.consumer_result.new_verdict for event in events
                if event.consumer_result is not None and event.consumer_result.new_verdict is not None]
    if flushed is not None and getattr(flushed, "new_verdict", None) is not None:
        verdicts.append(flushed.new_verdict)
    print(
        f"source={source_name} mode={_source_mode(source_name)} "
        f"realtime={realtime} samples={accepted} rejected={rejected}"
    )
    for verdict in verdicts:
        print(
            f"{verdict.window_id}: class={verdict.class_name} "
            f"confidence={max(verdict.class_probabilities):.3f} "
            f"delta_m_kg={verdict.delta_m_kg} k_f={verdict.k_f} "
            f"t={verdict.timestamp_s:.2f}s"
        )
    print(f"processed {len(events)} samples, {len(verdicts)} verdicts")
    print(f"detector_state={detector.state.value}")
    statuses = getattr(source, "statuses", ())
    for status in statuses:
        print(f"transport_status={status.kind} message={status.message}")
    return 0


def _source_mode(source_name: str) -> str:
    if source_name.startswith("tcp://"):
        return "separated"
    if source_name.startswith("local/"):
        return "co-located"
    return "replay"


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
    parser.add_argument(
        "--status-interval",
        type=float,
        default=0.5,
        help="seconds between live status lines; zero disables them",
    )
    parser.add_argument(
        "--tcp-listen",
        metavar="HOST:PORT",
        help="receive a separated PC stream over TCP",
    )
    parser.add_argument(
        "--transport-file",
        type=Path,
        help="replay a framed JSON transport capture",
    )
    args = parser.parse_args(argv)
    tcp_listen = None
    if args.tcp_listen:
        try:
            host, port_text = args.tcp_listen.rsplit(":", 1)
            tcp_listen = (host, int(port_text))
        except ValueError as exc:
            parser.error("--tcp-listen must use HOST:PORT")
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
        tcp_listen=tcp_listen,
        transport_file=args.transport_file,
        status_interval_s=args.status_interval,
    )


if __name__ == "__main__":
    sys.exit(main())

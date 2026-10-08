"""Send a local or file-backed sample stream to a Pi TCP receiver."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .local_plant import LocalScenario, build_source
    from .transport.adapters import FileSampleSource
    from .transport.tcp import TcpSampleSender
except ImportError:  # pragma: no cover
    from local_plant import LocalScenario, build_source
    from transport.adapters import FileSampleSource
    from transport.tcp import TcpSampleSender


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Pi hostname or address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--transport-file", type=Path)
    parser.add_argument("--scenario", choices=("nominal", "A", "B", "AB"), default="A")
    parser.add_argument("--change-type", choices=("constant", "step", "ramp"), default="step")
    parser.add_argument("--t-start", type=float, default=30.0)
    parser.add_argument("--t-end", type=float, default=0.0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--profile-seed", type=int, default=1)
    parser.add_argument("--disturbance-seed", type=int, default=1)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args(argv)

    if args.transport_file:
        source = FileSampleSource(args.transport_file, realtime=args.realtime)
        source_name = str(args.transport_file)
    else:
        source = build_source(
            LocalScenario(
                name=args.scenario,
                change_type=args.change_type,
                t_start_s=args.t_start,
                t_end_s=args.t_end,
                reverse=args.reverse,
                profile_seed=args.profile_seed,
                disturbance_seed=args.disturbance_seed,
            ),
            realtime=args.realtime,
        )
        source_name = f"local/{args.scenario}/{args.change_type}"

    sender = TcpSampleSender(args.host, args.port)
    sender.send_samples(source)
    print(f"sent source={source_name} destination=tcp://{args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

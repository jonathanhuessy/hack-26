"""Visualize a Python plant run and optionally send the same stream to a Pi."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

try:
    from .local_plant import LocalScenario, NOMINAL_PARAMS, VehicleEvent, build_source
    from .transport.adapters import FileSampleSink
    from .transport.tcp import TcpSampleSender
except ImportError:  # pragma: no cover
    from local_plant import LocalScenario, NOMINAL_PARAMS, VehicleEvent, build_source
    from transport.adapters import FileSampleSink
    from transport.tcp import TcpSampleSender


EVENT_NAMES = ("implement_attached", "tire_flat", "A", "B", "AB")


def parse_event_spec(
    spec: str,
    *,
    added_mass_kg: float,
    kf: float,
    reverse: bool = False,
) -> VehicleEvent:
    """Parse NAME:START[:END] using the training-represented event names."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("event must use NAME:START or NAME:START:END")
    name = parts[0]
    if name not in EVENT_NAMES:
        raise ValueError(f"unsupported event {name!r}; choose from {EVENT_NAMES}")
    try:
        start_s = float(parts[1])
        end_s = float(parts[2]) if len(parts) == 3 else None
    except ValueError as exc:
        raise ValueError("event times must be numeric seconds") from exc
    return VehicleEvent(
        name=name,
        start_s=start_s,
        end_s=end_s,
        reverse=reverse,
        added_mass_kg=added_mass_kg,
        kf=kf,
    )


def collect_samples(source: Iterable) -> list:
    """Materialize a deterministic run for plotting and repeatable sending."""
    return list(source)


def _series(samples: list, name: str, default: float = 0.0) -> np.ndarray:
    return np.asarray(
        [sample.diagnostics.get(name, default) for sample in samples],
        dtype=float,
    )


def _parameter_series(samples: list, index: int) -> np.ndarray:
    return np.asarray(
        [sample.diagnostics.get("params", NOMINAL_PARAMS)[index] for sample in samples],
        dtype=float,
    )


def plot_run(samples: list, *, title: str = "Python plant run"):
    """Create a four-panel figure for plant outputs and outgoing edge inputs."""
    if not samples:
        raise ValueError("cannot plot an empty sample stream")
    import matplotlib.pyplot as plt

    time_s = np.asarray([sample.timestamp_s for sample in samples])
    delta = np.asarray([sample.delta for sample in samples])
    vx = np.asarray([sample.vx for sample in samples])
    yaw_rate = np.asarray([sample.yaw_rate for sample in samples])
    ay = _series(samples, "ay")
    x = _series(samples, "X")
    y = _series(samples, "Y")

    figure, axes = plt.subplots(4, 1, figsize=(12, 10))
    axes[0].plot(x, y)
    axes[0].set_ylabel("Y [m]")
    axes[0].set_title(title)
    axes[0].axis("equal")
    axes[1].plot(time_s, delta, label="delta")
    axes[1].plot(time_s, vx, label="Vx")
    axes[1].set_ylabel("edge inputs")
    axes[1].legend(loc="upper right")
    axes[2].plot(time_s, yaw_rate, label="r")
    axes[2].plot(time_s, ay, label="ay")
    axes[2].set_ylabel("plant outputs")
    axes[2].legend(loc="upper right")
    axes[3].plot(time_s, _parameter_series(samples, 0), label="mass")
    axes[3].set_ylabel("mass [kg]")
    axes[3].set_xlabel("time [s]")

    _shade_events(axes[1:], samples, time_s)
    figure.tight_layout()
    return figure


def _shade_events(axes, samples: list, time_s: np.ndarray) -> None:
    labels = [str(sample.diagnostics.get("active_events", "")) for sample in samples]
    start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[start]:
            continue
        if labels[start]:
            for axis in axes:
                axis.axvspan(
                    time_s[start],
                    time_s[index - 1],
                    alpha=0.12,
                    color="tab:orange",
                )
        start = index


def _scenario_from_args(args: argparse.Namespace) -> LocalScenario:
    scenario = LocalScenario(
        name=args.scenario,
        change_type=args.change_type,
        added_mass_kg=args.added_mass_kg,
        kf=args.kf,
        t_start_s=args.t_start,
        t_end_s=args.t_end,
        reverse=args.reverse,
        profile_seed=args.profile_seed,
        disturbance_seed=args.disturbance_seed,
        events=(
            tuple(
                parse_event_spec(
                    item,
                    added_mass_kg=args.added_mass_kg,
                    kf=args.kf,
                    reverse=args.reverse,
                )
                for item in args.event
            )
            if args.event
            else None
        ),
    )
    if args.event and args.scenario != "nominal":
        raise ValueError("--event cannot be combined with --scenario")
    return scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("nominal", "A", "B", "AB"), default="nominal")
    parser.add_argument("--change-type", choices=("constant", "step", "ramp"), default="constant")
    parser.add_argument("--event", action="append", default=[], help="NAME:START[:END]; repeatable")
    parser.add_argument("--added-mass-kg", type=float, default=1000.0)
    parser.add_argument("--kf", type=float, default=0.8)
    parser.add_argument("--t-start", type=float, default=30.0)
    parser.add_argument("--t-end", type=float, default=0.0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--profile-seed", type=int, default=1)
    parser.add_argument("--disturbance-seed", type=int, default=1)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--tcp-host", help="send the generated stream to this Pi host")
    parser.add_argument("--tcp-port", type=int, default=8765)
    parser.add_argument("--capture", type=Path, help="also save the exact framed stream")
    parser.add_argument("--save", type=Path, help="save the visualization instead of showing it")
    args = parser.parse_args(argv)

    try:
        scenario = _scenario_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    samples = collect_samples(build_source(scenario))
    if args.tcp_host:
        TcpSampleSender(args.tcp_host, args.tcp_port).send_samples(
            samples,
            realtime=args.realtime,
        )
    if args.capture:
        sink = FileSampleSink(args.capture)
        try:
            for sample in samples:
                sink.send(sample)
        finally:
            sink.close()
    figure = plot_run(
        samples,
        title=f"Python plant: {'custom events' if args.event else args.scenario}",
    )
    if args.save:
        figure.savefig(args.save, dpi=140)
    else:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

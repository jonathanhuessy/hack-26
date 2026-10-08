# Phase 5 demo runbook

This runbook supports the Python edge demo in three modes:

1. co-located plant and detector;
2. a PC plant sending samples to the Pi over TCP;
3. prerecorded transport replay when live operation is unavailable.

The detector always receives the same three measured channels: `delta`, `Vx`,
and `r`. MATLAB/Simulink is used to generate and validate artifacts; it is not
required to run the Python demo.

## Prerequisites

- Python 3.10 or newer.
- NumPy and SciPy installed on the host or Pi.
- Repository root as the current directory.
- `models/export/weights.mat` for the trained model. Without it, the app uses
  deterministic dummy weights for plumbing tests.

Install the Python dependencies in the environment used for the demo:

```text
python -m pip install numpy scipy
```

## Preflight

Run this on the host and, before presenting, on the Pi:

```text
python -m pi.verify_phase5
```

The plant parity check requires the exported vector and can be added when it is
available:

```text
python -m pi.verify_phase5 --include-plant
```

## Demo A: co-located mode

Run the deterministic rear-ballast step at real-time speed:

```text
python -m pi.app --local --scenario A --change-type step --t-start 30 --realtime
```

The status lines show the mode, detector state, sample count, and sample age.
After each completed turn, the app prints the class, confidence, and available
regression values. A successful run ends with three verdicts and
`detector_state=idle`.

## Demo B: separated PC-to-Pi mode

On the Pi, start the receiver:

```text
python -m pi.app --tcp-listen 0.0.0.0:8765 --realtime
```

For a finite capture test, add `--tcp-no-reconnect` so the receiver exits
after the sender's `end` record:

```text
python -m pi.app --tcp-listen 0.0.0.0:8765 --tcp-no-reconnect
```

On the PC, send the same deterministic local plant:

```text
python -m pi.send_tcp --host PI_HOST --port 8765 \
  --scenario A --change-type step --t-start 30 --realtime
```

The sender and receiver must agree on the protocol, schema, units, feature,
and model versions during the handshake. A mismatch is rejected before any
sample reaches the detector. The receiver reports `connected`, `gap`,
`stale`, or `disconnected` transport status separately from the classifier
verdict.

To send an existing framed capture instead:

```text
python -m pi.send_tcp --host PI_HOST --port 8765 \
  --transport-file pi/test_vectors/demo_capture.ndjson --realtime
```

## Demo C: offline fallback

Use this when the PC-to-Pi link is unreliable:

```text
python -m pi.app --transport-file pi/test_vectors/demo_capture.ndjson
```

The capture is deterministic and uses the same handshake, sample validation,
and detector path as the TCP receiver. It should produce the documented
three-turn sequence for the `A` step scenario.

## Performance evidence

Host-side comparison:

```text
python -m pi.benchmark_local_demo --scenario A --change-type step
```

On the actual Pi, record the output of the same command plus:

```text
python --version
uname -a
free -h
```

Record hardware model, OS image, NumPy/SciPy versions, elapsed time, real-time
factor, peak memory, last verdict timestamp, and whether the run used local,
file, or TCP input. A real-time factor below `1.0` means the pipeline has
processing margin at 100 Hz.

## Troubleshooting

- `unsupported ... version`: use matching checked-in model and feature
  artifacts on both sides.
- `stale`: verify the sender is running, the address/port is reachable, and
  `--realtime` is not being starved by the host.
- `disconnected`: restart the receiver and sender; the detector does not
  fabricate a verdict or silently reset state.
- No verdicts: confirm the stream contains a complete maneuver with a turn and
  enough post-turn samples, then use the fallback capture to isolate transport
  from detector behavior.
- Low confidence: confirm `models/export/weights.mat` is present and rerun
  parity tests before changing demo inputs.

The planned Simulink streaming/dashboard artifact is outside this Python-first
Phase 5 scope. Python local, TCP, and file replay are the supported
presentation paths.

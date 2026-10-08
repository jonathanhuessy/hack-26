# Phase 5 evidence record

This record separates measurements made on the development host from
measurements that must be repeated on the target Raspberry Pi.

## Host verification

Command:

```text
python -m pi.verify_phase5
```

The command runs the package-safe Python test suite, including feature,
model, detector, local-demo, source-contract, and transport checks. Plant
numerical parity is a separate command because it uses the exported MATLAB
plant vector:

```text
python pi/test_plant_parity.py
```

## Host performance baseline

Command:

```text
python -m pi.benchmark_local_demo --scenario A --change-type step
```

Measured on the development host on 2026-10-07:

```text
duration_s=167.360000
elapsed_s=5.641472
realtime_factor=0.033709
peak_memory_mb=21.390292
samples=16737
verdicts=3
last_verdict_s=167.360000
```

The real-time factor is elapsed processing time divided by stream duration;
values below `1.0` have processing margin at 100 Hz. This is not a Raspberry
Pi measurement.

## Fallback replay evidence

Command:

```text
python -m pi.app --transport-file pi/test_vectors/demo_capture.ndjson --status-interval 0
```

The checked-in capture contains 16,737 samples, no rejected samples, and three
deterministic verdicts:

```text
turn-0: class=B confidence=0.866 k_f=0.7958766305268128 t=55.97s
turn-1: class=B confidence=0.881 k_f=0.741196071380733 t=111.76s
turn-2: class=B confidence=0.885 k_f=0.7260820780619457 t=167.36s
detector_state=idle
```

These values document the current checked-in model artifact and capture; they
are not a claim that the fallback scenario is a perfect physical label.

## Model evaluation status

The checked-in repository contains validation baselines in `plan.md` and the
exported model artifacts, but no completed `scripts/evaluate_models.m` test-set
report. Held-out confusion matrices, nominal false-alarm rate, regression MAE,
and step/ramp detection-delay plots must be generated in MATLAB before using
those metrics in slides. Do not substitute validation accuracy for held-out
accuracy.

## Raspberry Pi measurement record

Run the following on the target Pi and append the output and hardware details
to the presentation record:

```text
python --version
uname -a
free -h
python -m pi.verify_phase5
python pi/test_plant_parity.py
python -m pi.benchmark_local_demo --scenario A --change-type step
```

Also record the NumPy/SciPy versions, Pi model, OS image, input mode (local,
file, or TCP), and whether the full run stayed below real-time factor `1.0`.

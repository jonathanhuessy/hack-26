# Tractor property change detection (hackathon-26)

Detect changes in tractor properties (mounted rear ballast, front tire stiffness) from onboard signals, using a 3-state bicycle model as the simulator and small neural networks as classifier and regressor. Everything is built in MATLAB/Simulink and runs offline on a Raspberry Pi (manual Python port).

The full design, decisions and progress checklist are in [plan.md](plan.md). Read the "Progress" list there first.

## Status

Preparation phase (simulator, labelled raw dataset): see the checklist in `plan.md`. Hackathon-day work (features, training, Simulink inference model, Pi demo) is not started.

## Requirements

- MATLAB R2024b with Simulink, Control System Toolbox, Signal Processing Toolbox, Statistics and Machine Learning Toolbox, Deep Learning Toolbox.
- `arion_630_parameters.m` calls `tf2ss_observable` (from the `icons-engineering-toolbox`). It must be on the MATLAB path on your machine. It is not needed on the Raspberry Pi.
- Raspberry Pi side: Python 3 with NumPy and SciPy.

## Quickstart (MATLAB)

Open MATLAB in the repo root and run:

```matlab
addpath(pwd)                              % makes the +change_detector package visible
p0   = change_detector.nominalParams();   % nominal Arion 630 parameters

% 1. one maneuver (3 swaths with end-of-row turns), simulated on the plant
prof = change_detector.makeManeuverProfile(struct(), 1);      % seed 1
dist = change_detector.makeDisturbance(prof.t, struct(), 1);
out  = change_detector.simulatePlant(prof, dist, p0);
plot(out.X, out.Y); axis equal

% 2. same run with 1000 kg ballast on the rear hitch
pA   = change_detector.applyScenario(p0, struct('addedMassKg', 1000, 'mountXM', -1.2));
outA = change_detector.simulatePlant(prof, dist, pA);
```

Verification scripts (each prints PASS/FAIL):

| Script | Checks |
|---|---|
| `scripts/check_plant.m` | Simulink plant against `lsim`, steady-state yaw rate, parameter step and ramp |
| `scripts/check_trajectory.m` | Maneuver limits, turn geometry, parallel swaths; writes `data/check_trajectory.png` |
| `scripts/check_detectability.m` | How well each change shows up in the signals (see "Detectability" below); writes `data/check_detectability.png` |

## Dataset

Raw, noise-free time series at 100 Hz with per-sample labels. Sensor noise and features are added afterwards, so you can change them without re-simulating.

Google drive link: [here](https://drive.google.com/drive/folders/1F0uMHkr00N4wtjrQoFY74cG6VwrlfHsj?usp=drive_link)

```matlab
idx = load('data/index.mat').idx;                  % one row per run: split, class, dm, kf, seeds, ...
r   = load('data/runs/run_00001.mat');             % r.tt (timetable) and r.meta (struct)
tt  = change_detector.addSensorNoise(r.tt, [], 42);   % adds deltaMeas, VxMeas, rMeas, ayMeas
```

Measured signals the classifier may use: `deltaMeas` (front wheel angle), `VxMeas` (speed), `rMeas` (gyro yaw rate), `ayMeas` (lateral acceleration at the IMU). The clean columns (`delta`, `Vx`, `r`, `ay`) and the model states (`ydot`, `alphaF`, `X`, `Y`, `psi`) are for analysis only, they are not available on the tractor.

Labels:

| Column | Meaning |
|---|---|
| `cls` | 0 nominal, 1 A (rear ballast), 2 B (front tire stiffness), 3 A+B |
| `dm` | added mass at the rear hitch in kg (regression target A) |
| `kf` | B: front cornering stiffness factor $k_f = C_{af}/C_{af,nominal}$, 1 = nominal, 0.8 = front tires 20 % softer (regression target B) |
| `params` | true `[m lf lr Izz Caf Car sigmaF]` over time |
| `segment` | 1 straight at about 15 km/h, 2 speed transition, 3 end-of-row turn at about 5 km/h |
| `turnIdx`, `swath` | turn number (0 outside turns) and swath number |

`idx.split` is `train`, `val`, `test` or `demo`. Splits are per run, never per window. Demo runs have a change at a swath boundary (`changeType` = `step`) or a slow drift (`ramp`); `idx.reverse` = true means the change is removed instead of added. Use demo runs for the demo only, not for training.

Every run also contains soil variation (about +/-8 % on the tire stiffnesses), so "nominal" runs are not all identical. B values between 0.85 and 1.15 are deliberately not generated, that range is indistinguishable from soil variation.

Regenerate or extend the dataset (about 1400 runs, roughly 30 minutes, about 1 GB):

```matlab
addpath(pwd); addpath('scripts');
generate_dataset                       % skips run files that already exist
```

Each run is reproduced exactly from its row in `index.mat` (all seeds are stored): `change_detector.generateRun(idx(k,:), p0)`.

## Detectability

`check_detectability.m` compares a changed vehicle with the nominal one on the same run and computes $d^2 = \sum \Delta r^2/\sigma_r^2 + \sum \Delta a_y^2/\sigma_{a_y}^2$ (difference signal energy in units of sensor noise). $d^2 \ge 25$ counts as detectable. Results and the reasoning behind the dataset ranges are in `plan.md`, section P6.

## Raspberry Pi

`pi/plant.py` is a line-by-line Python port of the plant. On the Pi:

```bash
cd pi
python3 test_plant_parity.py     # compares against pi/test_vectors/plant_run.mat, prints PASS and the real-time factor
```

Regenerate the test vector after any change to the plant: run `scripts/export_pi_test_vectors.m` in MATLAB.

## Layout

```
+change_detector/   MATLAB package: parameters, scenarios, maneuver, disturbance, plant wrapper, dataset, sensor noise
models/             tractor_plant.slx (Simulink plant, fixed-step ode4, 0.01 s)
scripts/            checks, dataset generation, test vector export
data/               index.mat, config.mat, runs/ (large files are not in git)
pi/                 Python port for the Raspberry Pi and its parity test
plan.md             design, decisions, progress
```

## Conventions and assumptions

- y and the front wheel angle delta are positive left, yaw rate is positive counter-clockwise. Forward motion only (Vx > 0).
- The plant is driven directly by the measured front wheel angle (no steering actuator). Open-loop steering profile, no path-following controller.
- Tire stiffness does not change with added load. The IMU is assumed to sit above the rear axle.
- The disturbance levels (800 N lateral force, 400 Nm yaw moment) and sensor noise levels are assumptions, not calibrated to real logs.

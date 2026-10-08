# Tractor property change detection (hackathon-26)

Detect changes in tractor properties (mounted rear ballast, front tire stiffness) from onboard signals, using a 3-state bicycle model as the simulator and small neural networks as classifier and regressor. Everything is built in MATLAB/Simulink and runs offline on a Raspberry Pi (manual Python port).

The full design, decisions and progress checklist are in [plan.md](plan.md). Read the "Progress" list there first.

## Status

Preparation phase (simulator, labelled raw dataset): done, see the checklist in `plan.md`. Hackathon day: H1 (turn detection and features) is done; training, the Simulink inference model and the Pi demo are in progress.

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

### What is in the files

```
data/index.mat        idx     table, one row per run (1420 rows): scenario, labels, seeds
data/config.mat       cfg     struct: nominal parameters and settings used for generation
data/runs/run_XXXXX.mat
                      tt      timetable, the time series of one run (100 Hz)
                      meta    struct, the scenario and parameter sets of that run
```

#### `index.mat` → `idx` (table)

Use it to select runs (by split, class, magnitude) without opening the run files.

| Column | Meaning |
|---|---|
| `id` | run number, matches the file name `run_<id>.mat` |
| `split` | `train` (240 per class), `val` (60 per class), `test` (50 per class), `demo` (20) |
| `class` | `nominal`, `A` (rear ballast), `B` (front tire stiffness), `AB` (both) |
| `changeType` | `constant`: the changed parameters apply for the whole run (all train/val/test runs). `step`: the change happens at the end of turn `stepTurn`. `ramp`: linear drift from start to end of the run |
| `reverse` | demo only: `true` means the run starts changed and goes back to nominal (implement removed, tank draining) |
| `dm` | added mass in kg (0 if no A). For `step`/`ramp` runs this is the value at the changed end |
| `mountXM` | position of the added mass relative to the rear axle in m (−1.2 = 1.2 m behind it, the rear hitch) |
| `kf` | front cornering stiffness factor (1 if no B) |
| `soilCaf`, `soilCar` | soil variation factors on the front and rear cornering stiffness (0.92–1.08), present in every run, also nominal |
| `profileSeed`, `distSeed` | seeds of the maneuver (steering and speed) and of the soil and terrain disturbance |
| `stepTurn` | after which turn (1 or 2) a `step` change happens; drawn for all runs but only used by `step` runs |
| `duration` | run length in s (about 155–315 s) |
| `tStart`, `tEnd` | when the change happens in s. `constant`: 0/0. `step`: `tStart = tEnd`. `ramp`: 0 to the end of the run |
| `file` | file name in `data/runs/` |

The 20 demo runs: 8 × nominal → A, 4 × nominal → B, 3 × A → nominal (step, implement removed), 3 × A draining (ramp, reverse), 2 × A filling (ramp).

#### `config.mat` → `cfg` (struct)

| Field | Meaning |
|---|---|
| `nominalParams` | nominal Arion 630 parameters (without soil variation): `m` [kg], `L`, `lf`, `lr` [m], `Izz` [kg m²], `Caf`, `Car` [N/rad], `sigmaF` [m], steering limits, IMU position (`imuXFromRearAxleM` = 0: above the rear axle) |
| `sensorNoise` | headline sensor noise levels (gyro 0.01 rad/s, accelerometer 0.2 m/s², steering 0.2°). The complete noise model, including biases and vibration, is the default of `change_detector.addSensorNoise` |
| `matlabVersion`, `created` | provenance |
| `opts` | options passed to `generate_dataset` |

#### `run_XXXXX.mat` → `tt` (timetable)

One row every 0.01 s (about 15 000–31 000 rows). All values are `single`. Columns fall into three groups:

**1. What the tractor measures (inputs for features and the classifier).** These are stored clean; add noise with `change_detector.addSensorNoise`, which appends `deltaMeas`, `VxMeas`, `rMeas`, `ayMeas`.

| Column | Unit | Meaning |
|---|---|---|
| `delta` | rad | front wheel steering angle, positive left (plant input) |
| `Vx` | m/s | longitudinal speed (plant input) |
| `r` | rad/s | yaw rate $\dot\psi$, positive counter-clockwise |
| `ay` | m/s² | lateral acceleration at the IMU (above the rear axle) |

**2. Model internals (analysis and plots only, not measurable on the tractor).**

| Column | Unit | Meaning |
|---|---|---|
| `ydot` | m/s | lateral velocity at the centre of gravity |
| `alphaF` | rad | front slip angle (relaxation-length state) |
| `X`, `Y`, `psi` | m, m, rad | position and heading, integrated from the start of the run |

**3. Labels (ground truth per sample).** They change over time only in `step` and `ramp` runs; in `constant` runs they are the same in every row.

| Column | Meaning |
|---|---|
| `params` | 7 columns, the true plant parameters at that instant in the order of `meta.paramNames`: `[m lf lr Izz Caf Car sigmaF]` |
| `dm` | current added mass in kg (regression target A) |
| `kf` | current front stiffness factor (regression target B) |
| `cls` | current class: 0 nominal, 1 A, 2 B, 3 A+B. A counts when `dm` ≥ 250 kg, B when \|`kf` − 1\| ≥ 0.15 |
| `segment` | 1 straight (about 15 km/h), 2 speed transition, 3 end-of-row turn (about 5 km/h) |
| `turnIdx` | 1–3 inside a turn, 0 elsewhere. Use it to cut per-turn windows |
| `swath` | 1–3: swath number (a swath = straight + following turn) |

Typical use: build a window from `turnIdx == k`, take the noisy measured columns as input, and label the window with `dm`, `kf` and `cls` from the same rows (constant within a window except in demo runs).

#### `run_XXXXX.mat` → `meta` (struct)

| Field | Meaning |
|---|---|
| `run` | this run's row of `idx`, as a struct |
| `profile` | the drawn maneuver: `Vs`, `Vt` (straight and turn speed in m/s), `R` (turn radius in m), `w` (working width in m), `operatorStdRad` (size of the operator's steering corrections), `straightLengthM` (3 values), `turns` (per turn: `tStart`, `tEnd` in s, `dir` +1 left / −1 right, `isOmega`, `R`, `beta` = first arc angle of the omega turn) |
| `pStart` | parameter vector at the start of the run, `[m lf lr Izz Caf Car sigmaF]` |
| `pEnd` | parameter vector at the end of the run |
| `pNominal` | nominal parameter vector (the commissioning state, without soil variation) |
| `paramNames` | names of the 7 vector entries |
| `fs` | sample rate, 100 Hz |
| `matlabVersion` | provenance |

How `pStart` and `pEnd` relate to the scenario:

- `constant` runs: `pStart = pEnd` = nominal + soil variation + the change (A and/or B). For a nominal run they differ from `pNominal` only by the soil factors.
- `step`/`ramp` runs: `pStart` = nominal + soil, `pEnd` = nominal + soil + change. With `reverse = true` the two are swapped.
- `(pEnd - pNominal) ./ pNominal` gives the relative true change of each physical parameter. It shows, for example, that rear ballast changes `m`, `lf`, `lr` and `Izz` together, while B changes only `Caf`.

## Detectability

`check_detectability.m` compares a changed vehicle with the nominal one on the same run and computes $d^2 = \sum \Delta r^2/\sigma_r^2 + \sum \Delta a_y^2/\sigma_{a_y}^2$ (difference signal energy in units of sensor noise). $d^2 \ge 25$ counts as detectable. Results and the reasoning behind the dataset ranges are in `plan.md`, section P6.

## Turn detection and features (H1)

The classifier works per end-of-row turn. Each detected turn gives one feature vector computed from the measured signals only.

```matlab
addpath(pwd); addpath('scripts');
build_features          % all runs -> data/features.mat (about 2 min)
explore_features        % single-feature separation, baselines -> data/results/explore_features.png
select_features         % permutation importance and feature subsets (basis of the 18-feature choice)
plot_run(1100)          % visual check: path, detected turns, windows and trigger signals of one run
```

| Function | Purpose |
|---|---|
| `change_detector.turnTriggerConfig` | all trigger and window thresholds (shared by MATLAB, Simulink and the Pi) |
| `change_detector.findTurns` | turn trigger on measured $\delta$ and $r$: entry at filtered \|$\delta$\| > 5°, exit after 2 s below 3°, heading change ≥ 120° |
| `change_detector.turnWindow` | turn window (5 s before entry to 5 s after exit) and the preceding straight (last 20 s with $V_x$ > 3 m/s) |
| `change_detector.turnFeatures` | 35 features of one window; names in `turnFeatureNames` |
| `change_detector.selectedFeatureNames` | **the 18 features used for training and on the Pi**; they need only $\delta$, $V_x$ and $r$ |

`data/features.mat` contains `F` (one row per detected turn: `runId`, `split`, `class`, `changeType`, `turn`, `trueTurn`, `isOmega`, `cls`, `dm`, `kf` and the feature matrix `X`, 35 columns), `featureNames` and `info`. Sensor noise seed per run: 5000 + run id. The meaning of each selected feature and the validation results are in `plan.md`, H1.

Pick the selected columns by name:

```matlab
S = load('data/features.mat');
[~, cols] = ismember(change_detector.selectedFeatureNames(), S.featureNames);
X18 = S.F.X(:, cols);
```

## Raspberry Pi

`pi/plant.py` is a line-by-line Python port of the plant. On the Pi:

```bash
cd pi
python3 test_plant_parity.py     # compares against pi/test_vectors/plant_run.mat, prints PASS and the real-time factor
```

Regenerate the test vector after any change to the plant: run `scripts/export_pi_test_vectors.m` in MATLAB.

Test vectors for porting the turn trigger and features (`pi/features.py`): `pi/test_vectors/features_<class>.mat`, one validation run per class (nominal, A, B, AB) written by `scripts/export_feature_test_vectors.m`. Each file holds the measured signals of the whole run, the `findTurns` result, the window and straight row ranges, the expected 35 features per turn, the indices of the 18 selected features, and all filter coefficients. Indices are 1-based (MATLAB); subtract 1 in Python. The header of the script lists all fields.

## Layout

```
+change_detector/   MATLAB package: parameters, scenarios, maneuver, disturbance, plant wrapper, dataset, sensor noise,
                    turn trigger, windows, features
models/             tractor_plant.slx (Simulink plant, fixed-step ode4, 0.01 s)
scripts/            checks, dataset generation, features (build, explore, select, plot_run), test vector export
data/               index.mat, config.mat, features.mat, results/, runs/ (large files are not in git)
pi/                 Python port for the Raspberry Pi, parity tests, test_vectors/
plan.md             design, decisions, progress
```

## Conventions and assumptions

- y and the front wheel angle delta are positive left, yaw rate is positive counter-clockwise. Forward motion only (Vx > 0).
- The plant is driven directly by the measured front wheel angle (no steering actuator). Open-loop steering profile, no path-following controller.
- Tire stiffness does not change with added load. The IMU is assumed to sit above the rear axle.
- The disturbance levels (800 N lateral force, 400 Nm yaw moment) and sensor noise levels are assumptions, not calibrated to real logs.

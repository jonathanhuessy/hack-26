# Plan: Detecting tractor property changes with a bicycle model on a Raspberry Pi

**Summary:** Detect and quantify changes in tractor properties (mounted ballast/implement, front tire stiffness) from onboard signals, using a 3-state bicycle model ($\dot y$, $\dot\psi$, $\alpha_f$) as the simulator and small neural networks as classifier and regressor. Everything is built in MATLAB/Simulink and runs fully offline on a Raspberry Pi (no connectivity needed). The plant is driven directly by the measured front wheel angle, so no steering actuator model is needed. The trajectory is open-loop: 3 swaths, each a near-straight line at 15 km/h with soil and terrain disturbances, followed by an end-of-row omega turn at 5 km/h (as in the real logs).

Work is split into a **preparation phase (now)**, which delivers a validated simulator and a labelled raw dataset, and the **hackathon day**, which covers features, training, the Simulink inference model and the Pi demo. Several conclusions below are hand calculations from the parameter file; the preparation phase checks them.

## Signals (glossary)

| Symbol | Meaning | Source | Seen by classifier |
|---|---|---|---|
| $\delta$ | Front wheel steering angle (positive left) | Steering angle sensor | Yes (plant input, plus small sensor noise) |
| $V_x$ | Longitudinal speed | Wheel speed / radar / GNSS | Yes (plant input) |
| $r = \dot\psi$ | Yaw rate (positive counter-clockwise) | Gyro z-axis of the IMU | Yes |
| $a_y$ | Lateral acceleration at the IMU, $\ddot y + V_x r$ plus lever-arm terms | Accelerometer y-axis | Yes |
| $\dot y$, $\alpha_f$ | Lateral velocity, front slip angle | Model states | No (not measured) |
| $X, Y, \psi$ | Position and heading | Integrated kinematics | No (plots only) |

## Key physics finding: what each speed contributes (straights 15 km/h, turns 5 km/h)

These are rough estimates using [arion_630_parameters.m](arion_630_parameters.m) ($L=2.81$ m, $K_{us}\approx-0.005$ rad/(m/s²) nominal):

| Speed | $K_{us}V_x^2$ relative to $L$ (steady-state effect) | Yaw lag $\tau$ nominal | $\tau$ with +1.5 t rear |
|---|---|---|---|
| 5 km/h (1.39 m/s) | 0.4 % | 0.29 s | 0.11 s |
| 10 km/h (2.78 m/s) | 1.4 % | 0.15 s | 0.06 s |
| 15 km/h (4.17 m/s) | 3.2 % | 0.10 s | 0.04 s |

- **Steady-state turning:** the yaw gain is $\frac{V_x/L}{1+K_{us}V_x^2/L}$ times $\delta$. In the 5 km/h turns this is practically kinematic, so the steady part of the turn carries almost no information. At 15 km/h, +1.5 t on the rear changes $K_{us}$ from about −0.005 to about −0.010, a yaw-gain change of about 3 %. On the straights the only steering is the small operator corrections, so this effect is only visible if those corrections are large enough (see step 4).
- **Transients:** at low speed the rear tire damping dominates, and the yaw response is roughly a first-order lag:

  $$\tau \approx \frac{C_{ar}\,l_r^2\,\sigma_f}{C_{af}\,l_f^2\,V_x}$$

  At 5 km/h this lag is the strongest single signal: +1.5 t on the rear cuts it from about 0.29 s to about 0.11 s. It is observable at 100 Hz sampling because the measured wheel angle is the model input, so no actuator delay sits in between.
  - **Mass alone can't be identified from the lag:** $m$ and $I_{zz}$ nearly cancel out of $\tau$.
  - **Centre-of-gravity shift can be identified:** it enters as $(l_r/l_f)^2$.
- **Two speeds add information:** the steady-state term scales with $V_x^2$ and the lag with $1/V_x$. Turn-in and turn-out lags at 5 km/h plus small-signal gain on the 15 km/h straights give two different combinations of the parameters, which helps tell A from B.
- **Front tire changes can be identified,** but mostly as the ratio $C_{af}/\sigma_f$ in the lag; $C_{af}$ also enters $K_{us}$ on its own. Treat $C_{af}$ and $\sigma_f$ as one "front tire stiffness factor".
- **Speeds:** straights at 15 km/h (a realistic sprayer or spreader working speed), end-of-row turns at 5 km/h, matching the real logs. Lateral acceleration in the turn is only about 0.2 m/s² at $R$ = 9 m, so the tires are deep in the linear range.
- **Main risk:** ballast that moves the CG forward and a softer front tire both make $\tau$ larger, so they may be hard to tell apart. The preparation phase checks this (step P6).
- **Side notes:**
  - The nominal parameters describe a mildly oversteering vehicle (critical speed about 23 m/s). That's fine in the field.
  - The file computes $I_{zz}=m\,l_f\,l_r$. Ballast scenarios must update $m$, $l_f$, $l_r$ and $I_{zz}$ together (parallel-axis rule).

## Recommended targets (at most 2)

- **A. Mounted ballast or implement:** added mass $\Delta m$ at a known mount point (rear 3-point hitch, or front weight). From it, compute the new $m$, CG, and $I_{zz}$. Report $\Delta m$ and the CG shift. A towed cart is a different case: it makes the vehicle articulated, which the bicycle model doesn't cover.
- **B. Front cornering stiffness factor** $k_f = C_{af}/C_{af,0}$, for tire pressure, tire type or soil changes.
- **Fallbacks if P6 shows A and B can't be separated:** rear cornering stiffness, or reduce the task to classifying the direction of change (front-heavy vs rear-heavy vs tire).

## Sudden or slow changes?

- **Sudden (step) changes are simpler and the primary target:** for example attaching or removing a mounted implement or front weight, or changing tire pressure. Parameters are constant within each window, labels are clean, and every window gets exactly one answer.
- **Slow changes (drift)** such as a mounted sprayer tank draining or a mounted hopper filling: parameters vary inside a window, labels are time-varying, and you need tracking rather than classification.
- **Recommendation:** train on piecewise-constant parameters only. Because the regressor outputs a magnitude per window, a slow drift then shows up for free as a sequence of window estimates (for example $\Delta m$ decreasing from turn to turn while a tank drains). Generate a few drift runs in preparation, but use them only as a demo and stretch goal, not for training.

## Is the estimator overkill?

No, as long as it's just a regression output on the same features (two extra outputs, negligible effort). A full online estimator (EKF, recursive least squares) would be overkill. As an offline baseline and sanity check, add a per-window grey-box fit of the two parameters with `fminsearch` (base MATLAB; Optimization and System Identification toolboxes are not installed).

## Toolboxes (checked in MATLAB R2024b)

- **Installed:** MATLAB, Simulink, Control System, Signal Processing, Deep Learning, Simscape, Simscape Multibody. Statistics and Machine Learning is being reinstalled and assumed available.
- **Not installed:** System Identification, Optimization, Parallel Computing, MATLAB Coder, Simulink Coder, Embedded Coder.
- **Consequences:**
  - Statistics and Machine Learning serves exploration and model selection (feature ranking with `fscmrmr`, quick baselines with `fitcensemble`, `fitcnet`, `fitclinear`, `fitcdiscr`). The deployed models must be easy to port by hand: small MLPs (`fitcnet`/`fitrnet` or Deep Learning Toolbox `trainnet`) or linear models. No tree ensembles on the Pi.
  - Data generation runs serially with `sim` and Fast Restart (no `parsim`).
  - Control System Toolbox (`ss`, `lsim`) serves to cross-check the Simulink plant.
  - **Raspberry Pi deployment is a manual port** (no code generation). See H5.

## Inference architecture: buffer → features → classifier

Yes, that's the right structure. Recommended details:

- **Buffer:** a ring buffer of the 4 measured signals ($\delta$, $V_x$, $r$, $a_y$) at 100 Hz, holding the last 60 s or so (about 24 000 values), enough for a full omega turn plus margins.
- **Trigger:** event-based rather than "buffer full". When a turn ends (|$\delta$| falls back below a threshold with hysteresis), take the window from about 5 s before turn entry to about 5 s after turn exit, plus a summary of the preceding straight. Every window then contains the same kind of manoeuvre, which makes features much more consistent.
- **Fallback:** fixed sliding windows (e.g. 30 s, hop 10 s). Simpler, but windows mix straights and turns, so the classifier has to cope with very different content.
- Because the dataset stores labels per sample (P7), both windowing schemes can be tried on hackathon day.

## Frequency-band features

Yes, in two bands that match the physics:

- **Low band (below about 0.2 Hz):** steady-state yaw gain and $a_y / (V_x r)$. This carries the understeer change ($K_{us}$), mainly on the 15 km/h straights.
- **Mid band (about 0.2–1.5 Hz):** gain and phase of $\delta \to r$, i.e. the yaw lag $\tau$. This is excited by the turn-in and turn-out ramps at 5 km/h.
- **High band (above about 3 Hz):** mostly sensor noise and vibration with no parameter information. Use it at most to normalise for the noise level.
- **Implementation:** fixed IIR band-pass filters (coefficients designed once offline with Signal Processing Toolbox and stored as constants) plus band power and cross-power. On the Pi, `filter` maps one-to-one to `scipy.signal.lfilter`. `tfestimate` is fine for offline exploration but stays off the deployment path.
- **Portability rule for all deployed feature code:** only basic operations (filter with fixed coefficients, sums, max, cross-correlation written out explicitly), so the Python port is line-by-line.

## Preparation phase (now): validated simulator and labelled raw dataset

**Goal:** arrive at the hackathon with a trustworthy simulator and all raw data, so the day can focus on features, training and the Pi demo. The dataset stores **raw time series with per-sample labels, not features**, so feature design stays open on the day. Two cheap checks (P6 and P8) make sure the data actually contains the signal before you commit to generating everything.

### P1. Project skeleton
- Folders: `+tcd/` (functions), `models/` (Simulink), `scripts/`, `data/`.
- `tcd.nominalParams()` wraps [arion_630_parameters.m](arion_630_parameters.m).
- `tcd.applyScenario(p0, scen)` turns a scenario ($\Delta m$, mount position, $k_f$, soil variation) into a consistent parameter set: $m$, $l_f$, $l_r$, $I_{zz}$ via the parallel-axis rule, $C_{af}$, $C_{ar}$, $\sigma_f$.

### P2. Trajectory: steering angle and speed profile
- `tcd.makeManeuverProfile(cfg, seed)` returns $\delta(t)$ (the front wheel angle directly; no actuator model) and $V_x(t)$ at 100 Hz:
  - **3 swaths:** straight → end-of-row turn, repeated 3 times. The net turn direction alternates L, R, L.
  - **Straight:** about 100 m at 15 km/h (4.17 m/s), about 24 s. Steering near 0° plus slow operator corrections (low-pass noise below about 0.5 Hz, amplitude randomised per run at about 0.5–1.5° std, which matches the real logs). These corrections are the only excitation at 15 km/h; below about 0.3° the yaw response would get lost in gyro noise.
  - **Speed transitions:** 15 → 5 km/h at about 0.5 m/s² before each turn (about 5.5 s), and back up after it.
  - **End-of-row omega turn at 5 km/h (1.39 m/s)**, the primary turn type. It's the standard turn when the working width $w$ is smaller than the minimum turning diameter (about 16.4 m), which fits implements typical for an Arion 630 class tractor (3–6 m).
    - **Geometry:** three arcs of radius $R$: away from the next lane by $\beta$, towards it by $180^\circ + 2\beta$, away again by $\beta$. The tractor lands exactly on the next lane when $w = R(4\cos\beta - 2)$, i.e. $\beta = \arccos\frac{w/R + 2}{4}$. Example: $w$ = 6 m, $R$ = 9 m gives $\beta \approx 48^\circ$, about 59 m of path and about 42 s at 5 km/h.
    - **Steering:** $0 \to -17.3^\circ \to +17.3^\circ \to -17.3^\circ \to 0$, at most 25 °/s, $|\delta| \le 19^\circ$ (real steering limits). That is about 100° of total steering travel per turn, versus about 35° for a U-turn, including two full-lock reversals through zero. These reversals are the best mid-band excitation for the yaw-lag features, and having both signs in every turn helps cancel gyro bias.
  - **U-turn variant:** a single semicircle when $w \ge 2R$ (with a short straight between if $w > 2R$). It's the same generator with $\beta = 0$, so it costs nothing to add. Use it for about 30 % of turns so the features don't depend on one turn shape.
  - **Not used:** K-turns or fishtail turns. They need reversing and stopping, and the model has $1/|V_x|$ terms that become singular at standstill.
  - **Total run:** about 230 s with omega turns.
  - **Randomise per run:** straight length ±20 %, straight speed 13–16 km/h, turn speed 4–6 km/h, $R$ in 8.5–11 m, working width 3–6 m, turn type (about 70 % omega, 30 % U-turn), first turn direction, operator-correction seed.
- `tcd.makeDisturbance(cfg, seed)`: soil and terrain effects as band-limited noise entering as lateral force $F_{y,d}$ and yaw moment $M_{z,d}$. Size them so heading drifts only a few degrees per 100 m straight.

### P3. Simulink plant model `models/tractor_plant.slx`
- Fixed-step `ode4`, 0.01 s. All MATLAB Function blocks compatible with code generation.
- **Inputs:** $\delta$, $V_x$, $F_{y,d}$, $M_{z,d}$ from From Workspace or `setExternalInput`.
- **Parameter Scheduler:** outputs a parameter bus. Modes: constant, step at `t_change`, or linear ramp between `t_start` and `t_end` (for the drift demo runs).
- **Tractor Plant:** a MATLAB Function block builds $A(p, V_x)$ and $B(p, V_x)$ from the equations in the image at every step, plus disturbance inputs, followed by an Integrator for $[\dot y, \dot\psi, \alpha_f]$. Kinematics integrate X, Y, ψ.
- **Outputs (clean, without sensor noise):** $\delta$, $V_x$, $r$, $a_y$ at the IMU position, $\dot y$, $\alpha_f$, X, Y, ψ, and the true parameter signals ($m$, $l_f$, $l_r$, $I_{zz}$, $C_{af}$, $C_{ar}$, $\sigma_f$).
- **Sensor noise is added afterwards** by `tcd.addSensorNoise(run, noiseCfg, seed)`: gyro about 0.01 rad/s, accelerometer about 0.2 m/s² plus vibration, steering angle about 0.2°, plus biases. Keeping noise out of the stored data lets you change noise levels on the day without re-simulating.

### P4. Plant verification (`scripts/check_plant.m`)
- Simulink output matches `lsim` of the same `ss` model (Control System Toolbox) at constant $V_x$.
- The steady-state yaw rate in a constant-steering turn matches the analytic $\frac{V_x\delta}{L + K_{us}V_x^2}$.
- A parameter step at `t_change` shows up in the parameter signals and in the response.

### P5. Trajectory check (`scripts/check_trajectory.m`)
- XY plot shows 3 swaths and 3 turns, plus plots of $\delta(t)$, $V_x(t)$ and the steering rate.

### P6. Detectability and separability check (`scripts/check_detectability.m`)
This replaces the earlier "Phase 0" and is the go/no-go gate for the dataset design.
- Simulate the same trajectory and seed, without noise, for nominal and for a sweep of A, B and A+B magnitudes.
- **Detectability** of each change: $d^2 = \sum_k \Delta r_k^2/\sigma_r^2 + \sum_k \Delta a_{y,k}^2/\sigma_{a_y}^2$ over one run, where $\Delta$ is the difference from nominal and $\sigma$ the sensor noise. Report it separately for turn windows and straights.
- **Separability:** cosine similarity between the difference signals of A and B, $\cos\theta = \frac{\langle \Delta r_A, \Delta r_B\rangle}{\lVert\Delta r_A\rVert\,\lVert\Delta r_B\rVert}$, using magnitudes that give similar $d^2$.
- Outcomes:
  - The smallest change with $d^2 \ge 25$ (about a "5σ" detection) sets the lower end of the label grid.
  - If $|\cos\theta| > 0.9$, switch to the fallback targets before generating data.
  - Confirm that the operator-correction amplitude range (0.5–1.5°) makes the straights informative.

### P7. Scenario grid, labels and data generation (`scripts/generate_dataset.m`)
- **Classes:** Nominal / A / B / A+B. Every run, including nominal, gets soil variation of ±5–10 % on $C_{af}$ and $C_{ar}$.
- **Ranges:** A: $\Delta m$ from the P6 minimum up to 2000 kg at the rear hitch (about 1.2 m behind the rear axle), optionally front weights 0–1500 kg. B: $k_f$ 0.6–1.2, excluding a band around 1 sized by P6.
- **Change types:**
  - Constant parameters for the whole run: the bulk, used for training and validation.
  - Step at a random swath boundary: test and demo runs.
  - Slow ramp: a few drift demo runs.
- **Size:** for example 300 constant runs per class, 200 held-out test runs with unseen magnitudes and seeds, and 20 step and drift demo runs, about 1420 runs in total. Serial `sim` with Fast Restart.
- **Labels:**
  - **Per run** (index table): run id, split (train/val/test/demo), class, change type, $\Delta m$, mount position, $k_f$, soil factors, `t_change` or ramp times, all seeds, trajectory config (speeds, $R$, lengths, turn directions), file name.
  - **Per sample:** the true parameter signals, a segment label (straight / transition / turn), the turn index, and the current class. With these, any windowing scheme can be labelled automatically on the day.
- **Splits** are assigned per run (never per window), so no leakage.

### P8. Smoke test of a single feature (`scripts/smoke_feature.m`)
- On about 50 runs per class, add sensor noise and compute one simple feature: turn-in lag from cross-correlation of $\delta$ and $r$.
- Show histograms by class. This is not the final pipeline, only evidence that classification will work.

### P9. Storage
- `data/runs/run_00001.mat` … : one file per run, containing timetable `tt` (signals from P3, stored as `single` at 100 Hz) and struct `meta`. About 1 MB per run, about 1.4 GB in total.
- `data/index.mat`: run index table from P7.
- `data/config.mat`: generator configs, noise config, nominal parameters, MATLAB version, generation date.
- Access on the day with `fileDatastore` or a simple loop.

### P10. Pi readiness (manual port)
- Raspberry Pi OS with Python 3, NumPy and SciPy installed and tested on the actual Pi. Optionally matplotlib or a small terminal or web dashboard for the demo.
- Port the plant now (it's fixed): `pi/plant.py` with the same RK4 step as `ode4` at 0.01 s.
- Parity test: the same input profile run in Simulink and in `pi/plant.py` gives a maximum relative difference below 1e-6 on $r$ and $a_y$. Reference outputs are exported from MATLAB as test vectors (`.mat` read with `scipy.io.loadmat`, or CSV).
- Measure the Pi's real-time factor for the plant loop (target: well below 1).

### Preparation success criteria (exit checklist)
1. **Trajectory:** XY plot shows 3 swaths and 3 turns. $|\delta| \le 19^\circ$, $|\dot\delta| \le 25$ °/s, speeds 15 and 5 km/h ±randomisation. Net heading change per turn 180° ± 15°, and omega turns land within about 1 m of the next lane when run without disturbances. Heading drift below 5° per straight.
2. **Plant correct:** Simulink vs `lsim` maximum relative error below 1e-6. Steady-state yaw rate within 0.5 % of the analytic value. Parameter step and ramp visible in the logged parameter signals.
3. **Signal present:** P6 shows $d^2 \ge 25$ for the smallest change that will be labelled "changed", and $|\cos\theta_{AB}| < 0.9$ (or a documented switch to fallback targets).
4. **Dataset complete:** every run in the index has a file, no NaN or Inf, classes balanced, splits fixed. Regenerating any run from its seeds reproduces it exactly.
5. **Smoke test:** the turn-in lag feature visibly separates nominal from the largest A change.
6. **Pi ready:** Python environment works on the Pi, `pi/plant.py` passes the parity test against Simulink, and the real-time factor is measured.

## Hackathon day

### H1. Features (MATLAB functions, compatible with code generation)
- Windowing per turn (event-based) as described in the inference architecture section.
- Time-domain features:
  - Turn-in and turn-out lag between $\delta$ and $r$.
  - Steady-state yaw gain in the turn compared with $V_x\delta/L$.
  - Small-signal $\delta \to r$ gain on the preceding straight.
  - Residuals of the nominal model: RMS and peak for $r$ and $a_y$.
  - $a_y / (V_x r)$.
  - Turn speed.
- Band features: low and mid band gain and phase, as described in the frequency-band section.

### H2. Training
- Explore first with Statistics and Machine Learning: feature ranking (`fscmrmr`), quick baselines (`fitcensemble`, `fitcnet`) to see the achievable accuracy.
- Deployed models must be portable:
  - Classifier: small MLP with 4 classes. Either `fitcnet` (weights in `LayerWeights`/`LayerBiases`) or Deep Learning Toolbox (`featureInputLayer` → 2 × fully connected 16–32 ReLU → softmax).
  - Regressor: same structure with 2 outputs ($\Delta m$, $k_f$), via `fitrnet` or `trainnet`.
  - If a linear model (`fitclinear`, linear regression) is almost as good, prefer it: even simpler to port and to explain.
- Train on the train split, tune on the validation split.
- Export weights, biases and feature normalisation (mean, std) to `models/export/weights.mat`.

### H3. Evaluation
On the test split:
- Confusion matrix.
- False-alarm rate on nominal runs.
- Magnitude errors: $|\Delta m|$ MAE and $|k_f|$ MAE.
- Detection delay (in turns) on the step runs.
- Tracking on the drift runs.

### H4. Simulink inference model `models/tractor_change_detection.slx`
- References `tractor_plant.slx` as a Model block, followed by:
  - Sensors subsystem.
  - Ring buffer and Feature Extraction (MATLAB Function block).
  - Triggered Classifier and Regressor as MATLAB Function blocks doing the MLP forward pass with the exported weights (identical maths to the Pi port).
  - Median over the last K = 3 turns.
  - Dashboard: XY plot, $r$ measured vs nominal, current verdict.
- Demo: nominal for swath 1, +1500 kg rear before swath 2. The verdict should update after turns 2 and 3.

### H5. Raspberry Pi deployment (manual port to Python)
- `pi/plant.py` (ported in P10), `pi/features.py` (line-by-line port of the MATLAB feature functions), `pi/model.py` (MLP forward pass with NumPy, about 10 lines), `pi/app.py` (loop: plant with injected change → sensor noise → ring buffer → turn trigger → features → inference → verdict display).
- **Parity tests** against MATLAB test vectors: features (relative difference below 1e-6), class probabilities and magnitudes (below 1e-5), on at least one run per class.
- Runs fully offline on the Pi: no network needed at any point.

## Relevant files

- [arion_630_parameters.m](arion_630_parameters.m): source of the nominal values. Uses `frontTireRelaxationLengthM` and `vehicleZInertiaKgmm` (recompute it per ballast scenario instead of reusing it). The steering actuator fields are no longer used.
- New files:
  - `+tcd/`: `nominalParams`, `applyScenario`, `makeManeuverProfile`, `makeDisturbance`, `bicycleMatrices`, `addSensorNoise`, and on hackathon day the feature functions.
  - `models/tractor_plant.slx` (preparation), `models/tractor_change_detection.slx` (hackathon day).
  - `scripts/`: `check_plant.m`, `check_trajectory.m`, `check_detectability.m`, `generate_dataset.m`, `smoke_feature.m`, and on the day `train_models.m`, `run_demo.m`.
  - `data/`: `runs/`, `index.mat`, `config.mat`.
  - `pi/`: `plant.py` (preparation), `features.py`, `model.py`, `app.py`, parity tests (hackathon day).

## Verification

1. Preparation exit checklist above.
2. Held-out test set: class accuracy above 90 % after aggregating K turns, false alarms on nominal runs below 5 %, magnitude errors reported as $|\Delta m|$ MAE and $|k_f|$ MAE.
3. Simulink inference matches offline MATLAB `predict` on the same feature vectors.
4. Pi output matches MATLAB on the same input run (same verdicts, magnitudes within rounding).
5. Demo run: a change injected mid-run is detected within K turns.

## Decisions

- **Must deploy and run on a Raspberry Pi with no connectivity** (hackathon themes: "AI on the edge" with limited or intermittent connectivity, and "hardware + physical AI").
- No steering actuator model. The plant takes the front wheel angle directly; the trajectory respects the real limits (19°, 25 °/s).
- Open-loop steering (no path-following controller), 3 swaths with 3 end-of-row turns. Straights at 15 km/h, turns at 5 km/h (as in the real logs), prescribed speed profile.
- Simulation only.
- MATLAB/Simulink for simulation, features, training and inference. The Pi runs a manual Python/NumPy port (plant, features, MLP forward pass), verified by parity tests against MATLAB.
- Deployed models are small MLPs or linear models (easy to port); Statistics and Machine Learning is used for exploration and baselines.
- At most 2 properties. Primarily sudden (step) changes; slow drift only as a demo.
- Mass is always modelled together with its CG and $I_{zz}$ effects, never alone.
- The dataset stores clean raw signals with per-sample labels; sensor noise and features are applied afterwards.

## Out of scope

- Real sensor interfaces.
- Nonlinear tires and roll dynamics.
- Towed carts and trailers (articulated vehicle).
- Online EKF estimation.
- Path-following controller.
- Longitudinal dynamics (speed is a prescribed input).
- Road transport speeds (up to 35 km/h); possible extension if field speeds are not enough.
- Implement draft forces (for example a plough in the ground).

## Open questions

- None at the moment. The turn type is decided: omega primary, U-turn as a 30 % variant.

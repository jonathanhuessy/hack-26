# Repository instructions

## Project overview

This repository detects tractor property changes from measured steering angle,
forward speed, and yaw rate. MATLAB/Simulink generates and validates the
model; Python runs the deployed detector and classifier on a Raspberry Pi.

The PC is the TCP client. The Raspberry Pi is the TCP listener and runs the
edge pipeline, classifier, and regressors. The Pi receives only:

```text
[delta, Vx, r]
```

`ay`, labels, model states, and simulator diagnostics are optional replay
metadata and are not detector inputs.

## Supported operator workflows

### Local PC/Pi simulation

From the repository root:

```bash
python -m pi.app --local --scenario A --change-type step --t-start 30
python -m pi.benchmark_local_demo --scenario A --change-type step
```

### Streamlit PC UI

Install the PC UI dependencies and launch:

```bash
python -m pip install -r requirements-pc-ui.txt
streamlit run pi/streamlit_app.py
```

For the MATLAB demo A workflow:

1. In MATLAB, run `export_demo_playback_trajectories`.
2. Confirm `data/generated_trajectories/matlab/demo_a.mat` exists.
3. In Streamlit, select `MATLAB demo A`.
4. Set the Pi host/IP and TCP port `8765`.
5. Start at `1x` playback before increasing speed.

### Raspberry Pi listener

On the Pi:

```bash
cd ~/hack-26
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy
python -m pi.app \
  --tcp-listen 0.0.0.0:8765 \
  --realtime \
  --status-interval 0.5
```

The listener waits silently for the PC connection. Check its state with:

```bash
ss -ltnp | grep ':8765'
ss -tnp | grep ':8765'
```

If the port is occupied, identify and stop the stale listener:

```bash
ps -ef | grep '[p]i.app'
kill <PID>
```

## Validation

Run the Python checks from the repository root:

```bash
python -m unittest discover -s pi -p "test_*.py"
python -m pi.validate_streaming
```

The plant and transport checks should pass before a Pi TCP demo. Classifier
parity requires `models/export/weights.mat` and matching MATLAB-generated
fixtures in `pi/test_vectors/`.

## Model and artifact rules

- Do not train models on the Pi.
- Keep `models/export/weights.mat` synchronized between PC and Pi.
- Regenerate probability and detector fixtures whenever the exported model
  changes.
- Do not use the deterministic dummy model for a classification demo.
- Do not commit runtime logs, passwords, SSH keys, or other secrets.
- Treat generated captures and large simulation outputs as temporary unless
  explicitly requested for version control.

## Change and test expectations

- Preserve the three-input edge contract and the PC/Pi transport handshake.
- Keep local, file-replay, and TCP paths on the same `EdgePipeline`.
- Run focused tests for changed code, then the relevant full Python suite.
- Do not force-push or discard branch history unless the user explicitly
  approves it.

"""Parity test: pi/plant.py against the Simulink reference run. Also prints the real-time factor."""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from plant import simulate

LIMIT = 1e-6
ref = loadmat(Path(__file__).parent / "test_vectors" / "plant_run.mat", squeeze_me=True)

t0 = time.perf_counter()
out = simulate(ref["delta"], ref["Vx"], ref["Fyd"], ref["Mzd"], list(ref["p0"]), list(ref["p1"]),
               float(ref["tStart"]), float(ref["tEnd"]), float(ref["imuX"]))
elapsed = time.perf_counter() - t0
duration = ref["t"][-1]

ok = True
for name in ("r", "ay", "ydot", "alphaF", "X", "Y", "psi"):
    err = np.max(np.abs(out[name] - ref[name])) / np.max(np.abs(ref[name]))
    ok &= err < LIMIT
    print(f"{name:7s} max rel diff {err:.2e}")
err = np.max(np.abs(out["params"] - ref["params"])) / np.max(np.abs(ref["params"]))
ok &= err < LIMIT
print(f"params  max rel diff {err:.2e}")
print(f"real-time factor {elapsed / duration:.4f} ({elapsed:.2f} s for {duration:.0f} s of simulation)")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

"""Bicycle-model plant, ported line by line from models/tractor_plant.slx.

States x = [ydot, r, alphaF, X, Y, psi]; p = [m, lf, lr, Izz, Caf, Car, sigmaF].
Fixed-step RK4 like Simulink ode4: stage inputs are linearly interpolated between
samples, so the midpoint stages use the average of two neighbouring samples.
"""
import math

import numpy as np


def params_at(t, p0, p1, t_start, t_end):
    s = min(max((t - t_start) / max(t_end - t_start, 1e-6), 0.0), 1.0)
    return [a + s * (b - a) for a, b in zip(p0, p1)]


def deriv(x, delta, vx, fyd, mzd, p, imu_x):
    m, lf, lr, izz, caf, car, sig = p
    v, r, af, _, _, psi = x
    fyf = caf * af
    fyr = -car * (v - lr * r) / vx
    vdot = (fyf + fyr + fyd) / m - vx * r
    rdot = (lf * fyf - lr * fyr + mzd) / izz
    afdot = (vx * delta - v - lf * r - vx * af) / sig
    xdot = [vdot, rdot, afdot,
            vx * math.cos(psi) - v * math.sin(psi),
            vx * math.sin(psi) + v * math.cos(psi),
            r]
    ay = vdot + vx * r + (imu_x - lr) * rdot
    return xdot, ay


def simulate(delta, vx, fyd, mzd, p0, p1=None, t_start=0.0, t_end=0.0, imu_x=0.0, h=0.01):
    """Inputs are arrays sampled every h seconds starting at t = 0. Returns a dict of arrays."""
    n = len(delta)
    p1 = p0 if p1 is None else p1
    u = np.column_stack([delta, vx, fyd, mzd])
    x = [0.0] * 6
    out = {k: np.zeros(n) for k in ("r", "ay", "ydot", "alphaF", "X", "Y", "psi")}
    out["params"] = np.zeros((n, 7))
    for k in range(n):
        t = k * h
        p = params_at(t, p0, p1, t_start, t_end)
        _, ay = deriv(x, *u[k], p, imu_x)
        out["r"][k], out["ay"][k] = x[1], ay
        out["ydot"][k], out["alphaF"][k] = x[0], x[2]
        out["X"][k], out["Y"][k], out["psi"][k] = x[3], x[4], x[5]
        out["params"][k] = p
        if k == n - 1:
            break
        um = 0.5 * (u[k] + u[k + 1])
        pm = params_at(t + h / 2, p0, p1, t_start, t_end)
        pn = params_at(t + h, p0, p1, t_start, t_end)
        k1, _ = deriv(x, *u[k], p, imu_x)
        k2, _ = deriv([a + h / 2 * b for a, b in zip(x, k1)], *um, pm, imu_x)
        k3, _ = deriv([a + h / 2 * b for a, b in zip(x, k2)], *um, pm, imu_x)
        k4, _ = deriv([a + h * b for a, b in zip(x, k3)], *u[k + 1], pn, imu_x)
        x = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
    return out

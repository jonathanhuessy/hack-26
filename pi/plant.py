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


class StreamingPlant:
    """Advance the bicycle model one fixed-step sample at a time."""

    def __init__(
        self,
        *,
        parameter_schedule,
        imu_x: float = 0.0,
        h: float = 0.01,
    ):
        self.parameter_schedule = parameter_schedule
        self.imu_x = imu_x
        self.h = h
        self.reset()

    def reset(self) -> None:
        self.state = [0.0] * 6
        self.index = 0

    @property
    def time_s(self) -> float:
        return self.index * self.h

    def step(self, current_input, next_input=None):
        """Return the current output and integrate toward the next sample."""
        current_input = np.asarray(current_input, dtype=float)
        if current_input.shape != (4,):
            raise ValueError("current_input must contain delta, Vx, Fyd, and Mzd")
        next_input = current_input if next_input is None else np.asarray(next_input, dtype=float)
        if next_input.shape != (4,):
            raise ValueError("next_input must contain delta, Vx, Fyd, and Mzd")
        t = self.time_s
        p = list(self.parameter_schedule(t))
        _, ay = deriv(self.state, *current_input, p, self.imu_x)
        output = {
            "r": self.state[1],
            "ay": ay,
            "ydot": self.state[0],
            "alphaF": self.state[2],
            "X": self.state[3],
            "Y": self.state[4],
            "psi": self.state[5],
            "params": np.asarray(p, dtype=float),
        }
        k1, _ = deriv(self.state, *current_input, p, self.imu_x)
        midpoint_input = 0.5 * (current_input + next_input)
        midpoint_params = list(self.parameter_schedule(t + self.h / 2))
        next_params = list(self.parameter_schedule(t + self.h))
        k2, _ = deriv(
            [a + self.h / 2 * b for a, b in zip(self.state, k1)],
            *midpoint_input,
            midpoint_params,
            self.imu_x,
        )
        k3, _ = deriv(
            [a + self.h / 2 * b for a, b in zip(self.state, k2)],
            *midpoint_input,
            midpoint_params,
            self.imu_x,
        )
        k4, _ = deriv(
            [a + self.h * b for a, b in zip(self.state, k3)],
            *next_input,
            next_params,
            self.imu_x,
        )
        self.state = [
            a + self.h / 6 * (b + 2 * c + 2 * d + e)
            for a, b, c, d, e in zip(self.state, k1, k2, k3, k4)
        ]
        self.index += 1
        return output


def simulate(
    delta,
    vx,
    fyd,
    mzd,
    p0,
    p1=None,
    t_start=0.0,
    t_end=0.0,
    imu_x=0.0,
    h=0.01,
    parameter_schedule=None,
):
    """Inputs are arrays sampled every h seconds starting at t = 0. Returns a dict of arrays."""
    n = len(delta)
    supplied_p1 = p1
    p1 = p0 if p1 is None else p1
    if parameter_schedule is not None and (
        supplied_p1 is not None or t_start != 0.0 or t_end != 0.0
    ):
        raise ValueError("parameter_schedule cannot be combined with p1/t_start/t_end")
    u = np.column_stack([delta, vx, fyd, mzd])
    x = [0.0] * 6
    out = {k: np.zeros(n) for k in ("r", "ay", "ydot", "alphaF", "X", "Y", "psi")}
    out["params"] = np.zeros((n, 7))
    for k in range(n):
        t = k * h
        p = (
            list(parameter_schedule(t))
            if parameter_schedule is not None
            else params_at(t, p0, p1, t_start, t_end)
        )
        _, ay = deriv(x, *u[k], p, imu_x)
        out["r"][k], out["ay"][k] = x[1], ay
        out["ydot"][k], out["alphaF"][k] = x[0], x[2]
        out["X"][k], out["Y"][k], out["psi"][k] = x[3], x[4], x[5]
        out["params"][k] = p
        if k == n - 1:
            break
        um = 0.5 * (u[k] + u[k + 1])
        pm = (
            list(parameter_schedule(t + h / 2))
            if parameter_schedule is not None
            else params_at(t + h / 2, p0, p1, t_start, t_end)
        )
        pn = (
            list(parameter_schedule(t + h))
            if parameter_schedule is not None
            else params_at(t + h, p0, p1, t_start, t_end)
        )
        k1, _ = deriv(x, *u[k], p, imu_x)
        k2, _ = deriv([a + h / 2 * b for a, b in zip(x, k1)], *um, pm, imu_x)
        k3, _ = deriv([a + h / 2 * b for a, b in zip(x, k2)], *um, pm, imu_x)
        k4, _ = deriv([a + h * b for a, b in zip(x, k3)], *u[k + 1], pn, imu_x)
        x = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
    return out

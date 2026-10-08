"""Three-input port of the H1 selected turn features."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

import numpy as np
from scipy.signal import lfilter


_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = json.loads(
    (_ROOT / "models" / "export" / "feature_contract.json").read_text(encoding="utf-8")
)
FEATURE_NAMES = tuple(_CONTRACT["feature_names"])
WHEELBASE_M = float(_CONTRACT["wheelbase_m"])


def _dft_at(x: np.ndarray, frequencies: np.ndarray, fs: float) -> np.ndarray:
    n = x.size
    k = np.arange(n, dtype=float)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
    return np.exp(-2j * np.pi * (frequencies[:, None] / fs) * k) @ (
        (x - np.mean(x)) * window
    )


def _band_log_power(x: np.ndarray, bands: np.ndarray, fs: float, use: np.ndarray,
                    coefficients: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    result = np.zeros(len(bands))
    centered = x - x[0]
    for i, band in enumerate(bands):
        if coefficients is None:
            from scipy.signal import butter
            b, a = butter(2, band / (fs / 2), btype="bandpass")
        else:
            b, a = coefficients[i]
        filtered = lfilter(b, a, centered)
        result[i] = np.log10(np.mean(filtered[use] ** 2) + 1e-12)
    return result


def _spectral_peak(x: np.ndarray, frequencies: np.ndarray, fs: float) -> tuple[float, float, float]:
    power = np.abs(_dft_at(x, frequencies, fs)) ** 2 + 1e-20
    peak_index = int(np.argmax(power))
    peak = float(power[peak_index])
    f_peak = float(frequencies[peak_index])
    if 0 < peak_index < len(power) - 1:
        l1, l2, l3 = np.log(power[peak_index - 1:peak_index + 2])
        denominator = l1 - 2 * l2 + l3
        if denominator < 0:
            f_peak += 0.5 * (l1 - l3) / denominator * (frequencies[1] - frequencies[0])
    half = peak / 2
    left = peak_index
    while left > 0 and power[left - 1] >= half:
        left -= 1
    right = peak_index
    while right < len(power) - 1 and power[right + 1] >= half:
        right += 1
    zeta = (right - left + 1) * (frequencies[1] - frequencies[0]) / (2 * f_peak)
    return f_peak, float(np.log10(peak / np.median(power))), float(zeta)


def _ar2_mode(x: np.ndarray, fs: float, use: np.ndarray,
              coefficients: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[float, float]:
    if coefficients is None:
        from scipy.signal import butter
        b, a = butter(2, np.array([0.3, 3.0]) / (fs / 2), btype="bandpass")
    else:
        b, a = coefficients
    filtered = lfilter(b, a, x - x[0])
    y = filtered[use[0]:use[-1] + 1:5]
    fs2 = fs / 5
    if y.size < 3:
        return 0.0, 1.0
    phi = np.column_stack((y[1:-1], y[:-2]))
    coeff = np.linalg.lstsq(phi, y[2:], rcond=None)[0]
    a1, a2 = coeff
    if a1 * a1 + 4 * a2 < 0:
        radius = np.sqrt(-a2)
        theta = np.arccos(np.clip(a1 / (2 * radius), -1, 1))
        sigma = -np.log(radius) * fs2
        omega = theta * fs2
        wn = np.sqrt(sigma * sigma + omega * omega)
        return float(wn / (2 * np.pi)), float(sigma / wn)
    return 0.0, 1.0


def _cross_gain(x: np.ndarray, y: np.ndarray, bands: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    gains = np.zeros(len(bands))
    phases = np.zeros(len(bands))
    for i, band in enumerate(bands):
        frequencies = np.arange(band[0], band[1] + 1e-12, 0.02)
        X = _dft_at(x, frequencies, fs)
        Y = _dft_at(y, frequencies, fs)
        transfer = np.sum(Y * np.conj(X)) / max(np.sum(np.abs(X) ** 2), 1e-20)
        gains[i] = abs(transfer)
        phases[i] = np.angle(transfer)
    return gains, phases


def selected_features(
    window: np.ndarray,
    straight: np.ndarray,
    fs: float,
    *,
    wheelbase_m: float = WHEELBASE_M,
    margin_s: float = 5.0,
    filter_coefficients: dict | None = None,
) -> np.ndarray:
    """Return the 18 selected features from ``[delta, Vx, r]`` arrays.

    The MATLAB reference accepts a fourth ``ay`` column for its exploratory
    35-feature vector. This deployable function intentionally accepts exactly
    three columns and computes only the selected H1 features.
    """
    window = np.asarray(window, dtype=float)
    straight = np.asarray(straight, dtype=float)
    if window.ndim != 2 or window.shape[1] != 3:
        raise ValueError("window must have columns [delta, Vx, r]")
    if straight.ndim != 2 or straight.shape[1] != 3:
        raise ValueError("straight must have columns [delta, Vx, r]")
    if window.shape[0] < 2:
        raise ValueError("window is too short")

    n_margin = round(margin_s * fs)
    warm = round(2 * fs)
    core = np.arange(n_margin, window.shape[0] - n_margin)
    use = np.arange(warm, window.shape[0])
    if core.size == 0 or use.size == 0:
        raise ValueError("window does not contain enough samples for margins")

    delta, vx, yaw_rate = window.T
    residual = yaw_rate - vx * np.tan(delta) / wheelbase_m
    bands = np.array([[0.2, 0.6], [0.6, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 3.0]])
    frequencies = np.arange(0.5, 3.0 + 1e-12, 0.02)
    band_coeff = None
    ar_coeff = None
    if filter_coefficients:
        band_coeff = [
            (np.asarray(b), np.asarray(a))
            for b, a in zip(filter_coefficients["bBand"][:5], filter_coefficients["aBand"][:5])
        ]
        ar_coeff = (
            np.asarray(filter_coefficients["bAR"]),
            np.asarray(filter_coefficients["aAR"]),
        )
    bp_e = _band_log_power(residual, bands, fs, use, band_coeff)
    fpk_e, _, _ = _spectral_peak(residual[core], frequencies, fs)
    f_ar, zeta_ar = _ar2_mode(residual, fs, use, ar_coeff)

    v_turn = float(np.mean(vx[core]))
    gain_r, phase_r = _cross_gain(delta[use], yaw_rate[use],
                                   np.array([[0.1, 0.4], [0.4, 0.8], [0.8, 1.5]]), fs)
    gain_r /= v_turn / wheelbase_m
    selected_sign = np.sign(np.sum(yaw_rate[core]))
    opposite = float(np.mean(
        (np.sign(delta[core]) == -selected_sign) & (np.abs(delta[core]) > np.deg2rad(3))
    ))
    active = core[np.abs(delta[core]) > np.deg2rad(8)]
    if active.size:
        ss_gain = float(np.median(yaw_rate[active] / (vx[active] * np.tan(delta[active]) / wheelbase_m)))
    else:
        ss_gain = 1.0

    straight_features = np.zeros(4)
    if straight.shape[0] >= round(10 * fs):
        ds, vs, rs = straight.T
        use_s = np.arange(warm, straight.shape[0])
        v_mean = float(np.mean(vs))
        gain_s, phase_s = _cross_gain(ds[use_s], rs[use_s], np.array([[0.1, 0.5]]), fs)
        residual_s = rs - vs * np.tan(ds) / wheelbase_m
        bp_s = _band_log_power(residual_s, np.array([[1.0, 2.0]]), fs, use_s)
        fpk_s, _, _ = _spectral_peak(residual_s[use_s], frequencies, fs)
        straight_features = np.array([gain_s[0] / (v_mean / wheelbase_m), phase_s[0], bp_s[0], fpk_s])

    return np.array([
        v_turn, opposite, *bp_e[:5], fpk_e, f_ar, zeta_ar,
        gain_r[0], phase_r[0], phase_r[1], ss_gain,
        *straight_features,
    ])


def selected_feature_names() -> tuple[str, ...]:
    return FEATURE_NAMES

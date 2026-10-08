"""Streaming three-input turn detector and K-turn verdict aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.signal import butter, lfilter

try:
    from .contracts import MeasuredSample, SampleStatus
    from .features import selected_features
    from .model import ModelArtifact, Prediction, dummy_model
except ImportError:
    from contracts import MeasuredSample, SampleStatus
    from features import selected_features
    from model import ModelArtifact, Prediction, dummy_model


class DetectorState(str, Enum):
    IDLE = "idle"
    IN_TURN = "in_turn"
    WAITING_WINDOW = "waiting_window"


@dataclass(frozen=True)
class Verdict:
    class_probabilities: np.ndarray
    class_name: str
    delta_m_kg: float | None
    k_f: float | None
    turn_index: int
    timestamp_s: float
    window_id: str


@dataclass(frozen=True)
class DetectorEvent:
    status: SampleStatus
    state: DetectorState
    new_verdict: Verdict | None = None
    message: str = ""


class TurnDetector:
    """Consume ``MeasuredSample`` values and emit a verdict after each turn."""

    def __init__(
        self,
        model: ModelArtifact | None = None,
        *,
        sample_rate_hz: float = 100.0,
        margin_s: float = 5.0,
        straight_s: float = 20.0,
        straight_min_vx: float = 3.0,
        entry_deg: float = 5.0,
        exit_deg: float = 3.0,
        hold_s: float = 2.0,
        min_heading_deg: float = 120.0,
        aggregate_turns: int = 3,
    ):
        self.model = model or dummy_model()
        self.fs = sample_rate_hz
        self.period = 1.0 / sample_rate_hz
        self.margin_n = round(margin_s * sample_rate_hz)
        self.straight_n = round(straight_s * sample_rate_hz)
        self.straight_min_vx = straight_min_vx
        self.entry_rad = np.deg2rad(entry_deg)
        self.exit_rad = np.deg2rad(exit_deg)
        self.hold_n = round(hold_s * sample_rate_hz)
        self.min_heading_rad = np.deg2rad(min_heading_deg)
        self.aggregate_turns = aggregate_turns
        self._b_lp, self._a_lp = butter(2, 1.0 / (sample_rate_hz / 2))
        self.reset()

    def start(self) -> None:
        self._started = True

    def push(self, sample: MeasuredSample) -> DetectorEvent:
        if not self._started:
            self.start()
        self._samples.append(sample)
        index = len(self._samples) - 1
        self._process_trigger(index)
        verdict = self._maybe_emit(sample.timestamp_s)
        status = SampleStatus.ACCEPTED
        return DetectorEvent(status, self.state, verdict)

    def flush(self) -> DetectorEvent:
        if self._pending is not None:
            verdict = self._maybe_emit(self._samples[-1].timestamp_s if self._samples else 0.0, force=True)
            return DetectorEvent(SampleStatus.ACCEPTED, self.state, verdict)
        return DetectorEvent(SampleStatus.ACCEPTED, self.state)

    def reset(self) -> None:
        self._samples: list[MeasuredSample] = []
        self._started = False
        self.state = DetectorState.IDLE
        self._in_turn = False
        self._entry_index: int | None = None
        self._quiet = 0
        self._pending: tuple[int, int] | None = None
        self._turn_count = 0
        self._history: list[Prediction] = []
        self._filter_state = np.zeros(max(len(self._a_lp), len(self._b_lp)) - 1)

    def close(self) -> None:
        self._started = False

    def _process_trigger(self, index: int) -> None:
        filtered, self._filter_state = lfilter(
            self._b_lp,
            self._a_lp,
            [self._samples[index].delta],
            zi=self._filter_state,
        )
        magnitude = abs(filtered[0])
        if not self._in_turn:
            if magnitude > self.entry_rad:
                self._in_turn = True
                self._entry_index = index
                self._quiet = 0
                self.state = DetectorState.IN_TURN
            return
        if magnitude < self.exit_rad:
            self._quiet += 1
        else:
            self._quiet = 0
        if self._quiet >= self.hold_n:
            exit_index = index - self.hold_n + 1
            entry_index = self._entry_index
            assert entry_index is not None
            yaw = sum(sample.yaw_rate for sample in self._samples[entry_index:exit_index + 1])
            heading = yaw / self.fs
            if abs(heading) >= self.min_heading_rad:
                self._pending = (entry_index, exit_index)
                self.state = DetectorState.WAITING_WINDOW
            self._in_turn = False
            self._entry_index = None
            self._quiet = 0
            if self._pending is None:
                self.state = DetectorState.IDLE

    def _maybe_emit(self, timestamp_s: float, force: bool = False) -> Verdict | None:
        if self._pending is None:
            return None
        entry, exit_index = self._pending
        if not force and len(self._samples) - 1 < exit_index + self.margin_n:
            return None
        start = max(0, entry - self.margin_n)
        end = min(len(self._samples), exit_index + self.margin_n + 1)
        values = np.array(
            [[sample.delta, sample.vx, sample.yaw_rate] for sample in self._samples],
            dtype=float,
        )
        window = values[start:end]
        candidates = np.flatnonzero(values[:start, 1] > self.straight_min_vx)
        if candidates.size:
            candidates = candidates[-self.straight_n:]
            straight = values[candidates]
        else:
            straight = np.empty((0, 3))
        try:
            features = selected_features(window, straight, self.fs, margin_s=self.margin_n / self.fs)
            prediction = self.model.predict(features)
        except (ValueError, FloatingPointError) as exc:
            self._pending = None
            self.state = DetectorState.IDLE
            return None
        self._history.append(prediction)
        if len(self._history) > self.aggregate_turns:
            self._history.pop(0)
        probabilities = np.mean([item.class_probabilities for item in self._history], axis=0)
        class_index = int(np.argmax(probabilities))
        delta_values = [item.delta_m_kg for item in self._history if item.delta_m_kg is not None]
        k_values = [item.k_f for item in self._history if item.k_f is not None]
        verdict = Verdict(
            class_probabilities=probabilities,
            class_name=("nominal", "A", "B", "AB")[class_index],
            delta_m_kg=float(np.median(delta_values)) if delta_values else None,
            k_f=float(np.median(k_values)) if k_values else None,
            turn_index=self._turn_count,
            timestamp_s=timestamp_s,
            window_id=f"turn-{self._turn_count}",
        )
        self._turn_count += 1
        self._pending = None
        self.state = DetectorState.IDLE
        return verdict

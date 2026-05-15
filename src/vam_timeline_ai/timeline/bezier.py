"""Timeline Bezier curve evaluation used for technical baking."""

from __future__ import annotations

import bisect
from dataclasses import replace

import numpy as np

from vam_timeline_ai.timeline.codec import TimelineKeyframe


class CurveTypeValues:
    Undefined = -1
    LeaveAsIs = 0
    Flat = 1
    Linear = 2
    SmoothLocal = 3
    Bounce = 4
    LinearFlat = 5
    FlatLinear = 6
    CopyPrevious = 7
    Constant = 8
    FlatLong = 9
    SmoothGlobal = 10


class BezierCurve:
    def __init__(self, keys: list[TimelineKeyframe] | None = None, loop: bool = False):
        self.keys = sorted(keys or [], key=lambda k: k.time)
        self.loop = loop
        self.warnings: list[str] = []

    def add_edge_frames_if_missing(self, animation_length: float, default_curve_type: int = CurveTypeValues.SmoothLocal) -> bool:
        if not self.keys:
            self.keys = [
                TimelineKeyframe(0.0, 0.0, default_curve_type),
                TimelineKeyframe(float(animation_length), 0.0, default_curve_type),
            ]
            return True
        if len(self.keys) == 1:
            key = replace(self.keys[0], time=0.0)
            self.keys = [key, replace(key, time=float(animation_length))]
            return True
        dirty = False
        if self.keys[0].time > 0.0:
            self.keys.insert(0, TimelineKeyframe(0.0, self.keys[0].value, default_curve_type))
            dirty = True
        if self.keys[-1].time < animation_length:
            self.keys.append(TimelineKeyframe(float(animation_length), self.keys[-1].value, default_curve_type))
            dirty = True
        return dirty

    def compute_curves(self) -> list[str]:
        if not self.keys:
            return self.warnings
        if any(k.curve_type == CurveTypeValues.SmoothGlobal for k in self.keys):
            self.warnings.append("SmoothGlobal approximated as SmoothLocal")
        for idx in range(len(self.keys)):
            self._compute_key(idx)
        return self.warnings

    def evaluate(self, time: float) -> float:
        if not self.keys:
            return 0.0
        if len(self.keys) == 1:
            return float(self.keys[0].value)
        if time <= self.keys[0].time:
            return self._compute_value(self.keys[0], self.keys[1], time)
        if time >= self.keys[-1].time:
            return float(self.keys[-1].value)
        times = [k.time for k in self.keys]
        idx = max(0, min(bisect.bisect_right(times, time) - 1, len(self.keys) - 2))
        return self._compute_value(self.keys[idx], self.keys[idx + 1], time)

    def evaluate_many(self, times: np.ndarray) -> np.ndarray:
        return np.asarray([self.evaluate(float(t)) for t in times], dtype=np.float32)

    def _compute_key(self, idx: int) -> None:
        current = self.keys[idx]
        previous = self.keys[idx - 1] if idx > 0 else None
        next_key = self.keys[idx + 1] if idx < len(self.keys) - 1 else None
        curve_type = current.curve_type
        if curve_type == CurveTypeValues.CopyPrevious and previous is not None:
            current = replace(current, value=previous.value)
            curve_type = previous.curve_type if previous.curve_type != CurveTypeValues.CopyPrevious else CurveTypeValues.SmoothLocal
        if curve_type == CurveTypeValues.Linear:
            current = self._linear_interpolation(current, previous, next_key)
        elif curve_type in (CurveTypeValues.SmoothLocal, CurveTypeValues.SmoothGlobal):
            current = self._smooth_local_interpolation(current, previous, next_key)
        elif curve_type == CurveTypeValues.LinearFlat:
            cpi = current.value - (current.value - previous.value) / 3.0 if previous else current.value
            current = replace(current, control_point_in=cpi, control_point_out=current.value)
        elif curve_type in (CurveTypeValues.Flat, CurveTypeValues.FlatLong, CurveTypeValues.FlatLinear):
            current = replace(current, control_point_in=current.value, control_point_out=current.value)
        elif curve_type == CurveTypeValues.LeaveAsIs:
            if current.control_point_in is None or current.control_point_out is None:
                current = replace(current, control_point_in=current.value, control_point_out=current.value)
        elif current.control_point_in is None or current.control_point_out is None:
            current = replace(current, control_point_in=current.value, control_point_out=current.value)
        self.keys[idx] = current

    @staticmethod
    def _linear_interpolation(current: TimelineKeyframe, previous: TimelineKeyframe | None, next_key: TimelineKeyframe | None) -> TimelineKeyframe:
        cpi = current.value - (current.value - previous.value) / 3.0 if previous is not None else current.value
        cpo = current.value + (next_key.value - current.value) / 3.0 if next_key is not None else current.value
        return replace(current, control_point_in=cpi, control_point_out=cpo)

    @staticmethod
    def _smooth_local_interpolation(current: TimelineKeyframe, previous: TimelineKeyframe | None, next_key: TimelineKeyframe | None) -> TimelineKeyframe:
        if previous is not None and next_key is not None:
            cpi = current.value - (current.value - previous.value) / 3.0
            cpo = current.value + (next_key.value - current.value) / 3.0
            return replace(current, control_point_in=cpi, control_point_out=cpo)
        if previous is not None:
            return replace(current, control_point_in=current.value - (current.value - previous.value) / 3.0, control_point_out=current.value)
        if next_key is not None:
            return replace(current, control_point_in=current.value, control_point_out=current.value + (next_key.value - current.value) / 3.0)
        return replace(current, control_point_in=current.value, control_point_out=current.value)

    @staticmethod
    def _compute_value(current: TimelineKeyframe, next_key: TimelineKeyframe | None, time: float) -> float:
        if next_key is None:
            return float(current.value)
        denom = next_key.time - current.time
        if abs(denom) < 1e-8:
            return float(current.value)
        t = (time - current.time) / denom
        if current.curve_type == CurveTypeValues.Constant:
            return float(current.value)
        if current.curve_type in (CurveTypeValues.Linear, CurveTypeValues.FlatLinear):
            return float(current.value + (next_key.value - current.value) * t)
        w0 = current.value
        w1 = current.control_point_out if current.control_point_out is not None else current.value
        w2 = next_key.control_point_in if next_key.control_point_in is not None else next_key.value
        w3 = next_key.value
        mt = 1.0 - t
        return float(w0 * mt**3 + 3.0 * w1 * mt**2 * t + 3.0 * w2 * mt * t**2 + w3 * t**3)

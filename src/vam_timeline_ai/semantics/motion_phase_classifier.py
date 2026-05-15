"""Conservative motion-phase heuristics for review selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def classify_motion_phase(feature_row: dict[str, Any], body_quality: dict[str, Any] | None = None) -> dict[str, Any]:
    values = feature_row.get("feature_values", {}) or {}
    body_quality = body_quality or {}
    q = body_quality.get("body_motion_quality", "unknown")
    energy = _f(values.get("pelvis_movement_energy"))
    regular = _f(values.get("steady_rhythm_score_proxy"))
    irregular = _f(values.get("irregular_rhythm_score_proxy"))
    pause = _f(values.get("pause_hold_score_proxy"))
    accel = _f(values.get("pelvis_acceleration_peak_count"))
    speed_std = _f(values.get("pelvis_speed_std"))
    moving_parts = int(body_quality.get("moving_bodypart_count") or 0)

    warnings: list[str] = []
    if q in {"controller_only_whole_person_motion", "root_only_motion"}:
        phase = "root_only_motion_candidate"
        warnings.append("Root/controller-only motion is not clean body motion.")
    elif q == "static_or_micro_motion" or body_quality.get("static_or_micro_motion"):
        phase = "pose_hold_candidate"
        warnings.append("Static or micro-motion is not clean body motion.")
    elif q == "static_or_empty" or (pause >= 0.75 and energy < 0.02):
        phase = "pose_hold_candidate"
    elif irregular >= 0.7 or accel >= 6 or speed_std >= 0.25:
        phase = "transition_adjustment_candidate"
    elif q in {"good_body_motion", "partial_body_motion"} and moving_parts >= 2 and regular >= 0.45 and energy > 0.005:
        phase = "clean_repetitive_motion_candidate"
    elif irregular >= 0.9 and energy > 0.2:
        phase = "chaotic_unusable_candidate"
    else:
        phase = "unknown_phase"
    return {"motion_phase_candidate": phase, "warnings": warnings}


def _f(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0

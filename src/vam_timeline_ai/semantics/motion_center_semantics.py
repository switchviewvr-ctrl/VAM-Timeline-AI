"""Infer primary semantic motion centers from existing feature evidence."""

from __future__ import annotations

from typing import Any


def infer_primary_motion_center(candidate: dict[str, Any], relative_features: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    """Return a conservative primary motion center plus explanatory flags."""

    flags: list[str] = []
    rel = relative_features or {}
    feature_values = rel.get("feature_values") or {}
    hip_strength = _num(candidate.get("hip_motion_strength"), _num(candidate.get("pelvis_trajectory_strength"), 0.0))
    clean_score = _num(candidate.get("clean_motion_score"), _num(candidate.get("motion_score"), 0.0))
    head_motion = _num(feature_values.get("head_relative_to_chest_motion"), 0.0)
    hand_motion = _num(feature_values.get("hands_relative_to_chest_pelvis_head"), 0.0)
    pelvis_path = _num(feature_values.get("local_path_length"), _num(candidate.get("pelvis_trajectory_strength"), 0.0))
    semantic_family = str(candidate.get("semantic_family") or "")
    motion_subtype = str(candidate.get("motion_subtype") or "")

    if semantic_family == "bj_oral" or "bj" in motion_subtype or "oral" in motion_subtype:
        return "head_neck", ["semantic_bj_oral_hint"]
    if semantic_family in {"standing_hand_head", "hand_gesture", "head_gesture"}:
        if hand_motion >= head_motion:
            return "hands", ["standing_hand_head_hint"]
        return "head_neck", ["standing_head_hint"]
    if hand_motion > max(0.65, hip_strength + 0.25) and pelvis_path < 0.35:
        return "hands", ["hand_motion_exceeds_pelvis_motion"]
    if head_motion > max(0.45, hip_strength + 0.2) and pelvis_path < 0.35:
        return "head_neck", ["head_motion_exceeds_pelvis_motion"]
    if hip_strength >= 0.45 or clean_score >= 0.6 or pelvis_path >= 0.5:
        return "pelvis_hip", flags
    if pelvis_path < 0.15:
        return "static_pose", ["low_pelvis_path"]
    return "unknown", ["no_clear_primary_motion_center"]


def infer_motion_shape(candidate: dict[str, Any], relative_features: dict[str, Any] | None = None) -> str:
    rel = relative_features or {}
    fv = rel.get("feature_values") or {}
    if _num(fv.get("local_grind_score"), 0.0) >= 0.45:
        return "oval_or_grinding"
    if _num(fv.get("local_bounce_score"), 0.0) >= 0.25 or "bounce" in str(candidate.get("motion_subtype") or ""):
        return "vertical_bounce"
    if _num(fv.get("relative_pelvis_forward_back_amplitude"), 0.0) > _num(fv.get("relative_pelvis_vertical_amplitude"), 0.0):
        return "forward_back_rock"
    return "unknown"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

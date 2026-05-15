"""Pose semantic records for the clean_v3 semantic rescan.

Pose semantics describe body context only. They are intentionally separate from
motion semantics so that a kneeling pose can be Cowgirl-compatible, BJ/oral
compatible, a transition, or simply unknown depending on motion and interaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


POSE_FAMILIES = {
    "cowgirl",
    "bj_oral",
    "doggy",
    "standing",
    "lying_receiver",
    "kneeling_general",
    "hand_head_gesture",
    "transition",
    "unknown",
}


@dataclass
class PoseSemanticRecord:
    window_id: str
    sample_id: str | None = None
    source_scene_file: str | None = None
    technical_atom_id: str | None = None
    pose_family: str = "unknown"
    pose_subtype: str = "unknown"
    support_context: list[str] = field(default_factory=list)
    anchor_requirements: dict[str, Any] = field(default_factory=dict)
    pose_confidence: float = 0.0
    pose_generation_safe: bool = False
    warnings: list[str] = field(default_factory=list)
    is_human_ground_truth: bool = False
    is_training_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["pose_family"] not in POSE_FAMILIES:
            data["pose_family"] = "unknown"
        return data


def pose_subtype_from_context(pose_family: str, features: dict[str, Any], motion_hint: str = "") -> str:
    """Return a conservative pose subtype from score-like features."""
    kneeling = _num(features.get("kneeling_score"))
    squat = _num(features.get("squat_score"))
    standing = _num(features.get("standing_score"))
    lying_back = _num(features.get("lying_on_back_score"))
    lying_prone = _num(features.get("lying_prone_score"))
    lean = _num(features.get("torso_forward_lean_proxy"))
    hint = str(motion_hint or "").lower()
    if pose_family == "cowgirl":
        if lean >= 0.55 or "lean" in hint:
            return "cowgirl_lean_forward_supported"
        if squat >= max(kneeling, 0.45):
            return "cowgirl_squat"
        if standing >= 0.65:
            return "cowgirl_upright"
        return "cowgirl_kneeling"
    if pose_family == "bj_oral":
        return "bj_forward_lean" if lean >= 0.5 else "bj_kneeling"
    if pose_family == "standing":
        return "standing_upright"
    if pose_family == "lying_receiver":
        return "lying_on_back" if lying_back >= lying_prone else "lying_prone"
    if pose_family == "doggy":
        return "doggy_all_fours"
    if kneeling >= 0.45:
        return "cowgirl_kneeling" if "cowgirl" in hint else "unknown"
    return "unknown"


def support_context_from_features(features: dict[str, Any]) -> list[str]:
    context: list[str] = []
    if _num(features.get("hands_forward_support_score")) >= 0.45:
        context.append("hands_forward_support")
    if _num(features.get("knees_under_body_score")) >= 0.45:
        context.append("knees_anchored")
    if _num(features.get("feet_behind_body_score")) >= 0.35:
        context.append("feet_anchored")
    if not context:
        context.append("unknown")
    return context


def _num(value: Any) -> float:
    try:
        if value != value:
            return 0.0
        return float(value or 0.0)
    except Exception:
        return 0.0

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

POSE_SUBTYPES = {
    "cowgirl_kneeling",
    "cowgirl_squat",
    "cowgirl_upright",
    "cowgirl_lean_forward_supported",
    "cowgirl_lean_back",
    "cowgirl_lean_back_supported",
    "cowgirl_hands_behind_supported",
    "reverse_cowgirl_lean_back",
    "bj_kneeling",
    "bj_forward_lean",
    "doggy_all_fours",
    "standing_upright",
    "lying_on_back",
    "lying_prone",
    "unknown",
}

SUPPORT_CONTEXTS = {
    "hands_forward_support",
    "hands_on_partner",
    "hands_on_floor_or_bed",
    "hands_free",
    "knees_anchored",
    "feet_anchored",
    "hands_behind_support",
    "possible_hands_behind_support",
    "hands_on_partner_legs",
    "hands_on_partner_thighs",
    "hands_on_partner_legs_or_thighs",
    "hands_behind_on_floor_or_bed",
    "no_clear_support",
    "unknown",
}

FACING_CONTEXTS = {"front_cowgirl", "reverse_cowgirl", "side_facing", "unknown"}
TORSO_LEAN_DIRECTIONS = {"forward", "upright", "backward", "unknown"}


@dataclass
class PoseSemanticRecord:
    window_id: str
    sample_id: str | None = None
    source_scene_file: str | None = None
    technical_atom_id: str | None = None
    pose_family: str = "unknown"
    pose_subtype: str = "unknown"
    support_context: list[str] = field(default_factory=list)
    facing_context: str = "unknown"
    torso_lean_direction: str = "unknown"
    anchor_requirements: dict[str, Any] = field(default_factory=dict)
    pose_confidence: float = 0.0
    pose_generation_safe: bool = False
    lean_back_pose_confidence: float = 0.0
    hands_behind_support_confidence: float = 0.0
    partner_leg_support_confidence: float = 0.0
    facing_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    is_human_ground_truth: bool = False
    is_training_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["pose_family"] not in POSE_FAMILIES:
            data["pose_family"] = "unknown"
        if data["pose_subtype"] not in POSE_SUBTYPES:
            data["pose_subtype"] = "unknown"
        data["support_context"] = [item if item in SUPPORT_CONTEXTS else "unknown" for item in data["support_context"]]
        if data["facing_context"] not in FACING_CONTEXTS:
            data["facing_context"] = "unknown"
        if data["torso_lean_direction"] not in TORSO_LEAN_DIRECTIONS:
            data["torso_lean_direction"] = "unknown"
        return data


def pose_subtype_from_context(pose_family: str, features: dict[str, Any], motion_hint: str = "") -> str:
    """Return a conservative pose subtype from score-like features."""
    kneeling = _num(features.get("kneeling_score"))
    squat = _num(features.get("squat_score"))
    standing = _num(features.get("standing_score"))
    lying_back = _num(features.get("lying_on_back_score"))
    lying_prone = _num(features.get("lying_prone_score"))
    lean = max(_num(features.get("torso_forward_lean_score")), _num(features.get("torso_forward_lean_proxy")))
    lean_back = _num(features.get("torso_lean_back_score"))
    hands_behind = _num(features.get("hands_behind_support_score"))
    reverse = _num(features.get("rider_reverse_facing_proxy"))
    hint = str(motion_hint or "").lower()
    if pose_family == "cowgirl":
        if reverse >= 0.75 and lean_back >= 0.5:
            return "reverse_cowgirl_lean_back"
        if lean_back >= 0.55 and hands_behind >= 0.45:
            return "cowgirl_lean_back_supported"
        if hands_behind >= 0.6:
            return "cowgirl_hands_behind_supported"
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
    if _num(features.get("hands_on_partner_legs_score")) >= 0.5:
        context.append("hands_on_partner_legs")
    if _num(features.get("hands_on_partner_thighs_score")) >= 0.5:
        context.append("hands_on_partner_thighs")
    if _num(features.get("hands_near_partner_legs_score")) >= 0.45 or _num(features.get("hands_near_partner_thighs_score")) >= 0.45:
        context.append("hands_on_partner_legs_or_thighs")
    if _num(features.get("hands_behind_support_score")) >= 0.45:
        context.append("hands_behind_support")
    elif _num(features.get("hands_behind_body_score")) >= 0.45:
        context.append("possible_hands_behind_support")
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

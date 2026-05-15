"""Motion primitive abstraction records.

A motion primitive is an abstract relative motion pattern. It is not a Timeline
clip and it must not carry source-scene world coordinates as generation targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SEMANTIC_FAMILIES = {"cowgirl", "bj_oral", "doggy", "hand_gesture", "head_gesture", "transition", "unknown"}
PRIMITIVE_SUBTYPES = {
    "riding",
    "grinding",
    "oval_grind",
    "circular_grind",
    "vertical_bounce",
    "forward_back_rock",
    "intro_align",
    "hold",
    "transition",
    "unknown",
}


@dataclass
class MotionPrimitive:
    primitive_id: str
    semantic_family: str
    subtype: str
    source_window_ids: list[str]
    source_candidate_ids: list[str]
    learned_from_dataset: str
    duration_seconds: float
    relative_motion_summary: dict[str, Any]
    trajectory_shape: dict[str, Any]
    rhythm_profile: dict[str, Any]
    amplitude_profile: dict[str, Any]
    controller_role_map: dict[str, Any]
    anchor_requirements: dict[str, Any]
    safety_requirements: dict[str, Any]
    generation_parameters: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    is_timeline_clip: bool = False
    contains_absolute_world_coordinates: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MotionPrimitiveSet:
    primitive_set_id: str
    family: str
    subtype: str
    primitives: list[str]
    cluster_summary: dict[str, Any]
    variation_ranges: dict[str, Any]
    recommended_generation_use: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_family(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in SEMANTIC_FAMILIES else "unknown"


def normalize_subtype(value: Any, trajectory_shape: Any = None) -> str:
    text = str(value or "").strip().lower()
    shape = str(trajectory_shape or "").strip().lower()
    if text in PRIMITIVE_SUBTYPES:
        return text
    if "circular" in text or "circular" in shape:
        return "circular_grind"
    if "oval" in text or "ellipse" in shape or "oval" in shape:
        return "oval_grind"
    if "grind" in text:
        return "grinding"
    if "bounce" in text or "bounce" in shape:
        return "vertical_bounce"
    if "forward" in text or "rock" in text or "forward_back" in shape:
        return "forward_back_rock"
    if "intro" in text or "align" in text:
        return "intro_align"
    return "unknown"

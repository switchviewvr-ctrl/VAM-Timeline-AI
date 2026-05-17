"""Semantic stickman schemas for ontology sanity previews.

These structures are a comprehension/debugging layer only. They are schematic
relative poses, not VaM production controller targets and not Timeline export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Point3 = tuple[float, float, float]


CONTROLLER_NAMES = [
    "pelvis",
    "abdomen",
    "chest",
    "head",
    "lHand",
    "rHand",
    "lElbow",
    "rElbow",
    "lThigh",
    "rThigh",
    "lKnee",
    "rKnee",
    "lFoot",
    "rFoot",
]

PARTNER_REFERENCE_NAMES = [
    "partner_pelvis",
    "partner_chest",
    "partner_head",
    "partner_lThigh",
    "partner_rThigh",
    "partner_lLeg",
    "partner_rLeg",
]

SKELETON_EDGES = [
    ("pelvis", "abdomen"),
    ("abdomen", "chest"),
    ("chest", "head"),
    ("chest", "lElbow"),
    ("lElbow", "lHand"),
    ("chest", "rElbow"),
    ("rElbow", "rHand"),
    ("pelvis", "lThigh"),
    ("lThigh", "lKnee"),
    ("lKnee", "lFoot"),
    ("pelvis", "rThigh"),
    ("rThigh", "rKnee"),
    ("rKnee", "rFoot"),
]


@dataclass
class SemanticStickmanPose:
    concept_id: str
    family: str
    pose_subtype: str
    actor_role: str
    partner_role: str
    coordinate_frame: str
    facing_context: str = "unknown"
    torso_lean: str = "unknown"
    controller_points: dict[str, Point3] = field(default_factory=dict)
    partner_reference_points: dict[str, Point3] = field(default_factory=dict)
    anchors: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    followers: list[str] = field(default_factory=list)
    contact_targets: dict[str, str] = field(default_factory=dict)
    support_context: str = "unknown"
    not_labels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class SemanticStickmanMotion:
    concept_id: str
    family: str
    pose_subtype: str
    motion_subtype: str
    duration_seconds: float
    fps: int
    frames: list[dict[str, Any]] = field(default_factory=list)
    driver_curves: dict[str, Any] = field(default_factory=dict)
    follower_curves: dict[str, Any] = field(default_factory=dict)
    anchor_policy: dict[str, Any] = field(default_factory=dict)
    motion_trails: dict[str, list[Point3]] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    contact_targets: dict[str, str] = field(default_factory=dict)
    not_labels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def point_add(a: Point3, b: Point3) -> Point3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def point_lerp(a: Point3, b: Point3, t: float) -> Point3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def as_point3(value: Any) -> Point3:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return (0.0, 0.0, 0.0)

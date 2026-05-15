"""Coordinate-space guardrails for Timeline motion data.

Raw VaM/Timeline controller positions can be source-scene/world placement.
These helpers classify tracks conservatively so downstream features learn
relative body-controller motion rather than absolute scene coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vam_timeline_ai.motion.controller_mapping import map_controller_name


ALLOWED_BODY_PARTS = {
    "hip",
    "pelvis",
    "abdomen",
    "chest",
    "neck",
    "head",
    "left_hand",
    "right_hand",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "left_thigh",
    "right_thigh",
}
DISALLOWED_ROOT_TOKENS = {"control", "root", "rootcontrol", "person", "atom", "world", "worldcontrol"}
DISALLOWED_SUBSTRINGS = {"eyetarget", "world", "root"}


@dataclass(frozen=True)
class TrackCoordinateClassification:
    controller_name: str
    bodypart: str
    coordinate_space: str
    transform_type: str
    teleport_risk: str
    allowed_body_controller: bool
    disallowed_world_or_root: bool
    warnings: tuple[str, ...] = ()


def classify_controller_track(name: str, mapping: dict[str, Any] | None = None) -> TrackCoordinateClassification:
    token = _token(name)
    item = mapping if mapping is not None else map_controller_name(name)
    bodypart = str((item or {}).get("body_part") or "unknown")
    warnings: list[str] = []
    disallowed = token in DISALLOWED_ROOT_TOKENS or any(part in token for part in DISALLOWED_SUBSTRINGS)
    allowed = bodypart in ALLOWED_BODY_PARTS and not disallowed
    if disallowed:
        coordinate_space = "world_absolute"
        transform_type = "person_atom_transform" if token in {"control", "person", "atom", "root", "rootcontrol"} else "controller_world_transform"
        teleport_risk = "high"
        warnings.append("Person/root/world-like track is disallowed for final motion output.")
    elif allowed:
        coordinate_space = "body_relative" if str((item or {}).get("mapping_confidence")) == "high" else "unknown"
        transform_type = "controller_local_offset" if coordinate_space == "body_relative" else "controller_world_transform"
        teleport_risk = "low" if coordinate_space == "body_relative" else "medium"
        if coordinate_space == "unknown":
            warnings.append("Body controller mapping is not high-confidence; treat relative conversion conservatively.")
    else:
        coordinate_space = "unknown"
        transform_type = "unknown"
        teleport_risk = "unknown"
        warnings.append("Unknown controller track; not safe for final export.")
    return TrackCoordinateClassification(
        controller_name=str(name),
        bodypart=bodypart,
        coordinate_space=coordinate_space,
        transform_type=transform_type,
        teleport_risk=teleport_risk,
        allowed_body_controller=allowed,
        disallowed_world_or_root=disallowed,
        warnings=tuple(warnings),
    )


def is_disallowed_world_or_root_track(name: str, mapping: dict[str, Any] | None = None) -> bool:
    return classify_controller_track(name, mapping).disallowed_world_or_root


def is_allowed_body_controller_track(name: str, mapping: dict[str, Any] | None = None) -> bool:
    return classify_controller_track(name, mapping).allowed_body_controller


def needs_relative_conversion(name: str, mapping: dict[str, Any] | None = None) -> bool:
    c = classify_controller_track(name, mapping)
    return c.allowed_body_controller and c.coordinate_space in {"world_absolute", "unknown", "body_relative"}


def can_use_for_semantic_features(name: str, mapping: dict[str, Any] | None = None) -> bool:
    c = classify_controller_track(name, mapping)
    return c.allowed_body_controller and not c.disallowed_world_or_root


def can_use_for_final_export(name: str, mapping: dict[str, Any] | None = None) -> bool:
    c = classify_controller_track(name, mapping)
    return c.allowed_body_controller and c.coordinate_space == "body_relative" and c.teleport_risk == "low"


def _token(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())

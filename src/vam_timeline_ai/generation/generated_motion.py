"""Schema helpers for synthesized relative motion flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALLOWED_GENERATED_CONTROLLERS = {
    "hipControl",
    "pelvisControl",
    "abdomenControl",
    "chestControl",
    "headControl",
    "lHandControl",
    "rHandControl",
    "lElbowControl",
    "rElbowControl",
    "lKneeControl",
    "rKneeControl",
    "lFootControl",
    "rFootControl",
    "lThighControl",
    "rThighControl",
}


@dataclass
class GeneratedControllerTrack:
    controller_name: str
    bodypart: str
    role: str
    coordinate_space: str
    times: list[float]
    position_deltas: list[list[float]]
    rotation_deltas: list[list[float]] | None
    generation_method: str
    source_primitive_group: str
    safety_flags: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedMotionFlow:
    flow_id: str
    semantic_plan: dict[str, Any]
    selected_primitive_group: str
    duration_seconds: float
    fps: float
    controller_tracks: list[GeneratedControllerTrack]
    trajectory_shape: str
    rhythm_profile: dict[str, Any]
    amplitude_profile: dict[str, Any]
    anchor_policy: dict[str, Any]
    coordinate_space: str = "relative_body_motion"
    no_world_coordinates: bool = True
    no_person_root_tracks: bool = True
    clip_stitching_used: bool = False
    export_ready: bool = False
    timeline_export_performed: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["controller_tracks"] = [track.to_dict() for track in self.controller_tracks]
        return data


def controller_bodypart(name: str) -> str:
    text = str(name)
    mapping = {
        "hipControl": "hip",
        "pelvisControl": "pelvis",
        "abdomenControl": "abdomen",
        "chestControl": "chest",
        "headControl": "head",
        "lHandControl": "left_hand",
        "rHandControl": "right_hand",
        "lElbowControl": "left_elbow",
        "rElbowControl": "right_elbow",
        "lKneeControl": "left_knee",
        "rKneeControl": "right_knee",
        "lFootControl": "left_foot",
        "rFootControl": "right_foot",
        "lThighControl": "left_thigh",
        "rThighControl": "right_thigh",
    }
    return mapping.get(text, "unknown")


def is_allowed_generated_controller(name: str) -> bool:
    return str(name) in ALLOWED_GENERATED_CONTROLLERS

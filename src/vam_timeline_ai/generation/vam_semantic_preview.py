"""Schemas and constants for VaM semantic preview clips.

These previews are synthetic, review-only controller sketches derived from the
contact-aware semantic stickman examples. They are not final Timeline
generation and never use source scene/world coordinates or Person/root tracks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STICKMAN_TO_VAM_CONTROLLER = {
    "pelvis": "pelvisControl",
    "abdomen": "abdomenControl",
    "chest": "chestControl",
    "head": "headControl",
    "lHand": "lHandControl",
    "rHand": "rHandControl",
    "lElbow": "lElbowControl",
    "rElbow": "rElbowControl",
    "lThigh": "lThighControl",
    "rThigh": "rThighControl",
    "lKnee": "lKneeControl",
    "rKnee": "rKneeControl",
    "lFoot": "lFootControl",
    "rFoot": "rFootControl",
}

CORE_REVIEW_CONTROLLERS = {
    "pelvisControl",
    "abdomenControl",
    "chestControl",
    "headControl",
    "lHandControl",
    "rHandControl",
    "lKneeControl",
    "rKneeControl",
    "lFootControl",
    "rFootControl",
}

DISALLOWED_TRACK_TOKENS = ("person", "root", "world", "atom")


@dataclass
class VaMSemanticPreviewClip:
    clip_id: str
    family: str
    pose_subtype: str
    motion_subtype: str
    duration_seconds: float
    fps: int
    actor_atom: str = "Person"
    partner_atom_reference: str = "synthetic_partner_reference"
    coordinate_space: str = "synthetic_review_local"
    review_only: bool = True
    generated_from: str = "semantic_motion_examples_v2_contact_aware"
    controllers: list[str] = field(default_factory=list)
    controller_tracks: list[dict[str, Any]] = field(default_factory=list)
    partner_reference: dict[str, Any] = field(default_factory=dict)
    interaction_constraints: list[dict[str, Any]] = field(default_factory=list)
    alignment_validation: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    contact_targets: dict[str, Any] = field(default_factory=dict)
    support_targets: dict[str, Any] = field(default_factory=dict)
    target_points: dict[str, Any] = field(default_factory=dict)
    contact_zone: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)
    export_status: str = "exported"
    timeline_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def is_disallowed_timeline_track(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in DISALLOWED_TRACK_TOKENS)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value

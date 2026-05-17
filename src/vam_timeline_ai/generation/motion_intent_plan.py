"""Top-down motion intent plan schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import uuid


@dataclass
class MotionDriverSpec:
    primary_bodyparts: list[str] = field(default_factory=list)
    secondary_bodyparts: list[str] = field(default_factory=list)
    shape: str = "unknown"
    rhythm: str = "cyclic"
    tempo: str = "medium"
    amplitude: str = "medium"
    axis_priority: list[str] = field(default_factory=list)
    phase_relationships: list[str] = field(default_factory=list)


@dataclass
class MotionIntentPlan:
    plan_id: str
    source: str
    source_prompt: str | None
    family: str
    motion_subtype: str
    pose_subtype: str
    actor_role: str
    partner_role: str
    facing_context: str
    torso_lean: str
    coordinate_frame: str
    partner_relation_requirements: list[str] = field(default_factory=list)
    contact_support: str = "unknown"
    contact_targets: dict[str, str] = field(default_factory=dict)
    motion_driver: MotionDriverSpec = field(default_factory=MotionDriverSpec)
    followers: dict[str, str] = field(default_factory=dict)
    anchors: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    micro_states: list[str] = field(default_factory=list)
    limiter_rules: list[str] = field(default_factory=list)
    anomaly_guards: list[str] = field(default_factory=list)
    sourcebook_trace: dict[str, Any] = field(default_factory=dict)
    safety_rules: list[str] = field(default_factory=list)
    confidence: float = 0.0
    completeness: str = "incomplete"
    unresolved_requirements: list[str] = field(default_factory=list)
    invalid_mappings_prevented: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["motion_driver"] = asdict(self.motion_driver)
        return data


def new_motion_intent_plan_id() -> str:
    return f"motion_intent_v1::{uuid.uuid4().hex[:12]}"

"""Component schema objects for NLP-to-motion intent planning.

These dataclasses describe intent only. They do not generate Timeline keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AnatomyRef:
    semantic_region: str
    controllers: list[str] = field(default_factory=list)
    actor_scope: str = "primary"


@dataclass
class BaseState:
    family: str
    pose_subtype: str = "unknown"
    actor_role: str = "unknown"
    partner_role: str = "unknown"
    driver_region: str = "unknown"
    required_anchors: list[str] = field(default_factory=list)
    partner_relation: list[str] = field(default_factory=list)


@dataclass
class ActionConstraint:
    action_id: str
    effectors: list[str] = field(default_factory=list)
    target_node: str = "unknown"
    mode: str = "unknown"
    weight: float = 1.0
    relative_velocity: float | None = None


@dataclass
class MotionProfile:
    tempo_profile: str = "default"
    frequency_hz: list[float] = field(default_factory=list)
    amplitude_multiplier: float = 1.0
    curve_type: str = "smooth_sine"
    follower_lag: str = "medium"
    impact_profile: str = "none"


@dataclass
class SecondaryAction:
    action: str
    bodypart: str = "unknown"
    target: str = "unknown"
    timing: str = "overlap"
    can_overlap_with_base_motion: bool = True


@dataclass
class SequencePhase:
    phase_id: str
    duration_seconds: float | None = None
    base_state: BaseState | None = None
    motion_profile: MotionProfile | None = None
    constraints: list[ActionConstraint] = field(default_factory=list)
    secondary_actions: list[SecondaryAction] = field(default_factory=list)
    transition_to_next: dict[str, Any] | None = None


@dataclass
class MotionIntentPlan:
    sequence_id: str
    subject: str = "unknown"
    target: str = "unknown"
    phases: list[SequencePhase] = field(default_factory=list)
    unresolved_requirements: list[str] = field(default_factory=list)
    safety_rules: list[str] = field(default_factory=list)
    generated_timeline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

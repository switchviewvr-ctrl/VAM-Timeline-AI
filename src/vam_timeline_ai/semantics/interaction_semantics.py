"""Partner-relative interaction semantic records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class InteractionSemanticRecord:
    window_id: str
    pair_window_id: str | None = None
    rider_actor_id: str | None = None
    partner_actor_id: str | None = None
    actor_role: str = "unknown"
    partner_role: str = "unknown"
    interaction_family: str = "unknown"
    rider_pose_family: str = "unknown"
    partner_pose_family: str = "unknown"
    partner_relation: list[str] = field(default_factory=list)
    contact_targets: dict[str, Any] = field(default_factory=dict)
    support_context: str = "unknown"
    interaction_confidence: float = 0.0
    contact_support_confidence: float = 0.0
    contact_support_margin: float = 0.0
    contact_support_ambiguous: bool = False
    best_contact_target: str = "unknown"
    second_best_contact_target: str = "unknown"
    partner_context_confidence: float = 0.0
    hands_on_partner_legs_score: float = 0.0
    hands_on_partner_thighs_score: float = 0.0
    hands_behind_partner_support_score: float = 0.0
    partner_leg_thigh_approximation_used: bool = False
    warnings: list[str] = field(default_factory=list)
    is_human_ground_truth: bool = False
    is_training_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

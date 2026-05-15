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
    warnings: list[str] = field(default_factory=list)
    is_human_ground_truth: bool = False
    is_training_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

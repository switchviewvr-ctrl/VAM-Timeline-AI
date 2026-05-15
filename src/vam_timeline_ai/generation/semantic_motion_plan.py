"""Semantic motion plan schema for future generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import uuid


@dataclass
class PrimitiveQuery:
    family: str
    subtype: str
    trajectory_shape: str = "unknown"
    tempo: str = "medium"
    intensity: str = "medium"
    depth: str = "medium"
    amplitude: str = "medium"
    duration_range: dict[str, float] = field(default_factory=lambda: {"min_seconds": 4.0, "max_seconds": 8.0})
    generation_safe_required: bool = True
    anchor_complete_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticMotionPlanPhase:
    phase_id: str
    phase_type: str
    primitive_query: PrimitiveQuery
    duration: float
    style_parameters: dict[str, Any] = field(default_factory=dict)
    body_parameters: dict[str, Any] = field(default_factory=dict)
    contact_parameters: dict[str, Any] = field(default_factory=dict)
    safety_requirements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["primitive_query"] = self.primitive_query.to_dict()
        return data


@dataclass
class SemanticMotionPlan:
    plan_id: str
    source_prompt: str | None
    family: str
    requested_subtypes: list[str]
    sequence: list[SemanticMotionPlanPhase]
    warnings: list[str] = field(default_factory=list)
    is_final_text_to_animation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source_prompt": self.source_prompt,
            "family": self.family,
            "requested_subtypes": self.requested_subtypes,
            "sequence": [phase.to_dict() for phase in self.sequence],
            "warnings": self.warnings,
            "is_final_text_to_animation": self.is_final_text_to_animation,
        }


def new_plan_id(prefix: str = "plan_v0") -> str:
    return f"{prefix}::{uuid.uuid4().hex[:12]}"

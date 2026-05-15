"""Schemas for machine label proposals and silver labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


LabelGroup = Literal["movement", "role_candidate", "contact_candidate", "head_attention_candidate", "negative_candidate", "quality"]
ProposalType = Literal["positive", "negative", "uncertain", "role_candidate", "contact_candidate"]
ProposalSource = Literal["machine_rule_v1", "machine_pair_rule_v1", "machine_cluster_rule_v1"]


@dataclass
class MachineLabelProposal:
    proposal_id: str
    window_id: str
    pair_window_id: str | None
    sample_id: str
    source_id: str
    source_scene_file: str
    technical_atom_id: str
    label: str
    label_group: LabelGroup
    proposal_type: ProposalType
    confidence: float
    source: ProposalSource
    rule_id: str
    evidence_features: list[str] = field(default_factory=list)
    evidence_values: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    is_silver_candidate: bool = False
    is_human_ground_truth: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_human_ground_truth"] = False
        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
        return data


@dataclass
class SilverLabelRecord:
    window_id: str
    pair_window_id: str | None
    positive_labels: list[str] = field(default_factory=list)
    negative_labels: list[str] = field(default_factory=list)
    uncertain_labels: list[str] = field(default_factory=list)
    role_candidates: list[str] = field(default_factory=list)
    contact_candidates: list[str] = field(default_factory=list)
    confidence_by_label: dict[str, float] = field(default_factory=dict)
    rule_ids: list[str] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    label_source: str = "silver_machine_v1"
    is_human_ground_truth: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["label_source"] = "silver_machine_v1"
        data["is_human_ground_truth"] = False
        return data

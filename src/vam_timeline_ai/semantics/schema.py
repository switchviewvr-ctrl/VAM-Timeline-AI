"""Semantic data model for future Text -> VaM Timeline generation.

These dataclasses are intentionally schema-first. They describe what the
semantic database must store before ML, bridge playback, or Timeline generation
becomes useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticLabel:
    """A provisional semantic label that can be combined with other labels."""

    label_id: str
    description: str
    domain: str = "cowgirl_riding"
    provisional: bool = True


@dataclass
class ActorRoleGuess:
    """Role estimate for one technical VaM atom.

    Atom IDs are recorded only as technical identifiers. They must not be used
    as semantic truth.
    """

    technical_atom_id: str
    semantic_role_guess: str = "unknown"
    rider_score: float = 0.0
    receiver_score: float = 0.0
    focus_actor_score: float = 0.0
    partner_context_atom: str | None = None
    confidence: float = 0.0
    needs_manual_review: bool = True
    evidence: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class MovementWindow:
    """A first-class semantic window inside a source sample or clip."""

    sample_id: str
    start_seconds: float
    end_seconds: float
    window_seconds: float
    stride_seconds: float | None = None
    source_scene_file: str | None = None
    technical_actor_atom_id: str | None = None
    window_index: int | None = None

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("MovementWindow start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("MovementWindow end_seconds must be greater than start_seconds")
        if self.window_seconds <= 0:
            raise ValueError("MovementWindow window_seconds must be positive")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def key(self) -> str:
        return f"{self.sample_id}:{self.start_seconds:.3f}-{self.end_seconds:.3f}"


@dataclass
class MovementWindowFeatures:
    """Feature groups for a movement window.

    The groups match the semantic Cowgirl/Riding analysis plan. Values may be
    absent until the corresponding analyzer is implemented.
    """

    pelvis: dict[str, float | str | bool | None] = field(default_factory=dict)
    torso: dict[str, float | str | bool | None] = field(default_factory=dict)
    hands: dict[str, float | str | bool | None] = field(default_factory=dict)
    legs: dict[str, float | str | bool | None] = field(default_factory=dict)
    head_gaze: dict[str, float | str | bool | None] = field(default_factory=dict)
    rhythm_style: dict[str, float | str | bool | None] = field(default_factory=dict)
    computed: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ManualLabelOverride:
    """Manual correction at scene, actor, sample, or movement-window scope."""

    scene_file: str | None = None
    technical_atom_id: str | None = None
    sample_id: str | None = None
    window_key: str | None = None
    semantic_role: str | None = None
    focus_actor: bool | None = None
    include_for_cowgirl_db: bool | None = None
    labels: list[str] = field(default_factory=list)
    confidence: str = "manual"
    needs_manual_review: bool | None = None
    notes: str = ""


@dataclass
class CowgirlWindowRecord:
    """One semantic database row for a Cowgirl/Riding movement window."""

    window: MovementWindow
    actor_role: ActorRoleGuess
    features: MovementWindowFeatures
    labels: list[str] = field(default_factory=list)
    source_scene_file: str | None = None
    partner_context_atom: str | None = None
    manual_overrides: list[ManualLabelOverride] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        deduped: list[str] = []
        for label in self.labels:
            if label not in seen:
                deduped.append(label)
                seen.add(label)
        self.labels = deduped

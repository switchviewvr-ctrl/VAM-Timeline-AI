"""Research card helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class ResearchCard:
    source_url: str
    title: str
    category: str
    license_status: str = "unknown_needs_review"
    summary: str = ""
    extracted_concepts: list[str] = field(default_factory=list)
    maps_to_project: list[str] = field(default_factory=list)
    status: str = "candidate"
    do_not_use_as_training_data: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

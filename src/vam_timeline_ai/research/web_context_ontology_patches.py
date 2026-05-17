"""Build inactive ontology patch candidates from web research cards."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl
from vam_timeline_ai.research.web_context_research import dump_yaml
from vam_timeline_ai.semantics.ontology_loader import load_yaml


def build_web_context_ontology_patches_v1(
    research_dir: str | Path,
    current_ontology: str | Path,
    current_anatomy: str | Path,
    out_yaml: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    cards = load_jsonl(Path(research_dir) / "research_cards_v1.jsonl")
    ontology = load_yaml(current_ontology)
    anatomy = load_yaml(current_anatomy)
    patches = [_patch_from_card(card) for card in cards]
    data = {
        "schema": "web_context_ontology_patch_candidates_v1",
        "current_ontology": str(current_ontology),
        "current_anatomy": str(current_anatomy),
        "ontology_schema_seen": ontology.get("schema") or "unknown",
        "anatomy_schema_seen": anatomy.get("schema") or "unknown",
        "patch_candidates": patches,
        "lexicon_candidates": _lexicon_candidates(cards),
        "accepted": False,
        "requires_human_review": True,
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }
    dump_yaml(Path(out_yaml), data)
    summary = {
        "status": "ok",
        "cards": len(cards),
        "patch_candidates": len(patches),
        "lexicon_candidates": len(data["lexicon_candidates"]),
        "categories": dict(Counter(card.get("category") for card in cards)),
        "out_yaml": str(out_yaml),
        "report": str(report),
    }
    _write_report(Path(report), summary)
    return summary


def _patch_from_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch_id": f"candidate_{card.get('category')}_{abs(hash(card.get('source_url'))) % 100000}",
        "source_url": card.get("source_url"),
        "category": card.get("category"),
        "concepts": card.get("extracted_concepts") or [],
        "maps_to_project": card.get("maps_to_project") or [],
        "proposal": "review concept mapping; do not apply automatically",
        "accepted": False,
        "requires_human_review": True,
    }


def _lexicon_candidates(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for card in cards:
        for concept in card.get("extracted_concepts") or []:
            out.append(
                {
                    "id": f"external_candidate_{concept}",
                    "type": "external_candidate",
                    "terms": [str(concept).replace("_", " ")],
                    "maps_to": {"candidate_concept": concept},
                    "source_url": card.get("source_url"),
                    "active": False,
                    "accepted": False,
                    "requires_human_review": True,
                }
            )
    return out


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Web Context Ontology Patch Candidates v1",
        "",
        f"- Cards: `{summary['cards']}`",
        f"- Patch candidates: `{summary['patch_candidates']}`",
        f"- Lexicon candidates: `{summary['lexicon_candidates']}`",
        f"- Categories: `{summary['categories']}`",
        "- All candidates accepted: `false`",
        "- Requires human review: `true`",
        "- ML training performed: `false`",
        "- Timeline generation performed: `false`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

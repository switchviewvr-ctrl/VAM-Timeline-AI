"""Build project NLP lexicon from manual and candidate sources."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.semantics.ontology_loader import load_yaml, yaml


def build_nlp_lexicon_v1(
    manual: str | Path,
    sources: str | Path,
    out: str | Path,
    report: str | Path,
    allow_web: bool = False,
) -> dict[str, Any]:
    manual_data = load_yaml(manual)
    sources_data = load_yaml(sources)
    entries = []
    for entry in manual_data.get("entries", []):
        item = dict(entry)
        item.setdefault("source", "manual")
        item.setdefault("accepted", True)
        item.setdefault("requires_human_review", False)
        entries.append(item)
    # External candidate loading is intentionally conservative. No web terms are
    # activated by this command; candidate patches stay inactive until reviewed.
    for source in sources_data.get("sources", []):
        if source.get("type") != "external_candidate":
            continue
        candidate_path = Path(str(source.get("path") or ""))
        candidate_data = load_yaml(candidate_path)
        for candidate in candidate_data.get("lexicon_candidates", []):
            item = dict(candidate)
            item.setdefault("active", False)
            item["accepted"] = False
            item["requires_human_review"] = True
            item["source"] = source.get("source_id")
            entries.append(item)
    lexicon = {
        "schema": "nlp_lexicon_v1",
        "description": "Merged NLP lexicon. Manual entries are active; external candidates require review.",
        "allow_web_used": bool(allow_web),
        "entries": entries,
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        target.write_text(yaml.safe_dump(lexicon, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        import json

        target.write_text(json.dumps(lexicon, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "ok",
        "entries": len(entries),
        "active_entries": sum(1 for e in entries if e.get("active") is not False),
        "candidate_entries": sum(1 for e in entries if e.get("requires_human_review")),
        "entry_types": dict(Counter(e.get("type") for e in entries)),
        "out": str(out),
        "report": str(report),
    }
    _write_report(Path(report), summary, manual, sources)
    return summary


def _write_report(path: Path, summary: dict[str, Any], manual: str | Path, sources: str | Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NLP Lexicon v1",
        "",
        f"- Manual source: `{manual}`",
        f"- Source registry: `{sources}`",
        f"- Entries: `{summary['entries']}`",
        f"- Active entries: `{summary['active_entries']}`",
        f"- Candidate entries: `{summary['candidate_entries']}`",
        f"- Entry types: `{summary['entry_types']}`",
        "- External candidates default accepted=false.",
        "- ML training performed: `false`",
        "- Timeline generation performed: `false`",
        "- manual_labels.yaml modified: `false`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Ingest answers exported from the local semantic review UI."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import validate_answer


def ingest_review_ui_answers(
    answers: str | Path,
    review_dir: str | Path,
    out_ledger: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    answer_path = Path(answers)
    review = Path(review_dir)
    rows = _load_answers(answer_path)
    validated = [validate_answer(row) for row in rows if row.get("review_id")]
    system_rows = _system_rows(review)
    ledger_path = Path(out_ledger)
    existing = load_jsonl(ledger_path)
    existing_keys = {
        (row.get("source_review_folder"), row.get("review_id"), row.get("source") or "")
        for row in existing
    }
    new_records = []
    for answer in validated:
        record = _ledger_record(answer, review, system_rows.get(answer["review_id"]) or {})
        key = (record.get("source_review_folder"), record.get("review_id"), record.get("source") or "")
        if key not in existing_keys:
            new_records.append(record)
    merged = existing + new_records
    write_jsonl(ledger_path, merged)
    _write_report(validated, new_records, report)
    return {
        "status": "ok",
        "answers": len(validated),
        "new_ledger_records": len(new_records),
        "out_ledger": str(ledger_path),
        "report": str(report),
    }


def _load_answers(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        reviews = data.get("reviews") or {}
        return [dict({"review_id": rid}, **(row or {})) for rid, row in reviews.items()]
    return load_jsonl(path)


def _system_rows(review: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in [review / "semantic_review_010.jsonl", review / "vam_review_package" / "vam_review_manifest.jsonl"]:
        for row in load_jsonl(path):
            rid = row.get("review_id")
            if rid:
                rows.setdefault(rid, {}).update(row)
    return rows


def _ledger_record(answer: dict[str, Any], review: Path, system: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": answer["review_id"],
        "source": "review_ui",
        "source_review_folder": str(review),
        "run_id": _run_id(review),
        "source_scene_file": system.get("source_scene_file") or "",
        "technical_actor_id": system.get("technical_actor_id") or system.get("technical_atom_id") or "",
        "start_seconds": system.get("start_seconds"),
        "end_seconds": system.get("end_seconds"),
        "system_semantic_family": system.get("semantic_family") or "",
        "human_semantic_family": answer.get("actual_semantic_family") or "",
        "system_pose": _join(system.get("pose_family"), system.get("pose_subtype")),
        "human_pose": answer.get("actual_pose") or "",
        "system_motion": system.get("motion_subtype") or "",
        "human_motion": answer.get("actual_motion") or "",
        "system_partner_relation": _join(system.get("partner_relation")),
        "human_partner_relation": answer.get("actual_partner_relation") or "",
        "system_contact_support": system.get("contact_support") or "",
        "human_contact_support": answer.get("actual_contact_support") or "",
        "system_generation_safe": system.get("generation_safe"),
        "human_generation_safe": answer.get("actual_generation_safe") or "",
        "verdict": answer.get("verdict") or _derive_verdict(answer),
        "error_tags": answer.get("error_tags") or [],
        "notes": answer.get("notes") or "",
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _derive_verdict(answer: dict[str, Any]) -> str:
    checks = [
        answer.get("semantic_family_correct"),
        answer.get("pose_correct"),
        answer.get("motion_correct"),
        answer.get("partner_relation_correct"),
        answer.get("contact_support_correct"),
        answer.get("generation_safe_correct"),
    ]
    known = [str(v).lower() for v in checks if str(v).lower() not in {"", "unknown", "not_applicable", "none"}]
    if not known:
        return "unclear"
    if all(v == "true" for v in known):
        return "correct"
    if all(v == "false" for v in known):
        return "wrong"
    return "partially_correct"


def _write_report(answers: list[dict[str, Any]], new_records: list[dict[str, Any]], report: str | Path) -> None:
    verdicts = Counter((a.get("verdict") or _derive_verdict(a)) for a in answers)
    tags = Counter(tag for a in answers for tag in (a.get("error_tags") or []))
    family_known = [a for a in answers if a.get("semantic_family_correct") in {"true", "false"}]
    contact_known = [a for a in answers if a.get("contact_support_correct") in {"true", "false"}]
    lines = [
        "# Review UI Answer Ingestion Report",
        "",
        "Answers are audit findings only. They were not written to manual_labels.yaml.",
        "",
        f"- Answers read: {len(answers)}",
        f"- New ledger records appended: {len(new_records)}",
        "",
        "## Verdict Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in verdicts.most_common()) if verdicts else lines.append("- None")
    lines.extend(["", "## Common Error Tags", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in tags.most_common()) if tags else lines.append("- None")
    lines.extend(
        [
            "",
            "## Accuracy Fields Present",
            "",
            f"- Semantic family correctness answers: {len(family_known)}",
            f"- Contact/support correctness answers: {len(contact_known)}",
        ]
    )
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_id(path: Path) -> str:
    parts = list(path.parts)
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def _join(*values: Any) -> str:
    if len(values) == 1 and isinstance(values[0], list):
        return ";".join(str(v) for v in values[0])
    return " / ".join(str(v) for v in values if v)

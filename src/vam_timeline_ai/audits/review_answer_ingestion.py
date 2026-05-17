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
    overwrite: bool = False,
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
    duplicate_records = 0
    replaced_records = 0
    replacement_keys = set()
    for answer in validated:
        record = _ledger_record(answer, review, system_rows.get(answer["review_id"]) or {})
        key = (record.get("source_review_folder"), record.get("review_id"), record.get("source") or "")
        if key not in existing_keys:
            new_records.append(record)
        elif overwrite:
            new_records.append(record)
            replacement_keys.add(key)
            replaced_records += 1
        else:
            duplicate_records += 1
    merged_existing = [
        row
        for row in existing
        if (row.get("source_review_folder"), row.get("review_id"), row.get("source") or "") not in replacement_keys
    ]
    merged = merged_existing + new_records
    write_jsonl(ledger_path, merged)
    _write_report(validated, new_records, report, duplicate_records=duplicate_records, replaced_records=replaced_records)
    return {
        "status": "ok",
        "answers": len(validated),
        "new_ledger_records": len(new_records),
        "duplicates_skipped": duplicate_records,
        "records_replaced": replaced_records,
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
    derived = _derive_from_answer(answer)
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
        "human_semantic_family": answer.get("actual_semantic_family") or derived.get("semantic_family") or "",
        "system_pose": _join(system.get("pose_family"), system.get("pose_subtype")),
        "human_pose": answer.get("actual_pose") or derived.get("pose") or "",
        "system_motion": system.get("motion_subtype") or "",
        "human_motion": answer.get("actual_motion") or derived.get("motion") or "",
        "system_partner_relation": _join(system.get("partner_relation")),
        "human_partner_relation": answer.get("actual_partner_relation") or "",
        "system_contact_support": system.get("contact_support") or "",
        "human_contact_support": answer.get("actual_contact_support") or derived.get("contact_support") or "",
        "system_generation_safe": system.get("generation_safe"),
        "human_generation_safe": answer.get("actual_generation_safe") or derived.get("generation_safe") or "",
        "verdict": answer.get("verdict") or _derive_verdict(answer),
        "error_tags": list(dict.fromkeys((answer.get("error_tags") or []) + derived.get("error_tags", []))),
        "notes": answer.get("notes") or "",
        "screenshot_count": len(answer.get("screenshots") or []),
        "screenshots": answer.get("screenshots") or [],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _derive_verdict(answer: dict[str, Any]) -> str:
    labels = set(answer.get("review_labels") or [])
    notes = str(answer.get("notes") or "").lower()
    if labels & {"correct_clean_cowgirl_motion", "correct_short_cowgirl_motion", "correct_lean_back_supported_cowgirl", "front_cowgirl_not_reverse"}:
        return "correct"
    if labels & {"cowgirl_pose_only_low_motion", "cowgirl_transition_intro_alignment", "hands_behind_support_correct", "hands_on_partner_legs_or_thighs_correct"}:
        return "partially_correct"
    if labels & {"standing_hand_head_not_cowgirl", "bj_oral_not_cowgirl", "wrong_partner_context", "wrong_contact_support", "wrongly_marked_reverse_cowgirl", "broken_pose_or_bad_data"}:
        return "wrong"
    if "unknown_unclear" in labels:
        return "unclear"
    if any(bit in notes for bit in ["eindeutig bj", "bj animation", "hat mit cowgirl nix zu tun", "teleportiert"]):
        return "wrong"
    if any(bit in notes for bit in ["nicht zuzuordnen", "kann man absolut nicht", "fehlt im ordner", "keine bewegung"]):
        return "unclear"
    if "cowgirl" in notes and any(bit in notes for bit in ["grinding", "riding", "bounce", "oval", "hüftanimation", "entsprechender animation"]):
        return "correct"
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


def _derive_from_answer(answer: dict[str, Any]) -> dict[str, Any]:
    derived = _derive_from_review_labels(answer.get("review_labels") or [])
    text_derived = _derive_from_free_text(str(answer.get("notes") or ""))
    for key, value in text_derived.items():
        if key == "error_tags":
            derived["error_tags"] = list(dict.fromkeys((derived.get("error_tags") or []) + value))
        else:
            derived.setdefault(key, value)
    return derived


def _derive_from_review_labels(labels: list[str]) -> dict[str, Any]:
    label_set = set(labels)
    out: dict[str, Any] = {"error_tags": []}
    if label_set & {"correct_clean_cowgirl_motion", "correct_short_cowgirl_motion", "cowgirl_pose_only_low_motion", "cowgirl_transition_intro_alignment", "correct_lean_back_supported_cowgirl", "front_cowgirl_not_reverse"}:
        out["semantic_family"] = "cowgirl"
    if "bj_oral_not_cowgirl" in label_set:
        out["semantic_family"] = "bj_oral"
        out["error_tags"].append("bj_oral_as_cowgirl")
    if "standing_hand_head_not_cowgirl" in label_set:
        out["semantic_family"] = "standing_hand_head_gesture"
        out["error_tags"].append("standing_hand_head_as_cowgirl")
    if "correct_clean_cowgirl_motion" in label_set:
        out["motion"] = "clean_cowgirl_motion"
    if "correct_short_cowgirl_motion" in label_set:
        out["motion"] = "clean_cowgirl_motion_low_confidence_short"
    if "cowgirl_pose_only_low_motion" in label_set:
        out["motion"] = "low_motion_hold"
        out["error_tags"].append("low_motion_hold")
    if "cowgirl_transition_intro_alignment" in label_set:
        out["motion"] = "transition_intro_alignment"
        out["error_tags"].append("intro_alignment")
    if "correct_lean_back_supported_cowgirl" in label_set:
        out["pose"] = "cowgirl_lean_back_supported"
    if "hands_behind_support_correct" in label_set:
        out["contact_support"] = "hands_behind_support"
    if "hands_on_partner_legs_or_thighs_correct" in label_set:
        out["contact_support"] = "hands_on_partner_legs_or_thighs"
    if "wrong_partner_context" in label_set:
        out["error_tags"].append("partner_context_missing")
    if "wrong_contact_support" in label_set:
        out["error_tags"].append("contact_wrong_target")
    if "broken_pose_or_bad_data" in label_set:
        out["error_tags"].append("pose_broken")
    return out


def _derive_from_free_text(notes: str) -> dict[str, Any]:
    text = notes.lower()
    out: dict[str, Any] = {"error_tags": []}
    if not text.strip():
        return out
    if "bj" in text or "kopfbewegung" in text:
        out["semantic_family"] = "bj_oral"
        out["motion"] = "bj_oral_motion"
        out["error_tags"].append("bj_oral_as_cowgirl")
    elif any(bit in text for bit in ["hat mit cowgirl nix zu tun", "aufsteh-animation", "hinlegt", "nach vorn geschoben"]):
        out["semantic_family"] = "unknown"
        out["error_tags"].append("not_cowgirl")
    elif "cowgirl" in text:
        out["semantic_family"] = "cowgirl"

    if "upright cowgirl" in text:
        out["pose"] = "cowgirl_upright"
    elif "lean forward" in text or "hocke" in text:
        out["pose"] = "cowgirl_lean_forward_supported"
    elif "cowgirl pose" in text:
        out.setdefault("pose", "cowgirl_pose_context")

    if any(bit in text for bit in ["grinding", "riding", "bounce", "oval", "hüftanimation", "teasing"]):
        out.setdefault("motion", "clean_cowgirl_motion")
    if "transition" in text:
        out["motion"] = "transition_intro_alignment"
        out["error_tags"].append("intro_alignment")
    if any(bit in text for bit in ["hüften bewegen sich nicht", "keine cowgirl animation", "keine bewegung", "ohne bewegung", "nur chest und pelvis"]):
        out["motion"] = "low_motion_or_no_clear_hip_motion"
        out["error_tags"].append("no_clear_hip_motion")
    if "zu kurz" in text:
        out["motion"] = "clean_cowgirl_motion_low_confidence_short"
        out["error_tags"].append("short_window_low_confidence")

    if any(bit in text for bit in ["fuß controller fehlen", "controller fehlen", "teleportiert", "controller sind zuweit auseinander", "fehlt im ordner"]):
        out["generation_safe"] = "false"
        out["error_tags"].append("controller_missing_or_invalid")
    if any(bit in text for bit in ["nicht zuzuordnen", "kann man absolut nicht", "fehlt im ordner"]):
        out["error_tags"].append("unknown_unclear")
    return out


def _write_report(
    answers: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    report: str | Path,
    duplicate_records: int = 0,
    replaced_records: int = 0,
) -> None:
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
        f"- Duplicates skipped: {duplicate_records}",
        f"- Records replaced: {replaced_records}",
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

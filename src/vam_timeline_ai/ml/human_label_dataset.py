"""Human-reviewed labels for the Cowgirl review-assist ML baseline.

This module deliberately derives labels only from human review artifacts.
Rule/silver/machine labels may appear in reports, but they are not targets.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


TRUE = "true"
FALSE = "false"
UNKNOWN = "unknown"


def build_human_reviewed_ml_labels_v1(
    run_dir: str | Path,
    human_ledger: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    sources = []

    for row in load_jsonl(human_ledger):
        label = _derive_label_record(row, source_kind="human_review_ledger")
        key = _dedupe_key(label)
        if key not in seen:
            rows.append(label)
            seen.add(key)
    if Path(human_ledger).exists():
        sources.append(str(human_ledger))

    for answer_path in sorted(run.glob("audits/**/human_review_ui_answers.jsonl")):
        review_dir = answer_path.parent
        review_rows = _review_rows_by_id(review_dir)
        for answer in load_jsonl(answer_path):
            merged = dict(review_rows.get(str(answer.get("review_id") or ""), {}))
            merged.update(answer)
            merged["source_review_folder"] = str(review_dir)
            label = _derive_label_record(merged, source_kind="human_review_ui_answers")
            key = _dedupe_key(label)
            if key not in seen:
                rows.append(label)
                seen.add(key)
        sources.append(str(answer_path))

    for notes_path in sorted(run.glob("audits/**/semantic_review_010_human_notes.yaml")):
        data = _load_yaml(notes_path)
        for review_id, note in (data.get("reviews") or data or {}).items():
            if not isinstance(note, dict):
                continue
            note = dict(note)
            note.setdefault("review_id", review_id)
            note.setdefault("source_review_folder", str(notes_path.parent))
            label = _derive_label_record(note, source_kind="human_notes_yaml")
            key = _dedupe_key(label)
            if key not in seen:
                rows.append(label)
                seen.add(key)
        sources.append(str(notes_path))

    write_jsonl(out_jsonl, rows)
    summary = _summary(rows, sources)
    _write_report(report, summary)
    return summary


def _derive_label_record(row: dict[str, Any], source_kind: str) -> dict[str, Any]:
    labels = _listify(row.get("actual_labels")) + _listify(row.get("review_labels")) + _listify(row.get("error_tags"))
    text = " ".join(
        str(row.get(k) or "")
        for k in [
            "human_semantic_family",
            "semantic_family",
            "actual_semantic_family",
            "human_motion",
            "actual_motion",
            "verdict",
            "user_verdict",
            "notes",
        ]
    )
    text_l = text.lower()
    labels_l = {str(x).strip().lower() for x in labels if str(x).strip()}

    cowgirl_true_terms = {
        "cowgirl_true_segment",
        "correct_clean_cowgirl_motion",
        "correct_short_cowgirl_motion",
        "clean_cowgirl_motion",
        "clean_cowgirl_motion_low_confidence",
        "correct_lean_back_supported_cowgirl",
    }
    cowgirl_false_terms = {
        "not_cowgirl",
        "bj_oral_not_cowgirl",
        "standing_hand_head_not_cowgirl",
        "receiver_response_not_rider_motion",
        "unknown_unusable",
        "broken_pose_or_bad_data",
    }
    clean_false_terms = {
        "low_motion_hold",
        "cowgirl_pose_only_low_motion",
        "cowgirl_transition_intro_alignment",
        "intro_alignment",
        "transition_setup",
        "pose_context_only",
        "no_clear_hip_motion",
        "not_clean_motion",
    }
    gen_false_terms = {
        "pose_broken",
        "broken_pose_or_bad_data",
        "missing_required_controllers",
        "controller_missing",
        "foot_anchor_weird",
        "lower_body_anchor_not_stable",
        "generation_safe_false_positive",
        "not_generation_candidate",
        "controller_missing_or_invalid",
    }

    human_family = str(row.get("human_semantic_family") or row.get("actual_semantic_family") or row.get("semantic_family") or "").lower()
    if labels_l & cowgirl_true_terms or human_family == "cowgirl" or (_mentions_cowgirl_positive(text_l) and not labels_l & cowgirl_false_terms):
        cowgirl = TRUE
    elif labels_l & cowgirl_false_terms or human_family in {"bj_oral", "standing_hand_head_gesture", "receiver_response", "unknown"} or _mentions_non_cowgirl(text_l):
        cowgirl = FALSE
    else:
        cowgirl = UNKNOWN

    if cowgirl == TRUE and ((labels_l & cowgirl_true_terms) or "clean cowgirl" in text_l or "normal cowgirl" in text_l or "grinding" in text_l or "riding" in text_l):
        clean = TRUE
    elif labels_l & clean_false_terms or "nur setup" in text_l or "transition" in text_l or "low motion" in text_l or "fast keine motion" in text_l:
        clean = FALSE
    else:
        clean = UNKNOWN

    if labels_l & gen_false_terms:
        gen_safe = FALSE
    elif cowgirl == TRUE and clean == TRUE and not (labels_l & gen_false_terms):
        gen_safe = TRUE if ("pose_valid" in labels_l or "generation_safe_candidate" in labels_l) else UNKNOWN
    else:
        gen_safe = UNKNOWN

    return {
        "schema": "human_reviewed_labels_v1",
        "source_kind": source_kind,
        "review_id": row.get("review_id") or row.get("id") or "",
        "source_review_folder": row.get("source_review_folder") or row.get("review_folder") or "",
        "run_id": row.get("run_id") or "",
        "source_scene_file": row.get("source_scene_file") or row.get("scene") or "",
        "technical_actor_id": row.get("technical_actor_id") or row.get("technical_atom_id") or row.get("actor") or "",
        "sample_id": row.get("sample_id") or "",
        "window_id": row.get("window_id") or "",
        "pair_window_id": row.get("pair_window_id") or "",
        "start_seconds": _maybe_float(row.get("start_seconds")),
        "end_seconds": _maybe_float(row.get("end_seconds")),
        "label_cowgirl_candidate": cowgirl,
        "label_clean_motion": clean,
        "label_generation_safe": gen_safe,
        "human_semantic_family": row.get("human_semantic_family") or row.get("actual_semantic_family") or row.get("semantic_family") or "",
        "human_motion": row.get("human_motion") or row.get("actual_motion") or "",
        "human_generation_safe": row.get("human_generation_safe") or row.get("actual_generation_safe") or "",
        "verdict": row.get("verdict") or row.get("user_verdict") or "",
        "error_tags": sorted(labels_l),
        "human_notes": row.get("notes") or "",
        "is_human_reviewed_label": True,
        "is_training_truth_source": "human_review_audit_only",
        "manual_labels_yaml_modified": False,
    }


def _mentions_cowgirl_positive(text: str) -> bool:
    return "cowgirl" in text and not _mentions_non_cowgirl(text)


def _mentions_non_cowgirl(text: str) -> bool:
    negative_bits = [
        "not cowgirl",
        "not_cowgirl",
        "nicht cowgirl",
        "bj/oral",
        "standing",
        "steht",
        "nur ihre hände",
        "hand/head",
        "unknown unusable",
    ]
    return any(bit in text for bit in negative_bits)


def _review_rows_by_id(review_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for name in ["semantic_review_010.jsonl", "focused_review_manifest.jsonl", "strict_cowgirl_review_manifest.jsonl"]:
        for row in load_jsonl(review_dir / name):
            rid = str(row.get("review_id") or "")
            if rid:
                rows[rid] = row
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def _maybe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("window_id") or row.get("source_scene_file") or ""),
        str(row.get("review_id") or ""),
        str(row.get("human_notes") or ""),
    )


def _summary(rows: list[dict[str, Any]], sources: list[str]) -> dict[str, Any]:
    return {
        "status": "ok",
        "schema": "human_reviewed_labels_v1",
        "sources": sources,
        "total_human_reviewed": len(rows),
        "cowgirl_label_counts": dict(Counter(r["label_cowgirl_candidate"] for r in rows)),
        "clean_motion_label_counts": dict(Counter(r["label_clean_motion"] for r in rows)),
        "generation_safe_label_counts": dict(Counter(r["label_generation_safe"] for r in rows)),
        "usable_cowgirl_labels": sum(1 for r in rows if r["label_cowgirl_candidate"] in {TRUE, FALSE}),
        "usable_clean_motion_labels": sum(1 for r in rows if r["label_clean_motion"] in {TRUE, FALSE}),
        "usable_generation_safe_labels": sum(1 for r in rows if r["label_generation_safe"] in {TRUE, FALSE}),
        "class_imbalance_warnings": _imbalance_warnings(rows),
        "duplicate_warnings": _duplicate_warnings(rows),
        "review_assist_only": True,
    }


def _imbalance_warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for key in ["label_cowgirl_candidate", "label_clean_motion", "label_generation_safe"]:
        counts = Counter(r[key] for r in rows if r[key] in {TRUE, FALSE})
        if counts and min(counts.values()) < 5:
            warnings.append(f"{key} has fewer than 5 examples in one class: {dict(counts)}")
    return warnings


def _duplicate_warnings(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(r.get("window_id") or f"{r.get('source_scene_file')}:{r.get('start_seconds')}:{r.get('end_seconds')}" for r in rows)
    dupes = [k for k, v in counts.items() if k and v > 1]
    return [f"{len(dupes)} duplicate reviewed window/time keys found"] if dupes else []


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Human-reviewed ML Labels v1",
        "",
        "These labels are derived from human review artifacts only. They are audit labels for review-assist ML, not `manual_labels.yaml`.",
        "",
        f"- Total human reviewed records: {summary['total_human_reviewed']}",
        f"- Usable Cowgirl labels: {summary['usable_cowgirl_labels']}",
        f"- Usable clean-motion labels: {summary['usable_clean_motion_labels']}",
        f"- Usable generation-safe labels: {summary['usable_generation_safe_labels']}",
        f"- Cowgirl counts: `{summary['cowgirl_label_counts']}`",
        f"- Clean-motion counts: `{summary['clean_motion_label_counts']}`",
        f"- Generation-safe counts: `{summary['generation_safe_label_counts']}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = summary["class_imbalance_warnings"] + summary["duplicate_warnings"]
    lines.extend(f"- {w}" for w in warnings) if warnings else lines.append("- none")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

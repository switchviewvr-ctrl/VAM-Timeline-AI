"""Pose/export validity audit for semantic review batches.

This separates "the motion looks semantically right" from "the exported/reviewed
segment is reusable as a generation template."  Human review notes remain
audit-only and are never merged into manual labels.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


BROKEN_LABELS = {"pose_broken", "export_pose_validity_issue"}
EXPORT_UNAVAILABLE_LABELS = {"export_unavailable"}
LOW_MOTION_LABELS = {"low_motion_intro", "static_or_micro_motion", "static_or_empty", "minimal_head_motion", "minimal_hand_jitter"}
TOO_SHORT_LABELS = {"too_short_for_semantic_judgment"}
SEMANTIC_POSITIVE_LABELS = {"cowgirl_true_segment", "clean_cowgirl_motion"}
CONTEXT_LABELS = {"possible_cowgirl_context", "cowgirl_squat_or_crouch_pose", "cowgirl_intro_or_start_pose", "pose_context_cowgirl_but_motion_unclear"}
RECEIVER_LABELS = {"receiver_body_response", "passive_receiver_motion", "not_active_rider"}


def audit_pose_export_validity(
    run_dir: str | Path,
    review_dir: str | Path,
    sample_index: str | Path,
    relative_index: str | Path,
    body_quality: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    review_path = Path(review_dir)
    review_rows = load_jsonl(review_path / "semantic_review_010.jsonl")
    answers = _load_answers(review_path / "semantic_review_010_answer_sheet.yaml")
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    relative = {r.get("window_id"): r for r in load_jsonl(relative_index) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rows = [
        pose_export_validity_for_review_item(row, answers.get(row.get("review_id"), {}), samples.get(row.get("sample_id"), {}), relative.get(row.get("window_id"), {}), body.get(row.get("window_id"), {}))
        for row in review_rows
    ]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def pose_export_validity_for_review_item(
    review_row: dict[str, Any],
    answer: dict[str, Any] | None,
    sample: dict[str, Any] | None,
    relative: dict[str, Any] | None,
    body: dict[str, Any] | None,
) -> dict[str, Any]:
    answer = answer or {}
    sample = sample or {}
    relative = relative or {}
    body = body or {}
    labels = {str(x) for x in (answer.get("actual_labels") or [])}
    verdict = str(answer.get("user_verdict") or "unknown")
    has_export = bool(review_row.get("has_timeline_export"))
    export_unavailable = (not has_export) or bool(labels & EXPORT_UNAVAILABLE_LABELS)
    broken = bool(labels & BROKEN_LABELS)
    low_motion_intro = bool(labels & {"low_motion_intro", "cowgirl_intro_or_start_pose"})
    too_short = bool(labels & TOO_SHORT_LABELS) or float(review_row.get("duration_seconds") or 0.0) < 4.0
    receiver = bool(labels & RECEIVER_LABELS)
    semantic_positive = bool(labels & SEMANTIC_POSITIVE_LABELS)
    context_positive = bool(labels & CONTEXT_LABELS)
    if semantic_positive:
        semantic_valid: bool | str = True
    elif receiver or "not_cowgirl" in labels:
        semantic_valid = False
    elif context_positive or verdict in {"unclear", "correct_or_unclear"}:
        semantic_valid = "unknown"
    else:
        semantic_valid = "unknown"
    if export_unavailable:
        export_pose_validity = "export_unavailable"
    elif broken:
        export_pose_validity = "broken_pose"
    elif has_export and semantic_positive and not broken:
        export_pose_validity = "good"
    elif has_export:
        export_pose_validity = "review_only_absolute_pose"
    else:
        export_pose_validity = "unknown"
    allowed_count = int(relative.get("coordinate_space_assumptions", {}).get("allowed_body_controller_count") or len(relative.get("controllers", []) or []))
    stripped_count = int(relative.get("stripped_track_count") or 0)
    duration_ok = not too_short
    motion_strength = _motion_strength(body, relative)
    motion_strength_adequate = motion_strength >= 0.35 and not bool(body.get("static_or_micro_motion"))
    has_required = _has_required_body_controllers(relative)
    teleport_risk = str(relative.get("teleport_risk") or "unknown")
    generation_safe = bool(
        not export_unavailable
        and not broken
        and semantic_valid is True
        and has_required
        and duration_ok
        and motion_strength_adequate
        and bool(relative.get("safe_for_learning"))
        and teleport_risk in {"low", "medium"}
    )
    warnings = []
    if broken:
        warnings.append("Semantics may be correct, but reviewed/exported pose was broken.")
    if export_unavailable:
        warnings.append("Export unavailable is not counted as semantic wrong.")
    if low_motion_intro:
        warnings.append("Cowgirl context/intro is not clean motion.")
    if too_short:
        warnings.append("Window is too short for confident semantic motion judgment.")
    if not generation_safe:
        warnings.append("Not safe as generation template without further validation.")
    return {
        "review_id": review_row.get("review_id"),
        "window_id": review_row.get("window_id"),
        "sample_id": review_row.get("sample_id"),
        "source_id": review_row.get("source_id"),
        "source_scene_file": review_row.get("source_scene_file"),
        "technical_atom_id": review_row.get("technical_atom_id"),
        "semantic_motion_likely_valid": semantic_valid,
        "export_pose_validity": export_pose_validity,
        "generation_template_safe": generation_safe,
        "pose_broken_score": 1.0 if broken else 0.0,
        "teleport_or_world_coordinate_risk": teleport_risk,
        "allowed_body_controller_count": allowed_count,
        "stripped_root_world_track_count": stripped_count,
        "has_required_body_controllers": has_required,
        "duration_adequate_for_semantic_judgment": duration_ok,
        "motion_strength_adequate": motion_strength_adequate,
        "motion_strength_score": round(float(motion_strength), 6),
        "low_motion_intro_candidate": low_motion_intro,
        "too_short_for_semantic_judgment": too_short,
        "review_export_available": has_export,
        "uses_absolute_review_coordinates": has_export,
        "exported_as_relative_motion": False,
        "source_world_coords_stripped": bool(stripped_count),
        "human_audit_labels": sorted(labels),
        "user_verdict": verdict,
        "warnings": warnings,
        "is_training_label": False,
        "is_human_ground_truth": False,
    }


def _load_answers(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("reviews", {}) or {}


def _motion_strength(body: dict[str, Any], relative: dict[str, Any]) -> float:
    if body.get("static_or_micro_motion"):
        return 0.0
    ratio = float(body.get("meaningful_motion_duration_ratio") or 0.0)
    active = min(float(body.get("active_bodypart_count_above_threshold") or 0.0) / 3.0, 1.0)
    moving = min(float(relative.get("moving_controller_count_relative") or body.get("moving_controller_count") or 0.0) / 3.0, 1.0)
    return max(0.0, min(1.0, 0.45 * ratio + 0.30 * active + 0.25 * moving))


def _has_required_body_controllers(relative: dict[str, Any]) -> bool:
    parts = {str(x) for x in relative.get("bodyparts", []) or []}
    has_pelvis = bool(parts & {"hip", "pelvis", "abdomen"})
    has_support = bool(parts & {"chest", "left_thigh", "right_thigh", "left_knee", "right_knee", "left_hand", "right_hand", "head"})
    return has_pelvis and has_support


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    validity = Counter(r.get("export_pose_validity") for r in rows)
    semantic = Counter(str(r.get("semantic_motion_likely_valid")) for r in rows)
    generation = Counter("safe" if r.get("generation_template_safe") else "blocked" for r in rows)
    lines = [
        "# Pose / Export Validity Report",
        "",
        "This audit separates semantic correctness from export/pose/generation-template usability.",
        "",
        f"- Review items: {len(rows)}",
        "",
        "## Semantic Motion Likely Valid",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in semantic.most_common())
    lines.extend(["", "## Export Pose Validity", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in validity.most_common())
    lines.extend(["", "## Generation Template Safety", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in generation.most_common())
    lines.extend(["", "## Semantically Good But Not Generation-Safe", ""])
    examples = [r for r in rows if r.get("semantic_motion_likely_valid") is True and not r.get("generation_template_safe")]
    if examples:
        for row in examples:
            lines.append(f"- `{row.get('review_id')}` window=`{row.get('window_id')}` validity=`{row.get('export_pose_validity')}` warnings={row.get('warnings')}")
    else:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


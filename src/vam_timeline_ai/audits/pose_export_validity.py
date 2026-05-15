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
    controller_validity: str | Path | None = None,
    pose_anchor_completeness: str | Path | None = None,
) -> list[dict[str, Any]]:
    review_path = Path(review_dir)
    review_rows = load_jsonl(review_path / "semantic_review_010.jsonl")
    answers = _load_answers(review_path / "semantic_review_010_answer_sheet.yaml")
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    relative = {r.get("window_id"): r for r in load_jsonl(relative_index) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")} if controller_validity else {}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")} if pose_anchor_completeness else {}
    rows = [
        pose_export_validity_for_review_item(
            row,
            answers.get(row.get("review_id"), {}),
            samples.get(row.get("sample_id"), {}),
            relative.get(row.get("window_id"), {}),
            body.get(row.get("window_id"), {}),
            controller.get(row.get("window_id"), {}),
            anchors.get(row.get("window_id"), {}),
        )
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
    controller: dict[str, Any] | None = None,
    anchors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer = answer or {}
    sample = sample or {}
    relative = relative or {}
    body = body or {}
    controller = controller or {}
    anchors = anchors or {}
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
    elif controller.get("controller_validity_status") == "invalid" or controller.get("foot_controller_outlier"):
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
    controller_status = str(controller.get("controller_validity_status") or "unknown")
    foot_outlier = bool(controller.get("foot_controller_outlier") or "foot_controller_outlier" in labels)
    hand_outlier = bool(controller.get("hand_controller_outlier") or "hand_controller_outlier" in labels)
    missing_foot = bool(controller.get("missing_foot_controllers") or anchors.get("missing_foot_controllers") or "missing_foot_controllers" in labels)
    missing_knee = bool(controller.get("missing_knee_controllers") or anchors.get("missing_knee_controllers") or "missing_knee_controllers" in labels)
    anchor_incomplete = bool(controller.get("pose_anchor_incomplete") or anchors.get("pose_anchor_incomplete") or missing_foot or missing_knee)
    controller_outlier_count = int(controller.get("controller_outlier_count") or 0)
    if foot_outlier and not controller.get("foot_controller_outlier"):
        controller_outlier_count += 1
    if hand_outlier and not controller.get("hand_controller_outlier"):
        controller_outlier_count += 1
    controller_generation_ok = True
    if controller:
        controller_generation_ok = controller.get("generation_pose_valid") is True and controller_status == "valid" and not foot_outlier and not anchor_incomplete
    generation_safe = bool(
        not export_unavailable
        and not broken
        and controller_generation_ok
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
    if foot_outlier:
        warnings.append("Foot controller outlier blocks generation-template use but does not automatically make semantics wrong.")
    if hand_outlier:
        warnings.append("Hand controller outlier requires pose/controller inspection.")
    if missing_foot:
        warnings.append("Missing foot anchors block generation-template use; static feet still matter.")
    if missing_knee:
        warnings.append("Missing knee anchors make lower-body pose incomplete.")
    if export_unavailable:
        warnings.append("Export unavailable is not counted as semantic wrong.")
    if low_motion_intro:
        warnings.append("Cowgirl context/intro is not clean motion.")
    if too_short:
        warnings.append("Window is too short for confident semantic motion judgment.")
    if not generation_safe:
        warnings.append("Not safe as generation template without further validation.")
    invalid_reasons = []
    if broken:
        invalid_reasons.append("human_review_pose_broken")
    if foot_outlier:
        invalid_reasons.append("foot_controller_outlier")
    if hand_outlier:
        invalid_reasons.append("hand_controller_outlier")
    if missing_foot:
        invalid_reasons.append("missing_foot_controllers")
    if missing_knee:
        invalid_reasons.append("missing_knee_controllers")
    if anchor_incomplete:
        invalid_reasons.append("pose_anchor_incomplete")
    if controller_status == "invalid":
        invalid_reasons.append("controller_validity_invalid")
    if export_unavailable:
        invalid_reasons.append("export_unavailable")
    if low_motion_intro:
        invalid_reasons.append("low_motion_intro")
    if too_short:
        invalid_reasons.append("too_short")
    export_valid_reason = "valid_for_review_export" if export_pose_validity == "good" else "review_only_or_blocked"
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
        "controller_validity_status": controller_status,
        "controller_validity_score": controller.get("controller_validity_score"),
        "foot_controller_outlier": foot_outlier,
        "hand_controller_outlier": hand_outlier,
        "missing_foot_controllers": missing_foot,
        "missing_knee_controllers": missing_knee,
        "pose_anchor_incomplete": anchor_incomplete,
        "pose_anchor_completeness_score": anchors.get("pose_anchor_completeness_score") or controller.get("pose_anchor_completeness_score"),
        "generation_pose_anchor_safe": anchors.get("generation_pose_anchor_safe") if anchors else controller.get("generation_pose_anchor_safe"),
        "missing_required_anchor_controllers": anchors.get("missing_required_anchor_controllers") or controller.get("missing_required_anchor_controllers", []),
        "controller_outlier_count": controller_outlier_count,
        "generation_pose_invalid_reason": invalid_reasons,
        "export_pose_valid_reason": export_valid_reason,
        "controller_validity": controller,
        "pose_anchor_completeness": anchors,
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
    controller = Counter(r.get("controller_validity_status") for r in rows)
    foot = sum(1 for r in rows if r.get("foot_controller_outlier"))
    lines = [
        "# Pose / Export Validity Report",
        "",
        "This audit separates semantic correctness from export/pose/generation-template usability.",
        "",
        f"- Review items: {len(rows)}",
        f"- Foot/controller outlier items: {foot}",
        "",
        "## Semantic Motion Likely Valid",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in semantic.most_common())
    lines.extend(["", "## Export Pose Validity", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in validity.most_common())
    lines.extend(["", "## Generation Template Safety", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in generation.most_common())
    lines.extend(["", "## Controller Validity", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in controller.most_common()) if controller else lines.append("- None")
    lines.extend(["", "## Semantically Good But Not Generation-Safe", ""])
    examples = [r for r in rows if r.get("semantic_motion_likely_valid") is True and not r.get("generation_template_safe")]
    if examples:
        for row in examples:
            lines.append(f"- `{row.get('review_id')}` window=`{row.get('window_id')}` validity=`{row.get('export_pose_validity')}` warnings={row.get('warnings')}")
    else:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

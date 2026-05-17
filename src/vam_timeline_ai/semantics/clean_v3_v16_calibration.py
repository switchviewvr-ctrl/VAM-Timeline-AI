"""Audit-only clean_v3 calibration after semantic review v16."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import html

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.semantics.clean_v3_calibration_v1 import (
    _anchor_stability,
    _contact_confidence,
    _dedupe,
    _load_context,
    _near_duplicate_group,
    _num,
    _previous_review_windows,
    _update_vam_answer_sheet,
    _write_csv,
)
from vam_timeline_ai.ui.review_ui import build_static_review_ui
from vam_timeline_ai.reports.semantic_qa_dashboard import write_clean_v3_dashboard


def _same_as_review_001() -> dict[str, Any]:
    return {
        "user_verdict": "wrong",
        "semantic_family": "standing_hand_head_gesture",
        "actual_labels": ["standing_hand_head_gesture", "not_cowgirl", "not_cowgirl_pose", "not_generation_candidate"],
        "notes": "Standing, hands/head movement, not Cowgirl.",
    }


def _transition_setup() -> dict[str, Any]:
    return {
        "user_verdict": "partially_correct",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_pose_context", "transition_setup", "intro_alignment", "no_clear_hip_motion", "not_clean_motion"],
        "notes": "Cowgirl pose, but not clean Cowgirl animation; looks like transition/setup.",
    }


def _clean_cowgirl() -> dict[str, Any]:
    return {
        "user_verdict": "correct",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_true_segment", "clean_cowgirl_motion", "pose_valid"],
        "notes": "Cowgirl, pose ok.",
    }


V16_FINDINGS: dict[str, dict[str, Any]] = {
    "review_001": _same_as_review_001(),
    "review_002": _same_as_review_001(),
    "review_003": _clean_cowgirl(),
    "review_004": _transition_setup(),
    "review_005": _transition_setup(),
    "review_006": _transition_setup(),
    "review_007": _clean_cowgirl(),
    "review_008": _same_as_review_001(),
    "review_009": {
        "user_verdict": "correct_low_confidence",
        "semantic_family": "cowgirl",
        "actual_labels": [
            "cowgirl_true_segment",
            "cowgirl_short_motion_window",
            "clean_cowgirl_motion_low_confidence",
            "pose_valid",
        ],
        "notes": "Cowgirl and pose ok; hip movement is very short/hard to judge, but likely correct.",
    },
    "review_010": _clean_cowgirl(),
}

AUDIT_ONLY_LABELS = [
    "transition_setup",
    "no_clear_hip_motion",
    "cowgirl_short_motion_window",
    "clean_cowgirl_motion_low_confidence",
    "correct_low_confidence",
    "not_generation_candidate",
    "standing_hand_head_gesture",
    "not_cowgirl",
    "cowgirl_pose_context",
    "intro_alignment",
    "not_clean_motion",
]

V7_CATEGORIES = [
    "cowgirl_clean_motion_generation_safe",
    "cowgirl_clean_motion_low_confidence_short",
    "cowgirl_pose_context_low_motion",
    "cowgirl_intro_alignment",
    "cowgirl_transition_setup",
    "cowgirl_no_clear_hip_motion",
    "cowgirl_hands_on_partner_chest",
    "cowgirl_hands_on_partner_hips",
    "cowgirl_ambiguous_partner_contact",
    "cowgirl_missing_partner_context",
    "not_cowgirl_standing_hand_head",
    "not_cowgirl_bj_oral",
    "not_cowgirl_receiver_response",
    "unknown_or_unusable",
]


def ingest_v16_human_findings(review_dir: str | Path) -> dict[str, Any]:
    review_root = Path(review_dir)
    review_rows = {r.get("review_id"): r for r in load_jsonl(review_root / "semantic_review_010.jsonl") if r.get("review_id")}
    ui_answers = {r.get("review_id"): r for r in load_jsonl(review_root / "human_review_ui_answers.jsonl") if r.get("review_id")}
    notes = {
        "review_id": "semantic_review_010_v16",
        "audit_only": True,
        "is_human_ground_truth": False,
        "is_training_label": False,
        "do_not_merge_into_manual_labels": True,
        "audit_only_labels": AUDIT_ONLY_LABELS,
        "ui_answers_imported": len(ui_answers),
        "reviews": {},
    }
    for rid, finding in V16_FINDINGS.items():
        row = review_rows.get(rid, {})
        item = dict(finding)
        item["window_id"] = row.get("window_id")
        item["pair_window_id"] = row.get("pair_window_id")
        item["system_semantic_family"] = row.get("semantic_family")
        item["system_phase"] = row.get("phase")
        item["system_category"] = row.get("why_selected")
        if rid in ui_answers:
            item["ui_answer"] = ui_answers[rid]
        item["is_human_ground_truth"] = False
        item["is_training_label"] = False
        notes["reviews"][rid] = item
    yaml_path = review_root / "semantic_review_010_human_notes.yaml"
    yaml_path.write_text(yaml.safe_dump(notes, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_v16_summary(review_root / "semantic_review_010_human_summary.md", notes)
    _update_vam_answer_sheet(review_root / "vam_review_package" / "vam_review_answer_sheet.yaml", notes)
    _update_vam_answer_sheet(review_root / "semantic_review_010_answer_sheet.yaml", notes)
    return {
        "status": "ok",
        "review_items": len(notes["reviews"]),
        "ui_answers_imported": len(ui_answers),
        "notes_path": str(yaml_path),
        "summary_path": str(review_root / "semantic_review_010_human_summary.md"),
        "manual_labels_modified": False,
    }


def rebuild_clean_v3_semantic_actions_v2(run_dir: str | Path, previous_review: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    previous = Path(previous_review) if previous_review else run / "audits" / "semantic_review_010_v16"
    human_by_window = _human_findings_by_window(previous)
    context = _load_context_v2(run)
    source_actions = _load_first(run / "semantic_actions" / "semantic_actions_v1.jsonl", run / "semantic_actions" / "semantic_actions_v0.jsonl")
    rows = [_calibrate_action_v2(row, context, human_by_window.get(row.get("window_id"))) for row in source_actions]
    actions_path = run / "semantic_actions" / "semantic_actions_v2.jsonl"
    write_jsonl(actions_path, rows)
    _write_semantic_actions_v2_report(rows, run / "semantic_actions" / "semantic_actions_v2_report.md")
    semantic_db = _build_semantic_candidate_db_v2(rows)
    write_jsonl(run / "datasets" / "semantic_candidate_db_v2.jsonl", semantic_db)
    _write_semantic_db_v2_csv(semantic_db, run / "datasets" / "semantic_candidate_db_v2.csv")
    _write_semantic_db_v2_report(semantic_db, run / "datasets" / "semantic_candidate_db_v2_report.md")
    cowgirl_db = _build_cowgirl_db_v7(semantic_db)
    write_jsonl(run / "datasets" / "cowgirl_candidate_db_v7.jsonl", cowgirl_db)
    _write_cowgirl_v7_csv(cowgirl_db, run / "datasets" / "cowgirl_candidate_db_v7.csv")
    _write_cowgirl_v7_report(cowgirl_db, run / "datasets" / "cowgirl_candidate_db_v7_report.md")
    return {
        "status": "ok",
        "semantic_actions": len(rows),
        "semantic_action_counts": dict(Counter(r.get("semantic_family") for r in rows)),
        "clean_motion_gate_counts": dict(Counter(r.get("clean_motion_gate") for r in rows)),
        "semantic_db_counts": dict(Counter(r.get("semantic_family") for r in semantic_db)),
        "cowgirl_db_counts": dict(Counter(r.get("category") for r in cowgirl_db)),
        "standing_leakage_count": _standing_leakage_count(cowgirl_db),
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def export_semantic_review_v17(
    run_dir: str | Path,
    out_dir: str | Path,
    count: int = 10,
    build_vam_package: bool = True,
    previous_review: str | Path | None = None,
) -> dict[str, Any]:
    if count != 10:
        raise ValueError("v17 semantic review expects exactly 10 items")
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous_windows = _previous_review_windows(Path(previous_review)) if previous_review else set()
    context = _load_context_v2(run)
    cowgirl = load_jsonl(run / "datasets" / "cowgirl_candidate_db_v7.jsonl")
    semantic = load_jsonl(run / "datasets" / "semantic_candidate_db_v2.jsonl")
    selected, duplicate_summary = _select_v17(cowgirl, semantic, context, previous_windows)
    rows = [_review_row_v17(idx, row, context) for idx, row in enumerate(selected, start=1)]
    write_jsonl(out / "semantic_review_010.jsonl", rows)
    _write_review_md_v17(rows, out / "semantic_review_010.md")
    _write_review_html_v17(rows, out / "semantic_review_010_index.html")
    _write_review_answer_sheet(rows, out / "semantic_review_010_answer_sheet.yaml")
    _write_v17_selection_report(rows, duplicate_summary, out / "semantic_review_010_selection_report.md")
    package_summary = None
    static_summary = None
    if build_vam_package:
        package_summary = build_vam_review_package(
            out / "semantic_review_010.jsonl",
            run,
            run.parent / "clean_v2",
            out / "vam_review_package",
            attempt_timeline_segments=True,
        )
        static_summary = build_static_review_ui(run, out, out / "review_ui_static")
    return {
        "status": "ok",
        "review_items": len(rows),
        "category_counts": dict(Counter(r.get("why_selected") for r in rows)),
        "duplicate_summary": duplicate_summary,
        "vam_package": package_summary,
        "static_review_ui": static_summary,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def run_clean_v3_v16_calibration(run_dir: str | Path, previous_review: str | Path, out_review: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    ingest_summary = ingest_v16_human_findings(previous_review)
    rebuild_summary = rebuild_clean_v3_semantic_actions_v2(run, previous_review)
    review_summary = export_semantic_review_v17(run, out_review, count=10, build_vam_package=True, previous_review=previous_review)
    dashboard_summary = write_clean_v3_dashboard(
        run,
        run / "reports" / "clean_v3_semantic_dashboard.md",
        run / "reports" / "clean_v3_semantic_dashboard.html",
    )
    summary = {
        "status": "ok",
        "v16_findings": ingest_summary,
        "rebuild": rebuild_summary,
        "v17_review": review_summary,
        "dashboard": dashboard_summary,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    _write_pipeline_summary_v2(run / "reports" / "clean_v3_v16_calibration_summary.md", summary)
    dump_json(run / "reports" / "clean_v3_v16_calibration_summary.json", summary)
    return summary


def _load_context_v2(run: Path) -> dict[str, Any]:
    context = _load_context(run)
    context["pose_features"] = {r.get("window_id"): r for r in load_jsonl(run / "pose_semantics" / "pose_features_v0.jsonl") if r.get("window_id")}
    return context


def _calibrate_action_v2(action: dict[str, Any], context: dict[str, Any], human: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(action)
    wid = row.get("window_id")
    window = context["windows"].get(wid, {})
    relative = context["relative"].get(wid, {})
    trajectory = context["trajectory"].get(wid, {})
    pose_features = context["pose_features"].get(wid, {})
    partner = context["partner_by_pair"].get(row.get("pair_window_id")) or context["partner"].get(wid, {})
    anchors = context["pose_anchor"].get(wid, {})
    controller = context["controller"].get(wid, {})
    phase = _clean_motion_gate(row, relative, trajectory, pose_features, human)
    contact = _contact_confidence(row, partner)
    anchor = _anchor_stability(row, relative, anchors, controller, human)
    row.update({
        "source_scene_file": window.get("source_scene_file", row.get("source_scene_file")),
        "source_scene_path": window.get("source_scene_path", row.get("source_scene_path")),
        "technical_atom_id": window.get("technical_atom_id", row.get("technical_atom_id")),
        "sample_id": window.get("sample_id", row.get("sample_id")),
        "source_id": window.get("source_id", row.get("source_id")),
        "start_seconds": window.get("start_seconds", row.get("start_seconds")),
        "end_seconds": window.get("end_seconds", row.get("end_seconds")),
        "duration_seconds": window.get("duration_seconds", row.get("duration_seconds")),
        "motion_content_strength": phase["motion_content_strength"],
        "clean_motion_score": phase["clean_motion_score"],
        "low_motion_hold_score": phase["low_motion_hold_score"],
        "intro_alignment_score": phase["intro_alignment_score"],
        "insertion_setup_score": phase["insertion_setup_score"],
        "phase_confidence": phase["phase_confidence"],
        "hip_motion_strength": phase["hip_motion_strength"],
        "pelvis_trajectory_strength": phase["pelvis_trajectory_strength"],
        "pelvis_cycle_count": phase["pelvis_cycle_count"],
        "motion_duration_confidence": phase["motion_duration_confidence"],
        "clean_motion_gate": phase["clean_motion_gate"],
        "clean_motion_gate_reason": phase["clean_motion_gate_reason"],
        "contact_support_confidence": contact["contact_support_confidence"],
        "contact_support_margin": contact["contact_support_margin"],
        "contact_support_ambiguous": contact["contact_support_ambiguous"],
        "best_contact_target": contact["best_contact_target"],
        "second_best_contact_target": contact["second_best_contact_target"],
        "partner_context_confidence": contact["partner_context_confidence"],
        "foot_anchor_motion_score": anchor["foot_anchor_motion_score"],
        "knee_anchor_motion_score": anchor["knee_anchor_motion_score"],
        "lower_body_anchor_stability": anchor["lower_body_anchor_stability"],
        "anchor_motion_weird": anchor["anchor_motion_weird"],
        "anchor_motion_warning": anchor["anchor_motion_warning"],
    })
    row["phase"] = phase["phase"]
    row["contact_support"] = contact["contact_support"]
    row["warnings"] = _dedupe(list(row.get("warnings") or []) + phase["warnings"] + contact["warnings"] + anchor["warnings"])
    row["conflict_flags"] = _dedupe(list(row.get("conflict_flags") or []) + phase["conflict_flags"] + contact["conflict_flags"] + anchor["conflict_flags"])
    if human:
        _apply_v16_human_override(row, human)
    else:
        _apply_v2_generation_safety(row)
    row["is_human_ground_truth"] = False
    row["is_training_label"] = False
    return row


def _clean_motion_gate(
    action: dict[str, Any],
    relative: dict[str, Any],
    trajectory: dict[str, Any],
    pose_features: dict[str, Any],
    human: dict[str, Any] | None,
) -> dict[str, Any]:
    fv = relative.get("feature_values") or {}
    tv = trajectory.get("feature_values") or {}
    path = _num(fv.get("local_path_length"))
    velocity = _num(fv.get("local_velocity_mean"))
    energy = _num(fv.get("local_motion_energy"))
    grind = max(_num(fv.get("local_grind_score")), _num(tv.get("grind_pattern_score")))
    bounce = max(_num(fv.get("local_bounce_score")), _num(tv.get("bounce_pattern_score")))
    lateral = _num(fv.get("relative_pelvis_lateral_amplitude"))
    vertical = _num(fv.get("relative_pelvis_vertical_amplitude"))
    forward_back = _num(fv.get("relative_pelvis_forward_back_amplitude"))
    transition = _num(tv.get("transition_path_score"))
    cycles = _num(tv.get("cycle_count_estimate"))
    duration = _num(relative.get("duration_seconds") or action.get("duration_seconds"), 4.0)
    low_prev = _num(action.get("low_motion_hold_score"))
    clean_prev = _num(action.get("clean_motion_score"))
    intro_prev = _num(action.get("intro_alignment_score"))
    standing_score = _num(pose_features.get("standing_score"))

    amp_total = lateral + vertical + forward_back
    pelvis_strength = min(1.0, max(path / 0.9, velocity / 0.28, energy / 0.09, amp_total / 0.42))
    hip_strength = min(1.0, max(path / 1.0, (forward_back + vertical + lateral * 0.45) / 0.34, grind * 0.72, bounce * 0.55))
    duration_confidence = min(1.0, max(duration / 4.0, cycles / 2.0 if cycles else 0.0))
    motion_strength = min(1.0, max(_num(action.get("motion_content_strength")), pelvis_strength * 0.75 + hip_strength * 0.25))
    clean = min(1.0, max(clean_prev, 0.42 * hip_strength + 0.28 * pelvis_strength + 0.18 * min(1.0, cycles / 3.0) + 0.12 * max(grind, bounce)))
    low = max(low_prev, 1.0 - min(1.0, max(pelvis_strength, path / 0.6, energy / 0.04)))
    intro = max(intro_prev, min(1.0, transition + max(0.0, 0.66 - clean) * 0.45 + (0.2 if 0.25 <= pelvis_strength <= 0.65 else 0.0)))
    insertion = max(_num(action.get("insertion_setup_score")), intro * 0.75)

    warnings: list[str] = []
    conflicts: list[str] = []
    phase = "unknown"
    gate = "fail_low_motion"
    reason = "Hip/pelvis motion evidence is too weak for clean Cowgirl."

    standing_like = (
        action.get("semantic_family") in {"hand_gesture", "head_gesture"}
        or action.get("pose_family") == "standing"
        or standing_score >= 0.65
    )
    pose_cowgirl = action.get("pose_family") in {"cowgirl", "kneeling_general"} or action.get("semantic_family") == "cowgirl"
    if standing_like and hip_strength < 0.72:
        phase = "standing_gesture"
        gate = "fail_standing"
        reason = "Standing hand/head-style motion is not clean Cowgirl."
        conflicts.extend(["standing_hand_head_gesture", "not_cowgirl"])
    elif pose_cowgirl and (pelvis_strength < 0.34 or (path < 0.30 and energy < 0.015)):
        phase = "low_motion_hold"
        gate = "fail_low_motion"
        reason = "Cowgirl-compatible pose has almost no meaningful hip/pelvis motion."
        conflicts.append("not_clean_motion")
    elif pose_cowgirl and path >= 0.40 and cycles >= 1.0 and clean >= 0.45 and (path < 0.50 or cycles < 1.75 or duration_confidence < 0.85):
        phase = "clean_motion"
        gate = "soft_pass_short"
        reason = "Hip/pelvis motion is present but short or low-confidence."
        warnings.append("Short Cowgirl motion window; keep as low-confidence candidate, not clean generation-safe proof.")
    elif pose_cowgirl and (clean < 0.56 or (cycles < 1.25 and path < 0.55)):
        phase = "transition_setup" if intro >= 0.45 or transition >= 0.12 else "pose_context_only"
        gate = "fail_no_hip_motion"
        reason = "Pose/context evidence is stronger than cyclic hip/pelvis motion; treat as transition/setup."
        conflicts.extend(["no_clear_hip_motion", "not_clean_motion"])
    elif pose_cowgirl and clean >= 0.62 and pelvis_strength >= 0.55 and hip_strength >= 0.50:
        phase = "clean_motion"
        gate = "pass"
        reason = "Compatible pose plus meaningful cyclic hip/pelvis motion."
    elif pose_cowgirl and intro >= 0.56:
        phase = "intro_alignment"
        gate = "fail_no_hip_motion"
        reason = "Looks more like intro/alignment/setup than clean cyclic Cowgirl motion."
        conflicts.append("not_clean_motion")
    else:
        phase = action.get("phase") if action.get("phase") in {"clean_motion", "low_motion_hold", "intro_alignment", "transition_setup"} else "unknown"
        gate = "unknown"
        reason = "Insufficient evidence for calibrated clean-motion gate."

    if human:
        labels = set(human.get("actual_labels") or [])
        if "standing_hand_head_gesture" in labels:
            phase = "standing_gesture"
            gate = "fail_standing"
            reason = "Human v16 review: standing hand/head movement, not Cowgirl."
            conflicts.extend(["standing_hand_head_gesture", "not_cowgirl"])
            clean = min(clean, 0.2)
        elif "transition_setup" in labels or "no_clear_hip_motion" in labels:
            phase = "transition_setup"
            gate = "fail_no_hip_motion"
            reason = "Human v16 review: Cowgirl pose/context, but no clear matching hip motion."
            conflicts.extend(["transition_setup", "no_clear_hip_motion", "not_clean_motion"])
            clean = min(clean, 0.35)
            intro = max(intro, 0.9)
        elif "cowgirl_short_motion_window" in labels:
            phase = "clean_motion"
            gate = "soft_pass_short"
            reason = "Human v16 review: likely Cowgirl, but the motion window is very short/hard to judge."
            warnings.append("cowgirl_short_motion_window")
            clean = max(clean, 0.58)
        elif "clean_cowgirl_motion" in labels:
            phase = "clean_motion"
            gate = "pass"
            reason = "Human v16 review: clean Cowgirl motion and valid pose."
            clean = max(clean, 0.82)
            low = min(low, 0.15)

    if phase in {"low_motion_hold", "pose_context_only"}:
        warnings.append("Cowgirl pose/context without enough hip/pelvis motion is not clean motion.")
    if phase in {"intro_alignment", "transition_setup"}:
        warnings.append("Transition/setup phase must be separated from clean Cowgirl motion.")

    return {
        "phase": phase,
        "motion_content_strength": round(motion_strength, 6),
        "clean_motion_score": round(clean, 6),
        "low_motion_hold_score": round(low, 6),
        "intro_alignment_score": round(intro, 6),
        "insertion_setup_score": round(insertion, 6),
        "phase_confidence": round(max(clean, low, intro, insertion), 6),
        "hip_motion_strength": round(hip_strength, 6),
        "pelvis_trajectory_strength": round(pelvis_strength, 6),
        "pelvis_cycle_count": round(cycles, 6),
        "motion_duration_confidence": round(duration_confidence, 6),
        "clean_motion_gate": gate,
        "clean_motion_gate_reason": reason,
        "warnings": _dedupe(warnings),
        "conflict_flags": _dedupe(conflicts),
    }


def _apply_v16_human_override(row: dict[str, Any], human: dict[str, Any]) -> None:
    labels = set(human.get("actual_labels") or [])
    row["audit_calibration_source"] = "semantic_review_010_v16"
    row["audit_user_verdict"] = human.get("user_verdict")
    row["audit_actual_labels"] = human.get("actual_labels") or []
    row["audit_notes"] = human.get("notes")
    family = human.get("semantic_family")
    if family == "standing_hand_head_gesture":
        row["semantic_family"] = "hand_gesture"
        row["motion_family"] = "hand_gesture"
        row["pose_family"] = "standing"
        row["pose_subtype"] = "standing_upright"
        row["phase"] = "standing_gesture"
        row["clean_motion_gate"] = "fail_standing"
        row["clean_motion_gate_reason"] = "Human v16 review: standing hand/head movement, not Cowgirl."
        row["generation_safe"] = False
        row["conflict_flags"] = _dedupe(list(row.get("conflict_flags") or []) + ["standing_hand_head_gesture", "not_cowgirl"])
        return
    if family:
        row["semantic_family"] = "cowgirl" if family == "cowgirl" else family
        row["motion_family"] = row["semantic_family"] if row["semantic_family"] in {"cowgirl", "bj_oral", "receiver_response", "unknown"} else row.get("motion_family")
    if "cowgirl_pose_context" in labels:
        row["pose_family"] = "cowgirl"
        row.setdefault("pose_subtype", "cowgirl_pose_context")
    if "transition_setup" in labels or "no_clear_hip_motion" in labels:
        row["semantic_family"] = "cowgirl"
        row["motion_family"] = "cowgirl"
        row["phase"] = "transition_setup"
        row["clean_motion_gate"] = "fail_no_hip_motion"
        row["clean_motion_gate_reason"] = "Human v16 review: transition/setup, no matching clean hip motion."
        row["generation_safe"] = False
        row["conflict_flags"] = _dedupe(list(row.get("conflict_flags") or []) + ["transition_setup", "no_clear_hip_motion", "not_clean_motion"])
    if "cowgirl_short_motion_window" in labels:
        row["semantic_family"] = "cowgirl"
        row["motion_family"] = "cowgirl"
        row["pose_family"] = "cowgirl"
        row["phase"] = "clean_motion"
        row["clean_motion_gate"] = "soft_pass_short"
        row["clean_motion_gate_reason"] = "Human v16 review: short but likely valid Cowgirl motion."
        row["generation_safe"] = False
        row["conflict_flags"] = _dedupe([flag for flag in row.get("conflict_flags") or [] if flag not in {"not_clean_motion", "no_clear_hip_motion"}])
    if "clean_cowgirl_motion" in labels:
        row["semantic_family"] = "cowgirl"
        row["motion_family"] = "cowgirl"
        row["pose_family"] = "cowgirl"
        row["phase"] = "clean_motion"
        row["clean_motion_gate"] = "pass"
        row["clean_motion_gate_reason"] = "Human v16 review: clean Cowgirl motion."
        row["conflict_flags"] = _dedupe([flag for flag in row.get("conflict_flags") or [] if flag not in {"not_clean_motion", "not_cowgirl_bj_oral"}])
    _apply_v2_generation_safety(row)


def _apply_v2_generation_safety(row: dict[str, Any]) -> None:
    hard_conflicts = {
        "not_cowgirl_bj_oral",
        "standing_hand_head_gesture",
        "pose_broken",
        "not_cowgirl",
        "not_clean_motion",
        "transition_setup",
        "no_clear_hip_motion",
    }
    conflicts = set(row.get("conflict_flags") or [])
    is_cowgirl = row.get("semantic_family") == "cowgirl"
    pose_ok = row.get("pose_family") in {"cowgirl", "kneeling_general"}
    anchors_ok = _num(row.get("lower_body_anchor_stability"), 1.0) >= 0.3
    gate_ok = row.get("clean_motion_gate") == "pass"
    row["generation_safe"] = bool(is_cowgirl and pose_ok and anchors_ok and gate_ok and not (conflicts & hard_conflicts))
    if row.get("clean_motion_gate") == "soft_pass_short":
        row["generation_safe"] = False
    if row.get("semantic_family") != "cowgirl":
        row["generation_safe"] = False


def _build_semantic_candidate_db_v2(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    allowed = {"cowgirl", "bj_oral", "doggy", "hand_gesture", "head_gesture", "receiver_response", "transition", "unknown"}
    for row in actions:
        family = row.get("semantic_family") if row.get("semantic_family") in allowed else "unknown"
        rows.append({
            "candidate_id": f"semantic_action_v2::{row.get('window_id')}",
            "window_id": row.get("window_id"),
            "pair_window_id": row.get("pair_window_id"),
            "source_scene_file": row.get("source_scene_file"),
            "source_scene_path": row.get("source_scene_path"),
            "technical_actor_id": row.get("technical_atom_id"),
            "sample_id": row.get("sample_id"),
            "semantic_family": family,
            "pose_family": row.get("pose_family"),
            "pose_subtype": row.get("pose_subtype"),
            "motion_subtype": row.get("motion_subtype"),
            "partner_relation": row.get("partner_relation") or ["unknown"],
            "contact_support": row.get("contact_support") or "unknown",
            "phase": row.get("phase") or "unknown",
            "generation_safe": bool(row.get("generation_safe")),
            "safe_for_learning": bool(row.get("generation_safe")) and not row.get("conflict_flags"),
            "invalidity_reason": ";".join(row.get("conflict_flags") or []),
            "semantic_score": row.get("semantic_score"),
            "pose_score": row.get("pose_score"),
            "motion_score": row.get("motion_score"),
            "interaction_score": row.get("interaction_score"),
            "consistency_score": row.get("consistency_score"),
            "motion_content_strength": row.get("motion_content_strength"),
            "clean_motion_score": row.get("clean_motion_score"),
            "low_motion_hold_score": row.get("low_motion_hold_score"),
            "intro_alignment_score": row.get("intro_alignment_score"),
            "insertion_setup_score": row.get("insertion_setup_score"),
            "hip_motion_strength": row.get("hip_motion_strength"),
            "pelvis_trajectory_strength": row.get("pelvis_trajectory_strength"),
            "pelvis_cycle_count": row.get("pelvis_cycle_count"),
            "motion_duration_confidence": row.get("motion_duration_confidence"),
            "clean_motion_gate": row.get("clean_motion_gate"),
            "clean_motion_gate_reason": row.get("clean_motion_gate_reason"),
            "contact_support_confidence": row.get("contact_support_confidence"),
            "contact_support_margin": row.get("contact_support_margin"),
            "contact_support_ambiguous": row.get("contact_support_ambiguous"),
            "best_contact_target": row.get("best_contact_target"),
            "second_best_contact_target": row.get("second_best_contact_target"),
            "partner_context_confidence": row.get("partner_context_confidence"),
            "foot_anchor_motion_score": row.get("foot_anchor_motion_score"),
            "knee_anchor_motion_score": row.get("knee_anchor_motion_score"),
            "lower_body_anchor_stability": row.get("lower_body_anchor_stability"),
            "anchor_motion_weird": row.get("anchor_motion_weird"),
            "warnings": row.get("warnings") or [],
            "preserve_for_future_dataset": family in {"cowgirl", "bj_oral", "hand_gesture", "head_gesture", "receiver_response"},
            "audit_calibration_source": row.get("audit_calibration_source"),
            "is_human_ground_truth": False,
            "is_training_label": False,
        })
    rows.sort(key=lambda r: (r.get("semantic_family") != "cowgirl", _category_rank_v7(_cowgirl_v7_category(r)), -_num(r.get("semantic_score"))))
    return rows


def _build_cowgirl_db_v7(semantic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in semantic_rows:
        category = _cowgirl_v7_category(row)
        if category == "skip":
            continue
        rows.append({
            "candidate_id": f"cowgirl_v7::{row.get('window_id')}",
            "window_id": row.get("window_id"),
            "pair_window_id": row.get("pair_window_id"),
            "source_scene_file": row.get("source_scene_file"),
            "source_scene_path": row.get("source_scene_path"),
            "technical_actor_id": row.get("technical_actor_id"),
            "sample_id": row.get("sample_id"),
            "category": category,
            "semantic_family": row.get("semantic_family"),
            "pose_family": row.get("pose_family"),
            "pose_subtype": row.get("pose_subtype"),
            "motion_subtype": row.get("motion_subtype"),
            "phase": row.get("phase"),
            "partner_relation": row.get("partner_relation") or ["unknown"],
            "contact_support": row.get("contact_support"),
            "interaction_family": "cowgirl" if row.get("semantic_family") == "cowgirl" else row.get("semantic_family"),
            "hand_contact_target": row.get("best_contact_target"),
            "partner_relative_alignment_score": row.get("interaction_score"),
            "hands_on_partner_chest_score": row.get("contact_support_confidence") if row.get("contact_support") == "hands_on_partner_chest" else 0.0,
            "contact_support_confidence": row.get("contact_support_confidence"),
            "contact_support_margin": row.get("contact_support_margin"),
            "contact_support_ambiguous": row.get("contact_support_ambiguous"),
            "generation_requires_partner_targets": row.get("contact_support") in {"hands_on_partner_chest", "hands_on_partner_hips"},
            "generation_safe": bool(row.get("generation_safe")) and category == "cowgirl_clean_motion_generation_safe",
            "invalidity_reason": row.get("invalidity_reason") or "",
            "semantic_score": row.get("semantic_score"),
            "pose_score": row.get("pose_score"),
            "motion_score": row.get("motion_score"),
            "interaction_score": row.get("interaction_score"),
            "motion_content_strength": row.get("motion_content_strength"),
            "clean_motion_score": row.get("clean_motion_score"),
            "low_motion_hold_score": row.get("low_motion_hold_score"),
            "intro_alignment_score": row.get("intro_alignment_score"),
            "insertion_setup_score": row.get("insertion_setup_score"),
            "hip_motion_strength": row.get("hip_motion_strength"),
            "pelvis_trajectory_strength": row.get("pelvis_trajectory_strength"),
            "pelvis_cycle_count": row.get("pelvis_cycle_count"),
            "motion_duration_confidence": row.get("motion_duration_confidence"),
            "clean_motion_gate": row.get("clean_motion_gate"),
            "clean_motion_gate_reason": row.get("clean_motion_gate_reason"),
            "foot_anchor_motion_score": row.get("foot_anchor_motion_score"),
            "knee_anchor_motion_score": row.get("knee_anchor_motion_score"),
            "lower_body_anchor_stability": row.get("lower_body_anchor_stability"),
            "anchor_motion_weird": row.get("anchor_motion_weird"),
            "warnings": row.get("warnings") or [],
            "is_human_ground_truth": False,
            "is_training_label": False,
        })
    rows.sort(key=lambda r: (_category_rank_v7(r.get("category")), -_num(r.get("semantic_score")), -_num(r.get("clean_motion_score"))))
    return rows


def _cowgirl_v7_category(row: dict[str, Any]) -> str:
    family = row.get("semantic_family")
    if family == "bj_oral":
        return "not_cowgirl_bj_oral"
    if family == "receiver_response":
        return "not_cowgirl_receiver_response"
    if family in {"hand_gesture", "head_gesture"} or row.get("pose_family") == "standing":
        return "not_cowgirl_standing_hand_head"
    if family == "unknown":
        return "unknown_or_unusable"
    if family != "cowgirl":
        return "skip"
    gate = row.get("clean_motion_gate")
    phase = row.get("phase")
    if gate == "pass" and row.get("generation_safe"):
        return "cowgirl_clean_motion_generation_safe"
    if gate == "soft_pass_short":
        return "cowgirl_clean_motion_low_confidence_short"
    if gate == "fail_standing":
        return "not_cowgirl_standing_hand_head"
    if phase == "transition_setup":
        return "cowgirl_transition_setup"
    if phase == "intro_alignment":
        return "cowgirl_intro_alignment"
    if gate == "fail_no_hip_motion":
        return "cowgirl_no_clear_hip_motion"
    if phase in {"low_motion_hold", "pose_context_only"} or gate == "fail_low_motion":
        return "cowgirl_pose_context_low_motion"
    if row.get("contact_support") == "ambiguous_partner_contact":
        return "cowgirl_ambiguous_partner_contact"
    if row.get("contact_support") == "hands_on_partner_chest":
        return "cowgirl_hands_on_partner_chest"
    if row.get("contact_support") == "hands_on_partner_hips":
        return "cowgirl_hands_on_partner_hips"
    if row.get("contact_support") in {"unknown", "unknown_contact"}:
        return "cowgirl_missing_partner_context"
    return "unknown_or_unusable"


def _select_v17(
    cowgirl_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    context: dict[str, Any],
    previous_windows: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cowgirl_rows:
        rec = dict(row)
        rec["_source_pool"] = "cowgirl_candidate_db_v7"
        pools[str(rec.get("category"))].append(rec)
    for row in semantic_rows:
        category = (
            "not_cowgirl_bj_oral"
            if row.get("semantic_family") == "bj_oral"
            else "not_cowgirl_standing_hand_head"
            if row.get("semantic_family") in {"hand_gesture", "head_gesture"} or row.get("pose_family") == "standing"
            else "not_cowgirl_receiver_response"
            if row.get("semantic_family") == "receiver_response"
            else "unknown_or_unusable"
            if row.get("semantic_family") == "unknown"
            else None
        )
        if category:
            rec = dict(row)
            rec["category"] = category
            rec["_source_pool"] = "semantic_candidate_db_v2"
            pools[category].append(rec)
    for rows in pools.values():
        rows.sort(key=lambda r: (r.get("window_id") in previous_windows, -_num(r.get("semantic_score")), -_num(r.get("hip_motion_strength")), -_num(r.get("clean_motion_score"))))

    selected: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    sample_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    near_groups: set[str] = set()
    selected_categories: Counter[str] = Counter()
    rejected = Counter()

    def add_from(categories: str | list[str], limit: int) -> None:
        cats = [categories] if isinstance(categories, str) else categories
        added = 0
        for category in cats:
            for row in pools.get(category, []):
                if added >= limit or len(selected) >= 10:
                    return
                if _try_add(row, category):
                    added += 1

    def _try_add(row: dict[str, Any], category: str) -> bool:
        wid = str(row.get("window_id") or "")
        if not wid or wid in seen_windows:
            rejected["same_window"] += 1
            return False
        window = context["windows"].get(wid, {})
        sample = str(row.get("sample_id") or window.get("sample_id") or "")
        scene = str(row.get("source_scene_file") or window.get("source_scene_file") or "unknown")
        if sample and sample_counts[sample] >= 1:
            rejected["sample_cap"] += 1
            return False
        if scene_counts[scene] >= 2:
            rejected["scene_cap"] += 1
            return False
        if category == "not_cowgirl_standing_hand_head" and selected_categories[category] >= 1:
            rejected["standing_cap"] += 1
            return False
        if category == "cowgirl_pose_context_low_motion" and selected_categories[category] >= 1:
            rejected["low_motion_cap"] += 1
            return False
        group = _near_duplicate_group(row, window)
        if group in near_groups:
            rejected["near_duplicate"] += 1
            return False
        selected.append(row)
        seen_windows.add(wid)
        if sample:
            sample_counts[sample] += 1
        scene_counts[scene] += 1
        near_groups.add(group)
        selected_categories[category] += 1
        return True

    add_from("cowgirl_clean_motion_generation_safe", 4)
    add_from("cowgirl_clean_motion_low_confidence_short", 1)
    add_from("cowgirl_pose_context_low_motion", 1)
    add_from(["cowgirl_transition_setup", "cowgirl_intro_alignment", "cowgirl_no_clear_hip_motion"], 1)
    add_from("not_cowgirl_standing_hand_head", 1)
    add_from("not_cowgirl_bj_oral", 1)
    add_from(["unknown_or_unusable", "not_cowgirl_receiver_response"], 1)
    if len(selected) < 10:
        for category in [
            "cowgirl_clean_motion_generation_safe",
            "cowgirl_clean_motion_low_confidence_short",
            "cowgirl_hands_on_partner_chest",
            "cowgirl_hands_on_partner_hips",
            "cowgirl_ambiguous_partner_contact",
            "cowgirl_missing_partner_context",
            "not_cowgirl_receiver_response",
            "unknown_or_unusable",
        ]:
            add_from(category, 10 - len(selected))
            if len(selected) >= 10:
                break
    return selected[:10], {
        "selected": len(selected[:10]),
        "rejected_by_rule": dict(rejected),
        "scene_counts": dict(scene_counts),
        "sample_count": len(sample_counts),
        "selected_categories": dict(selected_categories),
        "previous_review_windows_deprioritized": len(previous_windows),
        "previous_review_windows_selected": sum(1 for row in selected[:10] if row.get("window_id") in previous_windows),
    }


def _review_row_v17(idx: int, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    window = context["windows"].get(row.get("window_id"), {})
    return {
        "review_id": f"review_{idx:03d}",
        "window_id": row.get("window_id"),
        "pair_window_id": row.get("pair_window_id"),
        "semantic_family": row.get("semantic_family"),
        "pose_semantics": {"family": row.get("pose_family"), "subtype": row.get("pose_subtype")},
        "motion_semantics": {"subtype": row.get("motion_subtype"), "phase": row.get("phase")},
        "partner_relation": row.get("partner_relation") or ["unknown"],
        "contact_support": row.get("contact_support"),
        "interaction_family": row.get("interaction_family") or ("cowgirl" if row.get("semantic_family") == "cowgirl" else row.get("semantic_family")),
        "generation_safe": bool(row.get("generation_safe")),
        "why_selected": row.get("category") or row.get("_source_pool"),
        "phase": row.get("phase"),
        "clean_motion_gate": row.get("clean_motion_gate"),
        "clean_motion_gate_reason": row.get("clean_motion_gate_reason"),
        "hip_motion_strength": row.get("hip_motion_strength"),
        "pelvis_trajectory_strength": row.get("pelvis_trajectory_strength"),
        "pelvis_cycle_count": row.get("pelvis_cycle_count"),
        "motion_duration_confidence": row.get("motion_duration_confidence"),
        "motion_content_strength": row.get("motion_content_strength"),
        "clean_motion_score": row.get("clean_motion_score"),
        "low_motion_hold_score": row.get("low_motion_hold_score"),
        "intro_alignment_score": row.get("intro_alignment_score"),
        "insertion_setup_score": row.get("insertion_setup_score"),
        "contact_support_confidence": row.get("contact_support_confidence"),
        "contact_support_margin": row.get("contact_support_margin"),
        "contact_support_ambiguous": row.get("contact_support_ambiguous"),
        "best_contact_target": row.get("best_contact_target") or row.get("hand_contact_target"),
        "second_best_contact_target": row.get("second_best_contact_target"),
        "partner_context_confidence": row.get("partner_context_confidence"),
        "foot_anchor_motion_score": row.get("foot_anchor_motion_score"),
        "knee_anchor_motion_score": row.get("knee_anchor_motion_score"),
        "lower_body_anchor_stability": row.get("lower_body_anchor_stability"),
        "anchor_motion_weird": row.get("anchor_motion_weird"),
        "rider_above_partner_score": None,
        "pelvis_alignment_score": row.get("partner_relative_alignment_score"),
        "hands_on_partner_chest_score": row.get("hands_on_partner_chest_score"),
        "hands_on_partner_hips_score": None,
        "partner_lying_score": None,
        "source_scene_file": row.get("source_scene_file") or window.get("source_scene_file"),
        "source_scene_path": row.get("source_scene_path") or window.get("source_scene_path"),
        "technical_atom_id": row.get("technical_actor_id") or window.get("technical_atom_id"),
        "sample_id": row.get("sample_id") or window.get("sample_id"),
        "start_seconds": window.get("start_seconds") or row.get("start_seconds"),
        "end_seconds": window.get("end_seconds") or row.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds") or row.get("duration_seconds"),
        "warnings": row.get("warnings") or [],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _human_findings_by_window(review_dir: Path) -> dict[str, dict[str, Any]]:
    notes_path = review_dir / "semantic_review_010_human_notes.yaml"
    if not notes_path.exists():
        return {}
    data = yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}
    return {item.get("window_id"): item for item in (data.get("reviews") or {}).values() if item.get("window_id")}


def _load_first(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def _write_v16_summary(path: Path, notes: dict[str, Any]) -> None:
    verdicts = Counter(item.get("user_verdict") for item in notes["reviews"].values())
    families = Counter(item.get("semantic_family") for item in notes["reviews"].values())
    labels = Counter(label for item in notes["reviews"].values() for label in item.get("actual_labels") or [])
    lines = [
        "# semantic_review_010_v16 Human Summary",
        "",
        "These are audit findings for calibration only. They are not manual training labels and must not be merged into `manual_labels.yaml`.",
        "",
        "## Key Outcomes",
        "",
        "- v16 is more useful than v15, but clean-motion gating is still too loose.",
        "- Cowgirl pose/context must be separated from clean Cowgirl hip/pelvis motion.",
        "- Standing hand/head examples leaked into Cowgirl-positive review slots.",
        "- Short but likely valid Cowgirl motion should be preserved as low-confidence, not discarded.",
        "",
        "## Verdict Counts",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in verdicts.most_common())
    lines.extend(["", "## Semantic Family Corrections", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in families.most_common())
    lines.extend(["", "## Audit Label Counts", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in labels.most_common())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_semantic_actions_v2_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Semantic Actions Report V2",
        "",
        "Calibration v2 adds an explicit clean-motion gate for hip/pelvis evidence, standing hand/head exclusion, and short-window Cowgirl preservation.",
        "The rows remain audit/silver candidates, not human training labels.",
        "",
        f"- Rows: {len(rows)}",
        f"- Generation-safe rows: {sum(1 for r in rows if r.get('generation_safe'))}",
        "",
        "## Semantic Families",
        "",
    ]
    lines.extend(_counter_lines(Counter(r.get("semantic_family") for r in rows)))
    lines.extend(["", "## Clean Motion Gate", ""])
    lines.extend(_counter_lines(Counter(r.get("clean_motion_gate") for r in rows)))
    lines.extend(["", "## Phases", ""])
    lines.extend(_counter_lines(Counter(r.get("phase") for r in rows)))
    lines.extend(["", "## Conflict/Warning Flags", ""])
    flags = Counter(flag for r in rows for flag in (r.get("conflict_flags") or []))
    lines.extend(_counter_lines(flags) if flags else ["- None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_semantic_db_v2_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "candidate_id",
        "window_id",
        "semantic_family",
        "pose_family",
        "pose_subtype",
        "motion_subtype",
        "phase",
        "clean_motion_gate",
        "contact_support",
        "generation_safe",
        "hip_motion_strength",
        "pelvis_trajectory_strength",
        "pelvis_cycle_count",
        "motion_duration_confidence",
        "invalidity_reason",
    ]
    _write_csv(rows, path, fields)


def _write_semantic_db_v2_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Semantic Candidate DB V2 Report",
        "",
        "Built from Semantic Actions v2. This is not ML training data.",
        "",
        f"- Records: {len(rows)}",
        "",
        "## Families",
        "",
    ]
    lines.extend(_counter_lines(Counter(r.get("semantic_family") for r in rows)))
    lines.extend(["", "## Clean Motion Gate", ""])
    lines.extend(_counter_lines(Counter(r.get("clean_motion_gate") for r in rows)))
    lines.extend(["", "## Phases", ""])
    lines.extend(_counter_lines(Counter(r.get("phase") for r in rows)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cowgirl_v7_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "candidate_id",
        "window_id",
        "category",
        "semantic_family",
        "pose_family",
        "pose_subtype",
        "motion_subtype",
        "phase",
        "clean_motion_gate",
        "contact_support",
        "generation_safe",
        "semantic_score",
        "clean_motion_score",
        "hip_motion_strength",
        "pelvis_trajectory_strength",
        "pelvis_cycle_count",
        "motion_duration_confidence",
        "clean_motion_gate_reason",
    ]
    _write_csv(rows, path, fields)


def _write_cowgirl_v7_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Cowgirl Candidate DB V7 Report",
        "",
        "V7 strengthens clean Cowgirl motion gating: meaningful hip/pelvis motion is required, standing hand/head is excluded, and short valid motion is preserved separately.",
        "",
        f"- Records: {len(rows)}",
        f"- Standing leakage into clean generation-safe: {_standing_leakage_count(rows)}",
        "",
        "## Categories",
        "",
    ]
    lines.extend(_counter_lines(Counter(r.get("category") for r in rows)))
    lines.extend(["", "## Clean Motion Gate", ""])
    lines.extend(_counter_lines(Counter(r.get("clean_motion_gate") for r in rows)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_md_v17(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Semantic Review 010 v17", "", "Audit review only; not manual training labels.", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['review_id']}",
                "",
                f"- Window: `{row.get('window_id')}`",
                f"- Family: `{row.get('semantic_family')}`",
                f"- Pose: `{(row.get('pose_semantics') or {}).get('family')}` / `{(row.get('pose_semantics') or {}).get('subtype')}`",
                f"- Motion: `{(row.get('motion_semantics') or {}).get('subtype')}`",
                f"- Phase: `{row.get('phase')}`",
                f"- Clean motion gate: `{row.get('clean_motion_gate')}`",
                f"- Gate reason: {row.get('clean_motion_gate_reason')}",
                f"- Hip motion strength: `{row.get('hip_motion_strength')}`",
                f"- Pelvis trajectory strength: `{row.get('pelvis_trajectory_strength')}`",
                f"- Pelvis cycles: `{row.get('pelvis_cycle_count')}`",
                f"- Contact/support: `{row.get('contact_support')}` confidence `{row.get('contact_support_confidence')}`",
                f"- Generation safe: `{row.get('generation_safe')}`",
                f"- Why selected: `{row.get('why_selected')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_html_v17(rows: list[dict[str, Any]], path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
<section>
<h2>{html.escape(row['review_id'])}</h2>
<p><strong>Family:</strong> <code>{html.escape(str(row.get('semantic_family')))}</code>
<strong>Phase:</strong> <code>{html.escape(str(row.get('phase')))}</code>
<strong>Why:</strong> <code>{html.escape(str(row.get('why_selected')))}</code></p>
<p><strong>Clean gate:</strong> <code>{html.escape(str(row.get('clean_motion_gate')))}</code>
<br>{html.escape(str(row.get('clean_motion_gate_reason')))}</p>
<p><strong>Hip/pelvis evidence:</strong>
hip <code>{html.escape(str(row.get('hip_motion_strength')))}</code>,
trajectory <code>{html.escape(str(row.get('pelvis_trajectory_strength')))}</code>,
cycles <code>{html.escape(str(row.get('pelvis_cycle_count')))}</code>,
duration confidence <code>{html.escape(str(row.get('motion_duration_confidence')))}</code></p>
<p><strong>Pose:</strong> <code>{html.escape(str((row.get('pose_semantics') or {}).get('family')))} / {html.escape(str((row.get('pose_semantics') or {}).get('subtype')))}</code>
<strong>Motion:</strong> <code>{html.escape(str((row.get('motion_semantics') or {}).get('subtype')))}</code></p>
<p><strong>Contact:</strong> <code>{html.escape(str(row.get('contact_support')))}</code>
confidence <code>{html.escape(str(row.get('contact_support_confidence')))}</code></p>
</section>
"""
        )
    text = """<!doctype html><meta charset="utf-8"><title>Semantic Review v17</title>
<style>body{font-family:system-ui,Segoe UI,sans-serif;margin:1.5rem;background:#f7f7f5;color:#202020}section{background:white;border:1px solid #ddd;border-radius:6px;padding:1rem;margin:1rem 0}code{background:#f0f0ea;padding:.1rem .25rem;border-radius:4px}</style>
<h1>Semantic Review 010 v17</h1><p>Audit review only. No ML training, no manual label merge.</p>
""" + "\n".join(cards)
    path.write_text(text, encoding="utf-8")


def _write_review_answer_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    data = {
        "reviews": {
            row["review_id"]: {
                "semantic_family_correct": "unknown",
                "pose_correct": "unknown",
                "motion_correct": "unknown",
                "partner_relation_correct": "unknown",
                "contact_support_correct": "unknown",
                "generation_safe_correct": "unknown",
                "clean_motion_gate_correct": "unknown",
                "notes": "",
            }
            for row in rows
        }
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_v17_selection_report(rows: list[dict[str, Any]], duplicate_summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Semantic Review v17 Selection Report",
        "",
        "Selection uses diversity caps: max 1 per sample, max 2 per scene, max 1 standing hand/head item, max 1 low-motion hold, and near-duplicate grouping.",
        "",
        f"- Items: {len(rows)}",
        f"- Categories: `{dict(Counter(r.get('why_selected') for r in rows))}`",
        f"- Duplicate prevention summary: `{duplicate_summary}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pipeline_summary_v2(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# clean_v3 v16 Calibration Summary",
        "",
        "Calibration stores v16 audit findings and rebuilds semantic actions/DBs without training ML or touching manual labels.",
        "",
        f"- v16 findings stored: {summary['v16_findings'].get('review_items')}",
        f"- Semantic actions v2: {summary['rebuild'].get('semantic_actions')}",
        f"- v17 review items: {summary['v17_review'].get('review_items')}",
        f"- Standing leakage count: {summary['rebuild'].get('standing_leakage_count')}",
        "",
        "## Clean Motion Gate Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in (summary["rebuild"].get("clean_motion_gate_counts") or {}).items())
    lines.extend(["", "## Cowgirl DB V7 Counts", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in (summary["rebuild"].get("cowgirl_db_counts") or {}).items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _category_rank_v7(category: str | None) -> int:
    try:
        return V7_CATEGORIES.index(str(category))
    except ValueError:
        return 50


def _counter_lines(counter: Counter[Any], limit: int | None = None) -> list[str]:
    items = counter.most_common(limit)
    return [f"- `{k}`: {v}" for k, v in items] if items else ["- None"]


def _standing_leakage_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("category") == "cowgirl_clean_motion_generation_safe"
        and (row.get("pose_family") == "standing" or row.get("semantic_family") in {"hand_gesture", "head_gesture"})
    )

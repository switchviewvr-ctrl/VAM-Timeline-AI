"""Audit-only clean_v3 calibration after semantic review v15."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import html

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl


V15_FINDINGS: dict[str, dict[str, Any]] = {
    "review_001": {
        "user_verdict": "partially_correct",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_true_segment", "pose_valid", "foot_anchor_motion_weird", "lower_body_anchor_not_stable"],
        "notes": "Cowgirl and pose ok, but feet are animated/weird.",
    },
    "review_002": {
        "user_verdict": "partially_correct",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_pose_context", "intro_alignment", "possible_insertion_setup", "not_clean_motion"],
        "notes": "Pose could be Cowgirl, but motion is intro/alignment/insertion setup, not clean Cowgirl animation.",
    },
    "review_003": {
        "user_verdict": "correct_family",
        "semantic_family": "bj_oral",
        "actual_labels": ["bj_oral_motion_candidate", "lying_or_low_pose_context", "support_near_partner_pelvis", "not_cowgirl"],
        "notes": "Clearly BJ/oral animation, not Cowgirl.",
    },
    "review_004": {
        "user_verdict": "correct",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_true_segment", "clean_cowgirl_motion", "pose_valid"],
        "notes": "Cowgirl, pose ok.",
    },
    "review_005": {
        "user_verdict": "wrong",
        "semantic_family": "standing_hand_head_gesture",
        "actual_labels": ["standing_hand_head_gesture", "not_cowgirl", "not_cowgirl_pose"],
        "notes": "Woman standing and moving hands/head.",
    },
    "review_006": {
        "user_verdict": "wrong_or_unclear",
        "semantic_family": "unknown",
        "actual_labels": ["unknown_unusable", "low_motion", "pose_broken"],
        "notes": "Cannot classify; almost no motion; pose broken.",
    },
    "review_007": {
        "user_verdict": "unclear",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_pose_context", "low_motion_hold", "not_clean_motion", "duplicate_like_low_motion_context"],
        "notes": "Almost no motion; pose probably Cowgirl.",
    },
    "review_008": {
        "user_verdict": "unclear",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_pose_context", "low_motion_hold", "not_clean_motion", "duplicate_like_low_motion_context"],
        "notes": "Almost no motion; pose probably Cowgirl.",
    },
    "review_009": {
        "user_verdict": "unclear",
        "semantic_family": "cowgirl",
        "actual_labels": ["cowgirl_pose_context", "low_motion_hold", "not_clean_motion", "duplicate_like_low_motion_context"],
        "notes": "Almost no motion; pose probably Cowgirl.",
    },
    "review_010": {
        "user_verdict": "unclear",
        "semantic_family": "cowgirl",
        "actual_labels": [
            "cowgirl_pose_context",
            "low_motion_hold",
            "not_clean_motion",
            "duplicate_like_low_motion_context",
            "repeated_duplicate_review_selection",
        ],
        "notes": "Same/similar low-motion Cowgirl pose repeated; should not be selected four times.",
    },
}

AUDIT_ONLY_LABELS = [
    "foot_anchor_motion_weird",
    "lower_body_anchor_not_stable",
    "intro_alignment",
    "possible_insertion_setup",
    "not_clean_motion",
    "cowgirl_pose_context",
    "low_motion_hold",
    "duplicate_like_low_motion_context",
    "repeated_duplicate_review_selection",
    "correct_family",
    "wrong_or_unclear",
]


def ingest_v15_human_findings(review_dir: str | Path) -> dict[str, Any]:
    review_root = Path(review_dir)
    review_rows = {r.get("review_id"): r for r in load_jsonl(review_root / "semantic_review_010.jsonl") if r.get("review_id")}
    notes = {
        "review_id": "semantic_review_010_v15",
        "audit_only": True,
        "is_human_ground_truth": False,
        "is_training_label": False,
        "do_not_merge_into_manual_labels": True,
        "audit_only_labels": AUDIT_ONLY_LABELS,
        "reviews": {},
    }
    for rid, finding in V15_FINDINGS.items():
        row = review_rows.get(rid, {})
        item = dict(finding)
        item["window_id"] = row.get("window_id")
        item["pair_window_id"] = row.get("pair_window_id")
        item["is_human_ground_truth"] = False
        item["is_training_label"] = False
        notes["reviews"][rid] = item
    yaml_path = review_root / "semantic_review_010_human_notes.yaml"
    yaml_path.write_text(yaml.safe_dump(notes, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_v15_summary(review_root / "semantic_review_010_human_summary.md", notes)
    _update_vam_answer_sheet(review_root / "vam_review_package" / "vam_review_answer_sheet.yaml", notes)
    return {
        "status": "ok",
        "review_items": len(notes["reviews"]),
        "notes_path": str(yaml_path),
        "summary_path": str(review_root / "semantic_review_010_human_summary.md"),
        "manual_labels_modified": False,
    }


def rebuild_clean_v3_semantic_actions_v1(run_dir: str | Path, previous_review: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    previous = Path(previous_review) if previous_review else run / "audits" / "semantic_review_010_v15"
    human_by_window = _human_findings_by_window(previous)
    context = _load_context(run)
    actions_v0 = load_jsonl(run / "semantic_actions" / "semantic_actions_v0.jsonl")
    rows = [_calibrate_action(row, context, human_by_window.get(row.get("window_id"))) for row in actions_v0]
    actions_path = run / "semantic_actions" / "semantic_actions_v1.jsonl"
    write_jsonl(actions_path, rows)
    _write_semantic_actions_report(rows, run / "semantic_actions" / "semantic_actions_v1_report.md")
    semantic_db = _build_semantic_candidate_db_v1(rows)
    write_jsonl(run / "datasets" / "semantic_candidate_db_v1.jsonl", semantic_db)
    _write_semantic_db_csv(semantic_db, run / "datasets" / "semantic_candidate_db_v1.csv")
    _write_semantic_db_report(semantic_db, run / "datasets" / "semantic_candidate_db_v1_report.md")
    cowgirl_db = _build_cowgirl_db_v6(semantic_db)
    write_jsonl(run / "datasets" / "cowgirl_candidate_db_v6.jsonl", cowgirl_db)
    _write_cowgirl_v6_csv(cowgirl_db, run / "datasets" / "cowgirl_candidate_db_v6.csv")
    _write_cowgirl_v6_report(cowgirl_db, run / "datasets" / "cowgirl_candidate_db_v6_report.md")
    return {
        "status": "ok",
        "semantic_actions": len(rows),
        "semantic_action_counts": dict(Counter(r.get("semantic_family") for r in rows)),
        "semantic_db_counts": dict(Counter(r.get("semantic_family") for r in semantic_db)),
        "cowgirl_db_counts": dict(Counter(r.get("category") for r in cowgirl_db)),
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def export_semantic_review_v16(
    run_dir: str | Path,
    out_dir: str | Path,
    count: int = 10,
    build_vam_package: bool = True,
    previous_review: str | Path | None = None,
) -> dict[str, Any]:
    if count != 10:
        raise ValueError("v16 semantic review expects exactly 10 items")
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous_windows = _previous_review_windows(Path(previous_review)) if previous_review else set()
    context = _load_context(run)
    cowgirl = load_jsonl(run / "datasets" / "cowgirl_candidate_db_v6.jsonl")
    semantic = load_jsonl(run / "datasets" / "semantic_candidate_db_v1.jsonl")
    selected, duplicate_summary = _select_v16(cowgirl, semantic, context, previous_windows)
    rows = [_review_row(idx, row, context) for idx, row in enumerate(selected, start=1)]
    write_jsonl(out / "semantic_review_010.jsonl", rows)
    _write_review_md(rows, out / "semantic_review_010.md")
    _write_review_html(rows, out / "semantic_review_010_index.html")
    _write_review_answer_sheet(rows, out / "semantic_review_010_answer_sheet.yaml")
    _write_v16_selection_report(rows, duplicate_summary, out / "semantic_review_010_selection_report.md")
    package_summary = None
    if build_vam_package:
        package_summary = build_vam_review_package(
            out / "semantic_review_010.jsonl",
            run,
            run.parent / "clean_v2",
            out / "vam_review_package",
            attempt_timeline_segments=True,
        )
    return {
        "status": "ok",
        "review_items": len(rows),
        "category_counts": dict(Counter(r.get("why_selected") for r in rows)),
        "duplicate_summary": duplicate_summary,
        "vam_package": package_summary,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def run_clean_v3_calibration_v1(run_dir: str | Path, previous_review: str | Path, out_review: str | Path) -> dict[str, Any]:
    ingest_summary = ingest_v15_human_findings(previous_review)
    rebuild_summary = rebuild_clean_v3_semantic_actions_v1(run_dir, previous_review)
    review_summary = export_semantic_review_v16(run_dir, out_review, count=10, build_vam_package=True, previous_review=previous_review)
    summary = {
        "status": "ok",
        "v15_findings": ingest_summary,
        "rebuild": rebuild_summary,
        "v16_review": review_summary,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    _write_pipeline_summary(Path(run_dir) / "reports" / "clean_v3_calibration_v1_summary.md", summary)
    dump_json(Path(run_dir) / "reports" / "clean_v3_calibration_v1_summary.json", summary)
    return summary


def _load_context(run: Path) -> dict[str, Any]:
    return {
        "run_dir": run,
        "windows": {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")},
        "relative": {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "relative_motion_features.jsonl") if r.get("window_id")},
        "trajectory": {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "trajectory_shape_features.jsonl") if r.get("window_id")},
        "pose_anchor": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "pose_anchor_completeness.jsonl") if r.get("window_id")},
        "controller": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "controller_validity.jsonl") if r.get("window_id")},
        "partner": {r.get("window_id"): r for r in load_jsonl(run / "interaction_semantics" / "partner_relative_features_v0.jsonl") if r.get("window_id")},
        "partner_by_pair": {r.get("pair_window_id"): r for r in load_jsonl(run / "interaction_semantics" / "partner_relative_features_v0.jsonl") if r.get("pair_window_id")},
    }


def _calibrate_action(action: dict[str, Any], context: dict[str, Any], human: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(action)
    wid = row.get("window_id")
    window = context["windows"].get(wid, {})
    relative = context["relative"].get(wid, {})
    trajectory = context["trajectory"].get(wid, {})
    partner = context["partner_by_pair"].get(row.get("pair_window_id")) or context["partner"].get(wid, {})
    anchors = context["pose_anchor"].get(wid, {})
    controller = context["controller"].get(wid, {})
    phase = _phase_scores(row, relative, trajectory, anchors, human)
    contact = _contact_confidence(row, partner)
    anchor = _anchor_stability(row, relative, anchors, controller, human)

    row.update({
        "source_scene_file": window.get("source_scene_file"),
        "source_scene_path": window.get("source_scene_path"),
        "technical_atom_id": window.get("technical_atom_id"),
        "sample_id": window.get("sample_id"),
        "source_id": window.get("source_id"),
        "start_seconds": window.get("start_seconds"),
        "end_seconds": window.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds"),
        "motion_content_strength": phase["motion_content_strength"],
        "clean_motion_score": phase["clean_motion_score"],
        "low_motion_hold_score": phase["low_motion_hold_score"],
        "intro_alignment_score": phase["intro_alignment_score"],
        "insertion_setup_score": phase["insertion_setup_score"],
        "phase_confidence": phase["phase_confidence"],
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
        _apply_human_audit_override(row, human)
    else:
        _apply_calibrated_generation_safety(row)
    row["is_human_ground_truth"] = False
    row["is_training_label"] = False
    return row


def _phase_scores(
    action: dict[str, Any],
    relative: dict[str, Any],
    trajectory: dict[str, Any],
    anchors: dict[str, Any],
    human: dict[str, Any] | None,
) -> dict[str, Any]:
    fv = relative.get("feature_values") or {}
    tv = trajectory.get("feature_values") or {}
    path = _num(fv.get("local_path_length"))
    velocity = _num(fv.get("local_velocity_mean"))
    energy = _num(fv.get("local_motion_energy"))
    grind = _num(fv.get("local_grind_score"))
    bounce = _num(fv.get("local_bounce_score"))
    root_removed = _num(fv.get("root_world_motion_removed"), 1.0)
    torso = _num(fv.get("torso_relative_to_pelvis_motion"))
    transition = _num(tv.get("transition_path_score"))
    cycles = _num(tv.get("cycle_count_estimate"))
    motion_strength = min(1.0, max(path / 1.2, velocity / 0.35, energy / 0.12) * (0.35 if root_removed < 0.5 and torso <= 0.001 else 1.0))
    clean = min(1.0, 0.45 * motion_strength + 0.25 * max(grind, bounce) + 0.2 * min(1.0, cycles / 3.0) + 0.1 * max(0.0, 1.0 - transition))
    low = max(0.0, 1.0 - motion_strength)
    if root_removed < 0.5 and torso <= 0.001:
        low = max(low, 0.75)
        clean *= 0.5
    intro = min(1.0, transition + max(0.0, 0.65 - clean) * 0.5 + (0.25 if 0.2 <= motion_strength <= 0.65 else 0.0))
    insertion = min(1.0, intro * 0.8 + (0.2 if action.get("contact_support") in {"hands_on_partner_chest", "hands_on_partner_hips"} else 0.0))
    phase = "clean_motion"
    warnings: list[str] = []
    conflicts: list[str] = []
    if low >= 0.72:
        phase = "low_motion_hold"
        warnings.append("Cowgirl-like pose/context with low or unreliable body-relative motion is not clean motion.")
        conflicts.append("not_clean_motion")
    elif intro >= 0.58 and clean < 0.68:
        phase = "intro_alignment"
        warnings.append("Motion looks like intro/alignment/setup rather than clean cyclic Cowgirl motion.")
        conflicts.append("not_clean_motion")
    elif clean < 0.45 and action.get("pose_family") == "cowgirl":
        phase = "pose_context_only"
        conflicts.append("not_clean_motion")
    if human:
        labels = set(human.get("actual_labels") or [])
        if "intro_alignment" in labels:
            phase = "intro_alignment"
            intro = max(intro, 0.9)
            clean = min(clean, 0.35)
            low = min(low, 0.4)
            conflicts.append("not_clean_motion")
        if "possible_insertion_setup" in labels:
            insertion = max(insertion, 0.9)
        if "low_motion_hold" in labels or "low_motion" in labels:
            phase = "low_motion_hold"
            low = max(low, 0.9)
            clean = min(clean, 0.25)
            conflicts.append("not_clean_motion")
        if "clean_cowgirl_motion" in labels:
            phase = "clean_motion"
            clean = max(clean, 0.85)
            low = min(low, 0.2)
    return {
        "phase": phase,
        "motion_content_strength": round(motion_strength, 6),
        "clean_motion_score": round(clean, 6),
        "low_motion_hold_score": round(low, 6),
        "intro_alignment_score": round(intro, 6),
        "insertion_setup_score": round(insertion, 6),
        "phase_confidence": round(max(clean, low, intro, insertion), 6),
        "warnings": warnings,
        "conflict_flags": _dedupe(conflicts),
    }


def _contact_confidence(action: dict[str, Any], partner: dict[str, Any]) -> dict[str, Any]:
    chest = _num(partner.get("hands_on_partner_chest_score"))
    hips = _num(partner.get("hands_on_partner_hips_score"))
    floor = _num(partner.get("hands_on_floor_or_bed_proxy"))
    context = _num(partner.get("partner_context_confidence"))
    scores = [("partner_chest", chest), ("partner_hips", hips), ("floor_or_bed", floor)]
    scores.sort(key=lambda item: item[1], reverse=True)
    best_name, best = scores[0]
    second_name, second = scores[1]
    margin = best - second
    support = str(action.get("contact_support") or "unknown")
    warnings: list[str] = []
    conflicts: list[str] = []
    ambiguous = False
    if not partner:
        support = "unknown_contact"
        confidence = 0.0
        warnings.append("Pair/contact evidence is missing; contact support remains unknown.")
    elif context < 0.45:
        support = "unknown_contact"
        confidence = context
        warnings.append("Partner context confidence is too weak for a contact/support claim.")
    elif best < 0.35:
        support = "unknown_contact"
        confidence = max(best, context * 0.3)
    elif best < 0.55:
        support = "possible_partner_contact"
        confidence = best * context
        warnings.append("Partner contact is possible but not strong enough for a specific support target.")
    elif margin < 0.18:
        support = "ambiguous_partner_contact"
        ambiguous = True
        confidence = best * context
        conflicts.append("ambiguous_partner_contact")
    elif best_name == "partner_chest":
        support = "hands_on_partner_chest"
        confidence = best * context
    elif best_name == "partner_hips":
        support = "hands_on_partner_hips"
        confidence = best * context
    else:
        support = "hands_on_floor_or_bed"
        confidence = best * context
    return {
        "contact_support": support,
        "contact_support_confidence": round(min(1.0, confidence), 6),
        "contact_support_margin": round(margin, 6),
        "contact_support_ambiguous": ambiguous,
        "best_contact_target": best_name,
        "second_best_contact_target": second_name,
        "partner_context_confidence": round(context, 6),
        "warnings": warnings,
        "conflict_flags": conflicts,
    }


def _anchor_stability(
    action: dict[str, Any],
    relative: dict[str, Any],
    anchors: dict[str, Any],
    controller: dict[str, Any],
    human: dict[str, Any] | None,
) -> dict[str, Any]:
    fv = relative.get("feature_values") or {}
    limb = _num(fv.get("limb_motion_relative_energy"))
    missing = anchors.get("missing_required_anchor_controllers") or controller.get("missing_required_anchor_controllers") or []
    foot_score = min(1.0, limb / 0.22)
    knee_score = min(1.0, limb / 0.28)
    stability = max(0.0, 1.0 - max(foot_score, knee_score) * 0.65)
    weird = bool(foot_score >= 0.9 and action.get("semantic_family") == "cowgirl")
    warnings: list[str] = []
    conflicts: list[str] = []
    if missing:
        stability *= 0.65
        warnings.append("Required lower-body anchor controllers are missing or incomplete.")
    if human and "foot_anchor_motion_weird" in set(human.get("actual_labels") or []):
        weird = True
        foot_score = max(foot_score, 0.95)
        stability = min(stability, 0.35)
    if weird:
        warnings.append("Foot/lower-body anchor motion is weird for a stable Cowgirl anchor context.")
        conflicts.append("foot_anchor_motion_weird")
    return {
        "foot_anchor_motion_score": round(foot_score, 6),
        "knee_anchor_motion_score": round(knee_score, 6),
        "lower_body_anchor_stability": round(stability, 6),
        "anchor_motion_weird": weird,
        "anchor_motion_warning": "foot_anchor_motion_weird" if weird else "",
        "warnings": warnings,
        "conflict_flags": conflicts,
    }


def _apply_human_audit_override(row: dict[str, Any], human: dict[str, Any]) -> None:
    labels = set(human.get("actual_labels") or [])
    row["audit_calibration_source"] = "semantic_review_010_v15"
    row["audit_user_verdict"] = human.get("user_verdict")
    row["audit_actual_labels"] = human.get("actual_labels") or []
    row["audit_notes"] = human.get("notes")
    family = human.get("semantic_family")
    if family == "standing_hand_head_gesture":
        row["semantic_family"] = "hand_gesture"
        row["motion_family"] = "hand_gesture"
        row["pose_family"] = "standing"
        row["phase"] = "gesture"
        row["generation_safe"] = False
        row["conflict_flags"] = _dedupe(list(row.get("conflict_flags") or []) + ["standing_hand_head_gesture", "not_cowgirl"])
        return
    if family:
        row["semantic_family"] = family
        row["motion_family"] = family if family in {"cowgirl", "bj_oral", "receiver_response", "unknown"} else row.get("motion_family")
    if "cowgirl_pose_context" in labels and row.get("pose_family") == "unknown":
        row["pose_family"] = "cowgirl"
        row["pose_subtype"] = "cowgirl_pose_context"
    if "unknown_unusable" in labels:
        row["semantic_family"] = "unknown"
        row["motion_family"] = "unknown"
        row["phase"] = "low_motion_hold" if "low_motion" in labels else "unknown"
    if "bj_oral_motion_candidate" in labels:
        row["semantic_family"] = "bj_oral"
        row["motion_family"] = "bj_oral"
        row["generation_safe"] = False
        row["conflict_flags"] = _dedupe(list(row.get("conflict_flags") or []) + ["not_cowgirl_bj_oral"])
    if "clean_cowgirl_motion" in labels:
        row["semantic_family"] = "cowgirl"
        row["motion_family"] = "cowgirl"
        row["phase"] = "clean_motion"
    if "intro_alignment" in labels:
        row["phase"] = "intro_alignment"
    if "low_motion_hold" in labels:
        row["phase"] = "low_motion_hold"
    if "pose_broken" in labels:
        row["conflict_flags"] = _dedupe(list(row.get("conflict_flags") or []) + ["pose_broken"])
    _apply_calibrated_generation_safety(row)


def _apply_calibrated_generation_safety(row: dict[str, Any]) -> None:
    clean = row.get("phase") == "clean_motion" and _num(row.get("clean_motion_score")) >= 0.45
    is_cowgirl = row.get("semantic_family") == "cowgirl"
    pose_ok = row.get("pose_family") in {"cowgirl", "kneeling_general"}
    no_hard_conflict = not any(
        flag in set(row.get("conflict_flags") or [])
        for flag in ["not_cowgirl_bj_oral", "standing_hand_head_gesture", "pose_broken", "not_clean_motion"]
    )
    anchors_ok = _num(row.get("lower_body_anchor_stability"), 1.0) >= 0.3
    row["generation_safe"] = bool(is_cowgirl and clean and pose_ok and no_hard_conflict and anchors_ok)
    if row.get("phase") != "clean_motion":
        row["generation_safe"] = False
    if row.get("semantic_family") != "cowgirl":
        row["generation_safe"] = False


def _build_semantic_candidate_db_v1(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in actions:
        family = row.get("semantic_family") if row.get("semantic_family") in {"cowgirl", "bj_oral", "doggy", "hand_gesture", "head_gesture", "receiver_response", "transition", "unknown"} else "unknown"
        rows.append({
            "candidate_id": f"semantic_action_v1::{row.get('window_id')}",
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
    rows.sort(key=lambda r: (r.get("semantic_family") != "cowgirl", r.get("phase") != "clean_motion", -_num(r.get("semantic_score"))))
    return rows


def _build_cowgirl_db_v6(semantic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in semantic_rows:
        category = _cowgirl_v6_category(row)
        if category == "skip":
            continue
        rows.append({
            "candidate_id": f"cowgirl_v6::{row.get('window_id')}",
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
            "foot_anchor_motion_score": row.get("foot_anchor_motion_score"),
            "knee_anchor_motion_score": row.get("knee_anchor_motion_score"),
            "lower_body_anchor_stability": row.get("lower_body_anchor_stability"),
            "anchor_motion_weird": row.get("anchor_motion_weird"),
            "warnings": row.get("warnings") or [],
            "is_human_ground_truth": False,
            "is_training_label": False,
        })
    rows.sort(key=lambda r: (_category_rank(r.get("category")), -_num(r.get("semantic_score"))))
    return rows


def _cowgirl_v6_category(row: dict[str, Any]) -> str:
    family = row.get("semantic_family")
    if family == "bj_oral":
        return "not_cowgirl_bj_oral"
    if family == "receiver_response":
        return "not_cowgirl_receiver_response"
    if family in {"hand_gesture", "head_gesture"} or row.get("pose_family") == "standing":
        return "not_cowgirl_standing_gesture"
    if family == "unknown":
        return "unknown_or_unusable"
    if family != "cowgirl":
        return "skip"
    if row.get("anchor_motion_weird"):
        return "cowgirl_anchor_motion_warning"
    if row.get("phase") == "low_motion_hold" or row.get("phase") == "pose_context_only":
        return "cowgirl_pose_context_low_motion"
    if row.get("phase") == "intro_alignment":
        return "cowgirl_possible_insertion_setup" if _num(row.get("insertion_setup_score")) >= 0.65 else "cowgirl_intro_alignment"
    if row.get("contact_support") == "ambiguous_partner_contact":
        return "cowgirl_ambiguous_partner_contact"
    if row.get("contact_support") == "hands_on_partner_chest":
        return "cowgirl_hands_on_partner_chest"
    if row.get("contact_support") == "hands_on_partner_hips":
        return "cowgirl_hands_on_partner_hips"
    if row.get("contact_support") == "hands_on_floor_or_bed":
        return "cowgirl_hands_on_floor_or_bed"
    if row.get("contact_support") in {"unknown", "unknown_contact"}:
        return "cowgirl_missing_partner_context"
    if row.get("generation_safe"):
        return "cowgirl_clean_motion_generation_safe"
    return "cowgirl_generation_unsafe"


def _select_v16(
    cowgirl_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    context: dict[str, Any],
    previous_windows: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_window = {r.get("window_id"): r for r in semantic_rows}
    for row in cowgirl_rows:
        row = dict(row)
        row["_source_pool"] = "cowgirl_db_v6"
        pools[str(row.get("category"))].append(row)
    for row in semantic_rows:
        category = (
            "bj_oral_motion"
            if row.get("semantic_family") == "bj_oral"
            else "standing_hand_head_gesture"
            if row.get("semantic_family") in {"hand_gesture", "head_gesture"}
            else "receiver_response"
            if row.get("semantic_family") == "receiver_response"
            else "unknown_or_unusable"
            if row.get("semantic_family") == "unknown"
            else "skip"
        )
        if category in {"bj_oral_motion", "standing_hand_head_gesture", "receiver_response", "unknown_or_unusable"}:
            rec = dict(row)
            rec["category"] = category
            rec["_source_pool"] = "semantic_candidate_db_v1"
            pools[category].append(rec)
    for rows in pools.values():
        rows.sort(key=lambda r: (r.get("window_id") in previous_windows, -_num(r.get("semantic_score")), -_num(r.get("clean_motion_score"))))

    quotas = [
        ("cowgirl_clean_motion_generation_safe", 3),
        ("cowgirl_pose_context_low_motion", 1),
        ("cowgirl_intro_alignment", 1),
        ("cowgirl_possible_insertion_setup", 1),
        ("cowgirl_hands_on_partner_chest", 1),
        ("cowgirl_ambiguous_partner_contact", 1),
        ("bj_oral_motion", 1),
        ("standing_hand_head_gesture", 1),
        ("receiver_response", 1),
        ("unknown_or_unusable", 1),
    ]
    selected: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    sample_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    low_motion_count = 0
    duplicate_groups: set[str] = set()
    rejected = Counter()

    def add_from(category: str, limit: int) -> None:
        nonlocal low_motion_count
        added = 0
        for row in pools.get(category, []):
            if added >= limit or len(selected) >= 10:
                return
            wid = row.get("window_id")
            if not wid or wid in seen_windows:
                rejected["same_window"] += 1
                continue
            window = context["windows"].get(wid, {})
            sample = str(row.get("sample_id") or window.get("sample_id") or "")
            scene = str(row.get("source_scene_file") or window.get("source_scene_file") or "unknown")
            if sample and sample_counts[sample] >= 1:
                rejected["sample_cap"] += 1
                continue
            if scene_counts[scene] >= 2:
                rejected["scene_cap"] += 1
                continue
            phase = str(row.get("phase") or "")
            if category == "cowgirl_pose_context_low_motion" or phase == "low_motion_hold":
                if low_motion_count >= 1:
                    rejected["low_motion_cap"] += 1
                    continue
            group = _near_duplicate_group(row, window)
            if group in duplicate_groups:
                rejected["near_duplicate"] += 1
                continue
            selected.append(row)
            seen_windows.add(str(wid))
            if sample:
                sample_counts[sample] += 1
            scene_counts[scene] += 1
            duplicate_groups.add(group)
            if category == "cowgirl_pose_context_low_motion" or phase == "low_motion_hold":
                low_motion_count += 1
            added += 1

    add_from("cowgirl_clean_motion_generation_safe", 3)
    add_from("cowgirl_pose_context_low_motion", 1)
    add_from("cowgirl_intro_alignment", 1)
    if not any(r.get("category") in {"cowgirl_intro_alignment", "cowgirl_possible_insertion_setup"} for r in selected):
        add_from("cowgirl_possible_insertion_setup", 1)
    if not any(r.get("category") in {"cowgirl_hands_on_partner_chest", "cowgirl_ambiguous_partner_contact"} for r in selected):
        add_from("cowgirl_hands_on_partner_chest", 1)
        if not any(r.get("category") == "cowgirl_hands_on_partner_chest" for r in selected):
            add_from("cowgirl_ambiguous_partner_contact", 1)
    add_from("bj_oral_motion", 1)
    add_from("standing_hand_head_gesture", 1)
    add_from("receiver_response", 1)
    add_from("unknown_or_unusable", 1)

    if len(selected) < 10:
        for category in [
            "cowgirl_hands_on_partner_hips",
            "cowgirl_anchor_motion_warning",
            "cowgirl_missing_partner_context",
            "not_cowgirl_bj_oral",
            "not_cowgirl_standing_gesture",
            "not_cowgirl_receiver_response",
        ]:
            add_from(category, 10 - len(selected))
            if len(selected) >= 10:
                break
    summary = {
        "selected": len(selected),
        "rejected_by_rule": dict(rejected),
        "scene_counts": dict(scene_counts),
        "sample_count": len(sample_counts),
        "low_motion_items": low_motion_count,
        "previous_review_windows_excluded_preference": len(previous_windows),
    }
    return selected[:10], summary


def _review_row(idx: int, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
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
        "technical_atom_id": row.get("technical_actor_id") or window.get("technical_atom_id"),
        "sample_id": row.get("sample_id") or window.get("sample_id"),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _near_duplicate_group(row: dict[str, Any], window: dict[str, Any]) -> str:
    start = round(_num(window.get("start_seconds") or row.get("start_seconds")) / 2.0) * 2
    return "|".join(
        [
            str(row.get("source_scene_file") or window.get("source_scene_file") or ""),
            str(row.get("technical_actor_id") or window.get("technical_atom_id") or ""),
            str(row.get("sample_id") or window.get("sample_id") or ""),
            str(start),
            str(row.get("pose_subtype") or ""),
            str(row.get("motion_subtype") or ""),
            str(row.get("phase") or ""),
        ]
    )


def _human_findings_by_window(review_dir: Path) -> dict[str, dict[str, Any]]:
    notes_path = review_dir / "semantic_review_010_human_notes.yaml"
    if not notes_path.exists():
        return {}
    data = yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}
    return {item.get("window_id"): item for item in (data.get("reviews") or {}).values() if item.get("window_id")}


def _previous_review_windows(review_dir: Path) -> set[str]:
    return {str(r.get("window_id")) for r in load_jsonl(review_dir / "semantic_review_010.jsonl") if r.get("window_id")}


def _write_v15_summary(path: Path, notes: dict[str, Any]) -> None:
    verdicts = Counter(item.get("user_verdict") for item in notes["reviews"].values())
    families = Counter(item.get("semantic_family") for item in notes["reviews"].values())
    lines = [
        "# semantic_review_010_v15 Human Summary",
        "",
        "These are audit findings for calibration only. They are not manual training labels and must not be merged into `manual_labels.yaml`.",
        "",
        "## Key Outcomes",
        "",
        "- clean_v3 architecture is still correct.",
        "- v15 over-selected low-motion Cowgirl pose/context windows.",
        "- Intro/alignment/setup must be separated from clean Cowgirl motion.",
        "- Contact/support confidence needs stronger evidence and ambiguity handling.",
        "- Foot/lower-body anchor weirdness must be tracked separately from semantic family.",
        "",
        "## Verdict Counts",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in verdicts.most_common())
    lines.extend(["", "## Semantic Family Corrections", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in families.most_common())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_vam_answer_sheet(path: Path, notes: dict[str, Any]) -> None:
    if not path.exists():
        return
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reviews = data.setdefault("reviews", {})
    for rid, finding in notes["reviews"].items():
        item = reviews.setdefault(rid, {})
        item["user_verdict"] = finding.get("user_verdict")
        item["semantic_family"] = finding.get("semantic_family")
        item["actual_labels"] = finding.get("actual_labels")
        item["notes"] = finding.get("notes")
        item["is_human_ground_truth"] = False
        item["is_training_label"] = False
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_semantic_actions_report(rows: list[dict[str, Any]], path: Path) -> None:
    families = Counter(r.get("semantic_family") for r in rows)
    phases = Counter(r.get("phase") for r in rows)
    contacts = Counter(r.get("contact_support") for r in rows)
    conflicts = Counter(flag for r in rows for flag in (r.get("conflict_flags") or []))
    lines = [
        "# Semantic Actions Report V1",
        "",
        "Calibration v1 separates clean motion from intro/alignment, low-motion holds, contact ambiguity, and lower-body anchor warnings.",
        "The rows remain audit/silver candidates, not human training labels.",
        "",
        f"- Rows: {len(rows)}",
        f"- Generation-safe rows: {sum(1 for r in rows if r.get('generation_safe'))}",
        "",
        "## Semantic Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common())
    lines.extend(["", "## Phases", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in phases.most_common())
    lines.extend(["", "## Contact/Support", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in contacts.most_common(12))
    lines.extend(["", "## Conflict/Warning Flags", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in conflicts.most_common(12)) if conflicts else lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_semantic_db_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "candidate_id",
        "window_id",
        "semantic_family",
        "pose_family",
        "pose_subtype",
        "motion_subtype",
        "phase",
        "contact_support",
        "generation_safe",
        "motion_content_strength",
        "contact_support_confidence",
        "lower_body_anchor_stability",
        "invalidity_reason",
    ]
    _write_csv(rows, path, fields)


def _write_semantic_db_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Semantic Candidate DB V1 Report",
        "",
        "Built from calibrated Semantic Actions v1. This is not ML training data.",
        "",
        f"- Records: {len(rows)}",
        "",
        "## Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in Counter(r.get("semantic_family") for r in rows).most_common())
    lines.extend(["", "## Phases", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in Counter(r.get("phase") for r in rows).most_common())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cowgirl_v6_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "candidate_id",
        "window_id",
        "category",
        "semantic_family",
        "pose_family",
        "pose_subtype",
        "motion_subtype",
        "phase",
        "contact_support",
        "generation_safe",
        "semantic_score",
        "clean_motion_score",
        "contact_support_confidence",
        "lower_body_anchor_stability",
        "anchor_motion_weird",
    ]
    _write_csv(rows, path, fields)


def _write_cowgirl_v6_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Cowgirl Candidate DB V6 Report",
        "",
        "V6 separates clean Cowgirl motion from pose-context holds, intro/setup, BJ/oral, standing gestures, receiver response, contact ambiguity, and anchor warnings.",
        "",
        f"- Records: {len(rows)}",
        "",
        "## Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in Counter(r.get("category") for r in rows).most_common())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_review_md(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Semantic Review 010 v16", "", "Audit review only; not manual training labels.", ""]
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
                f"- Contact/support: `{row.get('contact_support')}` confidence `{row.get('contact_support_confidence')}`",
                f"- Generation safe: `{row.get('generation_safe')}`",
                f"- Why selected: `{row.get('why_selected')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_html(rows: list[dict[str, Any]], path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
<section>
<h2>{html.escape(row['review_id'])}</h2>
<p><strong>Family:</strong> <code>{html.escape(str(row.get('semantic_family')))}</code>
<strong>Phase:</strong> <code>{html.escape(str(row.get('phase')))}</code>
<strong>Why:</strong> <code>{html.escape(str(row.get('why_selected')))}</code></p>
<p><strong>Pose:</strong> <code>{html.escape(str((row.get('pose_semantics') or {}).get('family')))} / {html.escape(str((row.get('pose_semantics') or {}).get('subtype')))}</code>
<strong>Motion:</strong> <code>{html.escape(str((row.get('motion_semantics') or {}).get('subtype')))}</code></p>
<p><strong>Contact:</strong> <code>{html.escape(str(row.get('contact_support')))}</code>
confidence <code>{html.escape(str(row.get('contact_support_confidence')))}</code>,
margin <code>{html.escape(str(row.get('contact_support_margin')))}</code></p>
<p><strong>Motion scores:</strong>
clean <code>{html.escape(str(row.get('clean_motion_score')))}</code>,
low-hold <code>{html.escape(str(row.get('low_motion_hold_score')))}</code>,
intro <code>{html.escape(str(row.get('intro_alignment_score')))}</code></p>
<p><strong>Anchor:</strong> stability <code>{html.escape(str(row.get('lower_body_anchor_stability')))}</code>,
weird <code>{html.escape(str(row.get('anchor_motion_weird')))}</code></p>
</section>
"""
        )
    text = """<!doctype html><meta charset="utf-8"><title>Semantic Review v16</title>
<style>body{font-family:system-ui,Segoe UI,sans-serif;margin:1.5rem;background:#f7f7f5;color:#202020}section{background:white;border:1px solid #ddd;border-radius:6px;padding:1rem;margin:1rem 0}code{background:#f0f0ea;padding:.1rem .25rem;border-radius:4px}</style>
<h1>Semantic Review 010 v16</h1><p>Audit review only. No ML training, no manual label merge.</p>
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
                "notes": "",
            }
            for row in rows
        }
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_v16_selection_report(rows: list[dict[str, Any]], duplicate_summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Semantic Review v16 Selection Report",
        "",
        "Selection uses diversity caps: max 1 per sample, max 2 per scene, max 1 low-motion hold, and near-duplicate grouping.",
        "",
        f"- Items: {len(rows)}",
        f"- Categories: `{dict(Counter(r.get('why_selected') for r in rows))}`",
        f"- Duplicate prevention summary: `{duplicate_summary}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pipeline_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# clean_v3 Calibration v1 Summary",
        "",
        "Calibration stores v15 audit findings and rebuilds semantic actions/DBs without training ML or touching manual labels.",
        "",
        f"- v15 findings stored: {summary['v15_findings'].get('review_items')}",
        f"- Semantic actions: {summary['rebuild'].get('semantic_actions')}",
        f"- v16 review items: {summary['v16_review'].get('review_items')}",
        "",
        "## Cowgirl DB V6 Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in (summary["rebuild"].get("cowgirl_db_counts") or {}).items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _category_rank(category: str | None) -> int:
    order = {
        "cowgirl_clean_motion_generation_safe": 0,
        "cowgirl_hands_on_partner_chest": 1,
        "cowgirl_hands_on_partner_hips": 2,
        "cowgirl_ambiguous_partner_contact": 3,
        "cowgirl_anchor_motion_warning": 4,
        "cowgirl_intro_alignment": 5,
        "cowgirl_possible_insertion_setup": 6,
        "cowgirl_pose_context_low_motion": 7,
        "not_cowgirl_bj_oral": 8,
        "not_cowgirl_standing_gesture": 9,
        "not_cowgirl_receiver_response": 10,
        "unknown_or_unusable": 11,
    }
    return order.get(str(category), 50)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value != value:
            return default
        return float(value)
    except Exception:
        return default


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(x) for x in value)
    if isinstance(value, dict):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return value


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(str(item))
    return out

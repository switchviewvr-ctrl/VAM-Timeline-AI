"""Curated Cowgirl candidate inventory.

This database is for audit/review triage.  It is not an ML training dataset and
does not promote audit labels into manual ground truth.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def build_cowgirl_candidate_db_v1(
    run_dir: str | Path,
    candidate_scores: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    pose_anchor_completeness: str | Path,
    controller_validity: str | Path,
    controller_orientation_validity: str | Path,
    controller_distance_validity: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    scores = {r.get("window_id"): r for r in load_jsonl(candidate_scores) if r.get("window_id")}
    rel = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    traj = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    orientation = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")}
    distance = {r.get("window_id"): r for r in load_jsonl(controller_distance_validity) if r.get("window_id")}
    rows = [
        _candidate_record(
            scores[wid],
            windows.get(wid, {}),
            rel.get(wid, {}),
            traj.get(wid, {}),
            body.get(wid, {}),
            anchors.get(wid, {}),
            controller.get(wid, {}),
            orientation.get(wid, {}),
            distance.get(wid, {}),
        )
        for wid in scores
    ]
    rows.sort(key=lambda r: (r.get("category") != "semantic_cowgirl_generation_safe", -float(r.get("generation_candidate_score") or 0.0), -float(r.get("semantic_cowgirl_score") or 0.0)))
    write_jsonl(out_jsonl, rows)
    _write_csv(rows, out_csv)
    _write_report(rows, report)
    return rows


def build_cowgirl_candidate_db_v2(
    run_dir: str | Path,
    candidate_scores: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    pose_anchor_completeness: str | Path,
    controller_validity: str | Path,
    controller_orientation_validity: str | Path,
    controller_distance_validity: str | Path,
    cowgirl_core_controllers: str | Path,
    bj_oral_trap_guard: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    scores = {r.get("window_id"): r for r in load_jsonl(candidate_scores) if r.get("window_id")}
    rel = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    traj = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    orientation = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")}
    distance = {r.get("window_id"): r for r in load_jsonl(controller_distance_validity) if r.get("window_id")}
    core = {r.get("window_id"): r for r in load_jsonl(cowgirl_core_controllers) if r.get("window_id")}
    traps = {r.get("window_id"): r for r in load_jsonl(bj_oral_trap_guard) if r.get("window_id")}
    rows = [
        _candidate_record_v2(
            scores[wid],
            windows.get(wid, {}),
            rel.get(wid, {}),
            traj.get(wid, {}),
            body.get(wid, {}),
            anchors.get(wid, {}),
            controller.get(wid, {}),
            orientation.get(wid, {}),
            distance.get(wid, {}),
            core.get(wid, {}),
            traps.get(wid, {}),
        )
        for wid in scores
    ]
    rows.sort(key=lambda r: (r.get("category") != "semantic_cowgirl_generation_safe", -float(r.get("generation_candidate_score") or 0.0), -float(r.get("semantic_cowgirl_score") or 0.0)))
    write_jsonl(out_jsonl, rows)
    _write_csv(rows, out_csv)
    _write_report_v2(rows, run / "datasets" / "cowgirl_candidate_db_v1.jsonl", report)
    return rows


def build_cowgirl_candidate_db_v3(
    run_dir: str | Path,
    candidate_scores: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    pose_anchor_completeness: str | Path,
    controller_validity: str | Path,
    controller_orientation_validity: str | Path,
    controller_distance_validity: str | Path,
    cowgirl_core_controllers: str | Path,
    bj_oral_domain: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    scores = {r.get("window_id"): r for r in load_jsonl(candidate_scores) if r.get("window_id")}
    rel = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    traj = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    orientation = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")}
    distance = {r.get("window_id"): r for r in load_jsonl(controller_distance_validity) if r.get("window_id")}
    core = {r.get("window_id"): r for r in load_jsonl(cowgirl_core_controllers) if r.get("window_id")}
    bj = {r.get("window_id"): r for r in load_jsonl(bj_oral_domain) if r.get("window_id")}
    rows = [
        _candidate_record_v3(
            scores[wid],
            windows.get(wid, {}),
            rel.get(wid, {}),
            traj.get(wid, {}),
            body.get(wid, {}),
            anchors.get(wid, {}),
            controller.get(wid, {}),
            orientation.get(wid, {}),
            distance.get(wid, {}),
            core.get(wid, {}),
            bj.get(wid, {}),
        )
        for wid in scores
    ]
    rows.sort(key=lambda r: (r.get("category") not in {"semantic_cowgirl_generation_safe", "semantic_cowgirl_core_soft_fail_generation_safe"}, -float(r.get("generation_candidate_score") or 0.0), -float(r.get("semantic_cowgirl_score") or 0.0)))
    write_jsonl(out_jsonl, rows)
    _write_csv(rows, out_csv)
    _write_report_v3(rows, Path(run_dir) / "datasets" / "cowgirl_candidate_db_v2.jsonl", report)
    return rows


def _candidate_record(
    score: dict[str, Any],
    window: dict[str, Any],
    relative: dict[str, Any],
    trajectory: dict[str, Any],
    body: dict[str, Any],
    anchors: dict[str, Any],
    controller: dict[str, Any],
    orientation: dict[str, Any],
    distance: dict[str, Any],
) -> dict[str, Any]:
    category = _db_category(score)
    subtype = _cowgirl_subtype(score, trajectory)
    invalidity = _invalidity_reasons(score, anchors, controller, orientation, distance)
    warnings = []
    for source in (score, body, anchors, controller, orientation, distance):
        warnings.extend(str(x) for x in (source.get("warnings") or [])[:3])
    return {
        "window_id": score.get("window_id"),
        "sample_id": score.get("sample_id") or window.get("sample_id"),
        "source_id": score.get("source_id") or window.get("source_id"),
        "source_scene_file": score.get("source_scene_file") or window.get("source_scene_file"),
        "technical_atom_id": score.get("technical_atom_id") or window.get("technical_atom_id"),
        "start_seconds": window.get("start_seconds") or score.get("start_seconds"),
        "end_seconds": window.get("end_seconds") or score.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds") or score.get("duration_seconds"),
        "category": category,
        "semantic_cowgirl_score": score.get("final_semantic_cowgirl_score_v9"),
        "clean_motion_score": score.get("final_clean_motion_score_v9"),
        "generation_candidate_score": score.get("final_generation_candidate_score_v9"),
        "trajectory_shape": trajectory.get("trajectory_shape_classification") or score.get("trajectory_shape_classification"),
        "cowgirl_subtype": subtype,
        "body_motion_quality": body.get("body_motion_quality") or score.get("body_motion_quality"),
        "rider_receiver_status": score.get("role_status") or score.get("rider_receiver_status"),
        "pose_anchor_status": anchors.get("generation_pose_anchor_status"),
        "controller_validity_status": controller.get("controller_validity_status") or score.get("controller_validity_status"),
        "orientation_validity_status": orientation.get("orientation_validity_status") or score.get("orientation_validity_status"),
        "distance_validity_status": distance.get("controller_distance_validity_status") or score.get("distance_validity_status"),
        "pose_anchor_completeness_score": anchors.get("pose_anchor_completeness_score") or score.get("pose_anchor_completeness_score"),
        "controller_validity_score": controller.get("controller_validity_score") or score.get("controller_validity_score"),
        "orientation_validity_score": orientation.get("orientation_validity_score") or score.get("orientation_validity_score"),
        "distance_validity_score": distance.get("controller_distance_validity_score") or score.get("distance_validity_score"),
        "safe_for_learning": bool((relative.get("feature_values") or {}).get("safe_for_learning") or score.get("safe_for_learning")),
        "generation_safe": category == "semantic_cowgirl_generation_safe",
        "export_review_safe": category not in {"unknown_or_unusable", "export_unavailable_or_unsafe"},
        "invalidity_reasons": invalidity,
        "warnings": _dedupe(warnings + [score.get("warning", "")]),
        "nearest_handmade_reference_ids": score.get("relative_nearest_handmade_references", []),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _candidate_record_v2(
    score: dict[str, Any],
    window: dict[str, Any],
    relative: dict[str, Any],
    trajectory: dict[str, Any],
    body: dict[str, Any],
    anchors: dict[str, Any],
    controller: dict[str, Any],
    orientation: dict[str, Any],
    distance: dict[str, Any],
    core: dict[str, Any],
    trap: dict[str, Any],
) -> dict[str, Any]:
    category = _db_category_v2(score)
    subtype = _cowgirl_subtype(score, trajectory)
    invalidity = _invalidity_reasons_v2(score, anchors, controller, orientation, distance, core, trap)
    warnings = []
    for source in (score, body, anchors, controller, orientation, distance, core, trap):
        warnings.extend(str(x) for x in (source.get("warnings") or [])[:3])
    return {
        "window_id": score.get("window_id"),
        "sample_id": score.get("sample_id") or window.get("sample_id"),
        "source_id": score.get("source_id") or window.get("source_id"),
        "source_scene_file": score.get("source_scene_file") or window.get("source_scene_file"),
        "technical_atom_id": score.get("technical_atom_id") or window.get("technical_atom_id"),
        "start_seconds": window.get("start_seconds") or score.get("start_seconds"),
        "end_seconds": window.get("end_seconds") or score.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds") or score.get("duration_seconds"),
        "category": category,
        "semantic_cowgirl_score": score.get("final_semantic_cowgirl_score_v10"),
        "clean_motion_score": score.get("final_clean_motion_score_v10"),
        "generation_candidate_score": score.get("final_generation_candidate_score_v10"),
        "trajectory_shape": trajectory.get("trajectory_shape_classification") or score.get("trajectory_shape_classification"),
        "cowgirl_subtype": subtype,
        "body_motion_quality": body.get("body_motion_quality") or score.get("body_motion_quality"),
        "rider_receiver_status": score.get("role_status") or score.get("rider_receiver_status"),
        "pose_anchor_status": anchors.get("generation_pose_anchor_status"),
        "controller_validity_status": controller.get("controller_validity_status") or score.get("controller_validity_status"),
        "orientation_validity_status": orientation.get("orientation_validity_status") or score.get("orientation_validity_status"),
        "distance_validity_status": distance.get("controller_distance_validity_status") or score.get("distance_validity_status"),
        "pose_anchor_completeness_score": anchors.get("pose_anchor_completeness_score") or score.get("pose_anchor_completeness_score"),
        "controller_validity_score": controller.get("controller_validity_score") or score.get("controller_validity_score"),
        "orientation_validity_score": orientation.get("orientation_validity_score") or score.get("orientation_validity_score"),
        "distance_validity_score": distance.get("controller_distance_validity_score") or score.get("distance_validity_score"),
        "core_controller_gate": score.get("core_controller_gate") if "core_controller_gate" in score else core.get("generation_safe_core_controller_gate"),
        "cowgirl_core_controller_status": score.get("cowgirl_core_controller_status") or core.get("cowgirl_core_controller_status"),
        "missing_core_controllers": score.get("missing_core_controllers") or core.get("missing_core_controllers", []),
        "bj_oral_trap_flag": bool(score.get("bj_oral_trap_flag") or trap.get("head_or_oral_domain_trap")),
        "arm_stretch_outlier_flag": bool(score.get("arm_stretch_outlier_flag") or distance.get("arm_stretch_outlier")),
        "safe_for_learning": bool((relative.get("feature_values") or {}).get("safe_for_learning") or score.get("safe_for_learning")),
        "generation_safe": category == "semantic_cowgirl_generation_safe",
        "export_review_safe": category not in {"unknown_or_unusable", "export_unavailable_or_unsafe"},
        "invalidity_reason": ";".join(invalidity),
        "invalidity_reasons": invalidity,
        "warnings": _dedupe(warnings + [score.get("warning", "")]),
        "nearest_handmade_reference_ids": score.get("relative_nearest_handmade_references", []),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _candidate_record_v3(
    score: dict[str, Any],
    window: dict[str, Any],
    relative: dict[str, Any],
    trajectory: dict[str, Any],
    body: dict[str, Any],
    anchors: dict[str, Any],
    controller: dict[str, Any],
    orientation: dict[str, Any],
    distance: dict[str, Any],
    core: dict[str, Any],
    bj: dict[str, Any],
) -> dict[str, Any]:
    category = _db_category_v3(score)
    invalidity = _invalidity_reasons_v2(score, anchors, controller, orientation, distance, core, bj)
    warnings = []
    for source in (score, body, anchors, controller, orientation, distance, core, bj):
        warnings.extend(str(x) for x in (source.get("warnings") or [])[:3])
    return {
        "window_id": score.get("window_id"),
        "sample_id": score.get("sample_id") or window.get("sample_id"),
        "source_id": score.get("source_id") or window.get("source_id"),
        "source_scene_file": score.get("source_scene_file") or window.get("source_scene_file"),
        "technical_atom_id": score.get("technical_atom_id") or window.get("technical_atom_id"),
        "start_seconds": window.get("start_seconds") or score.get("start_seconds"),
        "end_seconds": window.get("end_seconds") or score.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds") or score.get("duration_seconds"),
        "category": category,
        "semantic_family": score.get("semantic_family") or ("bj_oral" if score.get("not_cowgirl_bj_oral") else "cowgirl" if category.startswith("semantic_cowgirl") or category.startswith("cowgirl") else "unknown"),
        "excluded_from_cowgirl": bool(score.get("excluded_from_cowgirl")),
        "preserve_for_future_dataset": bool(score.get("preserve_for_future_dataset")),
        "semantic_cowgirl_score": score.get("final_semantic_cowgirl_score_v11"),
        "clean_motion_score": score.get("final_clean_motion_score_v11"),
        "generation_candidate_score": score.get("final_generation_candidate_score_v11"),
        "trajectory_shape": trajectory.get("trajectory_shape_classification") or score.get("trajectory_shape_classification"),
        "cowgirl_subtype": score.get("cowgirl_subtype") or _cowgirl_subtype(score, trajectory),
        "body_motion_quality": body.get("body_motion_quality") or score.get("body_motion_quality"),
        "rider_receiver_status": score.get("role_status") or score.get("rider_receiver_status"),
        "pose_anchor_status": anchors.get("generation_pose_anchor_status"),
        "controller_validity_status": controller.get("controller_validity_status") or score.get("controller_validity_status"),
        "orientation_validity_status": orientation.get("orientation_validity_status") or score.get("orientation_validity_status"),
        "distance_validity_status": distance.get("controller_distance_validity_status") or score.get("distance_validity_status"),
        "core_controller_gate": score.get("core_controller_gate") if "core_controller_gate" in score else core.get("generation_safe_core_controller_gate"),
        "core_gate_status": score.get("core_gate_status") or core.get("core_gate_status"),
        "core_gate_can_be_overridden": bool(score.get("core_gate_can_be_overridden") or core.get("core_gate_can_be_overridden")),
        "core_gate_override_reason": score.get("core_gate_override_reason") or core.get("core_gate_override_reason"),
        "missing_core_controllers": score.get("missing_core_controllers") or core.get("missing_core_controllers", []),
        "bj_oral_confidence": score.get("bj_oral_confidence") or bj.get("bj_oral_confidence"),
        "bj_oral_motion_candidate": bool(score.get("bj_oral_motion_candidate") or bj.get("bj_oral_motion_candidate")),
        "arm_stretch_outlier_flag": bool(score.get("arm_stretch_outlier_flag") or distance.get("arm_stretch_outlier")),
        "generation_safe": category in {"semantic_cowgirl_generation_safe", "semantic_cowgirl_core_soft_fail_generation_safe"},
        "export_review_safe": category not in {"unknown_or_unusable", "export_unavailable_or_unsafe"},
        "invalidity_reason": ";".join(invalidity),
        "invalidity_reasons": invalidity,
        "warnings": _dedupe(warnings + [score.get("warning", "")]),
        "nearest_handmade_reference_ids": score.get("relative_nearest_handmade_references", []),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _db_category(score: dict[str, Any]) -> str:
    if score.get("export_unavailable_for_generation"):
        return "export_unavailable_or_unsafe"
    raw = str(score.get("cowgirl_v9_category") or "")
    if raw == "standing_gesture_false_positive":
        return "standing_hand_head_negative"
    if raw in {
        "semantic_cowgirl_generation_safe",
        "semantic_cowgirl_pose_invalid",
        "semantic_cowgirl_anchor_incomplete",
        "semantic_cowgirl_orientation_invalid",
        "semantic_cowgirl_distance_invalid",
        "cowgirl_context_intro_low_motion",
        "receiver_response_negative",
        "unknown_or_unusable",
    }:
        return raw
    if score.get("semantic_cowgirl_distance_invalid"):
        return "semantic_cowgirl_distance_invalid"
    if score.get("semantic_cowgirl_pose_invalid"):
        return "semantic_cowgirl_pose_invalid"
    if score.get("semantic_cowgirl_generation_safe"):
        return "semantic_cowgirl_generation_safe"
    return "unknown_or_unusable"


def _db_category_v2(score: dict[str, Any]) -> str:
    if score.get("export_unavailable_for_generation"):
        return "export_unavailable_or_unsafe"
    raw = str(score.get("cowgirl_v10_category") or "")
    if raw in {
        "semantic_cowgirl_generation_safe",
        "semantic_cowgirl_core_controller_missing",
        "semantic_cowgirl_pose_invalid",
        "semantic_cowgirl_anchor_incomplete",
        "semantic_cowgirl_orientation_invalid",
        "semantic_cowgirl_distance_invalid",
        "cowgirl_context_intro_low_motion",
        "bj_oral_trap_negative",
        "standing_hand_head_negative",
        "receiver_response_negative",
        "unknown_or_unusable",
    }:
        return raw
    if raw == "standing_gesture_false_positive":
        return "standing_hand_head_negative"
    if score.get("semantic_cowgirl_core_controller_missing"):
        return "semantic_cowgirl_core_controller_missing"
    if score.get("bj_oral_trap_negative") or score.get("bj_oral_trap_flag"):
        return "bj_oral_trap_negative"
    if score.get("semantic_cowgirl_generation_safe"):
        return "semantic_cowgirl_generation_safe"
    return "unknown_or_unusable"


def _db_category_v3(score: dict[str, Any]) -> str:
    if score.get("export_unavailable_for_generation"):
        return "export_unavailable_or_unsafe"
    raw = str(score.get("cowgirl_v11_category") or "")
    allowed = {
        "semantic_cowgirl_generation_safe",
        "semantic_cowgirl_core_soft_fail_generation_safe",
        "semantic_cowgirl_core_hard_fail",
        "semantic_cowgirl_anchor_incomplete",
        "semantic_cowgirl_orientation_invalid",
        "semantic_cowgirl_distance_invalid",
        "semantic_cowgirl_pose_invalid",
        "cowgirl_context_intro_low_motion",
        "not_cowgirl_bj_oral",
        "standing_hand_head_negative",
        "receiver_response_negative",
        "unknown_or_unusable",
    }
    if raw in allowed:
        return raw
    return "unknown_or_unusable"


def _cowgirl_subtype(score: dict[str, Any], trajectory: dict[str, Any]) -> str:
    shape = str(trajectory.get("trajectory_shape_classification") or score.get("trajectory_shape_classification") or "")
    if score.get("likely_cowgirl_grinding") or "oval" in shape:
        return "oval_grind"
    if "circular" in shape:
        return "circular_grind"
    if score.get("likely_cowgirl_vertical_bounce") or "bounce" in shape:
        return "vertical_bounce"
    if score.get("likely_cowgirl_forward_back_rock"):
        return "forward_back_rock"
    return "unknown"


def _invalidity_reasons(score: dict[str, Any], anchors: dict[str, Any], controller: dict[str, Any], orientation: dict[str, Any], distance: dict[str, Any]) -> list[str]:
    reasons = []
    if score.get("missing_foot_controllers") or anchors.get("missing_foot_controllers"):
        reasons.append("missing_foot_controllers")
    if score.get("missing_knee_controllers") or anchors.get("missing_knee_controllers"):
        reasons.append("missing_knee_controllers")
    if score.get("controller_rotation_invalid") or orientation.get("controller_rotation_invalid"):
        reasons.append("controller_rotation_invalid")
    if score.get("controller_twist_invalid") or orientation.get("controller_twist_invalid"):
        reasons.append("controller_twist_invalid")
    if score.get("controller_distance_outlier") or distance.get("controller_distance_outlier"):
        reasons.append("controller_distance_outlier")
    if controller.get("foot_controller_outlier"):
        reasons.append("foot_controller_outlier")
    if score.get("standing_gesture_false_positive"):
        reasons.append("standing_hand_head_motion")
    if score.get("receiver_response_negative"):
        reasons.append("receiver_response_negative")
    if score.get("export_unavailable_for_generation"):
        reasons.append("export_unavailable")
    return _dedupe(reasons)


def _invalidity_reasons_v2(
    score: dict[str, Any],
    anchors: dict[str, Any],
    controller: dict[str, Any],
    orientation: dict[str, Any],
    distance: dict[str, Any],
    core: dict[str, Any],
    trap: dict[str, Any],
) -> list[str]:
    reasons = _invalidity_reasons(score, anchors, controller, orientation, distance)
    if score.get("missing_core_pelvis_motion_controllers") or core.get("cowgirl_core_controller_status") == "missing_core":
        reasons.append("missing_core_pelvis_motion_controllers")
    if score.get("missing_hip_thigh_pelvis_controllers"):
        reasons.append("missing_hip_thigh_pelvis_controllers")
    if score.get("arm_stretch_outlier_flag") or distance.get("arm_stretch_outlier"):
        reasons.append("arm_stretch_pose_invalid")
    if score.get("hand_controller_outlier") or distance.get("hand_controller_outlier"):
        reasons.append("hand_controller_outlier")
    if score.get("bj_oral_trap_flag") or score.get("not_cowgirl_bj_oral") or trap.get("bj_oral_motion_candidate") or trap.get("head_or_oral_domain_trap"):
        reasons.append("not_cowgirl_bj_oral")
    if trap.get("cowgirl_pose_but_bj_oral_motion") or trap.get("cowgirl_pose_false_positive"):
        reasons.append("cowgirl_pose_but_bj_oral_motion")
    return _dedupe(reasons)


def _write_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    target = Path(out_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_id",
        "sample_id",
        "source_scene_file",
        "technical_atom_id",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "category",
        "semantic_cowgirl_score",
        "clean_motion_score",
        "generation_candidate_score",
        "trajectory_shape",
        "cowgirl_subtype",
        "body_motion_quality",
        "rider_receiver_status",
        "pose_anchor_status",
        "controller_validity_status",
        "orientation_validity_status",
        "distance_validity_status",
        "core_controller_gate",
        "semantic_family",
        "excluded_from_cowgirl",
        "preserve_for_future_dataset",
        "core_gate_status",
        "core_gate_can_be_overridden",
        "cowgirl_core_controller_status",
        "bj_oral_trap_flag",
        "bj_oral_confidence",
        "arm_stretch_outlier_flag",
        "generation_safe",
        "invalidity_reasons",
    ]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            slim = {key: row.get(key) for key in fieldnames}
            slim["invalidity_reasons"] = ";".join(row.get("invalidity_reasons", []))
            writer.writerow(slim)


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    categories = Counter(r.get("category") for r in rows)
    subtypes = Counter(r.get("cowgirl_subtype") for r in rows)
    scenes = Counter(r.get("source_scene_file") for r in rows if r.get("category") == "semantic_cowgirl_generation_safe")
    reasons = Counter(reason for row in rows for reason in row.get("invalidity_reasons", []))
    generation = [r for r in rows if r.get("category") == "semantic_cowgirl_generation_safe"]
    lines = [
        "# Cowgirl Candidate Database V1 Report",
        "",
        "This is a curated candidate inventory for review. It is not ML training data and contains no human ground truth labels.",
        "",
        f"- Records: {len(rows)}",
        f"- Generation-safe Cowgirl candidates: {len(generation)}",
        "",
        "## Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in categories.most_common()) if categories else lines.append("- None")
    lines.extend(["", "## Subtypes", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in subtypes.most_common()) if subtypes else lines.append("- None")
    lines.extend(["", "## Invalidity Reasons", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in reasons.most_common()) if reasons else lines.append("- None")
    lines.extend(["", "## Top Generation-Safe Candidates", ""])
    for row in sorted(generation, key=lambda r: float(r.get("generation_candidate_score") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` generation={row.get('generation_candidate_score')} "
            f"semantic={row.get('semantic_cowgirl_score')} subtype={row.get('cowgirl_subtype')} scene=`{row.get('source_scene_file')}`"
        )
    if not generation:
        lines.append("- None")
    lines.extend(["", "## Generation-Safe Scene Distribution", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in scenes.most_common(25)) if scenes else lines.append("- None")
    lines.extend(["", "## Recommended Next Review Batch", "", "Use semantic_review_010_v11 to sample generation-safe, invalid, negative, and unknown examples from this database."])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v2(rows: list[dict[str, Any]], v1_path: Path, report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    categories = Counter(r.get("category") for r in rows)
    v1_categories = Counter(r.get("category") for r in load_jsonl(v1_path)) if v1_path.exists() else Counter()
    reasons = Counter(reason for row in rows for reason in row.get("invalidity_reasons", []))
    generation = [r for r in rows if r.get("category") == "semantic_cowgirl_generation_safe"]
    removed_core = sum(1 for r in rows if r.get("category") == "semantic_cowgirl_core_controller_missing")
    removed_trap = sum(1 for r in rows if r.get("category") == "bj_oral_trap_negative")
    removed_arm = sum(1 for r in rows if r.get("arm_stretch_outlier_flag"))
    lines = [
        "# Cowgirl Candidate Database V2 Report",
        "",
        "DB v2 is a curated candidate inventory for review. It is not ML training data and contains no human ground truth labels.",
        "",
        f"- Records: {len(rows)}",
        f"- Generation-safe Cowgirl candidates: {len(generation)}",
        f"- Removed from generation-safe by core controller gate: {removed_core}",
        f"- Removed by BJ/oral trap guard: {removed_trap}",
        f"- Hand/arm stretch outliers: {removed_arm}",
        "",
        "## V2 Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in categories.most_common()) if categories else lines.append("- None")
    lines.extend(["", "## V1 Categories For Comparison", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in v1_categories.most_common()) if v1_categories else lines.append("- V1 DB not found")
    lines.extend(["", "## Invalidity Reasons", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in reasons.most_common()) if reasons else lines.append("- None")
    lines.extend(["", "## Top Generation-Safe Candidates", ""])
    for row in sorted(generation, key=lambda r: float(r.get("generation_candidate_score") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` generation={row.get('generation_candidate_score')} "
            f"semantic={row.get('semantic_cowgirl_score')} subtype={row.get('cowgirl_subtype')} scene=`{row.get('source_scene_file')}`"
        )
    if not generation:
        lines.append("- None")
    lines.extend(["", "## Top Trap/Negative Candidates", ""])
    negatives = [r for r in rows if r.get("category") in {"bj_oral_trap_negative", "standing_hand_head_negative", "semantic_cowgirl_core_controller_missing"}]
    for row in sorted(negatives, key=lambda r: float(r.get("semantic_cowgirl_score") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` category={row.get('category')} "
            f"semantic={row.get('semantic_cowgirl_score')} reasons={row.get('invalidity_reasons')}"
        )
    if not negatives:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v3(rows: list[dict[str, Any]], v2_path: Path, report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    categories = Counter(r.get("category") for r in rows)
    families = Counter(r.get("semantic_family") for r in rows)
    subtypes = Counter(r.get("cowgirl_subtype") for r in rows if r.get("generation_safe"))
    v2_categories = Counter(r.get("category") for r in load_jsonl(v2_path)) if v2_path.exists() else Counter()
    bj = [r for r in rows if r.get("semantic_family") == "bj_oral"]
    soft = [r for r in rows if r.get("category") == "semantic_cowgirl_core_soft_fail_generation_safe"]
    lines = [
        "# Cowgirl Candidate Database V3 Report",
        "",
        "DB v3 adds semantic-family fields. BJ/oral candidates are excluded from Cowgirl but preserved for future BJ/oral datasets.",
        "",
        f"- Records: {len(rows)}",
        f"- Generation-safe Cowgirl candidates: {sum(1 for r in rows if r.get('generation_safe'))}",
        f"- Recovered core soft-fail candidates: {len(soft)}",
        f"- BJ/oral candidates excluded from Cowgirl: {len(bj)}",
        f"- BJ/oral candidates preserved: {sum(1 for r in bj if r.get('preserve_for_future_dataset'))}",
        "",
        "## V3 Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in categories.most_common()) if categories else lines.append("- None")
    lines.extend(["", "## V2 Categories For Comparison", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in v2_categories.most_common()) if v2_categories else lines.append("- V2 DB not found")
    lines.extend(["", "## Semantic Families", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common()) if families else lines.append("- None")
    lines.extend(["", "## Generation-Safe Cowgirl Subtypes", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in subtypes.most_common()) if subtypes else lines.append("- None")
    lines.extend(["", "## Recommended Larger Review Batch", "", "See `larger_review_batch_plan.md` for the prepared 65-item review proposal."])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out

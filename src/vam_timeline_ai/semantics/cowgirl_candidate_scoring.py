"""Clean Cowgirl candidate scoring for audit review selection.

Scores here are review triage, not labels and not training targets.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.domain_guards import evaluate_domain_guards
from vam_timeline_ai.semantics.motion_phase_classifier import classify_motion_phase


def score_cowgirl_candidates_v2(
    run_dir: str | Path,
    wild_reference_matches: str | Path,
    body_quality: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    matches = {r.get("window_id"): r for r in load_jsonl(wild_reference_matches) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window(feature_rows[wid], body.get(wid, {}), matches.get(wid, {}), windows.get(wid, {}))
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_clean_cowgirl_candidate_score") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def score_cowgirl_candidates_v3(
    run_dir: str | Path,
    wild_reference_matches: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    matches = {r.get("window_id"): r for r in load_jsonl(wild_reference_matches) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v3(feature_rows[wid], body.get(wid, {}), matches.get(wid, {}), windows.get(wid, {}), rider_receiver.get(wid, {}))
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_clean_cowgirl_rider_score_v3") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v3(rows, report)
    return rows


def score_cowgirl_candidates_v4(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    """Score clean Cowgirl candidates using relative motion and trajectory shape.

    This is still review triage.  It is intentionally not a classifier and not a
    training target.
    """
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v4(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_clean_cowgirl_score_v4") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v4(rows, report)
    return rows


def score_cowgirl_candidates_v5(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    pose_export_validity: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    """Score semantic Cowgirl separately from generation/export usability."""
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    pose_validity = {r.get("window_id"): r for r in load_jsonl(pose_export_validity) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v5(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
            pose_validity.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_semantic_cowgirl_score_v5") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v5(rows, report)
    return rows


def score_cowgirl_candidates_v6(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    pose_export_validity: str | Path,
    controller_validity: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    """Score Cowgirl semantics separately from controller/generation safety."""
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    pose_validity = {r.get("window_id"): r for r in load_jsonl(pose_export_validity) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v6(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
            pose_validity.get(wid, {}),
            controller.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_semantic_cowgirl_score_v6") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v6(rows, report)
    return rows


def score_cowgirl_candidates_v7(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    pose_export_validity: str | Path,
    controller_validity: str | Path,
    pose_anchor_completeness: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    pose_validity = {r.get("window_id"): r for r in load_jsonl(pose_export_validity) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v7(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
            pose_validity.get(wid, {}),
            controller.get(wid, {}),
            anchors.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_semantic_cowgirl_score_v7") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v7(rows, report)
    return rows


def score_cowgirl_candidates_v8(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    pose_export_validity: str | Path,
    controller_validity: str | Path,
    pose_anchor_completeness: str | Path,
    controller_orientation_validity: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    pose_validity = {r.get("window_id"): r for r in load_jsonl(pose_export_validity) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    orientations = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v8(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
            pose_validity.get(wid, {}),
            controller.get(wid, {}),
            anchors.get(wid, {}),
            orientations.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_semantic_cowgirl_score_v8") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v8(rows, report)
    return rows


def score_cowgirl_candidates_v9(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    pose_export_validity: str | Path,
    controller_validity: str | Path,
    pose_anchor_completeness: str | Path,
    controller_orientation_validity: str | Path,
    controller_distance_validity: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    pose_validity = {r.get("window_id"): r for r in load_jsonl(pose_export_validity) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    orientations = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")}
    distances = {r.get("window_id"): r for r in load_jsonl(controller_distance_validity) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v9(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
            pose_validity.get(wid, {}),
            controller.get(wid, {}),
            anchors.get(wid, {}),
            orientations.get(wid, {}),
            distances.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_semantic_cowgirl_score_v9") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v9(rows, report)
    return rows


def score_cowgirl_candidates_v10(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    pose_export_validity: str | Path,
    controller_validity: str | Path,
    pose_anchor_completeness: str | Path,
    controller_orientation_validity: str | Path,
    controller_distance_validity: str | Path,
    cowgirl_core_controllers: str | Path,
    bj_oral_trap_guard: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    pose_validity = {r.get("window_id"): r for r in load_jsonl(pose_export_validity) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    orientations = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")}
    distances = {r.get("window_id"): r for r in load_jsonl(controller_distance_validity) if r.get("window_id")}
    core = {r.get("window_id"): r for r in load_jsonl(cowgirl_core_controllers) if r.get("window_id")}
    traps = {r.get("window_id"): r for r in load_jsonl(bj_oral_trap_guard) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v10(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
            pose_validity.get(wid, {}),
            controller.get(wid, {}),
            anchors.get(wid, {}),
            orientations.get(wid, {}),
            distances.get(wid, {}),
            core.get(wid, {}),
            traps.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_semantic_cowgirl_score_v10") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v10(rows, report)
    return rows


def score_cowgirl_candidates_v11(
    run_dir: str | Path,
    relative_reference_matches: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    body_quality: str | Path,
    rider_receiver_scores: str | Path,
    pose_export_validity: str | Path,
    controller_validity: str | Path,
    pose_anchor_completeness: str | Path,
    controller_orientation_validity: str | Path,
    controller_distance_validity: str | Path,
    cowgirl_core_controllers: str | Path,
    bj_oral_domain: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    rel_matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    trajectories = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rider_receiver = {r.get("window_id"): r for r in load_jsonl(rider_receiver_scores) if r.get("window_id")}
    pose_validity = {r.get("window_id"): r for r in load_jsonl(pose_export_validity) if r.get("window_id")}
    controller = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    orientations = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")}
    distances = {r.get("window_id"): r for r in load_jsonl(controller_distance_validity) if r.get("window_id")}
    core = {r.get("window_id"): r for r in load_jsonl(cowgirl_core_controllers) if r.get("window_id")}
    bj_domain = {r.get("window_id"): r for r in load_jsonl(bj_oral_domain) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window_v11(
            feature_rows[wid],
            body.get(wid, {}),
            rel_matches.get(wid, {}),
            rel_features.get(wid, {}),
            trajectories.get(wid, {}),
            windows.get(wid, {}),
            rider_receiver.get(wid, {}),
            pose_validity.get(wid, {}),
            controller.get(wid, {}),
            anchors.get(wid, {}),
            orientations.get(wid, {}),
            distances.get(wid, {}),
            core.get(wid, {}),
            bj_domain.get(wid, {}),
        )
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_semantic_cowgirl_score_v11") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report_v11(rows, report)
    return rows


def score_window(feature_row: dict[str, Any], body: dict[str, Any], match: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    values = feature_row.get("feature_values", {}) or {}
    duration = _num(window.get("duration_seconds") or window.get("window_size_seconds") or 0.0)
    body_quality = body.get("body_motion_quality", "unknown")
    phase = classify_motion_phase(feature_row, body)
    guard = evaluate_domain_guards(feature_row, body)
    pelvis_range = max(_num(values.get("pelvis_total_position_range")), _num(values.get("pelvis_vertical_amplitude")), _num(values.get("pelvis_forward_back_amplitude")), _num(values.get("pelvis_lateral_amplitude")))
    pelvis_energy = _num(values.get("pelvis_movement_energy"))
    cowgirl_reference_score = _num(match.get("cowgirl_reference_score"))
    body_score = {"good_body_motion": 1.0, "partial_body_motion": 0.65}.get(str(body_quality), 0.0)
    pelvis_motion_score = max(min(pelvis_range / 0.12, 1.0), min(pelvis_energy / 0.04, 1.0)) if np.isfinite(pelvis_range) or np.isfinite(pelvis_energy) else 0.0
    duration_score = 1.0 if duration >= 8.0 else 0.85 if duration >= 4.0 else 0.45 if duration >= 3.0 else 0.18 if duration >= 2.0 else 0.0
    anti_static_score = 0.0 if body.get("body_motion_quality") in {"static_or_empty", "static_or_micro_motion"} or body.get("static_or_micro_motion") else 1.0
    anti_micro_motion_score = max(0.0, 1.0 - _num(body.get("micro_motion_score")))
    anti_head_only_score = 0.0 if body.get("minimal_head_motion_only") or "possible_non_cowgirl_head_dominant_motion" in guard.get("domain_guard_audit_labels", []) else 1.0
    anti_hand_jitter_score = 0.0 if body.get("minimal_hand_jitter_only") else 1.0
    transition_penalty = 0.45 if phase.get("motion_phase_candidate") == "transition_adjustment_candidate" or match.get("recommended_review_status") == "likely_transition_or_realign" else 0.0
    root_world_penalty = 1.0 if body_quality in {"controller_only_whole_person_motion", "root_only_motion"} else 0.0
    domain_guard_penalty = 0.55 if guard.get("cowgirl_confidence_multiplier", 1.0) < 0.8 else 0.0
    support_count = int(body.get("active_bodypart_count_above_threshold") or 0)
    support_penalty = 0.35 if support_count < 2 else 0.0
    raw_score = (
        0.30 * cowgirl_reference_score
        + 0.20 * body_score
        + 0.20 * pelvis_motion_score
        + 0.12 * duration_score
        + 0.08 * anti_static_score
        + 0.05 * anti_micro_motion_score
        + 0.03 * anti_head_only_score
        + 0.02 * anti_hand_jitter_score
    )
    penalty = max(0.0, 1.0 - transition_penalty - root_world_penalty - domain_guard_penalty - support_penalty)
    final_score = max(0.0, min(1.0, raw_score * penalty))
    reject_reasons = []
    if root_world_penalty:
        reject_reasons.append("root_or_controller_only")
    if anti_static_score == 0.0 or anti_micro_motion_score < 0.4:
        reject_reasons.append("static_or_micro_motion")
    if anti_head_only_score == 0.0:
        reject_reasons.append("head_only_or_head_dominant")
    if anti_hand_jitter_score == 0.0:
        reject_reasons.append("hand_jitter_only")
    if support_count < 2:
        reject_reasons.append("insufficient_supporting_bodyparts")
    if duration < 4.0:
        reject_reasons.append("too_short_for_semantic_judgment")
    if transition_penalty:
        reject_reasons.append("transition_or_realign_candidate")
    return {
        "window_id": feature_row.get("window_id"),
        "sample_id": feature_row.get("sample_id"),
        "source_id": feature_row.get("source_id"),
        "source_scene_file": feature_row.get("source_scene_file"),
        "technical_atom_id": feature_row.get("technical_atom_id"),
        "duration_seconds": duration,
        "cowgirl_reference_score": round(float(cowgirl_reference_score), 6),
        "body_motion_quality_score": round(float(body_score), 6),
        "pelvis_motion_score": round(float(pelvis_motion_score), 6),
        "duration_score": round(float(duration_score), 6),
        "anti_static_score": round(float(anti_static_score), 6),
        "anti_micro_motion_score": round(float(anti_micro_motion_score), 6),
        "anti_head_only_score": round(float(anti_head_only_score), 6),
        "anti_hand_jitter_score": round(float(anti_hand_jitter_score), 6),
        "transition_penalty": round(float(transition_penalty), 6),
        "root_world_penalty": round(float(root_world_penalty), 6),
        "domain_guard_penalty": round(float(domain_guard_penalty), 6),
        "support_penalty": round(float(support_penalty), 6),
        "final_clean_cowgirl_candidate_score": round(float(final_score), 6),
        "clean_cowgirl_candidate": bool(final_score >= 0.50 and not reject_reasons),
        "reject_reasons": reject_reasons,
        "body_motion_quality": body_quality,
        "static_or_micro_motion": bool(body.get("static_or_micro_motion")),
        "minimal_head_motion_only": bool(body.get("minimal_head_motion_only")),
        "minimal_hand_jitter_only": bool(body.get("minimal_hand_jitter_only")),
        "active_bodypart_count_above_threshold": support_count,
        "meaningful_motion_duration_ratio": body.get("meaningful_motion_duration_ratio"),
        "motion_phase_candidate": phase.get("motion_phase_candidate"),
        "domain_guard_warnings": guard.get("domain_guard_warnings", []),
        "reference_review_status": match.get("recommended_review_status"),
        "nearest_reference_families": match.get("nearest_reference_families", []),
        "is_human_ground_truth": False,
    }


def score_window_v3(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    match: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score clean active-rider Cowgirl candidates for review triage only."""
    base = score_window(feature_row, body, match, window)
    rider_receiver = rider_receiver or {}
    values = feature_row.get("feature_values", {}) or {}
    role_status = str(rider_receiver.get("rider_receiver_status") or "role_unclear")
    active = _num(rider_receiver.get("active_rider_score"))
    receiver = _num(rider_receiver.get("receiver_body_response_score"))
    role_unclear = _num(rider_receiver.get("role_unclear_score"))
    pair_uncertainty = 0.18 if role_status == "insufficient_pair_context" else 0.0
    receiver_penalty = 0.85 if role_status == "likely_receiver_body_response" else min(receiver * 0.55, 0.45)
    role_unclear_penalty = 0.22 if role_status == "role_unclear" else min(role_unclear * 0.12, 0.18)

    circularity = _num(values.get("pelvis_circularity_score_proxy"))
    grind = _num(values.get("pelvis_grind_score_proxy"))
    lateral = _num(values.get("pelvis_lateral_amplitude"))
    forward = _num(values.get("pelvis_forward_back_amplitude"))
    vertical = _num(values.get("pelvis_vertical_amplitude"))
    rock = _num(values.get("pelvis_rock_score_proxy"))
    bounce = _num(values.get("pelvis_bounce_score_proxy"))
    balanced_horizontal = min(lateral, forward) / max(lateral, forward, 1e-6)
    horizontal_dominance = (lateral + forward) / max(lateral + forward + vertical, 1e-6)
    cowgirl_grinding_score = min(1.0, 0.35 * circularity + 0.35 * grind + 0.20 * balanced_horizontal + 0.10 * horizontal_dominance)
    cowgirl_bounce_or_ride_score = min(1.0, 0.45 * min(vertical / 0.18, 1.0) + 0.25 * min(forward / 0.18, 1.0) + 0.15 * rock + 0.15 * bounce)

    likely_grinding = cowgirl_grinding_score >= 0.55 and cowgirl_grinding_score >= cowgirl_bounce_or_ride_score * 0.85
    likely_transition = bool(base.get("transition_penalty")) or base.get("motion_phase_candidate") == "transition_adjustment_candidate"
    likely_receiver = role_status == "likely_receiver_body_response"
    active_bonus = 0.16 * active if role_status in {"likely_active_rider", "insufficient_pair_context", "role_unclear"} else 0.0
    subtype_bonus = 0.04 * max(cowgirl_grinding_score, cowgirl_bounce_or_ride_score)
    raw = _num(base.get("final_clean_cowgirl_candidate_score")) + active_bonus + subtype_bonus
    penalty = receiver_penalty + role_unclear_penalty + pair_uncertainty
    final_score = max(0.0, min(1.0, raw * max(0.0, 1.0 - penalty)))

    reject_reasons = list(base.get("reject_reasons", []))
    if likely_receiver:
        reject_reasons.append("likely_receiver_body_response")
    if role_status == "role_unclear":
        reject_reasons.append("role_unclear")
    clean = bool(base.get("clean_cowgirl_candidate") and final_score >= 0.50 and not likely_receiver and role_status != "role_unclear")

    out = dict(base)
    out.update(
        {
            "active_rider_score": round(float(active), 6),
            "receiver_body_response_score": round(float(receiver), 6),
            "passive_context_score": rider_receiver.get("passive_context_score"),
            "role_unclear_score": round(float(role_unclear), 6),
            "role_status": role_status,
            "active_rider_score_bonus": round(float(active_bonus), 6),
            "receiver_body_response_penalty": round(float(receiver_penalty), 6),
            "role_unclear_penalty": round(float(role_unclear_penalty), 6),
            "pair_context_uncertainty_penalty": round(float(pair_uncertainty), 6),
            "cowgirl_grinding_score": round(float(cowgirl_grinding_score), 6),
            "cowgirl_bounce_or_ride_score": round(float(cowgirl_bounce_or_ride_score), 6),
            "likely_grinding_subtype": bool(likely_grinding),
            "likely_transition_context": bool(likely_transition),
            "likely_receiver_false_positive": bool(likely_receiver),
            "final_clean_cowgirl_rider_score_v3": round(float(final_score), 6),
            "clean_cowgirl_rider_candidate_v3": clean,
            "reject_reasons": _dedupe(reject_reasons),
            "rider_receiver_evidence": rider_receiver.get("evidence", {}),
            "rider_receiver_warnings": rider_receiver.get("warnings", []),
            "is_human_ground_truth": False,
        }
    )
    return out


def score_window_v4(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Relative/trajectory Cowgirl score for review selection only."""
    rider_receiver = rider_receiver or {}
    values = feature_row.get("feature_values", {}) or {}
    rel_values = relative_feature.get("feature_values", {}) or {}
    traj_values = trajectory.get("feature_values", {}) or {}
    duration = _num(window.get("duration_seconds") or feature_row.get("duration_seconds") or 0.0)
    body_quality = str(body.get("body_motion_quality") or "unknown")
    role_status = str(rider_receiver.get("rider_receiver_status") or "role_unclear")

    relative_cowgirl_reference_score = _num(relative_match.get("cowgirl_relative_score"))
    trajectory_grind_score = _num(relative_match.get("cowgirl_grind_trajectory_score") or traj_values.get("grind_pattern_score"))
    trajectory_bounce_score = _num(relative_match.get("cowgirl_bounce_trajectory_score") or traj_values.get("bounce_pattern_score"))
    trajectory_forward_back_score = _num(relative_match.get("cowgirl_forward_back_rock_score") or traj_values.get("forward_back_rock_pattern_score"))
    oval_path_score = _num(traj_values.get("oval_path_score"))
    ellipse_fit_score = _num(traj_values.get("ellipse_fit_score"))
    closed_loop_ratio = _num(traj_values.get("closed_loop_ratio"))
    safe_for_learning = bool(relative_feature.get("feature_values", {}).get("safe_for_learning") or relative_match.get("safe_for_learning"))
    safe_for_learning_score = 1.0 if safe_for_learning else 0.0
    active_rider_score = _num(rider_receiver.get("active_rider_score"))
    receiver_score = _num(rider_receiver.get("receiver_body_response_score"))
    static_micro = bool(body.get("static_or_micro_motion")) or body_quality in {"static_or_empty", "static_or_micro_motion"}
    head_only = bool(body.get("minimal_head_motion_only")) or relative_match.get("recommended_review_status") == "likely_not_cowgirl_head_or_bj"
    jitter = _num(traj_values.get("jitter_score") or relative_match.get("jitter_static_score"))
    transition = max(_num(traj_values.get("transition_path_score")), _num(relative_match.get("transition_trajectory_score")))
    root_world = body_quality in {"controller_only_whole_person_motion", "root_only_motion"} or not safe_for_learning
    duration_score = 1.0 if duration >= 8 else 0.88 if duration >= 4 else 0.35 if duration >= 2 else 0.0

    grind_subtype = max(trajectory_grind_score, oval_path_score * 0.75, ellipse_fit_score * 0.65)
    bounce_subtype = trajectory_bounce_score
    forward_subtype = trajectory_forward_back_score
    motion_shape_support = max(grind_subtype, bounce_subtype, forward_subtype)
    raw = (
        0.24 * relative_cowgirl_reference_score
        + 0.24 * motion_shape_support
        + 0.14 * oval_path_score
        + 0.10 * closed_loop_ratio
        + 0.12 * safe_for_learning_score
        + 0.10 * active_rider_score
        + 0.06 * duration_score
    )
    receiver_penalty = 0.75 if role_status == "likely_receiver_body_response" else min(receiver_score * 0.45, 0.40)
    static_micro_penalty = 0.80 if static_micro else 0.0
    root_world_penalty = 0.90 if root_world else 0.0
    transition_penalty = 0.45 if transition >= 0.58 else 0.20 if transition >= 0.42 else 0.0
    head_bj_penalty = 0.70 if head_only else min(max(_num(relative_match.get("bj_relative_score")), _num(relative_match.get("head_relative_score"))) * 0.35, 0.45)
    jitter_penalty = 0.65 if jitter >= 0.70 else min(jitter * 0.35, 0.35)
    penalty = receiver_penalty + static_micro_penalty + root_world_penalty + transition_penalty + head_bj_penalty + jitter_penalty
    final_score = max(0.0, min(1.0, raw * max(0.0, 1.0 - penalty)))

    likely_receiver = role_status == "likely_receiver_body_response" or receiver_penalty >= 0.65
    likely_grinding = bool(grind_subtype >= 0.50 and oval_path_score >= 0.35 and final_score >= 0.35)
    likely_bounce = bool(bounce_subtype >= 0.45 and bounce_subtype >= forward_subtype)
    likely_forward = bool(forward_subtype >= 0.40 and forward_subtype >= bounce_subtype)
    likely_transition = bool(transition_penalty >= 0.20)
    likely_static = bool(static_micro or jitter >= 0.70)
    likely_head = bool(head_only or head_bj_penalty >= 0.50)
    reject_reasons = []
    if likely_receiver:
        reject_reasons.append("likely_receiver_body_response")
    if root_world:
        reject_reasons.append("unsafe_root_world_or_no_relative_learning")
    if static_micro:
        reject_reasons.append("static_or_micro_motion")
    if head_only:
        reject_reasons.append("head_only_or_head_bj_false_positive")
    if jitter >= 0.70:
        reject_reasons.append("jitter_static_trajectory")
    if duration < 4:
        reject_reasons.append("too_short_for_semantic_judgment")
    clean = bool(final_score >= 0.50 and safe_for_learning and duration >= 4 and not likely_receiver and not likely_static and not likely_head and not root_world)
    return {
        "window_id": feature_row.get("window_id"),
        "sample_id": feature_row.get("sample_id"),
        "source_id": feature_row.get("source_id"),
        "source_scene_file": feature_row.get("source_scene_file"),
        "technical_atom_id": feature_row.get("technical_atom_id"),
        "duration_seconds": duration,
        "relative_cowgirl_reference_score": round(float(relative_cowgirl_reference_score), 6),
        "trajectory_grind_score": round(float(trajectory_grind_score), 6),
        "trajectory_bounce_score": round(float(trajectory_bounce_score), 6),
        "trajectory_forward_back_score": round(float(trajectory_forward_back_score), 6),
        "oval_path_score": round(float(oval_path_score), 6),
        "ellipse_fit_score": round(float(ellipse_fit_score), 6),
        "closed_loop_ratio": round(float(closed_loop_ratio), 6),
        "duration_score": round(float(duration_score), 6),
        "safe_for_learning_score": round(float(safe_for_learning_score), 6),
        "active_rider_score": round(float(active_rider_score), 6),
        "receiver_body_response_penalty": round(float(receiver_penalty), 6),
        "static_micro_penalty": round(float(static_micro_penalty), 6),
        "root_world_penalty": round(float(root_world_penalty), 6),
        "transition_penalty": round(float(transition_penalty), 6),
        "head_bj_penalty": round(float(head_bj_penalty), 6),
        "jitter_penalty": round(float(jitter_penalty), 6),
        "final_clean_cowgirl_score_v4": round(float(final_score), 6),
        "clean_cowgirl_candidate_v4": clean,
        "likely_cowgirl_grinding": likely_grinding,
        "likely_cowgirl_vertical_bounce": likely_bounce,
        "likely_cowgirl_forward_back_rock": likely_forward,
        "likely_transition_or_adjustment": likely_transition,
        "likely_receiver_response": likely_receiver,
        "likely_static_or_jitter": likely_static,
        "likely_head_or_bj_false_positive": likely_head,
        "trajectory_shape_classification": trajectory.get("trajectory_shape_classification"),
        "dominant_motion_plane": trajectory.get("dominant_motion_plane"),
        "safe_for_learning": safe_for_learning,
        "role_status": role_status,
        "body_motion_quality": body_quality,
        "relative_review_status": relative_match.get("recommended_review_status"),
        "nearest_handmade_references": relative_match.get("nearest_handmade_references", []),
        "reject_reasons": _dedupe(reject_reasons),
        "warning": "Cowgirl v4 score is review triage based on relative motion and trajectory shape, not ground truth.",
        "is_human_ground_truth": False,
    }


def score_window_v5(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
    pose_validity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v4 = score_window_v4(feature_row, body, relative_match, relative_feature, trajectory, window, rider_receiver)
    pose_validity = pose_validity or {}
    duration = _num(v4.get("duration_seconds"))
    fv = relative_feature.get("feature_values", {}) or {}
    tv = trajectory.get("feature_values", {}) or {}
    export_validity = str(pose_validity.get("export_pose_validity") or "unknown")
    semantic_hint = pose_validity.get("semantic_motion_likely_valid", "unknown")
    generation_template_safe = bool(pose_validity.get("generation_template_safe"))
    low_motion_intro = bool(pose_validity.get("low_motion_intro_candidate"))
    too_short = bool(pose_validity.get("too_short_for_semantic_judgment")) or duration < 4.0
    broken_pose = export_validity == "broken_pose" or bool(pose_validity.get("pose_broken_score"))
    export_unavailable = export_validity == "export_unavailable"
    review_only_absolute = export_validity == "review_only_absolute_pose"
    motion_strength = max(
        _num(pose_validity.get("motion_strength_score")),
        min(_num(fv.get("local_path_length")) / 0.8, 1.0),
        min(_num(fv.get("local_motion_energy")) / 2.0, 1.0),
    )
    clean_motion_strength_score = 0.0 if low_motion_intro else motion_strength
    intro_low_motion_penalty = 0.45 if low_motion_intro else 0.0
    too_short_penalty = 0.35 if too_short else 0.0
    broken_pose_penalty_for_export_only = 0.85 if broken_pose else 0.0
    export_pose_validity_score = {
        "good": 1.0,
        "review_only_absolute_pose": 0.55,
        "unknown": 0.60,
        "broken_pose": 0.05,
        "export_unavailable": 0.0,
    }.get(export_validity, 0.45)
    heuristic_generation_safe = bool(
        v4.get("safe_for_learning")
        and duration >= 4.0
        and not v4.get("likely_receiver_response")
        and not v4.get("likely_static_or_jitter")
        and not v4.get("likely_head_or_bj_false_positive")
        and motion_strength >= 0.35
    )
    generation_template_safety_score = 1.0 if generation_template_safe else (0.72 if heuristic_generation_safe and not broken_pose and not export_unavailable else 0.20 if heuristic_generation_safe else 0.0)
    semantic_cowgirl_motion_score = _num(v4.get("final_clean_cowgirl_score_v4"))
    if semantic_hint is True:
        semantic_cowgirl_motion_score = max(semantic_cowgirl_motion_score, 0.72)
    elif semantic_hint is False:
        semantic_cowgirl_motion_score *= 0.35
    final_semantic = max(0.0, min(1.0, semantic_cowgirl_motion_score * max(0.0, 1.0 - intro_low_motion_penalty - too_short_penalty)))
    generation_penalty = broken_pose_penalty_for_export_only + (0.55 if export_unavailable else 0.0) + intro_low_motion_penalty + too_short_penalty
    if review_only_absolute:
        generation_penalty += 0.20
    final_generation = max(0.0, min(1.0, final_semantic * generation_template_safety_score * max(0.0, clean_motion_strength_score) * max(0.0, 1.0 - generation_penalty)))
    out = dict(v4)
    out.update(
        {
            "semantic_cowgirl_motion_score": round(float(semantic_cowgirl_motion_score), 6),
            "export_pose_validity_score": round(float(export_pose_validity_score), 6),
            "generation_template_safety_score": round(float(generation_template_safety_score), 6),
            "clean_motion_strength_score": round(float(clean_motion_strength_score), 6),
            "intro_low_motion_penalty": round(float(intro_low_motion_penalty), 6),
            "too_short_penalty": round(float(too_short_penalty), 6),
            "broken_pose_penalty_for_export_only": round(float(broken_pose_penalty_for_export_only), 6),
            "final_semantic_cowgirl_score_v5": round(float(final_semantic), 6),
            "final_generation_candidate_score_v5": round(float(final_generation), 6),
            "semantic_cowgirl_candidate_v5": bool(final_semantic >= 0.50 and not v4.get("likely_receiver_response") and not v4.get("likely_head_or_bj_false_positive")),
            "generation_candidate_v5": bool(final_generation >= 0.20 and not broken_pose and not export_unavailable),
            "semantically_good_but_not_generation_safe": bool(final_semantic >= 0.50 and final_generation < 0.20),
            "cowgirl_context_low_motion_intro": bool(low_motion_intro),
            "export_pose_validity": export_validity,
            "generation_template_safe": generation_template_safe,
            "review_export_available": bool(pose_validity.get("review_export_available")) if pose_validity else None,
            "uses_absolute_review_coordinates": bool(pose_validity.get("uses_absolute_review_coordinates")) if pose_validity else None,
            "pose_export_validity": pose_validity,
            "trajectory_shape_classification": trajectory.get("trajectory_shape_classification") or v4.get("trajectory_shape_classification"),
            "warning": "V5 separates semantic Cowgirl likelihood from generation/export template usability.",
            "is_human_ground_truth": False,
        }
    )
    return out


def score_window_v6(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
    pose_validity: dict[str, Any] | None = None,
    controller_validity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """V6 keeps semantic, context, clean-motion, and generation scores apart."""
    v5 = score_window_v5(feature_row, body, relative_match, relative_feature, trajectory, window, rider_receiver, pose_validity)
    pose_validity = pose_validity or {}
    controller_validity = controller_validity or {}
    export_validity = str(pose_validity.get("export_pose_validity") or v5.get("export_pose_validity") or "unknown")
    controller_status = str(controller_validity.get("controller_validity_status") or pose_validity.get("controller_validity_status") or "unknown")
    controller_score = _num(controller_validity.get("controller_validity_score"))
    if not controller_validity:
        controller_score = 0.35
    foot_outlier = bool(controller_validity.get("foot_controller_outlier") or pose_validity.get("foot_controller_outlier"))
    hand_outlier = bool(controller_validity.get("hand_controller_outlier") or pose_validity.get("hand_controller_outlier"))
    controller_outlier_count = int(controller_validity.get("controller_outlier_count") or pose_validity.get("controller_outlier_count") or 0)
    if foot_outlier and controller_status == "valid":
        controller_status = "invalid"
    export_unavailable = export_validity == "export_unavailable"
    broken_pose = export_validity == "broken_pose" or foot_outlier or controller_status == "invalid"
    low_intro = bool(v5.get("cowgirl_context_low_motion_intro") or pose_validity.get("low_motion_intro_candidate"))
    too_short = bool(v5.get("too_short_penalty")) or bool(pose_validity.get("too_short_for_semantic_judgment"))
    receiver = bool(v5.get("likely_receiver_response")) or str(v5.get("role_status")) == "likely_receiver_body_response"

    semantic_cowgirl_motion_score = _num(v5.get("final_semantic_cowgirl_score_v5"))
    if receiver:
        semantic_cowgirl_motion_score *= 0.35
    clean_motion_score = _num(v5.get("clean_motion_strength_score"))
    if low_intro:
        clean_motion_score *= 0.30
    if too_short:
        clean_motion_score *= 0.55
    cowgirl_context_score = max(
        semantic_cowgirl_motion_score * (0.72 if low_intro or too_short else 0.45),
        _num(v5.get("final_clean_cowgirl_score_v4")) * 0.50,
    )
    if not (low_intro or too_short):
        cowgirl_context_score *= 0.55
    controller_validity_score = float(np.clip(controller_score, 0.0, 1.0))
    export_score = _num(v5.get("export_pose_validity_score"))
    if export_unavailable:
        export_score = 0.0
    if broken_pose:
        export_score = min(export_score, 0.08)
    controller_generation_multiplier = controller_validity_score
    if controller_status != "valid":
        controller_generation_multiplier = min(controller_generation_multiplier, 0.40 if controller_status == "warning" else 0.12)
    if foot_outlier:
        controller_generation_multiplier = min(controller_generation_multiplier, 0.04)
    if hand_outlier:
        controller_generation_multiplier = min(controller_generation_multiplier, 0.45)
    generation_candidate_score = semantic_cowgirl_motion_score * clean_motion_score * export_score * controller_generation_multiplier
    if export_unavailable:
        generation_candidate_score = 0.0
    final_semantic = float(np.clip(semantic_cowgirl_motion_score, 0.0, 1.0))
    final_generation = float(np.clip(generation_candidate_score, 0.0, 1.0))
    semantically_controller_invalid = bool(final_semantic >= 0.50 and (foot_outlier or controller_status == "invalid" or broken_pose) and not receiver)
    semantic_candidate = bool(final_semantic >= 0.50 and not receiver and not v5.get("likely_head_or_bj_false_positive"))
    generation_candidate = bool(
        semantic_candidate
        and clean_motion_score >= 0.35
        and final_generation >= 0.20
        and controller_status == "valid"
        and not foot_outlier
        and not hand_outlier
        and not export_unavailable
        and not broken_pose
        and not receiver
    )
    clean_candidate = bool(semantic_candidate and clean_motion_score >= 0.35 and not low_intro and not too_short)
    out = dict(v5)
    out.update(
        {
            "semantic_cowgirl_motion_score": round(float(semantic_cowgirl_motion_score), 6),
            "clean_motion_score": round(float(clean_motion_score), 6),
            "cowgirl_context_score": round(float(cowgirl_context_score), 6),
            "generation_candidate_score": round(float(generation_candidate_score), 6),
            "controller_validity_score": round(float(controller_validity_score), 6),
            "final_semantic_cowgirl_score_v6": round(float(final_semantic), 6),
            "final_generation_candidate_score_v6": round(float(final_generation), 6),
            "semantic_cowgirl_candidate_v6": semantic_candidate,
            "clean_motion_candidate_v6": clean_candidate,
            "generation_candidate_v6": generation_candidate,
            "semantically_cowgirl_but_controller_invalid": semantically_controller_invalid,
            "cowgirl_context_low_motion_intro": bool(low_intro),
            "export_unavailable_for_generation": bool(export_unavailable),
            "controller_validity_status": controller_status,
            "foot_controller_outlier": foot_outlier,
            "hand_controller_outlier": hand_outlier,
            "controller_outlier_count": controller_outlier_count,
            "generation_pose_valid": controller_validity.get("generation_pose_valid", pose_validity.get("generation_template_safe")),
            "controller_validity": controller_validity,
            "pose_export_validity": pose_validity,
            "warning": "V6 separates semantic Cowgirl/context/clean-motion scores from controller and generation safety.",
            "is_human_ground_truth": False,
        }
    )
    return out


def score_window_v7(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
    pose_validity: dict[str, Any] | None = None,
    controller_validity: dict[str, Any] | None = None,
    pose_anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v6 = score_window_v6(feature_row, body, relative_match, relative_feature, trajectory, window, rider_receiver, pose_validity, controller_validity)
    pose_anchor = pose_anchor or {}
    controller_validity = controller_validity or {}
    pose_validity = pose_validity or {}
    semantic = _num(v6.get("final_semantic_cowgirl_score_v6"))
    clean = _num(v6.get("clean_motion_score"))
    controller_score = _num(v6.get("controller_validity_score"))
    anchor_score = _num(pose_anchor.get("pose_anchor_completeness_score"))
    if not pose_anchor:
        anchor_score = 0.35
    missing_foot = bool(pose_anchor.get("missing_foot_controllers") or controller_validity.get("missing_foot_controllers") or pose_validity.get("missing_foot_controllers"))
    missing_knee = bool(pose_anchor.get("missing_knee_controllers") or controller_validity.get("missing_knee_controllers") or pose_validity.get("missing_knee_controllers"))
    anchor_incomplete = bool(pose_anchor.get("pose_anchor_incomplete") or controller_validity.get("pose_anchor_incomplete") or pose_validity.get("pose_anchor_incomplete") or missing_foot or missing_knee)
    anchor_safe = pose_anchor.get("generation_pose_anchor_safe") is True and not anchor_incomplete and not missing_foot and not missing_knee
    foot_outlier = bool(v6.get("foot_controller_outlier"))
    controller_outlier = bool(v6.get("controller_validity_status") == "invalid" or foot_outlier)
    receiver = bool(v6.get("likely_receiver_response"))
    export_unavailable = bool(v6.get("export_unavailable_for_generation"))
    context_low = bool(v6.get("cowgirl_context_low_motion_intro"))
    generation_multiplier = min(controller_score, anchor_score)
    if not anchor_safe:
        generation_multiplier = min(generation_multiplier, 0.08 if missing_foot else 0.20 if missing_knee else 0.35)
    if controller_outlier:
        generation_multiplier = min(generation_multiplier, 0.05)
    if export_unavailable:
        generation_multiplier = 0.0
    final_semantic = float(np.clip(semantic, 0.0, 1.0))
    final_clean = float(np.clip(clean * (0.35 if context_low else 1.0), 0.0, 1.0))
    final_generation = float(np.clip(final_semantic * final_clean * generation_multiplier, 0.0, 1.0))
    semantic_candidate = bool(v6.get("semantic_cowgirl_candidate_v6"))
    generation_safe = bool(semantic_candidate and final_generation >= 0.20 and anchor_safe and not controller_outlier and not receiver and not export_unavailable)
    semantic_anchor_incomplete = bool(semantic_candidate and anchor_incomplete and not controller_outlier and not receiver)
    semantic_controller_outlier = bool(semantic_candidate and controller_outlier and not receiver)
    context_intro = bool(v6.get("cowgirl_context_low_motion_intro"))
    receiver_negative = bool(receiver)
    unknown_unusable = bool(export_unavailable or (not semantic_candidate and not receiver_negative and final_semantic < 0.20))
    classification = "unknown_or_unusable"
    if generation_safe:
        classification = "semantic_cowgirl_generation_safe"
    elif semantic_controller_outlier:
        classification = "semantic_cowgirl_controller_outlier"
    elif semantic_anchor_incomplete:
        classification = "semantic_cowgirl_anchor_incomplete"
    elif context_intro:
        classification = "cowgirl_context_intro_low_motion"
    elif receiver_negative:
        classification = "receiver_response_negative"
    elif semantic_candidate:
        classification = "semantic_cowgirl_anchor_incomplete" if anchor_incomplete else "semantic_cowgirl_generation_safe"
    out = dict(v6)
    out.update(
        {
            "final_semantic_cowgirl_score_v7": round(final_semantic, 6),
            "final_clean_motion_score_v7": round(final_clean, 6),
            "final_generation_candidate_score_v7": round(final_generation, 6),
            "pose_anchor_completeness_score": round(float(anchor_score), 6),
            "controller_validity_score": round(float(controller_score), 6),
            "generation_pose_anchor_safe": bool(anchor_safe),
            "semantic_cowgirl_candidate_v7": semantic_candidate,
            "semantic_cowgirl_generation_safe": generation_safe,
            "semantic_cowgirl_anchor_incomplete": semantic_anchor_incomplete,
            "semantic_cowgirl_controller_outlier": semantic_controller_outlier,
            "cowgirl_context_intro_low_motion": context_intro,
            "receiver_response_negative": receiver_negative,
            "unknown_or_unusable": unknown_unusable,
            "cowgirl_v7_category": classification,
            "missing_foot_controllers": missing_foot,
            "missing_knee_controllers": missing_knee,
            "pose_anchor_incomplete": anchor_incomplete,
            "missing_required_anchor_controllers": pose_anchor.get("missing_required_anchor_controllers", []),
            "foot_controllers_present": pose_anchor.get("foot_controllers_present"),
            "knee_controllers_present": pose_anchor.get("knee_controllers_present"),
            "pose_anchor_completeness": pose_anchor,
            "warning": "V7 separates semantic Cowgirl from pose-anchor completeness and generation safety.",
            "is_human_ground_truth": False,
        }
    )
    return out


def score_window_v8(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
    pose_validity: dict[str, Any] | None = None,
    controller_validity: dict[str, Any] | None = None,
    pose_anchor: dict[str, Any] | None = None,
    orientation_validity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v7 = score_window_v7(feature_row, body, relative_match, relative_feature, trajectory, window, rider_receiver, pose_validity, controller_validity, pose_anchor)
    pose_validity = pose_validity or {}
    controller_validity = controller_validity or {}
    orientation_validity = orientation_validity or controller_validity.get("controller_orientation_validity") or {}
    semantic = _num(v7.get("final_semantic_cowgirl_score_v7"))
    clean = _num(v7.get("final_clean_motion_score_v7"))
    generation_v7 = _num(v7.get("final_generation_candidate_score_v7"))
    anchor_score = _num(v7.get("pose_anchor_completeness_score"))
    controller_score = _num(v7.get("controller_validity_score"))
    orientation_status = str(orientation_validity.get("orientation_validity_status") or controller_validity.get("orientation_validity_status") or pose_validity.get("orientation_validity_status") or "unknown")
    orientation_score = _num(orientation_validity.get("orientation_validity_score") or controller_validity.get("orientation_validity_score") or pose_validity.get("orientation_validity_score"))
    if not orientation_validity and not controller_validity.get("orientation_validity_score"):
        orientation_score = 0.35
    controller_rotation_invalid = bool(
        orientation_validity.get("controller_rotation_invalid")
        or controller_validity.get("controller_rotation_invalid")
        or pose_validity.get("controller_rotation_invalid")
        or orientation_status == "invalid"
    )
    controller_twist_invalid = bool(
        orientation_validity.get("controller_twist_invalid")
        or controller_validity.get("controller_twist_invalid")
        or pose_validity.get("controller_twist_invalid")
        or controller_rotation_invalid
    )
    twisted_controller_names = list(
        orientation_validity.get("twisted_controller_names")
        or controller_validity.get("twisted_controller_names")
        or pose_validity.get("twisted_controller_names")
        or []
    )
    if controller_rotation_invalid:
        orientation_status = "invalid"
        if not twisted_controller_names:
            twisted_controller_names = ["unknown_controller_rotation"]
    foot_rotation_outlier = bool(
        orientation_validity.get("foot_rotation_outlier")
        or controller_validity.get("foot_rotation_outlier")
        or pose_validity.get("foot_rotation_outlier")
    )
    missing_anchor = bool(v7.get("pose_anchor_incomplete") or v7.get("missing_foot_controllers") or v7.get("missing_knee_controllers"))
    receiver = bool(v7.get("receiver_response_negative") or v7.get("likely_receiver_response"))
    export_unavailable = bool(v7.get("export_unavailable_for_generation"))
    context_intro = bool(v7.get("cowgirl_context_intro_low_motion"))
    head_or_bj = bool(v7.get("likely_head_or_bj_false_positive"))
    body_quality = str(body.get("body_motion_quality") or "")
    minimal_head_or_hand = bool(body.get("minimal_head_motion_only") or body.get("minimal_hand_jitter_only"))
    movement_family = str((relative_match or {}).get("recommended_review_status") or "")
    standing_gesture = bool(
        head_or_bj
        or minimal_head_or_hand
        or movement_family in {"likely_isolated_gesture", "likely_not_cowgirl_head_or_bj"}
        or (body_quality in {"static_or_micro_motion", "static_or_empty"} and semantic < 0.45)
    )
    generation_multiplier = min(_bounded(controller_score), _bounded(anchor_score), _bounded(orientation_score))
    if controller_rotation_invalid:
        generation_multiplier = min(generation_multiplier, 0.04)
    elif orientation_status == "warning":
        generation_multiplier = min(generation_multiplier, 0.45)
    if standing_gesture:
        generation_multiplier = 0.0
    final_semantic = float(np.clip(semantic, 0.0, 1.0))
    final_clean = float(np.clip(clean * (0.35 if context_intro else 1.0), 0.0, 1.0))
    computed_generation = final_semantic * final_clean * generation_multiplier
    final_generation = float(np.clip(min(generation_v7, computed_generation) if generation_v7 > 0.0 else computed_generation, 0.0, 1.0))
    if controller_rotation_invalid or standing_gesture:
        final_generation = float(np.clip(final_semantic * final_clean * generation_multiplier, 0.0, 1.0))
    semantic_candidate = bool(v7.get("semantic_cowgirl_candidate_v7"))
    generation_pose_valid = bool(
        semantic_candidate
        and final_generation >= 0.20
        and v7.get("semantic_cowgirl_generation_safe")
        and not controller_rotation_invalid
        and not missing_anchor
        and not receiver
        and not export_unavailable
        and not standing_gesture
    )
    semantic_orientation_invalid = bool(semantic_candidate and controller_rotation_invalid and not receiver)
    semantic_anchor_incomplete = bool(semantic_candidate and missing_anchor and not controller_rotation_invalid and not receiver)
    semantic_controller_outlier = bool(v7.get("semantic_cowgirl_controller_outlier") and not semantic_orientation_invalid)
    classification = "unknown_or_unusable"
    if generation_pose_valid:
        classification = "semantic_cowgirl_generation_safe"
    elif semantic_orientation_invalid:
        classification = "semantic_cowgirl_orientation_invalid"
    elif semantic_anchor_incomplete:
        classification = "semantic_cowgirl_anchor_incomplete"
    elif semantic_controller_outlier:
        classification = "semantic_cowgirl_controller_outlier"
    elif context_intro:
        classification = "cowgirl_context_intro_low_motion"
    elif standing_gesture:
        classification = "standing_gesture_false_positive"
    elif receiver:
        classification = "receiver_response_negative"
    elif semantic_candidate:
        classification = "semantic_cowgirl_generation_safe" if not missing_anchor and not controller_rotation_invalid else "semantic_cowgirl_anchor_incomplete"
    out = dict(v7)
    out.update(
        {
            "final_semantic_cowgirl_score_v8": round(final_semantic, 6),
            "final_clean_motion_score_v8": round(final_clean, 6),
            "final_generation_candidate_score_v8": round(final_generation, 6),
            "pose_anchor_completeness_score": round(float(anchor_score), 6),
            "controller_validity_score": round(float(controller_score), 6),
            "orientation_validity_score": round(float(orientation_score), 6),
            "orientation_validity_status": orientation_status,
            "controller_rotation_invalid": controller_rotation_invalid,
            "controller_twist_invalid": controller_twist_invalid,
            "twisted_controller_names": twisted_controller_names,
            "foot_rotation_outlier": foot_rotation_outlier,
            "generation_pose_valid": generation_pose_valid,
            "semantic_cowgirl_candidate_v8": semantic_candidate,
            "semantic_cowgirl_generation_safe": generation_pose_valid,
            "semantic_cowgirl_anchor_incomplete": semantic_anchor_incomplete,
            "semantic_cowgirl_orientation_invalid": semantic_orientation_invalid,
            "semantic_cowgirl_controller_outlier": semantic_controller_outlier,
            "standing_gesture_false_positive": standing_gesture,
            "cowgirl_context_intro_low_motion": context_intro,
            "receiver_response_negative": receiver,
            "unknown_or_unusable": bool(classification == "unknown_or_unusable" or export_unavailable),
            "cowgirl_v8_category": classification,
            "controller_orientation_validity": orientation_validity,
            "warning": "V8 separates semantic Cowgirl from pose-anchor completeness, controller positions, and controller orientation/twist validity.",
            "is_human_ground_truth": False,
        }
    )
    return out


def score_window_v9(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
    pose_validity: dict[str, Any] | None = None,
    controller_validity: dict[str, Any] | None = None,
    pose_anchor: dict[str, Any] | None = None,
    orientation_validity: dict[str, Any] | None = None,
    distance_validity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v8 = score_window_v8(feature_row, body, relative_match, relative_feature, trajectory, window, rider_receiver, pose_validity, controller_validity, pose_anchor, orientation_validity)
    distance_validity = distance_validity or {}
    semantic = _bounded(v8.get("final_semantic_cowgirl_score_v8"))
    clean = _bounded(v8.get("final_clean_motion_score_v8"))
    generation_v8 = _bounded(v8.get("final_generation_candidate_score_v8"))
    distance_status = str(distance_validity.get("controller_distance_validity_status") or "unknown")
    distance_score = _num(distance_validity.get("controller_distance_validity_score"))
    if not distance_validity:
        distance_score = 0.35
    distance_invalid = bool(
        distance_validity.get("controller_distance_outlier")
        or distance_validity.get("foot_distance_outlier")
        or distance_validity.get("knee_distance_outlier")
        or distance_validity.get("head_distance_outlier")
        or distance_status == "invalid"
    )
    outlier_names = list(distance_validity.get("outlier_controller_names") or [])
    generation_multiplier = _bounded(distance_score)
    if distance_invalid:
        generation_multiplier = min(generation_multiplier, 0.04)
        distance_status = "invalid"
    elif distance_status == "warning":
        generation_multiplier = min(generation_multiplier, 0.45)
    final_generation = float(np.clip(min(generation_v8, semantic * clean * generation_multiplier), 0.0, 1.0))
    semantic_candidate = bool(v8.get("semantic_cowgirl_candidate_v8"))
    anchor_incomplete = bool(v8.get("semantic_cowgirl_anchor_incomplete"))
    orientation_invalid = bool(v8.get("semantic_cowgirl_orientation_invalid"))
    controller_outlier = bool(v8.get("semantic_cowgirl_controller_outlier"))
    standing = bool(v8.get("standing_gesture_false_positive"))
    receiver = bool(v8.get("receiver_response_negative"))
    context_intro = bool(v8.get("cowgirl_context_intro_low_motion"))
    export_unavailable = bool(v8.get("export_unavailable_for_generation"))
    semantic_or_context = bool(semantic_candidate or (semantic >= 0.45 and not standing and not receiver))
    distance_invalid_cowgirl = bool(semantic_or_context and distance_invalid and not receiver)
    pose_invalid = bool(semantic_or_context and not receiver and (anchor_incomplete or orientation_invalid or controller_outlier or distance_invalid))
    generation_safe = bool(
        semantic_candidate
        and v8.get("semantic_cowgirl_generation_safe")
        and final_generation >= 0.20
        and not distance_invalid
        and not anchor_incomplete
        and not orientation_invalid
        and not controller_outlier
        and not standing
        and not receiver
        and not export_unavailable
    )
    classification = "unknown_or_unusable"
    if generation_safe:
        classification = "semantic_cowgirl_generation_safe"
    elif distance_invalid_cowgirl:
        classification = "semantic_cowgirl_distance_invalid"
    elif orientation_invalid:
        classification = "semantic_cowgirl_orientation_invalid"
    elif anchor_incomplete:
        classification = "semantic_cowgirl_anchor_incomplete"
    elif pose_invalid:
        classification = "semantic_cowgirl_pose_invalid"
    elif context_intro:
        classification = "cowgirl_context_intro_low_motion"
    elif standing:
        classification = "standing_gesture_false_positive"
    elif receiver:
        classification = "receiver_response_negative"
    elif semantic_or_context:
        classification = "semantic_cowgirl_pose_invalid"
    out = dict(v8)
    out.update(
        {
            "final_semantic_cowgirl_score_v9": round(float(semantic), 6),
            "final_clean_motion_score_v9": round(float(clean), 6),
            "final_generation_candidate_score_v9": round(float(final_generation), 6),
            "pose_anchor_completeness_score": v8.get("pose_anchor_completeness_score"),
            "controller_validity_score": v8.get("controller_validity_score"),
            "orientation_validity_score": v8.get("orientation_validity_score"),
            "distance_validity_score": round(float(distance_score), 6),
            "distance_validity_status": distance_status,
            "controller_distance_outlier": bool(distance_invalid),
            "outlier_controller_names": outlier_names,
            "foot_distance_outlier": bool(distance_validity.get("foot_distance_outlier")),
            "knee_distance_outlier": bool(distance_validity.get("knee_distance_outlier")),
            "hand_distance_outlier": bool(distance_validity.get("hand_distance_outlier")),
            "head_distance_outlier": bool(distance_validity.get("head_distance_outlier")),
            "max_bodypart_distance_ratio": distance_validity.get("max_bodypart_distance_ratio"),
            "generation_pose_valid": generation_safe,
            "semantic_cowgirl_candidate_v9": semantic_or_context,
            "semantic_cowgirl_generation_safe": generation_safe,
            "semantic_cowgirl_anchor_incomplete": anchor_incomplete,
            "semantic_cowgirl_orientation_invalid": orientation_invalid,
            "semantic_cowgirl_distance_invalid": distance_invalid_cowgirl,
            "semantic_cowgirl_pose_invalid": pose_invalid,
            "standing_gesture_false_positive": standing,
            "cowgirl_context_intro_low_motion": context_intro,
            "receiver_response_negative": receiver,
            "unknown_or_unusable": bool(classification == "unknown_or_unusable" or export_unavailable),
            "cowgirl_v9_category": classification,
            "controller_distance_validity": distance_validity,
            "warning": "V9 separates semantic Cowgirl from anchor, orientation, distance, and generation safety. It is not an ML label.",
            "is_human_ground_truth": False,
        }
    )
    return out


def score_window_v10(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
    pose_validity: dict[str, Any] | None = None,
    controller_validity: dict[str, Any] | None = None,
    pose_anchor: dict[str, Any] | None = None,
    orientation_validity: dict[str, Any] | None = None,
    distance_validity: dict[str, Any] | None = None,
    core_controllers: dict[str, Any] | None = None,
    bj_oral_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v9 = score_window_v9(
        feature_row,
        body,
        relative_match,
        relative_feature,
        trajectory,
        window,
        rider_receiver,
        pose_validity,
        controller_validity,
        pose_anchor,
        orientation_validity,
        distance_validity,
    )
    core_controllers = core_controllers or {}
    bj_oral_guard = bj_oral_guard or {}
    controller_validity = controller_validity or {}
    distance_validity = distance_validity or {}
    semantic = _bounded(v9.get("final_semantic_cowgirl_score_v9"))
    clean = _bounded(v9.get("final_clean_motion_score_v9"))
    generation_v9 = _bounded(v9.get("final_generation_candidate_score_v9"))
    core_gate = core_controllers.get("generation_safe_core_controller_gate")
    missing_core = bool(core_gate is False or core_controllers.get("cowgirl_core_controller_status") == "missing_core")
    missing_core_controllers = list(core_controllers.get("missing_core_controllers") or [])
    bj_trap = bool(bj_oral_guard.get("head_or_oral_domain_trap") or bj_oral_guard.get("cowgirl_pose_false_positive"))
    arm_stretch = bool(
        distance_validity.get("arm_stretch_outlier")
        or distance_validity.get("arm_stretch_pose_invalid")
        or distance_validity.get("hand_controller_outlier")
        or controller_validity.get("hand_controller_outlier")
    )
    standing = bool(v9.get("standing_gesture_false_positive"))
    receiver = bool(v9.get("receiver_response_negative"))
    anchor_incomplete = bool(v9.get("semantic_cowgirl_anchor_incomplete"))
    orientation_invalid = bool(v9.get("semantic_cowgirl_orientation_invalid"))
    distance_invalid = bool(v9.get("semantic_cowgirl_distance_invalid") or v9.get("controller_distance_outlier"))
    context_intro = bool(v9.get("cowgirl_context_intro_low_motion"))
    export_unavailable = bool(v9.get("export_unavailable_for_generation"))
    generation_multiplier = 1.0
    if missing_core:
        generation_multiplier = min(generation_multiplier, 0.0)
    if bj_trap:
        generation_multiplier = min(generation_multiplier, 0.0)
    if arm_stretch:
        generation_multiplier = min(generation_multiplier, 0.04)
    final_generation = float(np.clip(generation_v9 * generation_multiplier, 0.0, 1.0))
    semantic_candidate = bool(v9.get("semantic_cowgirl_candidate_v9"))
    generation_safe = bool(
        semantic_candidate
        and v9.get("semantic_cowgirl_generation_safe")
        and final_generation >= 0.20
        and core_gate is True
        and not missing_core
        and not bj_trap
        and not arm_stretch
        and not standing
        and not receiver
        and not anchor_incomplete
        and not orientation_invalid
        and not distance_invalid
        and not export_unavailable
    )
    pose_invalid = bool(
        semantic_candidate
        and not generation_safe
        and not missing_core
        and not bj_trap
        and not standing
        and not receiver
        and (anchor_incomplete or orientation_invalid or distance_invalid or arm_stretch or v9.get("semantic_cowgirl_pose_invalid"))
    )
    classification = "unknown_or_unusable"
    if generation_safe:
        classification = "semantic_cowgirl_generation_safe"
    elif missing_core and semantic_candidate:
        classification = "semantic_cowgirl_core_controller_missing"
    elif anchor_incomplete and semantic_candidate:
        classification = "semantic_cowgirl_anchor_incomplete"
    elif orientation_invalid and semantic_candidate:
        classification = "semantic_cowgirl_orientation_invalid"
    elif distance_invalid and semantic_candidate:
        classification = "semantic_cowgirl_distance_invalid"
    elif pose_invalid:
        classification = "semantic_cowgirl_pose_invalid"
    elif context_intro:
        classification = "cowgirl_context_intro_low_motion"
    elif bj_trap:
        classification = "bj_oral_trap_negative"
    elif standing:
        classification = "standing_hand_head_negative"
    elif receiver:
        classification = "receiver_response_negative"
    out = dict(v9)
    out.update(
        {
            "final_semantic_cowgirl_score_v10": round(float(semantic), 6),
            "final_clean_motion_score_v10": round(float(clean), 6),
            "final_generation_candidate_score_v10": round(float(final_generation), 6),
            "core_controller_gate": core_gate,
            "cowgirl_core_controller_status": core_controllers.get("cowgirl_core_controller_status"),
            "missing_core_controllers": missing_core_controllers,
            "missing_core_pelvis_motion_controllers": missing_core,
            "missing_hip_thigh_pelvis_controllers": bool(missing_core and ("thigh_controls" in missing_core_controllers or "hipControl_or_pelvisControl" in missing_core_controllers)),
            "bj_oral_trap_flag": bj_trap,
            "head_or_oral_domain_trap": bool(bj_oral_guard.get("head_or_oral_domain_trap")),
            "cowgirl_pose_false_positive": bool(bj_oral_guard.get("cowgirl_pose_false_positive")),
            "arm_stretch_outlier_flag": arm_stretch,
            "hand_controller_outlier": bool(arm_stretch or v9.get("hand_controller_outlier")),
            "arm_stretch_pose_invalid": arm_stretch,
            "pose_validity": "valid" if generation_safe else "invalid" if pose_invalid or missing_core or bj_trap or arm_stretch else "unknown",
            "generation_safe": generation_safe,
            "semantic_cowgirl_candidate_v10": semantic_candidate,
            "semantic_cowgirl_generation_safe": generation_safe,
            "semantic_cowgirl_core_controller_missing": bool(missing_core and semantic_candidate),
            "semantic_cowgirl_anchor_incomplete": bool(anchor_incomplete and semantic_candidate),
            "semantic_cowgirl_orientation_invalid": bool(orientation_invalid and semantic_candidate),
            "semantic_cowgirl_distance_invalid": bool(distance_invalid and semantic_candidate),
            "semantic_cowgirl_pose_invalid": pose_invalid,
            "bj_oral_trap_negative": bj_trap,
            "standing_hand_head_negative": standing,
            "standing_gesture_false_positive": standing,
            "receiver_response_negative": receiver,
            "unknown_or_unusable": bool(classification == "unknown_or_unusable" or export_unavailable),
            "cowgirl_v10_category": classification,
            "cowgirl_core_controller_requirements": core_controllers,
            "bj_oral_trap_guard": bj_oral_guard,
            "warning": "V10 blocks generation-safe Cowgirl on missing core controllers, BJ/oral traps, and hand/arm stretch outliers. It is not an ML label.",
            "is_human_ground_truth": False,
        }
    )
    return out


def score_window_v11(
    feature_row: dict[str, Any],
    body: dict[str, Any],
    relative_match: dict[str, Any],
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    window: dict[str, Any],
    rider_receiver: dict[str, Any] | None = None,
    pose_validity: dict[str, Any] | None = None,
    controller_validity: dict[str, Any] | None = None,
    pose_anchor: dict[str, Any] | None = None,
    orientation_validity: dict[str, Any] | None = None,
    distance_validity: dict[str, Any] | None = None,
    core_controllers: dict[str, Any] | None = None,
    bj_oral_domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v10 = score_window_v10(
        feature_row,
        body,
        relative_match,
        relative_feature,
        trajectory,
        window,
        rider_receiver,
        pose_validity,
        controller_validity,
        pose_anchor,
        orientation_validity,
        distance_validity,
        core_controllers,
        bj_oral_domain,
    )
    core_controllers = core_controllers or {}
    bj_oral_domain = bj_oral_domain or {}
    semantic = _bounded(v10.get("final_semantic_cowgirl_score_v10"))
    clean = _bounded(v10.get("final_clean_motion_score_v10"))
    clean_evidence = max(clean, _bounded(v10.get("final_clean_cowgirl_score_v4")), _bounded(v10.get("clean_motion_strength_score")))
    generation_v10 = _bounded(v10.get("final_generation_candidate_score_v10"))
    core_status = str(core_controllers.get("core_gate_status") or "unknown")
    core_override = bool(core_controllers.get("core_gate_can_be_overridden"))
    core_reason = str(core_controllers.get("core_gate_override_reason") or "")
    bj_family = bool(bj_oral_domain.get("bj_oral_motion_candidate") or bj_oral_domain.get("semantic_family") == "bj_oral")
    bj_confidence = _bounded(bj_oral_domain.get("bj_oral_confidence") or bj_oral_domain.get("bj_oral_trap_confidence"))
    receiver = bool(v10.get("receiver_response_negative"))
    standing = bool(v10.get("standing_hand_head_negative") or v10.get("standing_gesture_false_positive"))
    orientation_invalid = bool(v10.get("semantic_cowgirl_orientation_invalid"))
    distance_invalid = bool(v10.get("semantic_cowgirl_distance_invalid"))
    anchor_incomplete = bool(v10.get("semantic_cowgirl_anchor_incomplete"))
    pose_invalid = bool(v10.get("semantic_cowgirl_pose_invalid"))
    context_intro = bool(v10.get("cowgirl_context_intro_low_motion"))
    semantic_candidate = bool(v10.get("semantic_cowgirl_candidate_v10"))
    soft_fail_accepted = bool(
        semantic_candidate
        and core_status == "soft_fail"
        and core_override
        and semantic >= 0.45
        and clean_evidence >= 0.35
        and not bj_family
        and not receiver
        and not standing
        and not orientation_invalid
        and not distance_invalid
    )
    hard_fail = bool(core_status == "hard_fail" or (v10.get("semantic_cowgirl_core_controller_missing") and not soft_fail_accepted))
    generation_safe = bool(
        (v10.get("semantic_cowgirl_generation_safe") or soft_fail_accepted)
        and not hard_fail
        and not bj_family
        and not receiver
        and not standing
        and not orientation_invalid
        and not distance_invalid
    )
    final_generation = generation_v10
    if soft_fail_accepted:
        final_generation = max(final_generation, float(np.clip(semantic * clean_evidence * 0.55, 0.0, 1.0)))
    if bj_family or hard_fail or receiver or standing:
        final_generation = 0.0
    subtype = _cowgirl_subtype(v10, trajectory)
    if subtype == "oval_grind":
        subtype = "grinding"
    elif subtype == "circular_grind":
        subtype = "circular_grind"
    elif subtype in {"vertical_bounce", "forward_back_rock"}:
        subtype = "riding"
    category = "unknown_or_unusable"
    semantic_family = "unknown"
    excluded = False
    preserve = False
    if bj_family:
        category = "not_cowgirl_bj_oral"
        semantic_family = "bj_oral"
        excluded = True
        preserve = True
    elif generation_safe and soft_fail_accepted:
        category = "semantic_cowgirl_core_soft_fail_generation_safe"
        semantic_family = "cowgirl"
    elif generation_safe:
        category = "semantic_cowgirl_generation_safe"
        semantic_family = "cowgirl"
    elif hard_fail and semantic_candidate:
        category = "semantic_cowgirl_core_hard_fail"
        semantic_family = "cowgirl"
    elif anchor_incomplete:
        category = "semantic_cowgirl_anchor_incomplete"
        semantic_family = "cowgirl"
    elif orientation_invalid:
        category = "semantic_cowgirl_orientation_invalid"
        semantic_family = "cowgirl"
    elif distance_invalid:
        category = "semantic_cowgirl_distance_invalid"
        semantic_family = "cowgirl"
    elif pose_invalid:
        category = "semantic_cowgirl_pose_invalid"
        semantic_family = "cowgirl"
    elif context_intro:
        category = "cowgirl_context_intro_low_motion"
        semantic_family = "cowgirl"
    elif standing:
        category = "standing_hand_head_negative"
        semantic_family = "hand_gesture" if body.get("minimal_hand_jitter_only") else "head_gesture"
        preserve = True
    elif receiver:
        category = "receiver_response_negative"
        semantic_family = "receiver_response"
        preserve = True
    out = dict(v10)
    out.update(
        {
            "final_semantic_cowgirl_score_v11": round(float(semantic), 6),
            "final_clean_motion_score_v11": round(float(clean), 6),
            "final_generation_candidate_score_v11": round(float(final_generation), 6),
            "semantic_family": semantic_family,
            "excluded_from_cowgirl": excluded,
            "preserve_for_future_dataset": preserve or semantic_family in {"cowgirl", "bj_oral", "receiver_response", "hand_gesture", "head_gesture"},
            "cowgirl_subtype": subtype,
            "bj_oral_confidence": round(float(bj_confidence), 6) if bj_family else None,
            "core_gate_status": core_status,
            "core_gate_can_be_overridden": core_override,
            "core_gate_override_reason": core_reason,
            "semantic_cowgirl_generation_safe": generation_safe,
            "semantic_cowgirl_core_soft_fail_generation_safe": bool(category == "semantic_cowgirl_core_soft_fail_generation_safe"),
            "semantic_cowgirl_core_hard_fail": bool(category == "semantic_cowgirl_core_hard_fail"),
            "not_cowgirl_bj_oral": bool(category == "not_cowgirl_bj_oral"),
            "bj_oral_motion_candidate": bj_family,
            "bj_oral_generation_candidate": bool(bj_oral_domain.get("bj_oral_generation_candidate")),
            "cowgirl_v11_category": category,
            "bj_oral_domain": bj_oral_domain,
            "warning": "V11 treats BJ/oral as a valid semantic family, excludes it from Cowgirl, and preserves it for future family-specific datasets.",
            "is_human_ground_truth": False,
        }
    )
    return out


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = [r for r in rows if r.get("clean_cowgirl_candidate")]
    rejection_counts = Counter(reason for r in rows for reason in r.get("reject_reasons", []))
    lines = [
        "# Cowgirl Candidate Score V2 Report",
        "",
        "Scores are review triage only. They are not labels and not ML targets.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Clean Cowgirl candidates: {len(clean)}",
        "",
        "## Rejection Reasons",
        "",
    ]
    for reason, count in rejection_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Top Candidates", ""])
    for row in rows[:20]:
        lines.append(f"- `{row.get('window_id')}` score={row.get('final_clean_cowgirl_candidate_score')} duration={row.get('duration_seconds')} scene=`{row.get('source_scene_file')}` reject={row.get('reject_reasons')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v3(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = [r for r in rows if r.get("clean_cowgirl_rider_candidate_v3")]
    receiver = [r for r in rows if r.get("likely_receiver_false_positive")]
    grinding = [r for r in rows if r.get("likely_grinding_subtype")]
    role_counts = Counter(r.get("role_status") for r in rows)
    rejection_counts = Counter(reason for r in rows for reason in r.get("reject_reasons", []))
    lines = [
        "# Cowgirl Candidate Score V3 Report",
        "",
        "Scores are review triage only. They are not labels and not ML targets.",
        "V3 adds rider/receiver body-response penalties while keeping grinding as a valid Cowgirl subtype.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Clean active-rider Cowgirl candidates: {len(clean)}",
        f"- Receiver/body-response false-positive candidates: {len(receiver)}",
        f"- Grinding subtype candidates: {len(grinding)}",
        "",
        "## Role Status Counts",
        "",
    ]
    for status, count in role_counts.most_common():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Rejection Reasons", ""])
    for reason, count in rejection_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Top Clean Active-Rider Candidates", ""])
    for row in clean[:20]:
        lines.append(
            f"- `{row.get('window_id')}` score={row.get('final_clean_cowgirl_rider_score_v3')} "
            f"role=`{row.get('role_status')}` grinding={row.get('cowgirl_grinding_score')} "
            f"scene=`{row.get('source_scene_file')}`"
        )
    lines.extend(["", "## High Receiver False-Positive Candidates", ""])
    for row in sorted(receiver, key=lambda r: float(r.get("receiver_body_response_score") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` receiver={row.get('receiver_body_response_score')} "
            f"active={row.get('active_rider_score')} v3={row.get('final_clean_cowgirl_rider_score_v3')} "
            f"scene=`{row.get('source_scene_file')}`"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v4(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = [r for r in rows if r.get("clean_cowgirl_candidate_v4")]
    grind = [r for r in rows if r.get("likely_cowgirl_grinding")]
    bounce = [r for r in rows if r.get("likely_cowgirl_vertical_bounce")]
    forward = [r for r in rows if r.get("likely_cowgirl_forward_back_rock")]
    receiver = [r for r in rows if r.get("likely_receiver_response")]
    rejection_counts = Counter(reason for r in rows for reason in r.get("reject_reasons", []))
    shape_counts = Counter(r.get("trajectory_shape_classification") for r in rows)
    lines = [
        "# Cowgirl Candidate Score V4 Report",
        "",
        "V4 uses relative motion, trajectory shape, and rider/receiver scores. It is review triage only, not ML training truth.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Clean Cowgirl candidates v4: {len(clean)}",
        f"- Grinding subtype candidates: {len(grind)}",
        f"- Vertical bounce subtype candidates: {len(bounce)}",
        f"- Forward/back rock subtype candidates: {len(forward)}",
        f"- Receiver/body-response rejected candidates: {len(receiver)}",
        "",
        "## Rejection Reasons",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in rejection_counts.most_common()) if rejection_counts else lines.append("- None")
    lines.extend(["", "## Trajectory Shapes", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in shape_counts.most_common()) if shape_counts else lines.append("- None")
    lines.extend(["", "## Top V4 Candidates", ""])
    for row in rows[:30]:
        lines.append(
            f"- `{row.get('window_id')}` score={row.get('final_clean_cowgirl_score_v4')} "
            f"shape={row.get('trajectory_shape_classification')} grind={row.get('trajectory_grind_score')} "
            f"role={row.get('role_status')} safe={row.get('safe_for_learning')} reject={row.get('reject_reasons')}"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v5(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    semantic = [r for r in rows if r.get("semantic_cowgirl_candidate_v5")]
    generation = [r for r in rows if r.get("generation_candidate_v5")]
    broken = [r for r in rows if r.get("export_pose_validity") == "broken_pose"]
    low_intro = [r for r in rows if r.get("cowgirl_context_low_motion_intro")]
    receiver = [r for r in rows if r.get("likely_receiver_response")]
    validity_counts = Counter(r.get("export_pose_validity") for r in rows)
    lines = [
        "# Cowgirl Candidate Score V5 Report",
        "",
        "V5 separates semantic Cowgirl likelihood from export/pose/generation-template usability. These are audit scores, not labels.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Semantic Cowgirl candidates: {len(semantic)}",
        f"- Generation candidate scores above threshold: {len(generation)}",
        f"- Known broken-pose review items: {len(broken)}",
        f"- Cowgirl context / low-motion intro candidates: {len(low_intro)}",
        f"- Receiver/body-response excluded candidates: {len(receiver)}",
        "",
        "## Export Pose Validity",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in validity_counts.most_common()) if validity_counts else lines.append("- None")
    lines.extend(["", "## Top Semantic Cowgirl Candidates", ""])
    for row in sorted(rows, key=lambda r: float(r.get("final_semantic_cowgirl_score_v5") or 0.0), reverse=True)[:20]:
        lines.append(f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v5')} generation={row.get('final_generation_candidate_score_v5')} validity={row.get('export_pose_validity')} shape={row.get('trajectory_shape_classification')}")
    lines.extend(["", "## Top Generation Candidate Scores", ""])
    for row in sorted(rows, key=lambda r: float(r.get("final_generation_candidate_score_v5") or 0.0), reverse=True)[:20]:
        lines.append(f"- `{row.get('window_id')}` generation={row.get('final_generation_candidate_score_v5')} semantic={row.get('final_semantic_cowgirl_score_v5')} safe={row.get('generation_template_safety_score')} validity={row.get('export_pose_validity')}")
    lines.extend(["", "## Semantically Good But Not Generation-Safe", ""])
    flagged = [r for r in rows if r.get("semantically_good_but_not_generation_safe")]
    lines.extend(f"- `{r.get('window_id')}` semantic={r.get('final_semantic_cowgirl_score_v5')} validity={r.get('export_pose_validity')}" for r in flagged[:20]) if flagged else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v6(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    semantic = [r for r in rows if r.get("semantic_cowgirl_candidate_v6")]
    clean = [r for r in rows if r.get("clean_motion_candidate_v6")]
    generation = [r for r in rows if r.get("generation_candidate_v6")]
    invalid = [r for r in rows if r.get("semantically_cowgirl_but_controller_invalid")]
    context = [r for r in rows if r.get("cowgirl_context_low_motion_intro")]
    foot = [r for r in rows if r.get("foot_controller_outlier")]
    status_counts = Counter(r.get("controller_validity_status") for r in rows)
    lines = [
        "# Cowgirl Candidate Score V6 Report",
        "",
        "V6 separates semantic Cowgirl, Cowgirl context, clean motion, and generation/controller safety. These are audit scores, not training labels.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Semantic Cowgirl candidates: {len(semantic)}",
        f"- Clean motion candidates: {len(clean)}",
        f"- Generation-safe candidates: {len(generation)}",
        f"- Semantically Cowgirl but controller-invalid candidates: {len(invalid)}",
        f"- Cowgirl context / low-motion intro candidates: {len(context)}",
        f"- Foot/controller outlier candidates: {len(foot)}",
        "",
        "## Controller Validity Status",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in status_counts.most_common()) if status_counts else lines.append("- None")
    lines.extend(["", "## Top Semantic Cowgirl Candidates", ""])
    for row in sorted(rows, key=lambda r: float(r.get("final_semantic_cowgirl_score_v6") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v6')} "
            f"generation={row.get('final_generation_candidate_score_v6')} controller=`{row.get('controller_validity_status')}` "
            f"foot_outlier={row.get('foot_controller_outlier')}"
        )
    lines.extend(["", "## Top Generation-Safe Candidates", ""])
    for row in sorted(generation, key=lambda r: float(r.get("final_generation_candidate_score_v6") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` generation={row.get('final_generation_candidate_score_v6')} "
            f"semantic={row.get('final_semantic_cowgirl_score_v6')} controller={row.get('controller_validity_score')}"
        )
    lines.extend(["", "## Semantically Cowgirl But Controller Invalid", ""])
    for row in sorted(invalid, key=lambda r: float(r.get("final_semantic_cowgirl_score_v6") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v6')} "
            f"controller=`{row.get('controller_validity_status')}` foot_outlier={row.get('foot_controller_outlier')} "
            f"scene=`{row.get('source_scene_file')}`"
        )
    if not invalid:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v7(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    semantic = [r for r in rows if r.get("semantic_cowgirl_candidate_v7")]
    generation = [r for r in rows if r.get("semantic_cowgirl_generation_safe")]
    anchor_incomplete = [r for r in rows if r.get("semantic_cowgirl_anchor_incomplete")]
    controller_outlier = [r for r in rows if r.get("semantic_cowgirl_controller_outlier")]
    foot_missing = [r for r in rows if r.get("missing_foot_controllers") and r.get("semantic_cowgirl_candidate_v7")]
    knee_missing = [r for r in rows if r.get("missing_knee_controllers") and r.get("semantic_cowgirl_candidate_v7")]
    category_counts = Counter(r.get("cowgirl_v7_category") for r in rows)
    lines = [
        "# Cowgirl Candidate Score V7 Report",
        "",
        "V7 adds pose-anchor completeness. Static foot/knee anchors can be generation-critical even when semantics are correct.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Semantic Cowgirl total: {len(semantic)}",
        f"- Generation-safe Cowgirl total: {len(generation)}",
        f"- Anchor-incomplete Cowgirl total: {len(anchor_incomplete)}",
        f"- Controller-outlier Cowgirl total: {len(controller_outlier)}",
        f"- Foot-missing Cowgirl total: {len(foot_missing)}",
        f"- Knee-missing Cowgirl total: {len(knee_missing)}",
        "",
        "## V7 Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in category_counts.most_common()) if category_counts else lines.append("- None")
    lines.extend(["", "## Generation-Safe Cowgirl Examples", ""])
    for row in sorted(generation, key=lambda r: float(r.get("final_generation_candidate_score_v7") or 0.0), reverse=True)[:20]:
        lines.append(f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v7')} generation={row.get('final_generation_candidate_score_v7')} anchors={row.get('pose_anchor_completeness_score')}")
    lines.extend(["", "## Anchor-Incomplete / V8-001-002-009-Like Examples", ""])
    examples = sorted([*anchor_incomplete, *controller_outlier], key=lambda r: float(r.get("final_semantic_cowgirl_score_v7") or 0.0), reverse=True)
    for row in examples[:20]:
        lines.append(f"- `{row.get('window_id')}` category={row.get('cowgirl_v7_category')} semantic={row.get('final_semantic_cowgirl_score_v7')} missing={row.get('missing_required_anchor_controllers')}")
    if not examples:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v8(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    semantic = [r for r in rows if r.get("semantic_cowgirl_candidate_v8")]
    generation = [r for r in rows if r.get("semantic_cowgirl_generation_safe")]
    anchor_incomplete = [r for r in rows if r.get("semantic_cowgirl_anchor_incomplete")]
    orientation_invalid = [r for r in rows if r.get("semantic_cowgirl_orientation_invalid")]
    controller_outlier = [r for r in rows if r.get("semantic_cowgirl_controller_outlier")]
    standing = [r for r in rows if r.get("standing_gesture_false_positive")]
    category_counts = Counter(r.get("cowgirl_v8_category") for r in rows)
    orientation_counts = Counter(r.get("orientation_validity_status") for r in rows)
    lines = [
        "# Cowgirl Candidate Score V8 Report",
        "",
        "V8 adds controller orientation/twist validity. Orientation invalidity lowers generation safety, not semantic Cowgirl evidence.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Semantic Cowgirl total: {len(semantic)}",
        f"- Generation-safe Cowgirl total: {len(generation)}",
        f"- Anchor-incomplete Cowgirl total: {len(anchor_incomplete)}",
        f"- Orientation-invalid Cowgirl total: {len(orientation_invalid)}",
        f"- Controller-outlier Cowgirl total: {len(controller_outlier)}",
        f"- Standing/gesture false positives: {len(standing)}",
        "",
        "## V8 Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in category_counts.most_common()) if category_counts else lines.append("- None")
    lines.extend(["", "## Orientation Validity", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in orientation_counts.most_common()) if orientation_counts else lines.append("- None")
    lines.extend(["", "## Generation-Safe Cowgirl Examples", ""])
    for row in sorted(generation, key=lambda r: float(r.get("final_generation_candidate_score_v8") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v8')} "
            f"generation={row.get('final_generation_candidate_score_v8')} orientation={row.get('orientation_validity_status')}"
        )
    if not generation:
        lines.append("- None")
    lines.extend(["", "## Review-002-Like Orientation Invalid Examples", ""])
    for row in sorted(orientation_invalid, key=lambda r: float(r.get("final_semantic_cowgirl_score_v8") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v8')} "
            f"twisted={row.get('twisted_controller_names')} orientation={row.get('orientation_validity_score')}"
        )
    if not orientation_invalid:
        lines.append("- None")
    lines.extend(["", "## Review-004/005-Like Anchor Incomplete Examples", ""])
    for row in sorted(anchor_incomplete, key=lambda r: float(r.get("final_semantic_cowgirl_score_v8") or 0.0), reverse=True)[:20]:
        lines.append(f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v8')} missing={row.get('missing_required_anchor_controllers')}")
    if not anchor_incomplete:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v9(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    semantic = [r for r in rows if r.get("semantic_cowgirl_candidate_v9")]
    generation = [r for r in rows if r.get("semantic_cowgirl_generation_safe")]
    anchor_incomplete = [r for r in rows if r.get("semantic_cowgirl_anchor_incomplete")]
    orientation_invalid = [r for r in rows if r.get("semantic_cowgirl_orientation_invalid")]
    distance_invalid = [r for r in rows if r.get("semantic_cowgirl_distance_invalid")]
    pose_invalid = [r for r in rows if r.get("semantic_cowgirl_pose_invalid")]
    standing = [r for r in rows if r.get("standing_gesture_false_positive")]
    category_counts = Counter(r.get("cowgirl_v9_category") for r in rows)
    distance_counts = Counter(r.get("distance_validity_status") for r in rows)
    lines = [
        "# Cowgirl Candidate Score V9 Report",
        "",
        "V9 adds controller-distance validity. Distance invalidity lowers generation safety, not semantic Cowgirl evidence.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Semantic Cowgirl total: {len(semantic)}",
        f"- Generation-safe Cowgirl total: {len(generation)}",
        f"- Pose-invalid Cowgirl total: {len(pose_invalid)}",
        f"- Anchor-incomplete Cowgirl total: {len(anchor_incomplete)}",
        f"- Orientation-invalid Cowgirl total: {len(orientation_invalid)}",
        f"- Distance-invalid Cowgirl total: {len(distance_invalid)}",
        f"- Standing/gesture false positives: {len(standing)}",
        "",
        "## V9 Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in category_counts.most_common()) if category_counts else lines.append("- None")
    lines.extend(["", "## Distance Validity", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in distance_counts.most_common()) if distance_counts else lines.append("- None")
    lines.extend(["", "## Generation-Safe Cowgirl Examples", ""])
    for row in sorted(generation, key=lambda r: float(r.get("final_generation_candidate_score_v9") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v9')} "
            f"generation={row.get('final_generation_candidate_score_v9')} distance={row.get('distance_validity_status')}"
        )
    if not generation:
        lines.append("- None")
    lines.extend(["", "## Review-009-Like Distance Invalid Examples", ""])
    for row in sorted(distance_invalid, key=lambda r: float(r.get("final_semantic_cowgirl_score_v9") or 0.0), reverse=True)[:20]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v9')} "
            f"distance={row.get('distance_validity_score')} outliers={row.get('outlier_controller_names')}"
        )
    if not distance_invalid:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v10(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    semantic = [r for r in rows if r.get("semantic_cowgirl_candidate_v10")]
    generation = [r for r in rows if r.get("semantic_cowgirl_generation_safe")]
    core_missing = [r for r in rows if r.get("semantic_cowgirl_core_controller_missing")]
    traps = [r for r in rows if r.get("bj_oral_trap_negative")]
    arm = [r for r in rows if r.get("arm_stretch_outlier_flag")]
    category_counts = Counter(r.get("cowgirl_v10_category") for r in rows)
    lines = [
        "# Cowgirl Candidate Score V10 Report",
        "",
        "V10 adds core pelvis/hip controller requirements, BJ/oral trap blocking, and arm-stretch outlier blocking for generation-safe Cowgirl. These are audit scores, not labels.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Semantic Cowgirl total: {len(semantic)}",
        f"- Generation-safe Cowgirl total: {len(generation)}",
        f"- Core-controller missing Cowgirl total: {len(core_missing)}",
        f"- BJ/oral trap negatives: {len(traps)}",
        f"- Arm-stretch outliers: {len(arm)}",
        "",
        "## V10 Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in category_counts.most_common()) if category_counts else lines.append("- None")
    lines.extend(["", "## Top Generation-Safe Candidates", ""])
    for row in sorted(generation, key=lambda r: float(r.get("final_generation_candidate_score_v10") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v10')} "
            f"generation={row.get('final_generation_candidate_score_v10')} core={row.get('core_controller_gate')} "
            f"scene=`{row.get('source_scene_file')}`"
        )
    if not generation:
        lines.append("- None")
    lines.extend(["", "## Core Missing / Review-006-Like Examples", ""])
    for row in sorted(core_missing, key=lambda r: float(r.get("final_semantic_cowgirl_score_v10") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` semantic={row.get('final_semantic_cowgirl_score_v10')} "
            f"missing={row.get('missing_core_controllers')} arm_stretch={row.get('arm_stretch_outlier_flag')}"
        )
    if not core_missing:
        lines.append("- None")
    lines.extend(["", "## BJ/Oral Trap Candidates", ""])
    for row in sorted(traps, key=lambda r: float((r.get("bj_oral_trap_guard") or {}).get("bj_oral_trap_confidence") or 0.0), reverse=True)[:25]:
        guard = row.get("bj_oral_trap_guard") or {}
        lines.append(
            f"- `{row.get('window_id')}` confidence={guard.get('bj_oral_trap_confidence')} "
            f"bj={guard.get('bj_reference_score')} head={guard.get('head_reference_score')} scene=`{row.get('source_scene_file')}`"
        )
    if not traps:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v11(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    generation = [r for r in rows if r.get("cowgirl_v11_category") in {"semantic_cowgirl_generation_safe", "semantic_cowgirl_core_soft_fail_generation_safe"}]
    soft = [r for r in rows if r.get("semantic_cowgirl_core_soft_fail_generation_safe")]
    hard = [r for r in rows if r.get("semantic_cowgirl_core_hard_fail")]
    bj = [r for r in rows if r.get("semantic_family") == "bj_oral"]
    categories = Counter(r.get("cowgirl_v11_category") for r in rows)
    families = Counter(r.get("semantic_family") for r in rows)
    subtypes = Counter(r.get("cowgirl_subtype") for r in generation)
    lines = [
        "# Cowgirl Candidate Score V11 Report",
        "",
        "V11 calibrates core-controller soft fails and treats BJ/oral as a valid semantic family. BJ/oral candidates are excluded from Cowgirl and preserved for future BJ/oral datasets.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Generation-safe Cowgirl candidates: {len(generation)}",
        f"- Core soft-fail accepted candidates: {len(soft)}",
        f"- Core hard-fail rejected candidates: {len(hard)}",
        f"- BJ/oral candidates excluded from Cowgirl: {len(bj)}",
        f"- BJ/oral candidates preserved for future dataset: {sum(1 for r in bj if r.get('preserve_for_future_dataset'))}",
        "",
        "## Categories",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in categories.most_common()) if categories else lines.append("- None")
    lines.extend(["", "## Semantic Families", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common()) if families else lines.append("- None")
    lines.extend(["", "## Generation-Safe Cowgirl Subtypes", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in subtypes.most_common()) if subtypes else lines.append("- None")
    lines.extend(["", "## Review-007-Like Core Soft-Fail Recoveries", ""])
    for row in sorted(soft, key=lambda r: float(r.get("final_generation_candidate_score_v11") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` generation={row.get('final_generation_candidate_score_v11')} "
            f"reason={row.get('core_gate_override_reason')} scene=`{row.get('source_scene_file')}`"
        )
    if not soft:
        lines.append("- None")
    lines.extend(["", "## Review-010-Like BJ/Oral Preserved Candidates", ""])
    for row in sorted(bj, key=lambda r: float(r.get("bj_oral_confidence") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` confidence={row.get('bj_oral_confidence')} "
            f"excluded_from_cowgirl={row.get('excluded_from_cowgirl')} scene=`{row.get('source_scene_file')}`"
        )
    if not bj:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _num(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _bounded(value: Any) -> float:
    return float(np.clip(_num(value), 0.0, 1.0))


def _cowgirl_subtype(score: dict[str, Any], trajectory: dict[str, Any]) -> str:
    shape = str(trajectory.get("trajectory_shape_classification") or score.get("trajectory_shape_classification") or "")
    if score.get("likely_cowgirl_grinding") or "oval" in shape:
        return "oval_grind"
    if "circular" in shape:
        return "circular_grind"
    if score.get("likely_cowgirl_vertical_bounce") or "bounce" in shape:
        return "vertical_bounce"
    if score.get("likely_cowgirl_forward_back_rock") or "forward_back" in shape:
        return "forward_back_rock"
    return "unknown"

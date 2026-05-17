"""Separate delta import for newly added VaM scenes.

This run intentionally scans only the requested new-scene folder and writes
all artifacts under a separate output run. It does not train ML or promote
machine/audit labels to manual ground truth.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import html

import yaml

from vam_timeline_ai.audits.body_motion_quality import audit_body_motion_quality
from vam_timeline_ai.audits.controller_validity import audit_controller_validity
from vam_timeline_ai.audits.pose_anchor_completeness import audit_pose_anchor_completeness
from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.cowgirl.feature_extractor_v1 import extract_cowgirl_features_v1
from vam_timeline_ai.cowgirl.pair_feature_extractor import extract_pair_features_v0
from vam_timeline_ai.datasets.window_dataset import build_movement_windows
from vam_timeline_ai.features.partner_relative_features import extract_partner_relative_features_v0
from vam_timeline_ai.features.pose_features import extract_pose_features_v0
from vam_timeline_ai.features.relative_features import extract_relative_motion_features
from vam_timeline_ai.features.trajectory_shape import analyze_trajectory_shapes
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.motion.baker import extract_motion_samples
from vam_timeline_ai.motion.controller_mapping import discover_controller_map
from vam_timeline_ai.motion.data_audit import audit_baked_samples
from vam_timeline_ai.motion.relative_motion import build_relative_motion_windows
from vam_timeline_ai.motion.source_inventory import build_motion_source_index
from vam_timeline_ai.reports.new_scene_delta_report import compare_new_scenes_to_clean_v3
from vam_timeline_ai.semantics.clean_v3_v16_calibration import rebuild_clean_v3_semantic_actions_v2
from vam_timeline_ai.semantics.context_pairing import build_context_pair_candidates
from vam_timeline_ai.semantics.interaction_classifier import classify_interactions_v0
from vam_timeline_ai.semantics.pair_windows import build_pair_windows_v1
from vam_timeline_ai.semantics.pose_classifier import classify_poses_v0
from vam_timeline_ai.ui.review_ui import build_static_review_ui


RUN_DIRS = [
    "audits",
    "baked",
    "semantic",
    "features",
    "relative_motion",
    "pose_semantics",
    "interaction_semantics",
    "semantic_actions",
    "datasets",
    "generation",
    "labels",
    "reports",
    "review",
]


def run_new_scenes_delta_import(raw_dir: str | Path, base_run: str | Path, out_run: str | Path) -> dict[str, Any]:
    raw = Path(raw_dir)
    base = Path(base_run)
    out = Path(out_run)
    _create_structure(raw, base, out)

    sources = build_motion_source_index(raw, out / "semantic" / "motion_source_index.jsonl", out / "semantic" / "motion_source_index_report.md", recursive=True)
    samples = extract_motion_samples(out / "semantic" / "motion_source_index.jsonl", out / "baked" / "samples", out / "baked" / "motion_sample_index.jsonl", fps=60.0)
    _write_sample_report(samples, out / "baked" / "motion_sample_index_report.md")
    audit_baked_samples(out / "baked" / "motion_sample_index.jsonl", out / "audits" / "baked_sample_audit.jsonl", out / "audits" / "baked_sample_audit_report.md")
    discover_controller_map(out / "baked" / "motion_sample_index.jsonl", out / "semantic" / "controller_inventory.json", out / "semantic" / "controller_bodypart_map.json", out / "semantic" / "controller_bodypart_map_report.md")
    windows = build_movement_windows(out / "baked" / "motion_sample_index.jsonl", out / "semantic" / "movement_windows.jsonl")
    _write_window_report(windows, out / "semantic" / "movement_windows_report.md")
    extract_cowgirl_features_v1(
        out / "semantic" / "movement_windows.jsonl",
        out / "baked" / "motion_sample_index.jsonl",
        out / "semantic" / "controller_bodypart_map.json",
        out / "features" / "cowgirl_window_features_v1.jsonl",
        out / "features" / "cowgirl_window_features_v1.npz",
        out / "features" / "cowgirl_window_features_v1_report.md",
    )
    audit_body_motion_quality(
        out,
        out / "baked" / "motion_sample_index.jsonl",
        out / "features" / "cowgirl_window_features_v1.jsonl",
        out / "semantic" / "controller_bodypart_map.json",
        out / "audits" / "body_motion_quality.jsonl",
        out / "audits" / "body_motion_quality_report.md",
    )
    build_context_pair_candidates(out / "baked" / "motion_sample_index.jsonl", out / "semantic" / "context_pair_candidates.jsonl", out / "semantic" / "context_pair_candidates_report.md")
    build_pair_windows_v1(
        out / "semantic" / "context_pair_candidates.jsonl",
        out / "semantic" / "movement_windows.jsonl",
        out / "baked" / "motion_sample_index.jsonl",
        out / "semantic" / "pair_windows_v1.jsonl",
        out / "semantic" / "pair_windows_v1_report.md",
    )
    extract_pair_features_v0(
        out / "semantic" / "pair_windows_v1.jsonl",
        out / "baked" / "motion_sample_index.jsonl",
        out / "semantic" / "controller_bodypart_map.json",
        out / "features" / "cowgirl_pair_features_v0.jsonl",
        out / "features" / "cowgirl_pair_features_v0.npz",
        out / "features" / "cowgirl_pair_features_v0_report.md",
    )
    build_relative_motion_windows(
        out,
        out / "baked" / "motion_sample_index.jsonl",
        out / "semantic" / "movement_windows.jsonl",
        out / "semantic" / "controller_bodypart_map.json",
        out / "audits" / "body_motion_quality.jsonl",
        out / "relative_motion" / "windows",
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "relative_motion" / "relative_motion_report.md",
    )
    extract_relative_motion_features(
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "relative_motion" / "relative_motion_features.jsonl",
        out / "relative_motion" / "relative_motion_features.npz",
        out / "relative_motion" / "relative_motion_feature_report.md",
    )
    analyze_trajectory_shapes(
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "relative_motion" / "relative_motion_features.jsonl",
        out / "relative_motion" / "trajectory_shape_features.jsonl",
        out / "relative_motion" / "trajectory_shape_features.npz",
        out / "relative_motion" / "trajectory_shape_report.md",
    )
    audit_pose_anchor_completeness(
        out,
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "baked" / "motion_sample_index.jsonl",
        out / "semantic" / "controller_bodypart_map.json",
        out / "audits" / "body_motion_quality.jsonl",
        out / "audits" / "pose_anchor_completeness.jsonl",
        out / "audits" / "pose_anchor_completeness_report.md",
    )
    audit_controller_validity(
        out,
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "baked" / "motion_sample_index.jsonl",
        out / "semantic" / "controller_bodypart_map.json",
        out / "audits" / "controller_validity.jsonl",
        out / "audits" / "controller_validity_report.md",
        out / "audits" / "pose_anchor_completeness.jsonl",
        None,
    )
    extract_pose_features_v0(
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "audits" / "body_motion_quality.jsonl",
        out / "audits" / "pose_anchor_completeness.jsonl",
        out / "audits" / "controller_validity.jsonl",
        out / "pose_semantics" / "pose_features_v0.jsonl",
        out / "pose_semantics" / "pose_feature_report_v0.md",
    )
    classify_poses_v0(
        out / "pose_semantics" / "pose_features_v0.jsonl",
        None,
        None,
        out / "pose_semantics" / "pose_semantics_v0.jsonl",
        out / "pose_semantics" / "pose_semantics_report_v0.md",
    )
    extract_partner_relative_features_v0(
        out / "semantic" / "pair_windows_v1.jsonl",
        out / "features" / "cowgirl_pair_features_v0.jsonl",
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "pose_semantics" / "pose_semantics_v0.jsonl",
        out / "interaction_semantics" / "partner_relative_features_v0.jsonl",
        out / "interaction_semantics" / "partner_relative_feature_report_v0.md",
    )
    classify_interactions_v0(
        out / "interaction_semantics" / "partner_relative_features_v0.jsonl",
        out / "pose_semantics" / "pose_semantics_v0.jsonl",
        out / "semantic_actions" / "semantic_actions_v0.jsonl",
        out / "interaction_semantics" / "interaction_semantics_v0.jsonl",
        out / "interaction_semantics" / "interaction_semantics_report_v0.md",
    )
    _build_initial_semantic_actions(out)
    _write_initial_semantic_db(out)
    # Calibrated clean_v3 logic writes semantic_candidate_db_v2 and cowgirl_candidate_db_v7.
    calibrated = rebuild_clean_v3_semantic_actions_v2(out, None)
    _alias_calibrated_outputs(out)
    delta = compare_new_scenes_to_clean_v3(base, out, out / "reports" / "new_scene_delta_report.md")
    review = export_new_scene_review_batch(out, out / "audits" / "semantic_review_new_scenes_020", count=20)
    summary = _summary(raw, base, out, sources, samples, windows, calibrated, delta, review)
    dump_json(out / "reports" / "new_scenes_delta_import_summary.json", summary)
    _write_summary(out / "reports" / "new_scenes_delta_import_summary.md", summary)
    return summary


def export_new_scene_review_batch(run_dir: str | Path, out_dir: str | Path, count: int = 20) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context = _context(run)
    semantic = load_jsonl(run / "datasets" / "semantic_candidate_db_v2.jsonl")
    cowgirl = load_jsonl(run / "datasets" / "cowgirl_candidate_db_v7.jsonl")
    selected, selection_summary = _select_review_items(cowgirl, semantic, context, count)
    rows = [_review_row(idx, row, context) for idx, row in enumerate(selected, start=1)]
    write_jsonl(out / "semantic_review_010.jsonl", rows)
    _write_review_md(rows, out / "semantic_review_010.md")
    _write_review_html(rows, out / "semantic_review_010_index.html")
    _write_answer_sheet(rows, out / "semantic_review_010_answer_sheet.yaml")
    _write_selection_report(rows, selection_summary, out / "semantic_review_selection_report.md")
    package = build_vam_review_package(
        out / "semantic_review_010.jsonl",
        run,
        run,
        out / "vam_review_package",
        attempt_timeline_segments=True,
    )
    ui = build_static_review_ui(run, out, out / "review_ui_static")
    return {
        "status": "ok",
        "review_items": len(rows),
        "category_counts": dict(Counter(r.get("why_selected") for r in rows)),
        "selection": selection_summary,
        "vam_package": package,
        "static_review_ui": ui,
    }


def _create_structure(raw: Path, base: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in RUN_DIRS:
        (out / name).mkdir(parents=True, exist_ok=True)
    scene_count = len(list(raw.rglob("*.json"))) if raw.exists() else 0
    manifest = {
        "run_name": out.name,
        "source_folder": str(raw),
        "parent_reference_run": base.name,
        "parent_reference_run_path": str(base),
        "purpose": "new_scene_delta_import",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene_count_expected": 24,
        "scene_count_found_at_manifest": scene_count,
        "constraints": {
            "manual_labels_modified": False,
            "ml_training_performed": False,
            "source_world_coords_as_learning_targets": False,
            "person_root_tracks_allowed_as_generation_targets": False,
        },
    }
    dump_json(out / "run_manifest.json", manifest)


def _build_initial_semantic_actions(run: Path) -> list[dict[str, Any]]:
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    poses = {r.get("window_id"): r for r in load_jsonl(run / "pose_semantics" / "pose_semantics_v0.jsonl") if r.get("window_id")}
    rel = {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "relative_motion_features.jsonl") if r.get("window_id")}
    traj = {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "trajectory_shape_features.jsonl") if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(run / "audits" / "body_motion_quality.jsonl") if r.get("window_id")}
    interactions_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(run / "interaction_semantics" / "interaction_semantics_v0.jsonl"):
        interactions_by_window.setdefault(str(row.get("window_id")), []).append(row)
    rows = []
    for wid, window in windows.items():
        pose = poses.get(wid, {})
        interaction = _best_interaction(interactions_by_window.get(str(wid), []))
        row = _initial_action_row(window, pose, interaction, rel.get(wid, {}), traj.get(wid, {}), body.get(wid, {}))
        rows.append(row)
    write_jsonl(run / "semantic_actions" / "semantic_actions_v0.jsonl", rows)
    _write_initial_actions_report(rows, run / "semantic_actions" / "semantic_actions_report_v0.md")
    return rows


def _initial_action_row(window: dict[str, Any], pose: dict[str, Any], interaction: dict[str, Any], rel: dict[str, Any], traj: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    fv = rel.get("feature_values") or {}
    tv = traj.get("feature_values") or {}
    pose_family = str(pose.get("pose_family") or "unknown")
    body_quality = str(body.get("body_motion_quality") or "unknown")
    local_path = _num(fv.get("local_path_length"))
    energy = _num(fv.get("local_motion_energy"))
    grind = max(_num(fv.get("local_grind_score")), _num(tv.get("grind_pattern_score")))
    bounce = max(_num(fv.get("local_bounce_score")), _num(tv.get("bounce_pattern_score")))
    head_motion = _num(fv.get("head_relative_to_chest_motion"))
    hand_motion = _num(fv.get("hands_relative_to_chest_pelvis_head"))
    torso_motion = _num(fv.get("torso_relative_to_pelvis_motion"))
    semantic_family = "unknown"
    motion_family = "unknown"
    motion_subtype = _motion_subtype(fv, tv)
    conflicts: list[str] = []
    warnings = list(window.get("warnings") or [])
    if pose_family == "standing" or body.get("minimal_head_motion_only") or body.get("minimal_hand_jitter_only"):
        semantic_family = "hand_gesture"
        motion_family = "hand_gesture"
        motion_subtype = "standing_hand_head"
    elif head_motion > max(local_path, torso_motion, 0.15) * 2.0 and grind < 0.55:
        semantic_family = "bj_oral"
        motion_family = "bj_oral"
        motion_subtype = "bj_head_dominant_motion"
        warnings.append("Head/oral-domain motion candidate preserved as BJ/oral, excluded from Cowgirl.")
    elif pose_family in {"cowgirl", "kneeling_general"} and (grind >= 0.45 or bounce >= 0.45 or local_path >= 0.35):
        semantic_family = "cowgirl"
        motion_family = "cowgirl"
    elif "receiver" in body_quality or str(interaction.get("actor_role")) == "receiver":
        semantic_family = "receiver_response"
        motion_family = "receiver_response"
    if semantic_family == "cowgirl" and pose_family not in {"cowgirl", "kneeling_general"}:
        conflicts.append("cowgirl_motion_wrong_pose")
    if semantic_family == "unknown" and energy <= 0.01:
        conflicts.append("low_motion_or_unusable")
    semantic_score = max(grind, bounce, min(1.0, local_path / 1.1))
    pose_score = _num(pose.get("pose_confidence"))
    interaction_score = _num(interaction.get("interaction_confidence"))
    generation_safe = semantic_family == "cowgirl" and pose_family in {"cowgirl", "kneeling_general"} and semantic_score >= 0.45 and not conflicts
    return {
        "window_id": window.get("window_id"),
        "pair_window_id": interaction.get("pair_window_id"),
        "semantic_family": semantic_family,
        "actor_role": interaction.get("actor_role") or ("rider" if semantic_family == "cowgirl" else "unknown"),
        "partner_role": interaction.get("partner_role") or "unknown",
        "pose_family": pose_family,
        "pose_subtype": pose.get("pose_subtype") or "unknown",
        "motion_family": motion_family,
        "motion_subtype": motion_subtype,
        "partner_relation": interaction.get("partner_relation") or ["unknown"],
        "contact_support": interaction.get("support_context") or interaction.get("contact_support") or "unknown",
        "phase": "clean_motion" if semantic_family in {"cowgirl", "bj_oral"} else "gesture" if semantic_family == "hand_gesture" else "unknown",
        "generation_safe": bool(generation_safe),
        "semantic_score": round(semantic_score, 6),
        "pose_score": round(pose_score, 6),
        "motion_score": round(min(1.0, max(local_path / 1.1, energy / 0.12, grind, bounce)), 6),
        "interaction_score": round(interaction_score, 6),
        "consistency_score": round(max(0.0, min(1.0, (semantic_score + pose_score + interaction_score) / 3.0)), 6),
        "conflict_flags": _dedupe(conflicts),
        "warnings": _dedupe(warnings),
        "source_scene_file": window.get("source_scene_file"),
        "source_scene_path": window.get("source_scene_path"),
        "technical_atom_id": window.get("technical_atom_id"),
        "sample_id": window.get("sample_id"),
        "source_id": window.get("source_id"),
        "start_seconds": window.get("start_seconds"),
        "end_seconds": window.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds"),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _write_initial_semantic_db(run: Path) -> None:
    actions = load_jsonl(run / "semantic_actions" / "semantic_actions_v0.jsonl")
    rows = []
    for row in actions:
        family = str(row.get("semantic_family") or "unknown")
        rows.append({
            "candidate_id": f"semantic_action_delta_v0::{row.get('window_id')}",
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
            "warnings": row.get("warnings") or [],
            "preserve_for_future_dataset": family in {"cowgirl", "bj_oral", "hand_gesture", "head_gesture", "receiver_response"},
            "is_human_ground_truth": False,
            "is_training_label": False,
        })
    rows.sort(key=lambda r: (r.get("semantic_family") != "cowgirl", r.get("semantic_family") != "bj_oral", -_num(r.get("semantic_score"))))
    write_jsonl(run / "datasets" / "semantic_candidate_db_v0.jsonl", rows)
    _write_semantic_csv(rows, run / "datasets" / "semantic_candidate_db_v0.csv")
    _write_semantic_report(rows, run / "datasets" / "semantic_candidate_db_v0_report.md", "Semantic Candidate DB V0")
    cowgirl = []
    for row in rows:
        category = _initial_cowgirl_category(row)
        cowgirl.append({
            "candidate_id": f"cowgirl_delta_v0::{row.get('window_id')}",
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
            "partner_relation": row.get("partner_relation"),
            "contact_support": row.get("contact_support"),
            "generation_safe": bool(row.get("generation_safe")) and category == "cowgirl_clean_motion_generation_safe",
            "semantic_score": row.get("semantic_score"),
            "pose_score": row.get("pose_score"),
            "motion_score": row.get("motion_score"),
            "interaction_score": row.get("interaction_score"),
            "warnings": row.get("warnings") or [],
            "is_human_ground_truth": False,
            "is_training_label": False,
        })
    write_jsonl(run / "datasets" / "cowgirl_candidate_db_v0.jsonl", cowgirl)
    _write_cowgirl_csv(cowgirl, run / "datasets" / "cowgirl_candidate_db_v0.csv")
    _write_cowgirl_report(cowgirl, run / "datasets" / "cowgirl_candidate_db_v0_report.md", "Cowgirl Candidate DB V0")


def _alias_calibrated_outputs(run: Path) -> None:
    sem = load_jsonl(run / "datasets" / "semantic_candidate_db_v2.jsonl")
    cow = load_jsonl(run / "datasets" / "cowgirl_candidate_db_v7.jsonl")
    # The requested v0 paths remain the delta-run public entrypoint; the rows
    # include calibrated clean-motion gate fields when v2/v7 are available.
    write_jsonl(run / "datasets" / "semantic_candidate_db_v0.jsonl", sem)
    _write_semantic_csv(sem, run / "datasets" / "semantic_candidate_db_v0.csv")
    _write_semantic_report(sem, run / "datasets" / "semantic_candidate_db_v0_report.md", "Semantic Candidate DB V0 (Calibrated Alias)")
    write_jsonl(run / "datasets" / "cowgirl_candidate_db_v0.jsonl", cow)
    _write_cowgirl_csv(cow, run / "datasets" / "cowgirl_candidate_db_v0.csv")
    _write_cowgirl_report(cow, run / "datasets" / "cowgirl_candidate_db_v0_report.md", "Cowgirl Candidate DB V0 (Calibrated Alias)")


def _select_review_items(cowgirl: list[dict[str, Any]], semantic: list[dict[str, Any]], context: dict[str, Any], count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cowgirl:
        rec = dict(row)
        rec["_pool"] = "cowgirl_candidate_db"
        pools[str(rec.get("category"))].append(rec)
    for row in semantic:
        family = row.get("semantic_family")
        category = (
            "bj_oral_candidate"
            if family == "bj_oral"
            else "receiver_response"
            if family == "receiver_response"
            else "standing_hand_head"
            if family in {"hand_gesture", "head_gesture"} or row.get("pose_family") == "standing"
            else "unknown_or_unusable"
            if family == "unknown"
            else None
        )
        if category:
            rec = dict(row)
            rec["category"] = category
            rec["_pool"] = "semantic_candidate_db"
            pools[category].append(rec)
    for rows in pools.values():
        rows.sort(key=lambda r: (-_num(r.get("semantic_score")), -_num(r.get("hip_motion_strength")), -_num(r.get("motion_score"))))
    quotas = [
        (["cowgirl_clean_motion_generation_safe", "cowgirl_clean_motion_low_confidence_short", "cowgirl_missing_partner_context"], 6),
        (["cowgirl_pose_context_low_motion", "cowgirl_transition_setup", "cowgirl_intro_alignment", "cowgirl_no_clear_hip_motion"], 4),
        (["not_cowgirl_bj_oral", "bj_oral_candidate"], 3),
        (["not_cowgirl_receiver_response", "receiver_response"], 2),
        (["not_cowgirl_standing_hand_head", "standing_hand_head"], 2),
        (["unknown_or_unusable"], 3),
    ]
    selected: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    sample_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    near_groups: set[str] = set()
    rejected = Counter()

    def add_from(categories: list[str], limit: int) -> None:
        added = 0
        for category in categories:
            for row in pools.get(category, []):
                if len(selected) >= count or added >= limit:
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
        if category in {"cowgirl_pose_context_low_motion", "unknown_or_unusable"} and _num(row.get("motion_score")) < 0.05:
            low_count = sum(1 for r in selected if r.get("category") == category and _num(r.get("motion_score")) < 0.05)
            if low_count >= 1:
                rejected["low_motion_spam"] += 1
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
        return True

    for categories, limit in quotas:
        add_from(categories, limit)
    if len(selected) < count:
        for category in list(pools):
            add_from([category], count - len(selected))
            if len(selected) >= count:
                break
    return selected[:count], {
        "selected": len(selected[:count]),
        "rejected_by_rule": dict(rejected),
        "scene_counts": dict(scene_counts),
        "sample_count": len(sample_counts),
        "category_counts": dict(Counter(r.get("category") for r in selected[:count])),
    }


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
        "why_selected": row.get("category") or row.get("_pool"),
        "phase": row.get("phase"),
        "clean_motion_gate": row.get("clean_motion_gate"),
        "clean_motion_gate_reason": row.get("clean_motion_gate_reason"),
        "hip_motion_strength": row.get("hip_motion_strength"),
        "pelvis_trajectory_strength": row.get("pelvis_trajectory_strength"),
        "pelvis_cycle_count": row.get("pelvis_cycle_count"),
        "motion_duration_confidence": row.get("motion_duration_confidence"),
        "semantic_score": row.get("semantic_score"),
        "pose_score": row.get("pose_score"),
        "motion_score": row.get("motion_score"),
        "interaction_score": row.get("interaction_score"),
        "contact_support_confidence": row.get("contact_support_confidence"),
        "rider_above_partner_score": None,
        "pelvis_alignment_score": row.get("partner_relative_alignment_score") or row.get("interaction_score"),
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


def _context(run: Path) -> dict[str, Any]:
    return {
        "windows": {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")},
    }


def _write_sample_report(rows: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(r.get("bake_status") for r in rows)
    lines = ["# Motion Sample Index Report", "", f"- Rows: {len(rows)}"]
    lines.extend(f"- `{k}`: {v}" for k, v in counts.most_common())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_window_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Movement Windows Report", "", f"- Windows: {len(rows)}", f"- Include for ML flag count: {sum(1 for r in rows if r.get('include_for_ml'))}", "- These are technical windows, not training labels."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_initial_actions_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Semantic Actions Report V0", "", "Initial delta-run semantic actions from pose, relative motion, and interaction proxies. Not manual truth.", "", f"- Rows: {len(rows)}", "", "## Families", ""]
    lines.extend(_counter_lines(Counter(r.get("semantic_family") for r in rows)))
    lines.extend(["", "## Phases", ""])
    lines.extend(_counter_lines(Counter(r.get("phase") for r in rows)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_semantic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["candidate_id", "window_id", "semantic_family", "pose_family", "pose_subtype", "motion_subtype", "phase", "clean_motion_gate", "contact_support", "generation_safe", "semantic_score", "pose_score", "motion_score", "interaction_score"]
    _write_csv(rows, path, fields)


def _write_cowgirl_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["candidate_id", "window_id", "category", "semantic_family", "pose_family", "pose_subtype", "motion_subtype", "phase", "clean_motion_gate", "contact_support", "generation_safe", "semantic_score", "pose_score", "motion_score", "interaction_score"]
    _write_csv(rows, path, fields)


def _write_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_semantic_report(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    lines = [f"# {title}", "", "Candidate inventory for review/calibration only. Not training truth.", "", f"- Rows: {len(rows)}", "", "## Families", ""]
    lines.extend(_counter_lines(Counter(r.get("semantic_family") for r in rows)))
    lines.extend(["", "## Generation Safe", "", f"- `true`: {sum(1 for r in rows if r.get('generation_safe'))}", f"- `false`: {sum(1 for r in rows if not r.get('generation_safe'))}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cowgirl_report(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    lines = [f"# {title}", "", "Cowgirl-focused candidate inventory for review/calibration only. Not training truth.", "", f"- Rows: {len(rows)}", "", "## Categories", ""]
    lines.extend(_counter_lines(Counter(r.get("category") for r in rows)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_md(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# New Scenes Semantic Review 020", "", "Audit review only; not manual training labels.", ""]
    for row in rows:
        lines.extend([
            f"## {row['review_id']}",
            "",
            f"- Scene: `{row.get('source_scene_file')}`",
            f"- Actor: `{row.get('technical_atom_id')}`",
            f"- Time: `{row.get('start_seconds')}` to `{row.get('end_seconds')}`",
            f"- Family: `{row.get('semantic_family')}`",
            f"- Pose: `{(row.get('pose_semantics') or {}).get('family')}` / `{(row.get('pose_semantics') or {}).get('subtype')}`",
            f"- Motion: `{(row.get('motion_semantics') or {}).get('subtype')}` / `{row.get('phase')}`",
            f"- Clean gate: `{row.get('clean_motion_gate')}`",
            f"- Contact/support: `{row.get('contact_support')}`",
            f"- Generation safe: `{row.get('generation_safe')}`",
            f"- Why selected: `{row.get('why_selected')}`",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_html(rows: list[dict[str, Any]], path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"<section><h2>{html.escape(row['review_id'])}</h2>"
            f"<p><strong>Scene:</strong> <code>{html.escape(str(row.get('source_scene_file')))}</code> "
            f"<strong>Time:</strong> <code>{html.escape(str(row.get('start_seconds')))}-{html.escape(str(row.get('end_seconds')))}s</code></p>"
            f"<p><strong>Family:</strong> <code>{html.escape(str(row.get('semantic_family')))}</code> "
            f"<strong>Why:</strong> <code>{html.escape(str(row.get('why_selected')))}</code></p>"
            f"<p><strong>Pose:</strong> <code>{html.escape(str((row.get('pose_semantics') or {}).get('family')))} / {html.escape(str((row.get('pose_semantics') or {}).get('subtype')))}</code></p>"
            f"<p><strong>Motion:</strong> <code>{html.escape(str((row.get('motion_semantics') or {}).get('subtype')))} / {html.escape(str(row.get('phase')))}</code> "
            f"<strong>Gate:</strong> <code>{html.escape(str(row.get('clean_motion_gate')))}</code></p>"
            f"</section>"
        )
    text = "<!doctype html><meta charset='utf-8'><title>New Scenes Review</title><style>body{font-family:system-ui;margin:1.5rem;background:#f7f7f5}section{background:white;border:1px solid #ddd;border-radius:6px;padding:1rem;margin:1rem 0}code{background:#f0f0ea;padding:.1rem .25rem;border-radius:4px}</style><h1>New Scenes Semantic Review 020</h1>" + "\n".join(cards)
    path.write_text(text, encoding="utf-8")


def _write_answer_sheet(rows: list[dict[str, Any]], path: Path) -> None:
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


def _write_selection_report(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = ["# New Scenes Review Selection Report", "", f"- Items: {len(rows)}", f"- Summary: `{summary}`", "", "## Categories", ""]
    lines.extend(_counter_lines(Counter(r.get("why_selected") for r in rows)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(raw: Path, base: Path, out: Path, sources: list[dict[str, Any]], samples: list[dict[str, Any]], windows: list[dict[str, Any]], calibrated: dict[str, Any], delta: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    semantic = load_jsonl(out / "datasets" / "semantic_candidate_db_v2.jsonl")
    cowgirl = load_jsonl(out / "datasets" / "cowgirl_candidate_db_v7.jsonl")
    raw_scene_files = len(list(raw.rglob("*.json"))) if raw.exists() else 0
    return {
        "status": "ok",
        "raw_dir": str(raw),
        "base_run": str(base),
        "out_run": str(out),
        "raw_scene_files_found": raw_scene_files,
        "scene_count_found": len({r.get("source_scene_path") or r.get("source_scene_file") for r in sources if r.get("source_scene_file")}),
        "motion_sources": len(sources),
        "samples": len(samples),
        "baked_ok": sum(1 for r in samples if r.get("bake_status") == "ok"),
        "movement_windows": len(windows),
        "semantic_family_counts": dict(Counter(r.get("semantic_family") for r in semantic)),
        "cowgirl_category_counts": dict(Counter(r.get("category") for r in cowgirl)),
        "clean_motion_gate_counts": calibrated.get("clean_motion_gate_counts"),
        "delta_report": delta,
        "review": review,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# New Scenes Delta Import Summary",
        "",
        f"- Raw dir: `{summary['raw_dir']}`",
        f"- Base run: `{summary['base_run']}`",
        f"- Out run: `{summary['out_run']}`",
        f"- Raw scene JSON files found: {summary.get('raw_scene_files_found', summary['scene_count_found'])}",
        f"- Scenes with motion sources: {summary['scene_count_found']}",
        f"- Motion sources: {summary['motion_sources']}",
        f"- Baked samples: {summary['samples']} / ok {summary['baked_ok']}",
        f"- Movement windows: {summary['movement_windows']}",
        "",
        "## Semantic Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in (summary.get("semantic_family_counts") or {}).items())
    lines.extend(["", "## Cowgirl Categories", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in (summary.get("cowgirl_category_counts") or {}).items())
    lines.extend(["", "## Review", "", f"- Items: {(summary.get('review') or {}).get('review_items')}", f"- Package: `{(((summary.get('review') or {}).get('vam_package') or {}).get('out_dir'))}`", f"- Static UI: `{(((summary.get('review') or {}).get('static_review_ui') or {}).get('index'))}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _initial_cowgirl_category(row: dict[str, Any]) -> str:
    family = row.get("semantic_family")
    if family == "bj_oral":
        return "not_cowgirl_bj_oral"
    if family == "receiver_response":
        return "not_cowgirl_receiver_response"
    if family in {"hand_gesture", "head_gesture"} or row.get("pose_family") == "standing":
        return "not_cowgirl_standing_hand_head"
    if family != "cowgirl":
        return "unknown_or_unusable"
    if row.get("generation_safe"):
        return "cowgirl_clean_motion_generation_safe"
    return "cowgirl_missing_partner_context"


def _motion_subtype(fv: dict[str, Any], tv: dict[str, Any]) -> str:
    scores = {
        "oval_grind": max(_num(fv.get("local_grind_score")), _num(tv.get("oval_path_score")), _num(tv.get("grind_pattern_score"))),
        "vertical_bounce": max(_num(fv.get("local_bounce_score")), _num(tv.get("bounce_pattern_score"))),
        "forward_back_rock": _num(tv.get("forward_back_rock_pattern_score")),
    }
    best, score = max(scores.items(), key=lambda item: item[1])
    return best if score >= 0.35 else "unknown"


def _best_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return sorted(rows, key=lambda r: _num(r.get("interaction_confidence")), reverse=True)[0]


def _near_duplicate_group(row: dict[str, Any], window: dict[str, Any]) -> str:
    start = round(_num(window.get("start_seconds") or row.get("start_seconds")) / 2.0) * 2
    return "|".join([
        str(row.get("source_scene_file") or window.get("source_scene_file") or ""),
        str(row.get("technical_actor_id") or window.get("technical_atom_id") or ""),
        str(row.get("sample_id") or window.get("sample_id") or ""),
        str(start),
        str(row.get("pose_subtype") or ""),
        str(row.get("motion_subtype") or ""),
        str(row.get("phase") or ""),
    ])


def _counter_lines(counter: Counter[Any], limit: int | None = None) -> list[str]:
    items = counter.most_common(limit)
    return [f"- `{k}`: {v}" for k, v in items] if items else ["- None"]


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

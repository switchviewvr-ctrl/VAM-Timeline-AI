from pathlib import Path

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.clean_v3_v16_calibration import (
    export_semantic_review_v17,
    ingest_v16_human_findings,
    rebuild_clean_v3_semantic_actions_v2,
)


def _base_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "data" / "runs" / "clean_v3"
    review = run / "audits" / "semantic_review_010_v16"
    for rel in [
        "semantic_actions",
        "semantic",
        "relative_motion",
        "pose_semantics",
        "audits",
        "interaction_semantics",
        "datasets",
        "audits/semantic_review_010_v16/vam_review_package",
    ]:
        (run / rel).mkdir(parents=True, exist_ok=True)
    return run, review


def _write_v16_review(review: Path) -> None:
    rows = []
    for idx in range(1, 11):
        rows.append(
            {
                "review_id": f"review_{idx:03d}",
                "window_id": f"w{idx}",
                "pair_window_id": f"p{idx}",
                "semantic_family": "cowgirl",
                "pose_semantics": {"family": "cowgirl", "subtype": "cowgirl_kneeling"},
                "motion_semantics": {"subtype": "grinding", "phase": "clean_motion"},
                "phase": "clean_motion",
                "why_selected": "fixture",
                "generation_safe": idx <= 4,
                "is_human_ground_truth": False,
                "is_training_label": False,
            }
        )
    write_jsonl(review / "semantic_review_010.jsonl", rows)
    answer = {"reviews": {row["review_id"]: {"notes": ""} for row in rows}}
    (review / "vam_review_package" / "vam_review_answer_sheet.yaml").write_text(
        yaml.safe_dump(answer, sort_keys=False),
        encoding="utf-8",
    )
    (review / "semantic_review_010_answer_sheet.yaml").write_text(yaml.safe_dump(answer, sort_keys=False), encoding="utf-8")


def _action(wid: str, family: str = "cowgirl", pose: str = "cowgirl", phase: str = "clean_motion") -> dict:
    return {
        "window_id": wid,
        "pair_window_id": f"p_{wid}",
        "semantic_family": family,
        "motion_family": family,
        "pose_family": pose,
        "pose_subtype": "cowgirl_kneeling" if pose == "cowgirl" else "standing_upright",
        "motion_subtype": "grinding",
        "partner_relation": ["rider_above_partner"] if family == "cowgirl" else ["unknown"],
        "contact_support": "unknown",
        "phase": phase,
        "generation_safe": family == "cowgirl",
        "semantic_score": 0.8,
        "pose_score": 0.7,
        "motion_score": 0.8,
        "interaction_score": 0.7,
        "consistency_score": 0.7,
        "conflict_flags": [],
        "warnings": [],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _rel(wid: str, path: float, energy: float, velocity: float, cycles: float, duration: float = 4.0) -> tuple[dict, dict]:
    rel = {
        "window_id": wid,
        "duration_seconds": duration,
        "feature_values": {
            "relative_pelvis_vertical_amplitude": max(0.02, path * 0.08),
            "relative_pelvis_forward_back_amplitude": max(0.03, path * 0.12),
            "relative_pelvis_lateral_amplitude": max(0.03, path * 0.12),
            "local_path_length": path,
            "local_motion_energy": energy,
            "local_velocity_mean": velocity,
            "local_grind_score": min(0.9, path),
            "local_bounce_score": 0.2,
            "torso_relative_to_pelvis_motion": 0.2,
            "limb_motion_relative_energy": 0.02,
            "root_world_motion_removed": 1.0,
        },
    }
    traj = {
        "window_id": wid,
        "feature_values": {
            "transition_path_score": 0.1,
            "cycle_count_estimate": cycles,
            "grind_pattern_score": min(0.9, path),
            "bounce_pattern_score": 0.2,
        },
    }
    return rel, traj


def _write_rebuild_inputs(run: Path) -> None:
    actions = [
        _action("standing", pose="standing"),
        _action("transition"),
        _action("short"),
        _action("clean"),
        _action("bj", family="bj_oral", pose="cowgirl"),
    ]
    rels = []
    trajs = []
    specs = {
        "standing": (0.6, 0.06, 0.2, 2.0),
        "transition": (0.34, 0.012, 0.08, 0.5),
        "short": (0.48, 0.04, 0.16, 1.5),
        "clean": (1.2, 0.15, 0.38, 4.0),
        "bj": (1.0, 0.1, 0.3, 4.0),
    }
    for wid, spec in specs.items():
        rel, traj = _rel(wid, *spec)
        rels.append(rel)
        trajs.append(traj)
    windows = [
        {
            "window_id": row["window_id"],
            "sample_id": f"s_{row['window_id']}",
            "source_id": f"src_{row['window_id']}",
            "source_scene_file": f"scene_{row['window_id']}.json",
            "source_scene_path": str(run / f"scene_{row['window_id']}.json"),
            "technical_atom_id": "Person",
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "duration_seconds": 4.0,
        }
        for row in actions
    ]
    poses = [
        {"window_id": "standing", "standing_score": 0.9},
        {"window_id": "transition", "standing_score": 0.0},
        {"window_id": "short", "standing_score": 0.0},
        {"window_id": "clean", "standing_score": 0.0},
        {"window_id": "bj", "standing_score": 0.0},
    ]
    write_jsonl(run / "semantic_actions" / "semantic_actions_v1.jsonl", actions)
    write_jsonl(run / "semantic" / "movement_windows.jsonl", windows)
    write_jsonl(run / "relative_motion" / "relative_motion_features.jsonl", rels)
    write_jsonl(run / "relative_motion" / "trajectory_shape_features.jsonl", trajs)
    write_jsonl(run / "pose_semantics" / "pose_features_v0.jsonl", poses)
    write_jsonl(run / "audits" / "pose_anchor_completeness.jsonl", [{"window_id": row["window_id"], "missing_required_anchor_controllers": []} for row in actions])
    write_jsonl(run / "audits" / "controller_validity.jsonl", [{"window_id": row["window_id"], "missing_required_anchor_controllers": []} for row in actions])
    write_jsonl(run / "interaction_semantics" / "partner_relative_features_v0.jsonl", [])


def test_ingest_v16_findings_stores_audit_only_notes(tmp_path):
    run, review = _base_run(tmp_path)
    _write_v16_review(review)

    summary = ingest_v16_human_findings(review)
    data = yaml.safe_load((review / "semantic_review_010_human_notes.yaml").read_text(encoding="utf-8"))
    answer = yaml.safe_load((review / "vam_review_package" / "vam_review_answer_sheet.yaml").read_text(encoding="utf-8"))

    assert summary["review_items"] == 10
    assert data["do_not_merge_into_manual_labels"] is True
    assert data["reviews"]["review_001"]["semantic_family"] == "standing_hand_head_gesture"
    assert "cowgirl_short_motion_window" in data["reviews"]["review_009"]["actual_labels"]
    assert answer["reviews"]["review_004"]["actual_labels"][-1] == "not_clean_motion"
    assert not (run / "labels" / "manual_labels.yaml").exists()


def test_rebuild_v2_clean_motion_gate_categories(tmp_path):
    run, review = _base_run(tmp_path)
    _write_rebuild_inputs(run)

    summary = rebuild_clean_v3_semantic_actions_v2(run, review)
    actions = {r["window_id"]: r for r in load_jsonl(run / "semantic_actions" / "semantic_actions_v2.jsonl")}
    cowgirl = {r["window_id"]: r for r in load_jsonl(run / "datasets" / "cowgirl_candidate_db_v7.jsonl")}

    assert summary["semantic_actions"] == 5
    assert actions["standing"]["clean_motion_gate"] == "fail_standing"
    assert cowgirl["standing"]["category"] == "not_cowgirl_standing_hand_head"
    assert actions["transition"]["clean_motion_gate"] in {"fail_no_hip_motion", "fail_low_motion"}
    assert cowgirl["transition"]["category"] in {"cowgirl_no_clear_hip_motion", "cowgirl_pose_context_low_motion", "cowgirl_transition_setup"}
    assert actions["short"]["clean_motion_gate"] == "soft_pass_short"
    assert cowgirl["short"]["category"] == "cowgirl_clean_motion_low_confidence_short"
    assert actions["clean"]["clean_motion_gate"] == "pass"
    assert cowgirl["clean"]["category"] == "cowgirl_clean_motion_generation_safe"
    assert cowgirl["bj"]["category"] == "not_cowgirl_bj_oral"
    assert summary["standing_leakage_count"] == 0
    assert not (run / "labels" / "manual_labels.yaml").exists()


def test_export_v17_caps_and_static_ui(tmp_path):
    run, review = _base_run(tmp_path)
    out = run / "audits" / "semantic_review_010_v17"
    windows = []
    cowgirl = []
    semantic = []
    categories = [
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_low_confidence_short",
        "cowgirl_pose_context_low_motion",
        "cowgirl_transition_setup",
        "not_cowgirl_standing_hand_head",
        "not_cowgirl_bj_oral",
        "unknown_or_unusable",
    ]
    for idx, category in enumerate(categories, start=1):
        wid = f"sel{idx}"
        family = "cowgirl"
        pose = "cowgirl"
        if "standing" in category:
            family = "hand_gesture"
            pose = "standing"
        elif "bj_oral" in category:
            family = "bj_oral"
        elif category == "unknown_or_unusable":
            family = "unknown"
        rec = {
            "window_id": wid,
            "sample_id": f"sample{idx}",
            "source_scene_file": f"scene_{idx}.json",
            "source_scene_path": str(run / f"scene_{idx}.json"),
            "technical_actor_id": "Person",
            "category": category,
            "semantic_family": family,
            "pose_family": pose,
            "pose_subtype": "cowgirl_kneeling" if pose == "cowgirl" else "standing_upright",
            "motion_subtype": "grinding",
            "phase": "clean_motion",
            "clean_motion_gate": "pass" if category == "cowgirl_clean_motion_generation_safe" else "soft_pass_short" if "short" in category else "fail_no_hip_motion",
            "clean_motion_gate_reason": "fixture",
            "hip_motion_strength": 0.8,
            "pelvis_trajectory_strength": 0.8,
            "pelvis_cycle_count": 3,
            "motion_duration_confidence": 1.0,
            "contact_support": "unknown_contact",
            "generation_safe": category == "cowgirl_clean_motion_generation_safe",
            "semantic_score": 1.0 - idx * 0.01,
            "clean_motion_score": 0.8,
            "partner_relation": ["rider_above_partner"] if family == "cowgirl" else ["unknown"],
        }
        cowgirl.append(rec)
        semantic.append(rec)
        windows.append(
            {
                "window_id": wid,
                "sample_id": f"sample{idx}",
                "source_id": f"src{idx}",
                "source_scene_file": f"scene_{idx}.json",
                "source_scene_path": str(run / f"scene_{idx}.json"),
                "technical_atom_id": "Person",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "duration_seconds": 4.0,
            }
        )
    write_jsonl(run / "datasets" / "cowgirl_candidate_db_v7.jsonl", cowgirl)
    write_jsonl(run / "datasets" / "semantic_candidate_db_v2.jsonl", semantic)
    write_jsonl(run / "semantic" / "movement_windows.jsonl", windows)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "old"}])

    summary = export_semantic_review_v17(run, out, previous_review=review, build_vam_package=True)
    selected = load_jsonl(out / "semantic_review_010.jsonl")

    assert summary["review_items"] == 10
    assert sum(1 for row in selected if row["why_selected"] == "not_cowgirl_standing_hand_head") == 1
    assert sum(1 for row in selected if row["why_selected"] == "cowgirl_pose_context_low_motion") == 1
    assert max(__import__("collections").Counter(row["source_scene_file"] for row in selected).values()) <= 2
    assert all("clean_motion_gate" in row for row in selected)
    assert (out / "vam_review_package" / "vam_review_index.html").exists()
    assert (out / "review_ui_static" / "index.html").exists()
    assert "clean_motion_gate" in (out / "review_ui_static" / "review_data.js").read_text(encoding="utf-8")

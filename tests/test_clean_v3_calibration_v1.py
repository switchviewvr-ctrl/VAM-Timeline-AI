from pathlib import Path

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.clean_v3_calibration_v1 import (
    export_semantic_review_v16,
    ingest_v15_human_findings,
    rebuild_clean_v3_semantic_actions_v1,
)


def _base_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "data" / "runs" / "clean_v3"
    review = run / "audits" / "semantic_review_010_v15"
    for rel in [
        "semantic_actions",
        "semantic",
        "relative_motion",
        "audits",
        "interaction_semantics",
        "datasets",
        "audits/semantic_review_010_v15/vam_review_package",
    ]:
        (run / rel).mkdir(parents=True, exist_ok=True)
    return run, review


def _write_v15_review(review: Path) -> None:
    rows = []
    for idx in range(1, 11):
        rows.append(
            {
                "review_id": f"review_{idx:03d}",
                "window_id": f"w{idx}",
                "pair_window_id": f"p{idx}" if idx <= 4 else None,
                "semantic_family": "cowgirl",
                "pose_semantics": {"family": "cowgirl", "subtype": "cowgirl_kneeling"},
                "motion_semantics": {"subtype": "grinding"},
                "partner_relation": ["rider_above_partner"],
                "contact_support": "hands_on_partner_chest" if idx in {2, 3} else "unknown",
                "generation_safe": idx <= 4,
                "why_selected": "fixture",
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


def _write_calibration_inputs(run: Path) -> None:
    actions = []
    windows = []
    rel = []
    traj = []
    anchors = []
    controllers = []
    partners = []
    for idx in range(1, 11):
        wid = f"w{idx}"
        actions.append(
            {
                "window_id": wid,
                "pair_window_id": f"p{idx}" if idx <= 4 else None,
                "semantic_family": "cowgirl" if idx not in {6} else "unknown",
                "motion_family": "cowgirl" if idx not in {6} else "unknown",
                "pose_family": "cowgirl" if idx <= 4 else "unknown",
                "pose_subtype": "cowgirl_kneeling" if idx <= 4 else "unknown",
                "motion_subtype": "grinding",
                "partner_relation": ["rider_above_partner"] if idx <= 4 else ["unknown"],
                "contact_support": "hands_on_partner_chest" if idx in {2, 3} else "unknown",
                "phase": "clean_motion",
                "generation_safe": idx <= 4,
                "semantic_score": 0.8,
                "pose_score": 0.7,
                "motion_score": 0.8,
                "interaction_score": 0.7 if idx <= 4 else 0.0,
                "consistency_score": 0.7,
                "conflict_flags": [],
                "warnings": [],
                "is_human_ground_truth": False,
                "is_training_label": False,
            }
        )
        windows.append(
            {
                "window_id": wid,
                "sample_id": f"s{idx}",
                "source_id": f"src{idx}",
                "source_scene_file": f"scene_{idx}.json",
                "source_scene_path": str(run / f"scene_{idx}.json"),
                "technical_atom_id": "Person",
                "start_seconds": float(idx),
                "end_seconds": float(idx + 4),
                "duration_seconds": 4.0,
            }
        )
        low = idx >= 7
        rel.append(
            {
                "window_id": wid,
                "feature_values": {
                    "local_path_length": 0.2 if low else 1.5,
                    "local_velocity_mean": 0.02 if low else 0.4,
                    "local_motion_energy": 0.002 if low else 0.2,
                    "local_grind_score": 0.2 if low else 0.8,
                    "local_bounce_score": 0.1 if low else 0.4,
                    "root_world_motion_removed": 0.0 if low else 1.0,
                    "torso_relative_to_pelvis_motion": 0.0 if low else 0.2,
                    "limb_motion_relative_energy": 0.02,
                },
            }
        )
        traj.append({"window_id": wid, "feature_values": {"transition_path_score": 0.1, "cycle_count_estimate": 3.0}})
        anchors.append({"window_id": wid, "missing_required_anchor_controllers": []})
        controllers.append({"window_id": wid, "missing_required_anchor_controllers": []})
        if idx <= 4:
            partners.append(
                {
                    "window_id": wid,
                    "pair_window_id": f"p{idx}",
                    "hands_on_partner_chest_score": 0.55,
                    "hands_on_partner_hips_score": 0.48,
                    "hands_on_floor_or_bed_proxy": 0.0,
                    "partner_context_confidence": 0.8,
                    "pelvis_alignment_score": 0.7,
                }
            )
    write_jsonl(run / "semantic_actions" / "semantic_actions_v0.jsonl", actions)
    write_jsonl(run / "semantic" / "movement_windows.jsonl", windows)
    write_jsonl(run / "relative_motion" / "relative_motion_features.jsonl", rel)
    write_jsonl(run / "relative_motion" / "trajectory_shape_features.jsonl", traj)
    write_jsonl(run / "audits" / "pose_anchor_completeness.jsonl", anchors)
    write_jsonl(run / "audits" / "controller_validity.jsonl", controllers)
    write_jsonl(run / "interaction_semantics" / "partner_relative_features_v0.jsonl", partners)


def test_ingest_v15_findings_stores_audit_only_notes(tmp_path):
    run, review = _base_run(tmp_path)
    _write_v15_review(review)

    summary = ingest_v15_human_findings(review)
    data = yaml.safe_load((review / "semantic_review_010_human_notes.yaml").read_text(encoding="utf-8"))
    answer = yaml.safe_load((review / "vam_review_package" / "vam_review_answer_sheet.yaml").read_text(encoding="utf-8"))

    assert summary["review_items"] == 10
    assert data["do_not_merge_into_manual_labels"] is True
    assert data["reviews"]["review_003"]["semantic_family"] == "bj_oral"
    assert answer["reviews"]["review_010"]["actual_labels"][-1] == "repeated_duplicate_review_selection"
    assert not (run / "labels" / "manual_labels.yaml").exists()


def test_rebuild_calibrates_low_motion_intro_bj_and_standing(tmp_path):
    run, review = _base_run(tmp_path)
    _write_v15_review(review)
    _write_calibration_inputs(run)
    ingest_v15_human_findings(review)

    summary = rebuild_clean_v3_semantic_actions_v1(run, review)
    actions = {r["window_id"]: r for r in load_jsonl(run / "semantic_actions" / "semantic_actions_v1.jsonl")}
    cowgirl = load_jsonl(run / "datasets" / "cowgirl_candidate_db_v6.jsonl")
    categories = {r["window_id"]: r["category"] for r in cowgirl}

    assert summary["semantic_actions"] == 10
    assert actions["w2"]["phase"] == "intro_alignment"
    assert actions["w2"]["generation_safe"] is False
    assert actions["w3"]["semantic_family"] == "bj_oral"
    assert actions["w5"]["semantic_family"] == "hand_gesture"
    assert actions["w7"]["phase"] == "low_motion_hold"
    assert categories["w7"] == "cowgirl_pose_context_low_motion"
    assert categories["w3"] == "not_cowgirl_bj_oral"
    assert categories["w5"] == "not_cowgirl_standing_gesture"


def test_export_v16_applies_duplicate_caps_and_builds_package(tmp_path):
    run, review = _base_run(tmp_path)
    out = run / "audits" / "semantic_review_010_v16"
    rows = []
    semantic = []
    windows = []
    categories = [
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_pose_context_low_motion",
        "cowgirl_pose_context_low_motion",
        "cowgirl_intro_alignment",
        "cowgirl_ambiguous_partner_contact",
        "bj_oral_motion",
        "standing_hand_head_gesture",
        "receiver_response",
        "unknown_or_unusable",
    ]
    for idx, category in enumerate(categories, start=1):
        wid = f"sel{idx}"
        scene = f"scene_{idx}.json"
        family = "cowgirl"
        if category == "bj_oral_motion":
            family = "bj_oral"
        elif category == "standing_hand_head_gesture":
            family = "hand_gesture"
        elif category == "receiver_response":
            family = "receiver_response"
        elif category == "unknown_or_unusable":
            family = "unknown"
        rec = {
            "window_id": wid,
            "sample_id": f"sample{idx}",
            "source_scene_file": scene,
            "source_scene_path": str(run / scene),
            "technical_actor_id": "Person",
            "category": category if category.startswith("cowgirl") else "not_cowgirl_" + family,
            "semantic_family": family,
            "pose_family": "cowgirl",
            "pose_subtype": "cowgirl_kneeling",
            "motion_subtype": "grinding",
            "phase": "low_motion_hold" if "low_motion" in category else "clean_motion",
            "contact_support": "ambiguous_partner_contact" if "ambiguous" in category else "possible_partner_contact",
            "generation_safe": category == "cowgirl_clean_motion_generation_safe",
            "semantic_score": 1.0 - idx * 0.01,
            "clean_motion_score": 0.8,
            "contact_support_confidence": 0.6,
            "lower_body_anchor_stability": 0.8,
            "partner_relation": ["rider_above_partner"],
        }
        if category.startswith("cowgirl"):
            rows.append(rec)
        semantic.append(rec)
        windows.append(
            {
                "window_id": wid,
                "sample_id": f"sample{idx}",
                "source_id": f"src{idx}",
                "source_scene_file": scene,
                "source_scene_path": str(run / scene),
                "technical_atom_id": "Person",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "duration_seconds": 4.0,
            }
        )
    write_jsonl(run / "datasets" / "cowgirl_candidate_db_v6.jsonl", rows)
    write_jsonl(run / "datasets" / "semantic_candidate_db_v1.jsonl", semantic)
    write_jsonl(run / "semantic" / "movement_windows.jsonl", windows)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "old"}])

    summary = export_semantic_review_v16(run, out, previous_review=review, build_vam_package=True)
    selected = load_jsonl(out / "semantic_review_010.jsonl")

    assert summary["review_items"] == 10
    assert sum(1 for row in selected if row["phase"] == "low_motion_hold") == 1
    assert max(__import__("collections").Counter(row["source_scene_file"] for row in selected).values()) <= 2
    assert len({row["sample_id"] for row in selected}) == 10
    assert (out / "vam_review_package" / "vam_review_index.html").exists()
    assert not (run / "labels" / "manual_labels.yaml").exists()

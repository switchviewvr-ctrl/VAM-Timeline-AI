import json
from pathlib import Path

import numpy as np

from vam_timeline_ai.audits.body_motion_quality import body_motion_quality_for_feature
from vam_timeline_ai.audits.semantic_review import _filter_safe_export_controllers
from vam_timeline_ai.audits.cowgirl_core_controller_requirements import cowgirl_core_controller_requirements_for_window
from vam_timeline_ai.datasets.cowgirl_candidate_database import build_cowgirl_candidate_db_v3
from vam_timeline_ai.datasets.semantic_candidate_database import build_semantic_candidate_db_v0
from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.references.handmade_import import parse_reference_filename
from vam_timeline_ai.references.handmade_parser import classify_timeline_target
from vam_timeline_ai.references.reference_matcher import compare_wild_to_handmade_references
from vam_timeline_ai.semantics.bj_oral_domain_classifier import bj_oral_domain_for_window
from vam_timeline_ai.semantics.domain_guards import evaluate_domain_guards
from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_window, score_window_v3, score_window_v11
from vam_timeline_ai.semantics.machine_label_proposals import _window_proposals
from vam_timeline_ai.semantics.motion_phase_classifier import classify_motion_phase
from vam_timeline_ai.semantics.rider_receiver_discrimination import score_window_role


def test_body_motion_quality_detects_root_only_synthetic_sample():
    row = {
        "window_id": "w1",
        "feature_values": {
            "pelvis_movement_energy": 1.0,
            "left_hand_motion_energy": 0.0,
            "right_hand_motion_energy": 0.0,
            "head_motion_energy": 0.0,
        },
    }
    sample = {"controller_names": ["control"]}
    result = body_motion_quality_for_feature(row, sample, {"control": {"body_part": "root"}})
    assert result["body_motion_quality"] == "controller_only_whole_person_motion"


def test_body_motion_quality_detects_multi_bodypart_motion():
    row = {
        "window_id": "w2",
        "feature_values": {
            "pelvis_movement_energy": 0.4,
            "left_hand_motion_energy": 0.2,
            "right_hand_motion_energy": 0.2,
            "head_motion_energy": 0.1,
            "knee_motion_energy_left": 0.1,
        },
    }
    sample = {"controller_names": ["hipControl", "lHandControl", "rHandControl", "headControl", "lKneeControl"]}
    cmap = {name: {"body_part": part} for name, part in [("hipControl", "hip"), ("lHandControl", "left_hand"), ("rHandControl", "right_hand"), ("headControl", "head"), ("lKneeControl", "left_knee")]}
    result = body_motion_quality_for_feature(row, sample, cmap)
    assert result["body_motion_quality"] == "good_body_motion"


def test_transition_classifier_separates_transition_from_clean_motion():
    row = {"feature_values": {"pelvis_movement_energy": 0.1, "irregular_rhythm_score_proxy": 0.9, "pelvis_acceleration_peak_count": 7}}
    phase = classify_motion_phase(row, {"body_motion_quality": "good_body_motion", "moving_bodypart_count": 3})
    assert phase["motion_phase_candidate"] == "transition_adjustment_candidate"


def test_domain_guard_triggers_on_head_dominant_motion():
    row = {"feature_values": {"head_motion_energy": 1.0, "pelvis_movement_energy": 0.01}}
    guard = evaluate_domain_guards(row, {"body_motion_quality": "partial_body_motion"})
    assert "possible_non_cowgirl_head_dominant_motion" in guard["domain_guard_audit_labels"]


def test_filename_label_parsing_for_handmade_references():
    assert parse_reference_filename("female_cowgirl_hard_vertical.json")["label_family"] == "cowgirl"
    assert parse_reference_filename("female bj_basic_deep.json")["label_family"] == "bj"
    assert parse_reference_filename("female_cowgirl_realign.json")["is_transition_or_realign"] is True


def test_timeline_export_strips_root_tracks():
    positions = np.zeros((3, 2, 3), dtype=np.float32)
    rotations = np.zeros((3, 2, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    pos, rot, names, safety = _filter_safe_export_controllers(positions, rotations, ["control", "hipControl"])
    assert names == ["hipControl"]
    assert pos.shape[1] == 1
    assert safety["stripped_atom_root_count"] == 1
    assert classify_timeline_target("control") == "disallowed_person_atom_or_root"


def test_machine_proposals_downgrade_root_only_motion():
    row = {"window_id": "w", "sample_id": "s", "feature_values": {"pelvis_movement_energy": 1.0, "left_hand_motion_energy": 0.0, "right_hand_motion_energy": 0.0, "head_motion_energy": 0.0}}
    thresholds = {key: {} for key in ["p20", "p35", "p40", "p50", "p60", "p70", "p75", "p80", "p85", "p90"]}
    proposals = _window_proposals(row, {}, {}, thresholds)
    labels = {p["label"] for p in proposals}
    assert "root_only_motion_false_positive" in labels or "controller_only_whole_person_motion" in labels
    assert "cowgirl_vertical_bounce" not in labels


def test_reference_matcher_does_not_mark_filename_labels_as_wild_truth(tmp_path):
    wild_features = tmp_path / "wild.jsonl"
    body = tmp_path / "body.jsonl"
    handmade = tmp_path / "handmade.jsonl"
    signatures = tmp_path / "signatures.json"
    out = tmp_path / "matches.jsonl"
    report = tmp_path / "report.md"
    wild_features.write_text(json.dumps({"window_id": "w", "feature_values": {"head_motion_energy": 1.0, "pelvis_movement_energy": 0.01}}) + "\n", encoding="utf-8")
    body.write_text(json.dumps({"window_id": "w", "body_motion_quality": "partial_body_motion"}) + "\n", encoding="utf-8")
    handmade.write_text("", encoding="utf-8")
    signatures.write_text(json.dumps({"families": {"head": {"feature_medians": {"head_motion_energy": 1.0, "pelvis_movement_energy": 0.0}}, "cowgirl": {"feature_medians": {"head_motion_energy": 0.0, "pelvis_movement_energy": 1.0}}}}), encoding="utf-8")
    rows = compare_wild_to_handmade_references(wild_features, body, handmade, signatures, out, report)
    assert rows[0]["recommended_review_status"] == "likely_not_cowgirl_head_or_bj"
    assert rows[0]["is_human_ground_truth"] is False


def test_static_or_micro_motion_detected_from_tiny_synthetic_window(tmp_path):
    npz = tmp_path / "tiny.npz"
    times = np.asarray([0.0, 1.0], dtype=np.float32)
    positions = np.zeros((2, 1, 3), dtype=np.float32)
    positions[1, 0, 1] = 0.0001
    rotations = np.zeros((2, 1, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    np.savez_compressed(npz, times=times, positions=positions, rotations=rotations, controller_names=np.asarray(["headControl"], dtype=object))
    row = {"window_id": "w", "sample_id": "s", "feature_values": {"head_motion_energy": 0.000001, "pelvis_movement_energy": 0.0}}
    sample = {"sample_id": "s", "baked_npz_path": str(npz), "controller_names": ["headControl"]}
    window = {"frame_start": 0, "frame_end": 2}
    result = body_motion_quality_for_feature(row, sample, {"headControl": {"body_part": "head"}}, window, tmp_path)
    assert result["body_motion_quality"] == "static_or_micro_motion"
    assert result["minimal_head_motion_only"] is True


def test_minimal_hand_jitter_detected(tmp_path):
    npz = tmp_path / "hand.npz"
    positions = np.zeros((3, 1, 3), dtype=np.float32)
    positions[1, 0, 0] = 0.0002
    rotations = np.zeros((3, 1, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    np.savez_compressed(npz, times=np.asarray([0.0, 0.5, 1.0], dtype=np.float32), positions=positions, rotations=rotations, controller_names=np.asarray(["lHandControl"], dtype=object))
    row = {"window_id": "w", "sample_id": "s", "feature_values": {"left_hand_motion_energy": 0.000001, "pelvis_movement_energy": 0.0}}
    sample = {"sample_id": "s", "baked_npz_path": str(npz), "controller_names": ["lHandControl"]}
    result = body_motion_quality_for_feature(row, sample, {"lHandControl": {"body_part": "left_hand"}}, {"frame_start": 0, "frame_end": 3}, tmp_path)
    assert result["minimal_hand_jitter_only"] is True
    assert result["body_motion_quality"] == "static_or_micro_motion"


def test_clean_cowgirl_scoring_rejects_static_micro_and_prefers_longer_windows():
    feature = {"window_id": "w", "feature_values": {"pelvis_total_position_range": 0.2, "pelvis_movement_energy": 0.1}}
    match = {"cowgirl_reference_score": 0.8, "recommended_review_status": "likely_cowgirl_candidate"}
    static_body = {"body_motion_quality": "static_or_micro_motion", "static_or_micro_motion": True, "micro_motion_score": 1.0, "active_bodypart_count_above_threshold": 0}
    rejected = score_window(feature, static_body, match, {"duration_seconds": 8.0})
    assert rejected["clean_cowgirl_candidate"] is False
    assert "static_or_micro_motion" in rejected["reject_reasons"]
    good_body = {"body_motion_quality": "good_body_motion", "static_or_micro_motion": False, "micro_motion_score": 0.0, "active_bodypart_count_above_threshold": 3}
    short = score_window(feature, good_body, match, {"duration_seconds": 2.0})
    long = score_window(feature, good_body, match, {"duration_seconds": 4.0})
    assert long["duration_score"] > short["duration_score"]
    assert "too_short_for_semantic_judgment" in short["reject_reasons"]
    assert long["clean_cowgirl_candidate"] is True


def test_rider_receiver_scoring_identifies_active_rider_vs_receiver_response():
    active_feature = {
        "window_id": "active",
        "feature_values": {
            "pelvis_total_position_range": 0.22,
            "pelvis_movement_energy": 0.08,
            "torso_motion_energy": 0.04,
            "left_hand_motion_energy": 0.03,
            "right_hand_motion_energy": 0.02,
            "knee_motion_energy_left": 0.02,
            "steady_rhythm_score_proxy": 0.8,
        },
    }
    active_body = {"body_motion_quality": "good_body_motion", "active_bodypart_count_above_threshold": 4, "moving_bodypart_count": 4}
    active = score_window_role(active_feature, active_body, {"cowgirl_reference_score": 0.75}, [{"active_support": 0.95, "receiver_support": 0.0, "passive_support": 0.0, "below_other_score": 0.0, "other_active_confidence": 0.0}])
    assert active["rider_receiver_status"] == "likely_active_rider"

    receiver_feature = {
        "window_id": "receiver",
        "feature_values": {
            "pelvis_total_position_range": 0.16,
            "pelvis_movement_energy": 0.04,
            "torso_motion_energy": 0.0,
            "left_hand_motion_energy": 0.0,
            "right_hand_motion_energy": 0.0,
            "knee_motion_energy_left": 0.0,
        },
    }
    receiver_body = {"body_motion_quality": "partial_body_motion", "active_bodypart_count_above_threshold": 1, "moving_bodypart_count": 1}
    receiver = score_window_role(receiver_feature, receiver_body, {"cowgirl_reference_score": 0.6}, [{"active_support": 0.0, "receiver_support": 0.9, "passive_support": 0.7, "below_other_score": 1.0, "other_active_confidence": 0.9}])
    assert receiver["rider_receiver_status"] == "likely_receiver_body_response"
    assert receiver["receiver_body_response_score"] > receiver["active_rider_score"]


def test_hip_only_motion_is_not_enough_for_active_rider_without_pair_context():
    feature = {"window_id": "hip", "feature_values": {"pelvis_total_position_range": 0.25, "pelvis_movement_energy": 0.08}}
    body = {"body_motion_quality": "partial_body_motion", "active_bodypart_count_above_threshold": 1, "moving_bodypart_count": 1}
    result = score_window_role(feature, body, {"cowgirl_reference_score": 0.7}, [])
    assert result["rider_receiver_status"] == "insufficient_pair_context"
    assert result["active_rider_score"] < 0.5
    assert any("No pair context" in warning for warning in result["warnings"])


def test_cowgirl_candidate_v3_penalizes_receiver_body_response():
    feature = {
        "window_id": "w",
        "feature_values": {
            "pelvis_total_position_range": 0.2,
            "pelvis_movement_energy": 0.08,
            "pelvis_circularity_score_proxy": 0.8,
            "pelvis_grind_score_proxy": 0.75,
            "pelvis_forward_back_amplitude": 0.12,
            "pelvis_lateral_amplitude": 0.11,
            "pelvis_vertical_amplitude": 0.02,
        },
    }
    body = {"body_motion_quality": "good_body_motion", "static_or_micro_motion": False, "micro_motion_score": 0.0, "active_bodypart_count_above_threshold": 3, "moving_bodypart_count": 3}
    match = {"cowgirl_reference_score": 0.8, "recommended_review_status": "likely_cowgirl_candidate"}
    rider = {"rider_receiver_status": "likely_receiver_body_response", "active_rider_score": 0.2, "receiver_body_response_score": 0.9}
    scored = score_window_v3(feature, body, match, {"duration_seconds": 8.0}, rider)
    assert scored["clean_cowgirl_rider_candidate_v3"] is False
    assert scored["likely_receiver_false_positive"] is True
    assert "likely_receiver_body_response" in scored["reject_reasons"]


def test_grinding_subtype_is_valid_cowgirl_subtype():
    feature = {
        "window_id": "grind",
        "feature_values": {
            "pelvis_total_position_range": 0.2,
            "pelvis_movement_energy": 0.08,
            "pelvis_circularity_score_proxy": 0.9,
            "pelvis_grind_score_proxy": 0.85,
            "pelvis_forward_back_amplitude": 0.12,
            "pelvis_lateral_amplitude": 0.11,
            "pelvis_vertical_amplitude": 0.01,
            "steady_rhythm_score_proxy": 0.75,
        },
    }
    body = {"body_motion_quality": "good_body_motion", "static_or_micro_motion": False, "micro_motion_score": 0.0, "active_bodypart_count_above_threshold": 3, "moving_bodypart_count": 3}
    match = {"cowgirl_reference_score": 0.8, "recommended_review_status": "likely_cowgirl_candidate"}
    rider = {"rider_receiver_status": "likely_active_rider", "active_rider_score": 0.85, "receiver_body_response_score": 0.05}
    scored = score_window_v3(feature, body, match, {"duration_seconds": 8.0}, rider)
    assert scored["likely_grinding_subtype"] is True
    assert scored["clean_cowgirl_rider_candidate_v3"] is True


def test_core_gate_soft_fail_can_be_overridden_when_motion_evidence_is_strong():
    row = {"window_id": "soft", "controllers": ["hipControl", "chestControl"], "bodyparts": ["hip", "chest"], "core_motion_amplitude": 0.08}
    body = {"body_motion_quality": "good_body_motion", "active_bodypart_count_above_threshold": 3, "moving_bodypart_count": 3}
    anchor = {"generation_pose_anchor_safe": True, "generation_pose_anchor_status": "complete"}
    result = cowgirl_core_controller_requirements_for_window(row, body, anchor)
    assert result["core_gate_status"] == "soft_fail"
    assert result["core_gate_can_be_overridden"] is True

    scored = score_window_v11(
        {"window_id": "soft", "feature_values": {"pelvis_total_position_range": 0.25, "pelvis_movement_energy": 0.1}},
        body,
        {"cowgirl_relative_score": 0.8, "cowgirl_grind_trajectory_score": 0.4},
        {"feature_values": {"safe_for_learning": True}},
        {"trajectory_shape_classification": "forward_back_rock"},
        {"window_id": "soft", "duration_seconds": 8.0},
        {"rider_receiver_status": "likely_active_rider", "active_rider_score": 0.9, "receiver_body_response_score": 0.0},
        {"generation_template_safe": True},
        {"controller_validity_score": 1.0, "controller_validity_status": "valid"},
        {"pose_anchor_completeness_score": 1.0, "generation_pose_anchor_safe": True},
        {"orientation_validity_score": 1.0, "orientation_validity_status": "valid"},
        {"controller_distance_validity_score": 1.0, "controller_distance_validity_status": "valid"},
        result,
        {},
    )
    assert scored["cowgirl_v11_category"] == "semantic_cowgirl_core_soft_fail_generation_safe"
    assert scored["semantic_family"] == "cowgirl"


def test_core_gate_hard_fail_rejects_hand_head_only_cases():
    result = cowgirl_core_controller_requirements_for_window(
        {"window_id": "hands", "controllers": ["lHandControl", "headControl"], "bodyparts": ["left_hand", "head"], "core_motion_amplitude": None},
        {"body_motion_quality": "partial_body_motion", "active_bodypart_count_above_threshold": 2, "moving_bodypart_count": 2},
        {},
    )
    assert result["core_gate_status"] == "hard_fail"
    assert result["core_gate_can_be_overridden"] is False


def test_bj_oral_domain_classifier_preserves_valid_family_outside_cowgirl():
    row = bj_oral_domain_for_window(
        {"window_id": "bj", "feature_values": {"head_relative_to_chest_motion": 0.5, "relative_pelvis_vertical_amplitude": 0.01, "limb_motion_relative_energy": 0.02}},
        {"window_id": "bj", "trajectory_shape_classification": "head_motion"},
        {"window_id": "bj", "bj_relative_score": 0.7, "head_relative_score": 0.65, "cowgirl_relative_score": 0.2},
        {"window_id": "bj", "core_gate_status": "hard_fail", "has_knee_controls": True},
    )
    assert row["semantic_family"] == "bj_oral"
    assert row["excluded_from_cowgirl"] is True
    assert row["preserve_for_future_dataset"] is True
    assert row["bj_oral_motion_candidate"] is True


def test_cowgirl_v11_excludes_bj_oral_but_preserves_family():
    scored = score_window_v11(
        {"window_id": "bj", "feature_values": {"pelvis_total_position_range": 0.05, "pelvis_movement_energy": 0.01}},
        {"body_motion_quality": "partial_body_motion", "active_bodypart_count_above_threshold": 2, "moving_bodypart_count": 2},
        {"cowgirl_relative_score": 0.3},
        {"feature_values": {"safe_for_learning": True}},
        {"trajectory_shape_classification": "head_motion"},
        {"window_id": "bj", "duration_seconds": 6.0},
        {"rider_receiver_status": "role_unclear", "active_rider_score": 0.2, "receiver_body_response_score": 0.1},
        {},
        {"controller_validity_score": 1.0, "controller_validity_status": "valid"},
        {"pose_anchor_completeness_score": 1.0, "generation_pose_anchor_safe": True},
        {"orientation_validity_score": 1.0, "orientation_validity_status": "valid"},
        {"controller_distance_validity_score": 1.0, "controller_distance_validity_status": "valid"},
        {"core_gate_status": "hard_fail", "generation_safe_core_controller_gate": False, "missing_core_controllers": ["hipControl_or_pelvisControl"]},
        {"semantic_family": "bj_oral", "bj_oral_motion_candidate": True, "bj_oral_generation_candidate": True, "bj_oral_confidence": 0.8},
    )
    assert scored["cowgirl_v11_category"] == "not_cowgirl_bj_oral"
    assert scored["semantic_family"] == "bj_oral"
    assert scored["excluded_from_cowgirl"] is True
    assert scored["preserve_for_future_dataset"] is True
    assert scored["final_generation_candidate_score_v11"] == 0.0


def test_candidate_db_v3_and_global_semantic_db_preserve_families(tmp_path):
    run = tmp_path / "data" / "runs" / "clean_v2"
    (run / "semantic").mkdir(parents=True)
    (run / "datasets").mkdir(parents=True)
    write_jsonl(run / "semantic" / "movement_windows.jsonl", [
        {"window_id": "cow", "sample_id": "s1", "source_scene_file": "a.json", "duration_seconds": 8.0},
        {"window_id": "bj", "sample_id": "s2", "source_scene_file": "b.json", "duration_seconds": 8.0},
    ])
    scores = tmp_path / "scores.jsonl"
    rel = tmp_path / "rel.jsonl"
    traj = tmp_path / "traj.jsonl"
    body = tmp_path / "body.jsonl"
    anchors = tmp_path / "anchors.jsonl"
    ctrl = tmp_path / "ctrl.jsonl"
    orient = tmp_path / "orient.jsonl"
    dist = tmp_path / "dist.jsonl"
    core = tmp_path / "core.jsonl"
    bj = tmp_path / "bj.jsonl"
    write_jsonl(scores, [
        {"window_id": "cow", "cowgirl_v11_category": "semantic_cowgirl_generation_safe", "semantic_family": "cowgirl", "final_semantic_cowgirl_score_v11": 0.9, "final_clean_motion_score_v11": 0.8, "final_generation_candidate_score_v11": 0.7, "cowgirl_subtype": "riding", "semantic_cowgirl_generation_safe": True},
        {"window_id": "bj", "cowgirl_v11_category": "not_cowgirl_bj_oral", "semantic_family": "bj_oral", "final_semantic_cowgirl_score_v11": 0.1, "final_clean_motion_score_v11": 0.1, "final_generation_candidate_score_v11": 0.0, "not_cowgirl_bj_oral": True, "excluded_from_cowgirl": True, "preserve_for_future_dataset": True, "bj_oral_confidence": 0.75},
    ])
    for path in [rel, traj, body, anchors, ctrl, orient, dist, core]:
        write_jsonl(path, [{"window_id": "cow"}, {"window_id": "bj"}])
    write_jsonl(bj, [{"window_id": "bj", "semantic_family": "bj_oral", "bj_oral_motion_candidate": True, "bj_oral_confidence": 0.75, "excluded_from_cowgirl": True, "preserve_for_future_dataset": True}])
    db = build_cowgirl_candidate_db_v3(run, scores, rel, traj, body, anchors, ctrl, orient, dist, core, bj, tmp_path / "db.jsonl", tmp_path / "db.csv", tmp_path / "db.md")
    assert {row["semantic_family"] for row in db} == {"cowgirl", "bj_oral"}
    assert any(row["category"] == "not_cowgirl_bj_oral" for row in db)
    semantic = build_semantic_candidate_db_v0(run, tmp_path / "db.jsonl", bj, rel, traj, tmp_path / "semantic.jsonl", tmp_path / "semantic.csv", tmp_path / "semantic.md")
    assert {"cowgirl", "bj_oral"} <= {row["semantic_family"] for row in semantic}
    assert (run / "datasets" / "larger_review_batch_plan.md").exists()

import json
from pathlib import Path

import numpy as np

from vam_timeline_ai.audits.body_motion_quality import body_motion_quality_for_feature
from vam_timeline_ai.audits.semantic_review import _filter_safe_export_controllers
from vam_timeline_ai.references.handmade_import import parse_reference_filename
from vam_timeline_ai.references.handmade_parser import classify_timeline_target
from vam_timeline_ai.references.reference_matcher import compare_wild_to_handmade_references
from vam_timeline_ai.semantics.domain_guards import evaluate_domain_guards
from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_window
from vam_timeline_ai.semantics.machine_label_proposals import _window_proposals
from vam_timeline_ai.semantics.motion_phase_classifier import classify_motion_phase


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

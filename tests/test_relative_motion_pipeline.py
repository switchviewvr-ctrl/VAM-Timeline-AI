from pathlib import Path

import numpy as np

from vam_timeline_ai.audits.semantic_review import export_semantic_review_010
from vam_timeline_ai.features.relative_features import relative_features_from_arrays
from vam_timeline_ai.features.trajectory_shape import trajectory_shape_for_points
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.motion.coordinate_spaces import can_use_for_final_export, is_allowed_body_controller_track, is_disallowed_world_or_root_track
from vam_timeline_ai.motion.relative_motion import build_relative_motion_window_row
from vam_timeline_ai.references.relative_matcher import compare_relative_wild_to_handmade
from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_window_v4


def _times(n=121):
    return np.arange(n, dtype=np.float32) / 60.0


def _oval(n=121, offset=(0.0, 0.0, 0.0)):
    t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = np.cos(t) * 0.20
    pts[:, 2] = np.sin(t) * 0.12
    pts[:, 1] = np.sin(t * 2) * 0.02
    return pts + np.asarray(offset, dtype=np.float32)


def test_coordinate_classifier_disallows_root_and_allows_body_controller():
    assert is_disallowed_world_or_root_track("Person")
    assert is_disallowed_world_or_root_track("rootControl")
    assert is_disallowed_world_or_root_track("eyeTargetControl")
    assert is_allowed_body_controller_track("hipControl")
    assert can_use_for_final_export("hipControl")


def test_relative_motion_delta_removes_absolute_world_offset(tmp_path):
    n = 61
    times = _times(n)
    names = ["hipControl", "chestControl"]
    base_path = _oval(n)
    positions_a = np.stack([base_path, base_path + np.array([0.0, 0.5, 0.0])], axis=1).astype(np.float32)
    positions_b = positions_a + np.array([10.0, -3.0, 7.0], dtype=np.float32)
    rotations = np.zeros((n, 2, 4), dtype=np.float32)
    window = {"window_id": "win", "sample_id": "sample", "start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0, "frame_start": 0, "frame_end": n}
    sample = {"sample_id": "sample", "source_id": "src", "source_scene_file": "scene.json", "technical_atom_id": "Person"}
    body = {"body_motion_quality": "good_body_motion"}
    row_a = build_relative_motion_window_row(window, sample, positions_a, rotations, times, names, {}, body, tmp_path / "a")
    row_b = build_relative_motion_window_row(window, sample, positions_b, rotations, times, names, {}, body, tmp_path / "b")
    with np.load(row_a["relative_npz_path"], allow_pickle=True) as a, np.load(row_b["relative_npz_path"], allow_pickle=True) as b:
        assert np.allclose(a["normalized_position_delta"], b["normalized_position_delta"], atol=1e-5)


def test_root_only_movement_becomes_unsafe_for_learning(tmp_path):
    n = 30
    positions = np.zeros((n, 1, 3), dtype=np.float32)
    positions[:, 0, 0] = np.linspace(0, 1, n)
    rotations = np.zeros((n, 1, 4), dtype=np.float32)
    window = {"window_id": "win", "sample_id": "sample", "frame_start": 0, "frame_end": n}
    sample = {"sample_id": "sample"}
    row = build_relative_motion_window_row(window, sample, positions, rotations, _times(n), ["control"], {}, {}, tmp_path)
    assert not row["safe_for_learning"]
    assert "no_allowed_body_controller_tracks" in row["unsafe_reasons"]


def test_bodypart_motion_remains_safe_for_learning(tmp_path):
    n = 61
    hip = _oval(n)
    chest = hip + np.array([0.0, 0.4, 0.0])
    positions = np.stack([hip, chest], axis=1).astype(np.float32)
    rotations = np.zeros((n, 2, 4), dtype=np.float32)
    window = {"window_id": "win", "sample_id": "sample", "frame_start": 0, "frame_end": n}
    sample = {"sample_id": "sample"}
    row = build_relative_motion_window_row(window, sample, positions, rotations, _times(n), ["hipControl", "chestControl"], {}, {"body_motion_quality": "good_body_motion"}, tmp_path)
    assert row["safe_for_learning"]
    assert row["controllers"] == ["hipControl", "chestControl"]


def test_trajectory_shape_detects_oval_bounce_and_jitter():
    oval_values, oval_quality = trajectory_shape_for_points(_oval(181), _times(181), safe_for_learning=True)
    assert oval_quality["trajectory_shape_classification"] in {"oval_grind", "circular_grind"}
    assert oval_values["oval_path_score"] > 0.4

    t = np.linspace(0, 4 * np.pi, 181, dtype=np.float32)
    bounce = np.zeros((181, 3), dtype=np.float32)
    bounce[:, 1] = np.sin(t) * 0.25
    bounce_values, bounce_quality = trajectory_shape_for_points(bounce, _times(181), safe_for_learning=True)
    assert bounce_quality["trajectory_shape_classification"] in {"vertical_bounce", "unknown"}
    assert bounce_values["bounce_pattern_score"] > bounce_values["grind_pattern_score"]

    jitter = np.zeros((181, 3), dtype=np.float32)
    jitter[:, 0] = np.sin(t * 3) * 0.002
    jitter_values, jitter_quality = trajectory_shape_for_points(jitter, _times(181), safe_for_learning=True)
    assert jitter_quality["trajectory_shape_classification"] == "jitter/static"
    assert jitter_values["jitter_score"] > 0.5


def test_cowgirl_v4_rewards_oval_and_penalizes_receiver_static_head():
    feature = {"window_id": "win", "feature_values": {}, "sample_id": "s"}
    body = {"body_motion_quality": "good_body_motion"}
    rel = {"feature_values": {"safe_for_learning": 1.0}}
    traj = {"trajectory_shape_classification": "oval_grind", "feature_values": {"grind_pattern_score": 0.8, "oval_path_score": 0.8, "ellipse_fit_score": 0.7, "closed_loop_ratio": 0.8, "jitter_score": 0.1}}
    match = {"cowgirl_relative_score": 0.7, "cowgirl_grind_trajectory_score": 0.8, "safe_for_learning": True}
    window = {"duration_seconds": 4.0}
    rider = {"rider_receiver_status": "likely_active_rider", "active_rider_score": 0.8, "receiver_body_response_score": 0.1}
    good = score_window_v4(feature, body, match, rel, traj, window, rider)
    assert good["clean_cowgirl_candidate_v4"]
    assert good["likely_cowgirl_grinding"]

    receiver = score_window_v4(feature, body, match, rel, traj, window, {"rider_receiver_status": "likely_receiver_body_response", "active_rider_score": 0.1, "receiver_body_response_score": 0.9})
    assert not receiver["clean_cowgirl_candidate_v4"]
    assert "likely_receiver_body_response" in receiver["reject_reasons"]

    static = score_window_v4(feature, {"body_motion_quality": "static_or_micro_motion", "static_or_micro_motion": True}, match, rel, traj, window, rider)
    assert not static["clean_cowgirl_candidate_v4"]


def test_relative_matcher_uses_relative_space(tmp_path):
    wild_rel = tmp_path / "wild_rel.jsonl"
    wild_traj = tmp_path / "wild_traj.jsonl"
    hand_rel = tmp_path / "hand_rel.jsonl"
    hand_traj = tmp_path / "hand_traj.jsonl"
    out = tmp_path / "out.jsonl"
    report = tmp_path / "report.md"
    write_jsonl(hand_rel, [
        {"reference_id": "cow", "label_family": "cowgirl", "feature_values": {"relative_pelvis_lateral_amplitude": 0.4, "relative_pelvis_forward_back_amplitude": 0.3, "safe_for_learning": 1.0}},
        {"reference_id": "head", "label_family": "head", "feature_values": {"head_relative_to_chest_motion": 0.5, "safe_for_learning": 1.0}},
    ])
    write_jsonl(hand_traj, [
        {"reference_id": "cow", "feature_values": {"grind_pattern_score": 0.8, "oval_path_score": 0.7}},
        {"reference_id": "head", "feature_values": {"jitter_score": 0.2}},
    ])
    write_jsonl(wild_rel, [{"window_id": "w", "feature_values": {"relative_pelvis_lateral_amplitude": 0.4, "relative_pelvis_forward_back_amplitude": 0.3, "safe_for_learning": 1.0}}])
    write_jsonl(wild_traj, [{"window_id": "w", "safe_for_learning": True, "feature_values": {"grind_pattern_score": 0.8, "oval_path_score": 0.7, "jitter_score": 0.1}}])
    rows = compare_relative_wild_to_handmade(wild_rel, wild_traj, hand_rel, hand_traj, out, report)
    assert rows[0]["cowgirl_relative_score"] > rows[0]["head_relative_score"]
    assert rows[0]["recommended_review_status"] == "likely_cowgirl_candidate"


def test_semantic_review_v6_includes_relative_trajectory_evidence(tmp_path):
    from tests.test_semantic_review import _make_run

    run = _make_run(tmp_path)
    # Add the minimum v6 artifacts needed for selection.
    rel_rows = []
    traj_rows = []
    match_rows = []
    score_rows = []
    rider_rows = []
    for idx in range(12):
        wid = f"win_{idx:02d}"
        rel_rows.append({"window_id": wid, "feature_values": {"safe_for_learning": 1.0}, "feature_quality": {"teleport_risk": "low"}})
        status = "likely_cowgirl_candidate"
        flags = {"clean_cowgirl_candidate_v4": idx < 4, "likely_cowgirl_grinding": idx == 0, "role_status": "likely_active_rider"}
        rider_status = "likely_active_rider"
        if idx in {4, 5}:
            status = "likely_transition_or_realign"
            flags["likely_transition_or_adjustment"] = True
        if idx == 6:
            flags["likely_receiver_response"] = True
            rider_status = "likely_receiver_body_response"
        if idx == 7:
            status = "likely_not_cowgirl_head_or_bj"
            flags["likely_head_or_bj_false_positive"] = True
        if idx == 8:
            status = "likely_isolated_gesture"
            flags["likely_static_or_jitter"] = True
        if idx == 9:
            status = "unsafe_relative_motion"
            flags["safe_for_learning"] = False
        traj_rows.append({"window_id": wid, "safe_for_learning": idx != 9, "trajectory_shape_classification": "oval_grind", "dominant_motion_plane": "horizontal_local_xz", "feature_values": {"oval_path_score": 0.7, "ellipse_fit_score": 0.7, "closed_loop_ratio": 0.7, "grind_pattern_score": 0.7, "transition_path_score": 0.8 if idx in {4, 5} else 0.1, "jitter_score": 0.8 if idx == 8 else 0.1}})
        match_rows.append({"window_id": wid, "cowgirl_relative_score": 0.7, "bj_relative_score": 0.8 if idx == 7 else 0.1, "head_relative_score": 0.8 if idx == 7 else 0.1, "cowgirl_grind_trajectory_score": 0.7, "transition_trajectory_score": 0.8 if idx in {4, 5} else 0.1, "jitter_static_score": 0.8 if idx == 8 else 0.1, "recommended_review_status": status, "safe_for_learning": idx != 9})
        flags.setdefault("safe_for_learning", True)
        score_rows.append({"window_id": wid, "duration_seconds": 4.0, "final_clean_cowgirl_score_v4": 0.8 - idx * 0.01, "trajectory_shape_classification": "oval_grind", **flags})
        rider_rows.append({"window_id": wid, "rider_receiver_status": rider_status, "active_rider_score": 0.8 if rider_status == "likely_active_rider" else 0.1, "receiver_body_response_score": 0.8 if rider_status == "likely_receiver_body_response" else 0.1})
    write_jsonl(run / "relative_motion" / "relative_motion_features.jsonl", rel_rows)
    write_jsonl(run / "relative_motion" / "trajectory_shape_features.jsonl", traj_rows)
    write_jsonl(run / "relative_motion" / "relative_reference_matches.jsonl", match_rows)
    write_jsonl(run / "audits" / "cowgirl_candidate_scores_v4.jsonl", score_rows)
    write_jsonl(run / "audits" / "rider_receiver_scores_v1.jsonl", rider_rows)
    out = run / "audits" / "semantic_review_010_v6"
    export_semantic_review_010(run, out, count=10, attempt_timeline_export=False, use_relative_motion_features=True, use_trajectory_shape_features=True, use_relative_reference_matches=True, use_cowgirl_candidate_score_v4=True, use_rider_receiver_discrimination=True)
    rows = load_jsonl(out / "semantic_review_010.jsonl")
    positives = [r for r in rows if r["category"] == "likely_cowgirl_candidate"]
    assert positives
    assert all(r["system_semantic_guess"]["safe_for_learning"] for r in positives)
    assert all("trajectory_shape_features" in r["evidence"] for r in rows)

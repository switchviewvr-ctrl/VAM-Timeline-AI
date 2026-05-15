from pathlib import Path

import numpy as np
import yaml

from vam_timeline_ai.audits.semantic_review import export_semantic_review_010, summarize_semantic_review_010
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def _sample_npz(path: Path, frames: int = 121) -> None:
    t = np.arange(frames, dtype=np.float32) / 60.0
    positions = np.zeros((frames, 2, 3), dtype=np.float32)
    positions[:, 0, 1] = np.sin(t * 6.0) * 0.05
    positions[:, 1, 0] = np.cos(t * 3.0) * 0.02
    rotations = np.zeros((frames, 2, 4), dtype=np.float32)
    rotations[:, :, 3] = 1.0
    np.savez_compressed(path, positions=positions, rotations=rotations, times=t, controller_names=np.asarray(["hipControl", "chestControl"], dtype=object))


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "data" / "runs" / "clean_v2"
    for rel in ["semantic", "features", "baked/samples", "audits", "labels/machine_proposals"]:
        (run / rel).mkdir(parents=True, exist_ok=True)
    windows = []
    features = []
    samples = []
    weak = []
    window_scores = []
    silver_win = []
    pair_windows = []
    pair_features = []
    pair_scores = []
    silver_pair = []
    specs = [
        ("likely_positive", "cowgirl_vertical_bounce", 0.95, "timeline_controller_motion"),
        ("likely_positive", "cowgirl_forward_back_rock", 0.93, "timeline_controller_motion"),
        ("pair_contact", "cowgirl_circular_grind", 0.8, "timeline_controller_motion"),
        ("pair_contact", "cowgirl_lateral_sway", 0.8, "timeline_controller_motion"),
        ("suspicious_problem", "cowgirl_fast_shallow", 0.83, "vam_native_motion_animation"),
        ("suspicious_problem", "cowgirl_vertical_bounce", 0.81, "timeline_controller_motion"),
        ("negative_control", "", 0.0, "vam_native_motion_animation"),
        ("negative_control", "", 0.0, "timeline_controller_motion"),
        ("borderline_unclear", "cowgirl_adjustment_transition", 0.72, "timeline_controller_motion"),
        ("borderline_unclear", "cowgirl_pause_hold", 0.7, "timeline_controller_motion"),
        ("fallback", "cowgirl_irregular_human_motion", 0.86, "timeline_controller_motion"),
        ("fallback", "cowgirl_circular_grind", 0.88, "timeline_controller_motion"),
    ]
    for idx, (category, label, score, source_type) in enumerate(specs):
        sample_id = f"sample_{idx:02d}"
        wid = f"win_{idx:02d}"
        scene = f"scene_{idx:02d}.json"
        npz = run / "baked" / "samples" / f"{sample_id}.npz"
        if source_type == "timeline_controller_motion":
            _sample_npz(npz)
        samples.append({
            "sample_id": sample_id,
            "source_id": f"src_{idx:02d}",
            "source_type": source_type,
            "source_scene_file": scene,
            "source_scene_path": str(tmp_path / scene),
            "technical_atom_id": "man" if idx % 2 else "Person",
            "controller_names": ["hipControl", "chestControl"],
            "baked_npz_path": str(npz),
            "bake_status": "ok",
        })
        windows.append({
            "window_id": wid,
            "sample_id": sample_id,
            "source_id": f"src_{idx:02d}",
            "source_scene_file": scene,
            "source_scene_path": str(tmp_path / scene),
            "technical_atom_id": "man" if idx % 2 else "Person",
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "duration_seconds": 2.0,
            "frame_start": 0,
            "frame_end": 120,
        })
        energy = 0.5 if category in {"likely_positive", "pair_contact"} else (0.01 if category == "negative_control" else 0.2)
        features.append({
            "window_id": wid,
            "sample_id": sample_id,
            "source_id": f"src_{idx:02d}",
            "source_scene_file": scene,
            "technical_atom_id": "man" if idx % 2 else "Person",
            "feature_values": {
                "pelvis_movement_energy": energy,
                "pelvis_mean_speed": energy,
                "pelvis_vertical_amplitude": 0.3 if "vertical" in label else 0.02,
                "pelvis_forward_back_amplitude": 0.3 if "forward" in label else 0.02,
                "pause_hold_score_proxy": 0.8 if category in {"negative_control", "borderline_unclear"} else 0.1,
                "irregular_rhythm_score_proxy": 0.8 if category == "borderline_unclear" else 0.2,
            },
            "feature_quality": {"root_mapping_confidence": "low" if category == "suspicious_problem" else "high"},
            "missing_controller_groups": ["hands"] if category == "suspicious_problem" else [],
        })
        weak.append({"window_id": wid, "weak_labels": [{"label": "weak_v2_hint", "confidence": 0.7}]})
        if label:
            rec_status = "reject_conflict" if category == "suspicious_problem" else ("review_only" if category == "borderline_unclear" else "silver_positive_candidate")
            window_scores.append({
                "window_id": wid,
                "window_ids": [wid],
                "label": label,
                "final_score": score,
                "max_confidence": score,
                "recommended_status": rec_status,
                "proposal_types": ["positive"],
                "rule_ids": ["rule"],
                "conflict_flags": ["synthetic_conflict"] if rec_status == "reject_conflict" else [],
            })
            silver_win.append({"window_id": wid, "positive_labels": [label], "scores_by_label": {label: score}, "is_human_ground_truth": False})
    for idx, wid in enumerate(["win_02", "win_03", "win_04"]):
        pid = f"pwin_{idx:02d}"
        pair_windows.append({
            "pair_window_id": pid,
            "window_id_a": wid,
            "window_id_b": f"win_{idx + 7:02d}",
            "sample_id_a": f"sample_{int(wid[-2:]):02d}",
            "sample_id_b": f"sample_{idx + 7:02d}",
            "technical_atom_id_a": "Person",
            "technical_atom_id_b": "Partner",
            "pairing_reasons": ["synthetic pair"],
            "pair_confidence": 0.8,
        })
        pair_features.append({"pair_window_id": pid, "feature_values": {"activity_ratio_a_over_b": 3.0, "a_hands_near_b_chest_proxy": 0.8}, "feature_quality": {"active_actor_candidate": "a", "active_actor_confidence": 0.8, "has_hand_to_partner_features": True}})
        pair_scores.append({"pair_window_id": pid, "window_ids": [wid], "label": "cowgirl_hand_supported_on_partner", "final_score": 0.9, "max_confidence": 0.9, "recommended_status": "silver_positive_candidate"})
        silver_pair.append({"pair_window_id": pid, "window_ids": [wid], "positive_labels": ["cowgirl_hand_supported_on_partner"], "scores_by_label": {"cowgirl_hand_supported_on_partner": 0.9}, "is_human_ground_truth": False})
    write_jsonl(run / "baked" / "motion_sample_index.jsonl", samples)
    write_jsonl(run / "semantic" / "movement_windows.jsonl", windows)
    write_jsonl(run / "features" / "cowgirl_window_features_v1.jsonl", features)
    write_jsonl(run / "semantic" / "weak_labels_v2.jsonl", weak)
    write_jsonl(run / "semantic" / "pair_windows_v1.jsonl", pair_windows)
    write_jsonl(run / "features" / "cowgirl_pair_features_v0.jsonl", pair_features)
    write_jsonl(run / "labels" / "machine_proposals" / "machine_window_label_scores_v2.jsonl", window_scores)
    write_jsonl(run / "labels" / "machine_proposals" / "machine_pair_label_scores_v2.jsonl", pair_scores)
    write_jsonl(run / "labels" / "machine_proposals" / "silver_window_labels_v2.jsonl", silver_win)
    write_jsonl(run / "labels" / "machine_proposals" / "silver_pair_labels_v2.jsonl", silver_pair)
    write_jsonl(run / "audits" / "baked_sample_audit.jsonl", [{"sample_id": "sample_04", "suspiciously_static": False, "suspiciously_huge_motion": True}])
    return run


def test_semantic_review_exports_exactly_10_with_guesses_and_answers(tmp_path):
    run = _make_run(tmp_path)
    out = run / "audits" / "semantic_review_010"

    summary = export_semantic_review_010(run, out, count=10, attempt_timeline_export=True)
    rows = load_jsonl(out / "semantic_review_010.jsonl")
    answers = yaml.safe_load((out / "semantic_review_010_answer_sheet.yaml").read_text(encoding="utf-8"))

    assert summary["review_items"] == 10
    assert len(rows) == 10
    assert {"likely_positive", "pair_contact", "suspicious_problem", "negative_control", "borderline_unclear"} <= {r["category"] for r in rows}
    assert all(r["system_semantic_guess"]["warning"] for r in rows)
    assert set(answers["reviews"]) == {r["review_id"] for r in rows}


def test_semantic_review_timeline_export_status_success_and_unavailable(tmp_path):
    run = _make_run(tmp_path)
    out = run / "audits" / "semantic_review_010"
    export_semantic_review_010(run, out, count=10, attempt_timeline_export=True)
    rows = load_jsonl(out / "semantic_review_010.jsonl")

    assert (out / "timeline_segment_export_status.md").exists()
    successful = [r for r in rows if r["has_timeline_export"]]
    unavailable = [r for r in rows if not r["has_timeline_export"]]
    assert successful
    assert unavailable
    for row in successful:
        folder = out / "timeline_segments" / row["review_id"]
        assert (folder / f"{row['review_id']}.timeline_meta.json").exists()
        assert row["timeline_export_validation_status"] == "ok"
    for row in unavailable:
        assert (out / "timeline_segments" / row["review_id"] / "export_unavailable.md").exists()


def test_semantic_review_summary_unknown_answers_is_not_completed(tmp_path):
    run = _make_run(tmp_path)
    out = run / "audits" / "semantic_review_010"
    export_semantic_review_010(run, out, count=10, attempt_timeline_export=True)

    summary = summarize_semantic_review_010(out / "semantic_review_010_answer_sheet.yaml", out / "semantic_review_010.jsonl", out / "semantic_review_010_result.md")

    assert summary["status"] == "not_completed"
    assert "not completed" in (out / "semantic_review_010_result.md").read_text(encoding="utf-8")


def test_semantic_review_does_not_create_manual_labels_or_infer_from_atom_names(tmp_path):
    run = _make_run(tmp_path)
    out = run / "audits" / "semantic_review_010"
    export_semantic_review_010(run, out, count=10, attempt_timeline_export=True)
    rows = load_jsonl(out / "semantic_review_010.jsonl")

    assert not (run / "labels" / "manual_labels.yaml").exists()
    assert all(r["is_human_ground_truth"] is False for r in rows)
    assert not any(r.get("semantic_role") in {"rider", "receiver"} for r in rows)


def test_semantic_review_v5_likely_positives_exclude_receiver_body_response(tmp_path):
    run = _make_run(tmp_path)
    (run / "references" / "handmade_animations").mkdir(parents=True, exist_ok=True)
    body_rows = []
    match_rows = []
    rr_rows = []
    v3_rows = []
    for idx in range(12):
        wid = f"win_{idx:02d}"
        quality = "good_body_motion"
        body_rows.append({
            "window_id": wid,
            "sample_id": f"sample_{idx:02d}",
            "body_motion_quality": quality,
            "static_or_micro_motion": idx == 6,
            "minimal_head_motion_only": False,
            "minimal_hand_jitter_only": False,
            "active_bodypart_count_above_threshold": 3,
            "moving_bodypart_count": 3,
            "micro_motion_score": 0.0,
        })
        status = "likely_cowgirl_candidate"
        if idx in {8, 9}:
            status = "likely_transition_or_realign"
        if idx == 5:
            status = "likely_not_cowgirl_head_or_bj"
        if idx == 6:
            status = "likely_isolated_gesture"
        if idx == 7:
            status = "unknown_needs_review"
        match_rows.append({
            "window_id": wid,
            "cowgirl_reference_score": 0.8 if idx < 5 else 0.2,
            "doggy_reference_score": 0.2,
            "bj_reference_score": 0.8 if idx == 5 else 0.1,
            "head_reference_score": 0.7 if idx == 5 else 0.1,
            "hand_reference_score": 0.1,
            "recommended_review_status": status,
        })
        role_status = "likely_active_rider" if idx in {0, 1, 2, 3} else "role_unclear"
        active = 0.85 if idx in {0, 1, 2, 3} else 0.2
        receiver = 0.05
        if idx == 4:
            role_status = "likely_receiver_body_response"
            active = 0.2
            receiver = 0.9
        rr_rows.append({
            "window_id": wid,
            "rider_receiver_status": role_status,
            "active_rider_score": active,
            "receiver_body_response_score": receiver,
            "role_unclear_score": 0.1,
            "pair_evidence": [{"pair_window_id": "pwin_02"}] if idx == 4 else [],
        })
        v3_rows.append({
            "window_id": wid,
            "sample_id": f"sample_{idx:02d}",
            "duration_seconds": 4.0 if idx != 0 else 8.0,
            "final_clean_cowgirl_rider_score_v3": 0.9 if idx in {0, 1, 2, 3} else 0.1,
            "clean_cowgirl_rider_candidate_v3": idx in {0, 1, 2, 3},
            "likely_receiver_false_positive": idx == 4,
            "role_status": role_status,
            "likely_grinding_subtype": idx == 1,
            "cowgirl_grinding_score": 0.85 if idx == 1 else 0.2,
            "reject_reasons": ["likely_receiver_body_response"] if idx == 4 else [],
        })
    write_jsonl(run / "audits" / "body_motion_quality.jsonl", body_rows)
    write_jsonl(run / "references" / "handmade_animations" / "wild_reference_matches.jsonl", match_rows)
    write_jsonl(run / "audits" / "rider_receiver_scores_v1.jsonl", rr_rows)
    write_jsonl(run / "audits" / "cowgirl_candidate_scores_v3.jsonl", v3_rows)

    out = run / "audits" / "semantic_review_010_v5"
    export_semantic_review_010(
        run,
        out,
        count=10,
        attempt_timeline_export=False,
        use_body_motion_quality=True,
        use_handmade_reference_matches=True,
        prefer_longer_cowgirl_windows=True,
        use_cowgirl_candidate_score_v3=True,
        use_rider_receiver_discrimination=True,
    )
    rows = load_jsonl(out / "semantic_review_010.jsonl")

    likely = [row for row in rows if row["category"] == "likely_cowgirl_candidate"]
    assert len(likely) == 4
    assert all(row["system_semantic_guess"]["rider_receiver_status"] != "likely_receiver_body_response" for row in likely)
    assert any(row["category"] == "receiver_body_response" for row in rows)

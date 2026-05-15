from pathlib import Path

import yaml

from vam_timeline_ai.audits.reality_audit import export_reality_audit_100, summarize_reality_audit
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def _make_synthetic_run(tmp_path: Path) -> Path:
    run = tmp_path / "data" / "runs" / "clean_v2"
    for rel in [
        "semantic",
        "features",
        "baked",
        "audits",
        "labels/machine_proposals",
    ]:
        (run / rel).mkdir(parents=True, exist_ok=True)
    mappings = {
        "controller_mappings": {
            "hipControl": {"body_part": "hip", "mapping_confidence": "high"},
            "chestControl": {"body_part": "chest", "mapping_confidence": "high"},
            "lHandControl": {"body_part": "left_hand", "mapping_confidence": "high"},
            "rHandControl": {"body_part": "right_hand", "mapping_confidence": "high"},
            "headControl": {"body_part": "head", "mapping_confidence": "high"},
        }
    }
    (run / "semantic" / "controller_bodypart_map.json").write_text(__import__("json").dumps(mappings), encoding="utf-8")
    samples = []
    windows = []
    features = []
    weak = []
    window_scores = []
    silver_win = []
    pair_windows = []
    pair_features = []
    pair_scores = []
    silver_pair = []
    labels = [
        "cowgirl_vertical_bounce",
        "cowgirl_forward_back_rock",
        "cowgirl_lateral_sway",
        "cowgirl_circular_grind",
        "cowgirl_fast_shallow",
        "cowgirl_pause_hold",
        "cowgirl_adjustment_transition",
    ]
    for sample_idx in range(60):
        sample_id = f"sample_{sample_idx:03d}"
        scene = f"scene_{sample_idx % 15:02d}.json"
        atom = "man" if sample_idx % 2 else "Person"
        samples.append({
            "sample_id": sample_id,
            "source_id": f"src_{sample_idx:03d}",
            "source_scene_file": scene,
            "technical_atom_id": atom,
            "controller_names": ["hipControl", "chestControl", "lHandControl", "rHandControl", "headControl"],
            "baked_npz_path": str(tmp_path / "missing.npz"),
            "bake_status": "ok",
        })
        for win_idx in range(3):
            global_idx = sample_idx * 3 + win_idx
            wid = f"win_{global_idx:03d}"
            start = float(win_idx * 2)
            windows.append({
                "window_id": wid,
                "sample_id": sample_id,
                "source_id": f"src_{sample_idx:03d}",
                "source_scene_file": scene,
                "technical_atom_id": atom,
                "start_seconds": start,
                "end_seconds": start + 2.0,
                "duration_seconds": 2.0,
                "frame_start": int(start * 60),
                "frame_end": int((start + 2.0) * 60),
            })
            energy = 0.02 + (global_idx % 20) * 0.01
            if global_idx % 17 == 0:
                energy = 0.0
            values = {
                "pelvis_vertical_amplitude": 0.01 * (global_idx % 10),
                "pelvis_forward_back_amplitude": 0.02 * ((global_idx + 2) % 10),
                "pelvis_lateral_amplitude": 0.015 * ((global_idx + 4) % 10),
                "pelvis_movement_energy": energy,
                "pelvis_mean_speed": energy * 2.0,
                "pelvis_max_speed": energy * 4.0,
                "pause_hold_score_proxy": 0.9 if global_idx % 19 == 0 else 0.1,
                "irregular_rhythm_score_proxy": 0.8 if global_idx % 13 == 0 else 0.2,
                "left_hand_motion_energy": 0.4 if global_idx % 7 == 0 else 0.02,
                "right_hand_motion_energy": 0.35 if global_idx % 7 == 0 else 0.02,
                "torso_motion_energy": 0.3 if global_idx % 11 == 0 else 0.02,
                "head_motion_energy": 0.25 if global_idx % 12 == 0 else 0.02,
            }
            missing = ["hands"] if global_idx % 23 == 0 else []
            features.append({
                "window_id": wid,
                "sample_id": sample_id,
                "source_id": f"src_{sample_idx:03d}",
                "source_scene_file": scene,
                "technical_atom_id": atom,
                "feature_values": values,
                "feature_quality": {"has_pelvis_features": True, "root_mapping_confidence": "low" if global_idx % 29 == 0 else "high"},
                "controllers_used": {"pelvis": ["hipControl"]},
                "missing_controller_groups": missing,
                "warnings": ["synthetic warning"] if missing else [],
            })
            weak.append({"window_id": wid, "weak_labels": [{"label": "weak_v2_high_vertical_motion", "confidence": 0.8}]})
            label = labels[global_idx % len(labels)]
            window_scores.append({
                "window_id": wid,
                "window_ids": [wid],
                "label": label,
                "final_score": 0.95,
                "max_confidence": 0.98,
                "recommended_status": "silver_positive_candidate",
                "proposal_types": ["positive"],
                "rule_ids": ["synthetic_rule"],
                "conflict_flags": [],
                "high_risk_proxy_label": False,
            })
            if global_idx % 14 == 0:
                window_scores.append({
                    "window_id": wid,
                    "window_ids": [wid],
                    "label": "cowgirl_fast_shallow",
                    "final_score": 0.85,
                    "max_confidence": 0.9,
                    "recommended_status": "reject_conflict",
                    "proposal_types": ["positive"],
                    "rule_ids": ["conflict_rule"],
                    "conflict_flags": ["fast_and_slow"],
                    "high_risk_proxy_label": False,
                })
            silver_win.append({
                "window_id": wid,
                "positive_labels": [label],
                "negative_labels": [],
                "review_only_labels": [],
                "default_trainable_labels": [label],
                "scores_by_label": {label: 0.95},
                "label_source": "silver_machine_v2",
                "is_human_ground_truth": False,
            })
    for idx in range(40):
        pwin = f"pwin_{idx:03d}"
        wid_a = f"win_{idx:03d}"
        wid_b = f"win_{idx + 40:03d}"
        pair_windows.append({
            "pair_window_id": pwin,
            "pair_id": f"pair_{idx:03d}",
            "source_scene_file": f"scene_{idx % 15:02d}.json",
            "sample_id_a": f"sample_{idx // 3:03d}",
            "sample_id_b": f"sample_{(idx + 40) // 3:03d}",
            "technical_atom_id_a": "man",
            "technical_atom_id_b": "Person",
            "window_id_a": wid_a,
            "window_id_b": wid_b,
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "frame_start_a": 0,
            "frame_end_a": 120,
            "frame_start_b": 0,
            "frame_end_b": 120,
            "pair_confidence": 0.8,
            "pairing_reasons": ["synthetic pair"],
        })
        pair_features.append({
            "pair_window_id": pwin,
            "window_id_a": wid_a,
            "window_id_b": wid_b,
            "feature_values": {"activity_ratio_a_over_b": 4.0, "a_hands_near_b_chest_proxy": 0.9},
            "feature_quality": {"has_hand_to_partner_features": True, "active_actor_candidate": "a", "active_actor_confidence": 0.9},
        })
        pair_scores.append({
            "pair_window_id": pwin,
            "window_ids": [wid_a, wid_b],
            "label": "cowgirl_hand_supported_on_partner",
            "final_score": 0.9,
            "max_confidence": 0.95,
            "recommended_status": "silver_positive_candidate",
            "rule_ids": ["pair_rule"],
        })
        if idx % 2 == 0:
            pair_scores.append({
                "pair_window_id": pwin,
                "window_ids": [wid_a, wid_b],
                "label": "rider_active",
                "final_score": 0.88,
                "max_confidence": 0.91,
                "recommended_status": "silver_positive_candidate",
                "rule_ids": ["role_rule"],
            })
        silver_pair.append({
            "pair_window_id": pwin,
            "window_ids": [wid_a, wid_b],
            "positive_labels": ["cowgirl_hand_supported_on_partner"],
            "scores_by_label": {"cowgirl_hand_supported_on_partner": 0.9},
            "label_source": "silver_machine_v2",
            "is_human_ground_truth": False,
        })
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
    write_jsonl(run / "audits" / "baked_sample_audit.jsonl", [{"sample_id": "sample_000", "suspiciously_static": True}, {"sample_id": "sample_001", "suspiciously_huge_motion": True}])
    return run


def test_reality_audit_exports_exactly_100_and_respects_limits(tmp_path):
    run = _make_synthetic_run(tmp_path)
    out = run / "audits" / "reality_audit_001"

    summary = export_reality_audit_100(run, out, count=100, render_previews=False)
    rows = load_jsonl(out / "reality_audit_batch.jsonl")

    assert summary["audit_items"] == 100
    assert len(rows) == 100
    scene_counts = {}
    sample_counts = {}
    for row in rows:
        scene_counts[row["source_scene_file"]] = scene_counts.get(row["source_scene_file"], 0) + 1
        sample_counts[row["sample_id"]] = sample_counts.get(row["sample_id"], 0) + 1
    assert max(scene_counts.values()) <= 10
    assert max(sample_counts.values()) <= 3
    assert sum(1 for row in rows if row.get("pair_window_id")) >= 15


def test_reality_audit_includes_categories_and_stub_ids(tmp_path):
    run = _make_synthetic_run(tmp_path)
    out = run / "audits" / "reality_audit_001"
    export_reality_audit_100(run, out, count=100, render_previews=False)

    rows = load_jsonl(out / "reality_audit_batch.jsonl")
    categories = {row["category"] for row in rows}
    stub = yaml.safe_load((out / "reality_audit_annotation.stub.yaml").read_text(encoding="utf-8"))

    assert {"random_baseline", "high_confidence_movement", "pair_contact", "suspicious_problem", "negative_control"} <= categories
    assert set(stub["audit_items"]) == {row["audit_id"] for row in rows}
    assert all(item["machine_labels_plausible"]["true_labels"] == [] for item in stub["audit_items"].values())


def test_reality_audit_summary_missing_annotations_is_graceful(tmp_path):
    run = _make_synthetic_run(tmp_path)
    out = run / "audits" / "reality_audit_001"
    export_reality_audit_100(run, out, count=100, render_previews=False)

    result = summarize_reality_audit(out / "reality_audit_annotation.edited.yaml", out / "reality_audit_batch.jsonl", out / "reality_audit_result.md")

    assert result["status"] == "not_completed"
    assert "not completed" in (out / "reality_audit_result.md").read_text(encoding="utf-8")


def test_reality_audit_keeps_machine_labels_as_hints_and_no_manual_file(tmp_path):
    run = _make_synthetic_run(tmp_path)
    out = run / "audits" / "reality_audit_001"
    export_reality_audit_100(run, out, count=100, render_previews=False)
    rows = load_jsonl(out / "reality_audit_batch.jsonl")

    assert not (run / "labels" / "manual_labels.yaml").exists()
    assert all(row["is_human_ground_truth"] is False for row in rows)
    assert all("machine_label_warning" in row for row in rows)
    assert not any(row.get("semantic_role") in {"rider", "receiver"} for row in rows)

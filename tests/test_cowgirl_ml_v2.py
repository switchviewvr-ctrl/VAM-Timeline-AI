from __future__ import annotations

import json
from pathlib import Path

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ml.cowgirl_ml_feature_table_v2 import build_cowgirl_ml_feature_table_v2
from vam_timeline_ai.ml.cowgirl_ml_label_dataset_v2 import build_cowgirl_ml_labels_v2
from vam_timeline_ai.ml.cowgirl_ml_model_v2 import score_new_scenes_cowgirl_ml_v2, train_cowgirl_ml_v2
from vam_timeline_ai.ml.cowgirl_ml_review_v2 import export_ml_assisted_cowgirl_review_v2


def test_label_builder_separates_semantic_and_completeness(tmp_path: Path) -> None:
    base = tmp_path / "base"
    new = tmp_path / "new"
    (new / "audits" / "cowgirl_motion_cycles_v1_010").mkdir(parents=True)
    (new / "audits" / "bj_doggy_missionary_motion_cycles_v1").mkdir(parents=True)
    cow = _candidate("w_cow", "cowgirl_clean_cyclic_motion", "cowgirl", "hipControl")
    bj = _candidate("w_bj", "bj_oral_clean_head_cycle", "bj_oral", "headControl")
    write_jsonl(new / "audits" / "cowgirl_motion_cycles_v1_010" / "semantic_review_010.jsonl", [cow])
    write_jsonl(new / "audits" / "bj_doggy_missionary_motion_cycles_v1" / "semantic_review_010.jsonl", [bj])
    write_jsonl(base / "audits" / "human_review_ledger.jsonl", [])
    write_jsonl(base / "manual_pose_ground_truth_v1" / "manual_pose_ground_truth_v1.jsonl", [])

    summary = build_cowgirl_ml_labels_v2(
        base,
        new,
        base / "audits" / "human_review_ledger.jsonl",
        base / "manual_pose_ground_truth_v1" / "manual_pose_ground_truth_v1.jsonl",
        new / "labels.jsonl",
        new / "report.md",
    )
    rows = load_jsonl(new / "labels.jsonl")
    by_window = {r["window_id"]: r for r in rows}
    assert summary["label_counts"]["label_cowgirl_semantic_family"]["true"] == 1
    assert by_window["w_cow"]["label_cowgirl_semantic_family"] == "true"
    assert by_window["w_cow"]["label_pose_incomplete_missing_controllers"] == "unknown"
    assert by_window["w_bj"]["label_cowgirl_semantic_family"] == "false"
    assert by_window["w_bj"]["label_not_cowgirl_bj_oral"] == "true"


def test_feature_table_train_score_and_review_export(tmp_path: Path) -> None:
    labels = []
    candidates = []
    cycles = []
    for idx in range(4):
        candidates.append(_candidate(f"w_pos_{idx}", "cowgirl_clean_cyclic_motion", "cowgirl", "hipControl", sample=f"s_pos_{idx}", scene=f"scene_pos_{idx}.json"))
        labels.append(_label(f"w_pos_{idx}", "true", "true", "false", "false"))
        cycles.append(_cycle(f"w_pos_{idx}", hip_cycles=2.0, head_cycles=0.0, hand_cycles=0.0))
    for idx in range(4):
        candidates.append(_candidate(f"w_neg_{idx}", "bj_oral_clean_head_cycle", "bj_oral", "headControl", sample=f"s_neg_{idx}", scene=f"scene_neg_{idx}.json"))
        labels.append(_label(f"w_neg_{idx}", "false", "false", "true", "false"))
        cycles.append(_cycle(f"w_neg_{idx}", hip_cycles=0.0, head_cycles=2.0, hand_cycles=0.0))
    write_jsonl(tmp_path / "labels.jsonl", labels)
    write_jsonl(tmp_path / "candidates.jsonl", candidates)
    write_jsonl(tmp_path / "pose.jsonl", candidates)
    write_jsonl(tmp_path / "motion.jsonl", candidates)
    write_jsonl(tmp_path / "cycles.jsonl", cycles)
    write_jsonl(tmp_path / "manual_gt.jsonl", [])

    table = build_cowgirl_ml_feature_table_v2(
        tmp_path / "labels.jsonl",
        tmp_path / "pose.jsonl",
        tmp_path / "cycles.jsonl",
        tmp_path / "motion.jsonl",
        tmp_path / "candidates.jsonl",
        tmp_path / "manual_gt.jsonl",
        tmp_path / "features.npz",
        tmp_path / "meta.jsonl",
        tmp_path / "feature_report.md",
    )
    assert table["shape"][0] == 8
    assert table["features"] > 0
    assert table["label_counts"]["label_not_cowgirl_bj_oral"]["true"] == 4

    model = train_cowgirl_ml_v2(tmp_path / "features.npz", tmp_path / "meta.jsonl", tmp_path / "model")
    assert model["trained"]
    scores = score_new_scenes_cowgirl_ml_v2(tmp_path / "model", tmp_path / "features.npz", tmp_path / "meta.jsonl", tmp_path / "scores.jsonl", tmp_path / "score_report.md")
    assert scores["rows"] == 8

    review = export_ml_assisted_cowgirl_review_v2(tmp_path, tmp_path / "scores.jsonl", tmp_path / "candidates.jsonl", tmp_path / "review", 4, build_static_ui=False, build_vam_package=False)
    assert review["selected"] == 4
    rows = load_jsonl(tmp_path / "review" / "semantic_review_010.jsonl")
    assert all(row["review_only"] for row in rows)
    assert all(not row["ml_training_performed"] for row in rows)


def _candidate(window: str, category: str, family: str, driver: str, sample: str = "sample", scene: str = "scene.json") -> dict:
    return {
        "window_id": window,
        "sample_id": sample,
        "source_id": f"src_{sample}",
        "source_scene_file": scene,
        "technical_actor_id": "Actor",
        "start_seconds": 0.0,
        "end_seconds": 4.0,
        "category": category,
        "resolved_motion_family": family,
        "resolved_motion_subtype": category,
        "resolved_semantic_family": family,
        "motion_state": "clean_motion",
        "pose_family": family,
        "pose_subtype": f"{family}_pose",
        "primary_driver_controller": driver,
        "primary_motion_center": "pelvis_hip" if driver == "hipControl" else "head_neck",
        "cycle_count": 2.0,
        "cyclicity_score": 0.9,
        "transition_score": 0.1,
        "final_clean_motion_gate": "pass",
        "contact_support": "hands_free",
    }


def _label(window: str, cowgirl: str, clean: str, bj_negative: str, incomplete: str) -> dict:
    return {
        "window_id": window,
        "source_kind": "test",
        "label_cowgirl_semantic_family": cowgirl,
        "label_cowgirl_clean_motion": clean,
        "label_cowgirl_pose_context": "false",
        "label_cowgirl_transition": "false",
        "label_not_cowgirl_bj_oral": bj_negative,
        "label_not_cowgirl_handjob": "false",
        "label_not_cowgirl_standing_hand_head": "false",
        "label_generation_safe_or_complete": "unknown",
        "label_pose_incomplete_missing_controllers": incomplete,
    }


def _cycle(window: str, hip_cycles: float, head_cycles: float, hand_cycles: float) -> dict:
    return {
        "window_id": window,
        "controller_metrics": {
            "hipControl": _metric(hip_cycles),
            "pelvisControl": _metric(hip_cycles * 0.2),
            "headControl": _metric(head_cycles),
            "chestControl": _metric(head_cycles * 0.5),
            "lHandControl": _metric(hand_cycles),
            "rHandControl": _metric(hand_cycles),
            "lFootControl": _metric(0.0),
            "rFootControl": _metric(0.0),
        },
    }


def _metric(cycles: float) -> dict:
    return {
        "max_displacement_range": cycles,
        "total_path_length": cycles * 2,
        "cyclicity_score": min(1.0, cycles / 2.0),
        "transition_score": 0.1 if cycles else 0.0,
        "pose_hold_score": 0.0 if cycles else 1.0,
        "estimated_cycle_count": cycles,
        "estimated_frequency_hz": cycles / 4.0,
        "dominant_axis": "y",
        "axis_metrics": {"x": {}, "y": {"displacement_range": cycles, "estimated_cycle_count": cycles, "cyclicity_score": min(1.0, cycles / 2.0), "transition_score": 0.1}, "z": {}},
    }

import json
from pathlib import Path

import numpy as np
import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ml.dataset_v3 import build_ml_dataset_v3
from vam_timeline_ai.ml.silver_baseline import train_silver_baseline_v0
from vam_timeline_ai.semantics.machine_labeling_v1 import run_machine_labeling_v1
from vam_timeline_ai.semantics.machine_label_proposals import generate_machine_label_proposals_v1
from vam_timeline_ai.semantics.machine_label_schema import MachineLabelProposal, SilverLabelRecord
from vam_timeline_ai.semantics.machine_proposal_review_batch import build_machine_proposal_review_batch
from vam_timeline_ai.semantics.silver_labels import build_silver_labels_v1


def _write_synthetic_run(run: Path) -> None:
    for name in ["features", "semantic", "labels/machine_proposals", "labels/batches", "ml/datasets", "ml/reports", "ml/models"]:
        (run / name).mkdir(parents=True, exist_ok=True)
    write_jsonl(
        run / "semantic" / "movement_windows.jsonl",
        [
            {"window_id": "win_a", "sample_id": "sample_a", "source_id": "src_a", "source_scene_file": "scene_a.json", "technical_atom_id": "ActorA", "start_seconds": 0, "end_seconds": 4},
            {"window_id": "win_b", "sample_id": "sample_b", "source_id": "src_b", "source_scene_file": "scene_b.json", "technical_atom_id": "ActorB", "start_seconds": 0, "end_seconds": 4},
        ],
    )
    write_jsonl(
        run / "features" / "cowgirl_window_features_v1.jsonl",
        [
            {
                "window_id": "win_a",
                "sample_id": "sample_a",
                "source_id": "src_a",
                "source_scene_file": "scene_a.json",
                "technical_atom_id": "ActorA",
                "feature_values": {
                    "pelvis_vertical_amplitude": 1.0,
                    "pelvis_forward_back_amplitude": 0.08,
                    "pelvis_lateral_amplitude": 0.05,
                    "pelvis_total_position_range": 1.0,
                    "pelvis_movement_energy": 2.0,
                    "pelvis_mean_speed": 0.4,
                    "pelvis_rock_score_proxy": 0.1,
                    "pelvis_circularity_score_proxy": 0.1,
                    "pelvis_grind_score_proxy": 0.1,
                    "slow_motion_score_proxy": 0.9,
                    "fast_motion_score_proxy": 0.1,
                    "pause_hold_score_proxy": 0.1,
                    "irregular_rhythm_score_proxy": 0.2,
                    "pelvis_acceleration_peak_count": 2,
                    "torso_lean_forward_proxy": 0.1,
                    "torso_lean_back_proxy": 0.1,
                },
                "feature_quality": {"has_pelvis_features": True, "has_torso_features": True},
            },
            {
                "window_id": "win_b",
                "sample_id": "sample_b",
                "source_id": "src_b",
                "source_scene_file": "scene_b.json",
                "technical_atom_id": "ActorB",
                "feature_values": {
                    "pelvis_vertical_amplitude": 0.01,
                    "pelvis_forward_back_amplitude": 0.01,
                    "pelvis_lateral_amplitude": 0.01,
                    "pelvis_total_position_range": 0.02,
                    "pelvis_movement_energy": 0.01,
                    "pelvis_mean_speed": 0.01,
                    "pelvis_rock_score_proxy": 0.0,
                    "pelvis_circularity_score_proxy": 0.0,
                    "pelvis_grind_score_proxy": 0.0,
                    "slow_motion_score_proxy": 0.1,
                    "fast_motion_score_proxy": 0.0,
                    "pause_hold_score_proxy": 0.8,
                    "irregular_rhythm_score_proxy": 0.0,
                    "pelvis_acceleration_peak_count": 0,
                    "torso_lean_forward_proxy": 0.0,
                    "torso_lean_back_proxy": 0.0,
                },
                "feature_quality": {"has_pelvis_features": True},
            },
        ],
    )
    write_jsonl(
        run / "semantic" / "weak_labels_v2.jsonl",
        [
            {"window_id": "win_a", "weak_labels": [{"label": "weak_v2_high_vertical_motion", "confidence": 0.9}, {"label": "weak_v2_slow_motion_candidate", "confidence": 0.8}]},
            {"window_id": "win_b", "weak_labels": [{"label": "weak_v2_pause_hold_candidate", "confidence": 0.8}]},
        ],
    )
    write_jsonl(
        run / "semantic" / "pair_windows_v1.jsonl",
        [{"pair_window_id": "pwin_ab", "window_id_a": "win_a", "window_id_b": "win_b", "sample_id_a": "sample_a", "sample_id_b": "sample_b", "source_scene_file": "scene_a.json"}],
    )
    write_jsonl(
        run / "features" / "cowgirl_pair_features_v0.jsonl",
        [
            {
                "pair_window_id": "pwin_ab",
                "window_id_a": "win_a",
                "window_id_b": "win_b",
                "sample_id_a": "sample_a",
                "sample_id_b": "sample_b",
                "technical_atom_id_a": "ActorA",
                "technical_atom_id_b": "ActorB",
                "source_scene_file": "scene_a.json",
                "feature_values": {
                    "a_left_hand_to_b_chest_distance_mean": 0.1,
                    "a_right_hand_to_b_chest_distance_mean": 0.1,
                    "a_left_hand_to_b_pelvis_distance_mean": 0.5,
                    "a_right_hand_to_b_pelvis_distance_mean": 0.5,
                    "a_static_hand_support_on_b_candidate_proxy": 0.9,
                    "a_motion_energy": 2.0,
                    "b_motion_energy": 0.1,
                },
                "feature_quality": {"active_actor_candidate": "a", "active_actor_confidence": 0.82, "has_hand_to_partner_features": True},
            }
        ],
    )


def test_machine_proposal_schema_marks_not_human_truth():
    prop = MachineLabelProposal(
        proposal_id="p",
        window_id="w",
        pair_window_id=None,
        sample_id="s",
        source_id="src",
        source_scene_file="scene",
        technical_atom_id="Actor",
        label="cowgirl_vertical_bounce",
        label_group="movement",
        proposal_type="positive",
        confidence=2.0,
        source="machine_rule_v1",
        rule_id="r",
        is_human_ground_truth=True,
    ).to_dict()
    silver = SilverLabelRecord(window_id="w", pair_window_id=None, positive_labels=["cowgirl_vertical_bounce"], is_human_ground_truth=True).to_dict()

    assert prop["is_human_ground_truth"] is False
    assert prop["confidence"] == 1.0
    assert silver["is_human_ground_truth"] is False
    assert silver["label_source"] == "silver_machine_v1"


def test_machine_proposals_never_emit_weak_labels(tmp_path):
    run = tmp_path / "run"
    _write_synthetic_run(run)
    rows = generate_machine_label_proposals_v1(
        run,
        run / "features" / "cowgirl_window_features_v1.jsonl",
        run / "features" / "cowgirl_pair_features_v0.jsonl",
        run / "semantic" / "weak_labels_v2.jsonl",
        run / "semantic" / "movement_windows.jsonl",
        run / "semantic" / "pair_windows_v1.jsonl",
        run / "labels" / "machine_proposals" / "props.jsonl",
        run / "labels" / "machine_proposals" / "props.yaml",
        run / "labels" / "machine_proposals" / "props.md",
    )

    assert rows
    assert all(not row["label"].startswith("weak_") for row in rows)
    assert all(row["is_human_ground_truth"] is False for row in rows)
    assert not (run / "labels" / "manual_labels.yaml").exists()


def test_silver_labels_are_separate_and_machine_sourced(tmp_path):
    run = tmp_path / "run"
    _write_synthetic_run(run)
    proposals = generate_machine_label_proposals_v1(
        run,
        run / "features" / "cowgirl_window_features_v1.jsonl",
        run / "features" / "cowgirl_pair_features_v0.jsonl",
        run / "semantic" / "weak_labels_v2.jsonl",
        run / "semantic" / "movement_windows.jsonl",
        run / "semantic" / "pair_windows_v1.jsonl",
        run / "labels" / "machine_proposals" / "props.jsonl",
        run / "labels" / "machine_proposals" / "props.yaml",
        run / "labels" / "machine_proposals" / "props.md",
    )
    silver = build_silver_labels_v1(run / "labels" / "machine_proposals" / "props.jsonl", run / "labels" / "machine_proposals" / "silver.jsonl", run / "labels" / "machine_proposals" / "silver.yaml", run / "labels" / "machine_proposals" / "silver.md", min_confidence=0.75)
    yaml_data = yaml.safe_load((run / "labels" / "machine_proposals" / "silver.yaml").read_text(encoding="utf-8"))

    assert proposals
    assert silver
    assert yaml_data["metadata"]["label_source"] == "silver_machine_v1"
    assert yaml_data["metadata"]["is_human_ground_truth"] is False


def test_dataset_v3_separates_manual_silver_and_weak(tmp_path):
    run = tmp_path / "run"
    _write_synthetic_run(run)
    write_jsonl(run / "semantic" / "movement_windows_labeled.jsonl", load_jsonl(run / "semantic" / "movement_windows.jsonl"))
    (run / "labels" / "manual_labels.yaml").write_text(
        yaml.safe_dump({"windows": {"win_a": {"labels": ["cowgirl_vertical_bounce"], "negative_labels": ["not_cowgirl"], "confidence": 0.8, "include_for_ml": True}}}),
        encoding="utf-8",
    )
    write_jsonl(
        run / "labels" / "machine_proposals" / "silver.jsonl",
        [{"window_id": "win_a", "pair_window_id": None, "positive_labels": ["cowgirl_deep_slow"], "negative_labels": [], "role_candidates": ["rider_active"], "contact_candidates": [], "confidence_by_label": {"cowgirl_deep_slow": 0.8, "rider_active": 0.82}, "label_source": "silver_machine_v1", "is_human_ground_truth": False}],
    )

    summary = build_ml_dataset_v3(
        run / "features" / "cowgirl_window_features_v1.jsonl",
        run / "semantic" / "movement_windows_labeled.jsonl",
        run / "semantic" / "weak_labels_v2.jsonl",
        run / "labels" / "manual_labels.yaml",
        run / "labels" / "machine_proposals" / "silver.jsonl",
        run / "ml" / "datasets" / "dataset_v3.npz",
        run / "ml" / "reports" / "dataset_v3.md",
    )
    data = np.load(run / "ml" / "datasets" / "dataset_v3.npz", allow_pickle=True)

    assert summary["manual_label_count"] == 2
    assert summary["silver_label_count"] == 2
    assert data["manual_y_positive_multilabel"].sum() == 1
    assert data["manual_y_negative_multilabel"].sum() == 1
    assert data["silver_y_positive_multilabel"].sum() == 2
    assert data["weak_y_multilabel"].sum() == 3
    metadata = json.loads(str(data["metadata_json"].item()))
    assert metadata["silver_is_human_ground_truth"] is False


def test_silver_baseline_blocks_or_reports_proxy_only(tmp_path):
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        X=np.ones((2, 1), dtype=np.float32),
        window_ids=np.asarray(["w1", "w2"], dtype=object),
        feature_names=np.asarray(["f"], dtype=object),
        silver_label_names=np.asarray(["cowgirl_vertical_bounce"], dtype=object),
        silver_y_positive_multilabel=np.asarray([[1], [0]], dtype=np.int8),
        silver_y_negative_multilabel=np.asarray([[0], [1]], dtype=np.int8),
        group_scene=np.asarray(["scene_a", "scene_b"], dtype=object),
        group_sample=np.asarray(["sample_a", "sample_b"], dtype=object),
        silver_confidence=np.asarray([[0.8], [0.0]], dtype=np.float32),
    )

    result = train_silver_baseline_v0(dataset, tmp_path / "models", tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert result["trained"] is False
    assert result["is_human_supervised"] is False
    assert "not human-supervised" in text or "not human" in text


def test_machine_proposal_review_batch_keeps_manual_stub_empty(tmp_path):
    run = tmp_path / "run"
    _write_synthetic_run(run)
    generate_machine_label_proposals_v1(
        run,
        run / "features" / "cowgirl_window_features_v1.jsonl",
        run / "features" / "cowgirl_pair_features_v0.jsonl",
        run / "semantic" / "weak_labels_v2.jsonl",
        run / "semantic" / "movement_windows.jsonl",
        run / "semantic" / "pair_windows_v1.jsonl",
        run / "labels" / "machine_proposals" / "props.jsonl",
        run / "labels" / "machine_proposals" / "props.yaml",
        run / "labels" / "machine_proposals" / "props.md",
    )
    build_silver_labels_v1(run / "labels" / "machine_proposals" / "props.jsonl", run / "labels" / "machine_proposals" / "silver.jsonl", run / "labels" / "machine_proposals" / "silver.yaml", run / "labels" / "machine_proposals" / "silver.md")
    rows = build_machine_proposal_review_batch(run, run / "labels" / "machine_proposals" / "props.jsonl", run / "labels" / "machine_proposals" / "silver.jsonl", run / "labels" / "batches" / "machine")
    stub = yaml.safe_load((run / "labels" / "batches" / "machine" / "manual_labels.stub.yaml").read_text(encoding="utf-8"))
    machine_review = yaml.safe_load((run / "labels" / "batches" / "machine" / "machine_label_review.yaml").read_text(encoding="utf-8"))

    assert rows
    assert all(entry["labels"] == [] for entry in stub["windows"].values())
    assert machine_review["metadata"]["is_human_ground_truth"] is False


def test_run_machine_labeling_v1_works_on_synthetic_run(tmp_path):
    run = tmp_path / "run"
    _write_synthetic_run(run)

    result = run_machine_labeling_v1(run, min_silver_confidence=0.75, train_silver_baseline=False)

    assert result["status"] == "ok"
    assert result["proposal_count"] > 0
    assert result["manual_labels_modified"] is False
    assert (run / "labels" / "machine_proposals" / "machine_labeling_v1_summary.md").exists()
    assert not (run / "labels" / "manual_labels.yaml").exists()

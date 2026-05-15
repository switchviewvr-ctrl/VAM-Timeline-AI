import yaml
import numpy as np

from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.ml.dataset_v2 import build_ml_dataset_v2
from vam_timeline_ai.ml.supervised_baseline import train_supervised_baseline_v0
from vam_timeline_ai.ml.supervised_readiness import analyze_supervised_readiness
from vam_timeline_ai.semantics.active_review_batch import build_active_review_batch_v3
from vam_timeline_ai.semantics.edited_label_batch import inspect_edited_label_batch
from vam_timeline_ai.semantics.manual_label_merge import merge_manual_label_batch
from vam_timeline_ai.semantics.manual_label_validation import validate_manual_labels_v2


def _write_windows(path):
    write_jsonl(path, [
        {"window_id": "win_clean_0000", "sample_id": "sample_a", "source_id": "src_a", "source_scene_file": "scene_a.json", "technical_atom_id": "A", "include_for_ml": True},
        {"window_id": "win_clean_0001", "sample_id": "sample_b", "source_id": "src_b", "source_scene_file": "scene_b.json", "technical_atom_id": "B", "include_for_ml": True},
    ])


def _write_features(path):
    write_jsonl(path, [
        {"window_id": "win_clean_0000", "sample_id": "sample_a", "source_id": "src_a", "source_scene_file": "scene_a.json", "technical_atom_id": "A", "feature_values": {"pelvis_movement_energy": 1.0, "head_motion_energy": 0.2}, "feature_quality": {"has_pelvis_features": True}},
        {"window_id": "win_clean_0001", "sample_id": "sample_b", "source_id": "src_b", "source_scene_file": "scene_b.json", "technical_atom_id": "B", "feature_values": {"pelvis_movement_energy": 0.1, "head_motion_energy": 0.8}, "feature_quality": {"has_head_features": True}},
    ])


def test_edited_batch_inspection_detects_empty_stub_and_weak_labels(tmp_path):
    windows = tmp_path / "windows.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    _write_windows(windows)
    write_jsonl(pairs, [{"pair_window_id": "pairwin_clean_0000"}])
    stub = tmp_path / "manual_labels.stub.yaml"
    edited = tmp_path / "manual_labels.edited.yaml"
    stub.write_text(yaml.safe_dump({"windows": {"win_clean_0000": {"labels": [], "confidence": 0.0, "notes": ""}}}), encoding="utf-8")
    edited.write_text(stub.read_text(encoding="utf-8"), encoding="utf-8")

    same = inspect_edited_label_batch(stub, edited, windows, pairs, tmp_path / "same.md")
    assert same["safe_to_merge"] is False
    assert any("byte-identical" in error for error in same["errors"])

    edited.write_text(yaml.safe_dump({"windows": {"win_clean_0000": {"labels": ["weak_v2_fast_motion_candidate"], "confidence": 0.8}}}), encoding="utf-8")
    weak = inspect_edited_label_batch(stub, edited, windows, pairs, tmp_path / "weak.md")
    assert weak["safe_to_merge"] is False
    assert any("weak labels" in error for error in weak["errors"])


def test_merge_preserves_negative_and_uncertain_and_ignores_empty(tmp_path):
    base = tmp_path / "manual_labels.yaml"
    batch = tmp_path / "edited.yaml"
    batch.write_text(yaml.safe_dump({"windows": {
        "win_clean_0000": {"labels": [], "confidence": 0.0, "notes": ""},
        "win_clean_0001": {"labels": ["cowgirl_pause_hold"], "negative_labels": ["not_cowgirl"], "uncertain_labels": ["cowgirl_lean_forward"], "confidence": 0.75, "notes": "reviewed"},
    }}), encoding="utf-8")

    result = merge_manual_label_batch(base, batch, base, backup=True, report=tmp_path / "merge.md")
    merged = yaml.safe_load(base.read_text(encoding="utf-8"))

    assert result["merged_windows"] == 1
    assert "win_clean_0000" not in merged["windows"]
    assert merged["windows"]["win_clean_0001"]["negative_labels"] == ["not_cowgirl"]
    assert merged["windows"]["win_clean_0001"]["uncertain_labels"] == ["cowgirl_lean_forward"]


def test_validation_rejects_unknown_and_stale_ids(tmp_path):
    windows = tmp_path / "windows.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    labels = tmp_path / "manual.yaml"
    schema = tmp_path / "schema.yaml"
    _write_windows(windows)
    write_jsonl(pairs, [])
    schema.write_text("allowed_manual_labels: []", encoding="utf-8")
    labels.write_text(yaml.safe_dump({"windows": {"old_window": {"labels": ["cowgirl_pause_hold"], "confidence": 0.8}}}), encoding="utf-8")

    result = validate_manual_labels_v2(labels, schema, windows, pairs, tmp_path / "validation.md")

    assert result["status"] == "error"
    assert any("unknown window_id" in error for error in result["errors"])


def test_dataset_v2_separates_manual_and_weak_labels(tmp_path):
    features = tmp_path / "features.jsonl"
    windows = tmp_path / "windows.jsonl"
    weak = tmp_path / "weak.jsonl"
    labels = tmp_path / "manual.yaml"
    out = tmp_path / "dataset.npz"
    _write_features(features)
    _write_windows(windows)
    write_jsonl(weak, [{"window_id": "win_clean_0000", "weak_labels": [{"label": "weak_v2_pause_hold_candidate"}]}])
    labels.write_text(yaml.safe_dump({"windows": {"win_clean_0000": {"labels": ["cowgirl_pause_hold"], "negative_labels": ["not_cowgirl"], "uncertain_labels": ["cowgirl_lean_forward"], "confidence": 0.7, "include_for_ml": True}}}), encoding="utf-8")

    summary = build_ml_dataset_v2(features, windows, weak, labels, out, tmp_path / "dataset.md")
    data = np.load(out, allow_pickle=True)

    assert summary["manual_label_count"] == 3
    assert data["manual_y_positive_multilabel"].sum() == 1
    assert data["manual_y_negative_multilabel"].sum() == 1
    assert data["manual_y_uncertain_multilabel"].sum() == 1
    assert data["weak_y_multilabel"].sum() == 1


def test_supervised_readiness_and_baseline_block_when_insufficient(tmp_path):
    dataset = tmp_path / "dataset.npz"
    split = tmp_path / "split.json"
    labels = tmp_path / "manual.yaml"
    np.savez_compressed(
        dataset,
        X=np.ones((2, 1), dtype=np.float32),
        window_ids=np.asarray(["w1", "w2"], dtype=object),
        manual_label_names=np.asarray(["cowgirl_pause_hold"], dtype=object),
        manual_y_positive_multilabel=np.asarray([[1], [0]], dtype=np.int8),
        manual_y_negative_multilabel=np.asarray([[0], [1]], dtype=np.int8),
        manual_y_uncertain_multilabel=np.asarray([[0], [0]], dtype=np.int8),
        group_scene=np.asarray(["scene_a", "scene_b"], dtype=object),
        group_sample=np.asarray(["sample_a", "sample_b"], dtype=object),
        group_source=np.asarray(["src_a", "src_b"], dtype=object),
        include_for_ml=np.asarray([True, True]),
        confidence=np.asarray([0.8, 0.8], dtype=np.float32),
    )
    split.write_text('{"random_window_split_allowed": false, "can_plan_supervised_split": false}', encoding="utf-8")
    labels.write_text("windows: {}\n", encoding="utf-8")

    readiness = analyze_supervised_readiness(dataset, labels, split, tmp_path / "ready.md")
    baseline = train_supervised_baseline_v0(dataset, split, tmp_path / "models", tmp_path / "baseline.md")

    assert readiness["eligible_labels"] == []
    assert baseline["trained"] is False


def test_active_review_batch_avoids_already_labeled_windows(tmp_path):
    windows = tmp_path / "windows.jsonl"
    features = tmp_path / "features.jsonl"
    weak = tmp_path / "weak.jsonl"
    pair_windows = tmp_path / "pairs.jsonl"
    pair_features = tmp_path / "pair_features.jsonl"
    manual = tmp_path / "manual.yaml"
    _write_windows(windows)
    _write_features(features)
    write_jsonl(weak, [
        {"window_id": "win_clean_0000", "weak_labels": [{"label": "weak_v2_pause_hold_candidate"}]},
        {"window_id": "win_clean_0001", "weak_labels": [{"label": "weak_v2_high_intensity"}]},
    ])
    write_jsonl(pair_windows, [])
    write_jsonl(pair_features, [])
    manual.write_text(yaml.safe_dump({"windows": {"win_clean_0000": {"labels": ["cowgirl_pause_hold"], "confidence": 0.8}}}), encoding="utf-8")

    rows = build_active_review_batch_v3(windows, features, weak, pair_windows, pair_features, manual, tmp_path / "ready.md", tmp_path / "batch", batch_size=2)

    assert rows
    assert all(row["window_id"] != "win_clean_0000" for row in rows)

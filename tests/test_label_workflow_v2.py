import builtins
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from vam_timeline_ai.audits.integrity import audit_data_integrity
from vam_timeline_ai.cowgirl.pair_feature_extractor import extract_pair_features_v0
from vam_timeline_ai.io.json_utils import dump_json, write_jsonl
from vam_timeline_ai.ml.splits import plan_ml_splits_v1
from vam_timeline_ai.semantics.manual_label_merge import merge_manual_label_batch
from vam_timeline_ai.semantics.manual_label_summary import summarize_manual_labels
from vam_timeline_ai.semantics.manual_label_validation import validate_manual_labels_v2
from vam_timeline_ai.semantics.manual_labels import write_manual_label_schema_v2
from vam_timeline_ai.semantics.pair_windows import build_pair_windows_v1
from vam_timeline_ai.semantics.review_batch import build_review_batch_v2
from vam_timeline_ai.semantics.weak_label_calibration import calibrate_weak_labels_v2
from vam_timeline_ai.visualization.review_previews import render_review_previews_v1


def _sample_npz(path: Path, offset: float) -> None:
    names = np.asarray(["pelvisControl", "chestControl", "headControl", "lHandControl", "rHandControl"], dtype=object)
    times = np.linspace(0, 4, 241, dtype=np.float32)
    positions = np.zeros((len(times), len(names), 3), dtype=np.float32)
    positions[:, 0, 1] = offset + 0.1 * np.sin(times * 3)
    positions[:, 1, :] = positions[:, 0, :] + np.array([0, 0.5, 0.1], dtype=np.float32)
    positions[:, 2, :] = positions[:, 0, :] + np.array([0, 0.8, 0.1], dtype=np.float32)
    positions[:, 3, :] = positions[:, 0, :] + np.array([-0.2, 0.35, 0.1], dtype=np.float32)
    positions[:, 4, :] = positions[:, 0, :] + np.array([0.2, 0.35, 0.1], dtype=np.float32)
    rotations = np.zeros((len(times), len(names), 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    np.savez_compressed(path, times=times, positions=positions, rotations=rotations, controller_names=names)


def _basic_files(tmp_path: Path):
    a_npz = tmp_path / "a.npz"
    b_npz = tmp_path / "b.npz"
    _sample_npz(a_npz, 0.2)
    _sample_npz(b_npz, -0.2)
    samples = tmp_path / "samples.jsonl"
    write_jsonl(samples, [
        {"sample_id": "a", "source_id": "sa", "source_scene_file": "scene.json", "technical_atom_id": "A", "bake_status": "ok", "baked_npz_path": str(a_npz), "duration_seconds": 4.0, "fps": 60, "controller_names": ["pelvisControl", "chestControl", "headControl", "lHandControl", "rHandControl"], "clip_name": "clip"},
        {"sample_id": "b", "source_id": "sb", "source_scene_file": "scene.json", "technical_atom_id": "B", "bake_status": "ok", "baked_npz_path": str(b_npz), "duration_seconds": 4.0, "fps": 60, "controller_names": ["pelvisControl", "chestControl", "headControl", "lHandControl", "rHandControl"], "clip_name": "clip"},
    ])
    windows = tmp_path / "windows.jsonl"
    write_jsonl(windows, [
        {"window_id": "a:0.000-2.000", "sample_id": "a", "source_id": "sa", "source_scene_file": "scene.json", "technical_atom_id": "A", "start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0, "frame_start": 0, "frame_end": 120},
        {"window_id": "b:0.000-2.000", "sample_id": "b", "source_id": "sb", "source_scene_file": "scene.json", "technical_atom_id": "B", "start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0, "frame_start": 0, "frame_end": 120},
    ])
    cmap = tmp_path / "map.json"
    dump_json(cmap, {"controller_mappings": {"pelvisControl": {"body_part": "pelvis"}, "chestControl": {"body_part": "chest"}, "headControl": {"body_part": "head"}, "lHandControl": {"body_part": "left_hand"}, "rHandControl": {"body_part": "right_hand"}}})
    return samples, windows, cmap


def test_data_integrity_audit_synthetic(tmp_path):
    sources = tmp_path / "sources.jsonl"
    samples, windows, _ = _basic_files(tmp_path)
    features = tmp_path / "features.jsonl"
    dataset = tmp_path / "dataset.npz"
    write_jsonl(sources, [{"source_id": "sa", "source_type": "timeline_controller_motion"}, {"source_id": "sb", "source_type": "timeline_controller_motion"}])
    write_jsonl(features, [{"window_id": "a:0.000-2.000", "sample_id": "a"}, {"window_id": "b:0.000-2.000", "sample_id": "b"}])
    np.savez_compressed(dataset, X=np.ones((2, 1), dtype=np.float32), sample_ids=np.asarray(["a", "b"], dtype=object), window_ids=np.asarray(["a:0.000-2.000", "b:0.000-2.000"], dtype=object), group_scene=np.asarray(["scene.json", "scene.json"], dtype=object), group_source=np.asarray(["sa", "sb"], dtype=object), metadata_json="{}")

    summary = audit_data_integrity(sources, samples, windows, features, dataset, tmp_path / "integrity.md")

    assert summary["source_records"] == 2
    assert summary["successful_baked_samples"] == 2
    assert summary["dataset"]["shape"] == [2, 1]


def test_weak_label_calibration_v2_prefixes_only(tmp_path):
    features = tmp_path / "features.jsonl"
    weak = tmp_path / "weak.jsonl"
    rows = []
    for i in range(20):
        rows.append({"window_id": f"w{i}", "feature_values": {"pelvis_vertical_amplitude": i / 100, "pelvis_forward_back_amplitude": 0.01, "pelvis_lateral_amplitude": 0.01, "intensity_score_proxy": i / 20}})
    write_jsonl(features, rows)
    write_jsonl(weak, [{"window_id": "w1", "weak_labels": [{"label": "weak_high_energy"}]}])

    out = calibrate_weak_labels_v2(features, weak, tmp_path / "weak2.jsonl", tmp_path / "report.md")
    labels = [item["label"] for row in out for item in row["weak_labels"]]

    assert labels
    assert all(label.startswith("weak_v2_") for label in labels)
    assert not any(label.startswith("cowgirl_") for label in labels)


def test_pair_window_and_pair_features_do_not_assign_roles(tmp_path):
    samples, windows, cmap = _basic_files(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl(pairs, [{"pair_id": "p", "source_scene_file": "scene.json", "sample_id_a": "a", "sample_id_b": "b", "technical_atom_id_a": "A", "technical_atom_id_b": "B", "clip_name_a": "clip", "clip_name_b": "clip", "pair_confidence": 1.0, "pairing_reasons": ["test"]}])

    pair_rows = build_pair_windows_v1(pairs, windows, samples, tmp_path / "pair_windows.jsonl", tmp_path / "pair_windows.md")
    features = extract_pair_features_v0(tmp_path / "pair_windows.jsonl", samples, cmap, tmp_path / "pair_features.jsonl", tmp_path / "pair_features.npz", tmp_path / "pair_features.md")

    assert pair_rows[0]["semantic_role_a"] == "unknown"
    assert pair_rows[0]["semantic_role_b"] == "unknown"
    assert features[0]["feature_quality"]["active_actor_candidate"] in {"a", "b", "unknown"}
    assert "semantic_role" not in features[0]["feature_quality"]


def test_manual_schema_and_validation_reject_unknown_labels(tmp_path):
    _, windows, _ = _basic_files(tmp_path)
    pair_windows = tmp_path / "pair_windows.jsonl"
    write_jsonl(pair_windows, [])
    labels = tmp_path / "manual_labels.yaml"
    labels.write_text(yaml.safe_dump({"windows": {"a:0.000-2.000": {"labels": ["not_a_real_label"], "confidence": 0.5}}}), encoding="utf-8")

    write_manual_label_schema_v2(tmp_path / "schema.yaml", tmp_path / "template.yaml", tmp_path / "guide.md")
    result = validate_manual_labels_v2(labels, tmp_path / "schema.yaml", windows, pair_windows, tmp_path / "validation.md")
    template_result = validate_manual_labels_v2(tmp_path / "template.yaml", tmp_path / "schema.yaml", windows, pair_windows, tmp_path / "template_validation.md")

    assert result["status"] == "error"
    assert any("unknown manual label" in e for e in result["errors"])
    assert template_result["status"] == "error"


def test_review_batch_limits_and_keeps_stub_empty(tmp_path):
    _, windows, _ = _basic_files(tmp_path)
    features = tmp_path / "features.jsonl"
    weak = tmp_path / "weak.jsonl"
    pair_windows = tmp_path / "pair_windows.jsonl"
    pair_features = tmp_path / "pair_features.jsonl"
    clusters = tmp_path / "clusters.jsonl"
    write_jsonl(features, [{"window_id": "a:0.000-2.000", "sample_id": "a", "source_scene_file": "scene.json", "technical_atom_id": "A", "feature_values": {"pelvis_movement_energy": 2.0}}, {"window_id": "b:0.000-2.000", "sample_id": "b", "source_scene_file": "scene.json", "technical_atom_id": "B", "feature_values": {"pelvis_movement_energy": 1.0}}])
    write_jsonl(weak, [{"window_id": "a:0.000-2.000", "weak_labels": [{"label": "weak_v2_high_intensity"}]}, {"window_id": "b:0.000-2.000", "weak_labels": [{"label": "weak_v2_high_intensity"}]}])
    write_jsonl(pair_windows, [])
    write_jsonl(pair_features, [])
    write_jsonl(clusters, [])

    rows = build_review_batch_v2(windows, features, weak, pair_windows, pair_features, clusters, tmp_path / "batch", batch_size=2, max_per_scene=1, max_per_sample=1)
    stub = yaml.safe_load((tmp_path / "batch" / "manual_labels.stub.yaml").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert stub["windows"][rows[0]["window_id"]]["labels"] == []


def test_preview_renderer_missing_matplotlib_graceful(tmp_path):
    samples, _, cmap = _basic_files(tmp_path)
    batch = tmp_path / "batch.jsonl"
    write_jsonl(batch, [{"review_id": "r1", "window_id": "a:0.000-2.000", "sample_id": "a", "source_scene_file": "scene.json", "technical_atom_id": "A", "start_seconds": 0, "end_seconds": 2, "weak_labels_v2": []}])
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib blocked")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        summary = render_review_previews_v1(batch, samples, cmap, tmp_path / "previews")

    assert summary["matplotlib_available"] is False
    assert (tmp_path / "previews" / "preview_report.md").exists()


def test_manual_merge_summary_and_split_plan(tmp_path):
    _, windows, _ = _basic_files(tmp_path)
    pair_windows = tmp_path / "pair_windows.jsonl"
    write_jsonl(pair_windows, [])
    base = tmp_path / "manual_labels.yaml"
    batch = tmp_path / "batch.yaml"
    batch.write_text(yaml.safe_dump({"windows": {"a:0.000-2.000": {"labels": [], "confidence": 0.0, "notes": ""}, "b:0.000-2.000": {"labels": ["cowgirl_pause_hold"], "negative_labels": ["not_cowgirl"], "confidence": 0.7, "notes": "reviewed"}}}), encoding="utf-8")

    merged = merge_manual_label_batch(base, batch, base, backup=True, report=tmp_path / "merge.md")
    summary = summarize_manual_labels(base, windows, pair_windows, tmp_path / "summary.md")
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(dataset, X=np.ones((2, 1), dtype=np.float32), window_ids=np.asarray(["a:0.000-2.000", "b:0.000-2.000"], dtype=object), group_scene=np.asarray(["scene.json", "scene.json"], dtype=object), group_sample=np.asarray(["a", "b"], dtype=object), group_source=np.asarray(["sa", "sb"], dtype=object))
    plan = plan_ml_splits_v1(dataset, base, tmp_path / "split.json", tmp_path / "split.md")

    assert merged["merged_windows"] == 1
    assert summary["total_labeled_windows"] == 1
    assert plan["random_window_split_allowed"] is False

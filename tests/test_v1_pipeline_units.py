import json
from pathlib import Path

import numpy as np

from vam_timeline_ai.cowgirl.feature_extractor_v1 import extract_cowgirl_features_v1
from vam_timeline_ai.io.json_utils import dump_json, write_jsonl
from vam_timeline_ai.ml.dataset_v1 import build_ml_dataset_v1
from vam_timeline_ai.ml.readiness_v1 import analyze_ml_v1
from vam_timeline_ai.motion.controller_mapping import discover_controller_map
from vam_timeline_ai.motion.data_audit import audit_baked_samples
from vam_timeline_ai.semantics.review_queue import build_review_queue_v1
from vam_timeline_ai.semantics.weak_labels import generate_weak_labels_v1


def _synthetic_sample_npz(path: Path, controller_names: list[str]) -> None:
    times = np.linspace(0, 4, 241, dtype=np.float32)
    positions = np.zeros((len(times), len(controller_names), 3), dtype=np.float32)
    rotations = np.zeros((len(times), len(controller_names), 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    name_to_idx = {name: idx for idx, name in enumerate(controller_names)}
    if "pelvisControl" in name_to_idx:
        idx = name_to_idx["pelvisControl"]
        positions[:, idx, 1] = 0.1 * np.sin(times * 4)
        positions[:, idx, 2] = 0.04 * np.cos(times * 2)
    if "chestControl" in name_to_idx:
        positions[:, name_to_idx["chestControl"], :] = positions[:, name_to_idx["pelvisControl"], :] + np.array([0.0, 0.5, 0.08], dtype=np.float32)
    if "headControl" in name_to_idx:
        positions[:, name_to_idx["headControl"], :] = positions[:, name_to_idx["pelvisControl"], :] + np.array([0.0, 0.8, 0.1], dtype=np.float32)
        rotations[:, name_to_idx["headControl"], 1] = 0.05 * np.sin(times)
        rotations[:, name_to_idx["headControl"], 3] = np.sqrt(1.0 - rotations[:, name_to_idx["headControl"], 1] ** 2)
    for name in ["lHandControl", "rHandControl"]:
        if name in name_to_idx:
            side = -1.0 if name.startswith("l") else 1.0
            positions[:, name_to_idx[name], :] = positions[:, name_to_idx["pelvisControl"], :] + np.array([0.25 * side, 0.35, 0.1], dtype=np.float32)
    for name in ["lKneeControl", "rKneeControl", "lFootControl", "rFootControl"]:
        if name in name_to_idx:
            side = -1.0 if name.startswith("l") else 1.0
            positions[:, name_to_idx[name], :] = positions[:, name_to_idx["pelvisControl"], :] + np.array([0.18 * side, -0.35, 0.0], dtype=np.float32)
    np.savez_compressed(
        path,
        times=times,
        positions=positions,
        rotations=rotations,
        controller_names=np.asarray(controller_names, dtype=object),
        metadata_json="{}",
    )


def test_baked_sample_audit_detects_motion_and_static(tmp_path):
    moving = tmp_path / "moving.npz"
    static = tmp_path / "static.npz"
    _synthetic_sample_npz(moving, ["pelvisControl"])
    times = np.linspace(0, 2, 121, dtype=np.float32)
    np.savez_compressed(
        static,
        times=times,
        positions=np.zeros((len(times), 1, 3), dtype=np.float32),
        rotations=np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (len(times), 1, 1)),
        controller_names=np.asarray(["pelvisControl"], dtype=object),
    )
    index = tmp_path / "samples.jsonl"
    write_jsonl(
        index,
        [
            {"sample_id": "moving", "bake_status": "ok", "baked_npz_path": str(moving), "duration_seconds": 4.0},
            {"sample_id": "static", "bake_status": "ok", "baked_npz_path": str(static), "duration_seconds": 2.0},
        ],
    )

    rows = audit_baked_samples(index, tmp_path / "audit.jsonl", tmp_path / "audit.md")

    assert rows[0]["moving_controller_count"] == 1
    assert rows[0]["audit_status"] == "ok"
    assert rows[1]["suspiciously_static"] is True


def test_controller_mapping_is_conservative(tmp_path):
    npz = tmp_path / "sample.npz"
    _synthetic_sample_npz(npz, ["pelvisControl", "lHandControl", "customThingControl"])
    index = tmp_path / "samples.jsonl"
    write_jsonl(index, [{"sample_id": "s", "bake_status": "ok", "baked_npz_path": str(npz), "controller_names": ["pelvisControl", "lHandControl", "customThingControl"]}])

    discover_controller_map(index, tmp_path / "inventory.json", tmp_path / "map.json", tmp_path / "report.md")
    mapping = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))["controller_mappings"]

    assert mapping["pelvisControl"]["body_part"] == "pelvis"
    assert mapping["lHandControl"]["body_part"] == "left_hand"
    assert mapping["customThingControl"]["body_part"] == "unknown"


def test_feature_extraction_v1_computes_groups_and_missing_groups(tmp_path):
    npz = tmp_path / "sample.npz"
    names = ["pelvisControl", "chestControl", "headControl", "lHandControl", "rHandControl", "lKneeControl", "rKneeControl", "lFootControl", "rFootControl"]
    _synthetic_sample_npz(npz, names)
    sample_index = tmp_path / "samples.jsonl"
    windows = tmp_path / "windows.jsonl"
    cmap = tmp_path / "map.json"
    write_jsonl(sample_index, [{"sample_id": "s", "source_id": "src", "source_scene_file": "scene.json", "technical_atom_id": "Actor", "bake_status": "ok", "baked_npz_path": str(npz), "controller_names": names}])
    write_jsonl(windows, [{"window_id": "s:0.000-2.000", "sample_id": "s", "source_id": "src", "source_scene_file": "scene.json", "technical_atom_id": "Actor", "frame_start": 0, "frame_end": 120}])
    discover_controller_map(sample_index, tmp_path / "inventory.json", cmap, tmp_path / "map.md")

    rows = extract_cowgirl_features_v1(windows, sample_index, cmap, tmp_path / "features.jsonl", tmp_path / "features.npz", tmp_path / "features.md")

    quality = rows[0]["feature_quality"]
    assert quality["has_pelvis_features"] is True
    assert quality["has_torso_features"] is True
    assert quality["has_hand_features"] is True
    assert quality["has_leg_features"] is True
    assert quality["has_head_features"] is True
    assert np.isfinite(rows[0]["feature_values"]["pelvis_vertical_amplitude"])


def test_feature_extraction_v1_missing_controllers_warns_without_fake_values(tmp_path):
    npz = tmp_path / "sample.npz"
    _synthetic_sample_npz(npz, ["pelvisControl"])
    sample_index = tmp_path / "samples.jsonl"
    windows = tmp_path / "windows.jsonl"
    cmap = tmp_path / "map.json"
    write_jsonl(sample_index, [{"sample_id": "s", "bake_status": "ok", "baked_npz_path": str(npz), "controller_names": ["pelvisControl"]}])
    write_jsonl(windows, [{"window_id": "s:0.000-2.000", "sample_id": "s", "frame_start": 0, "frame_end": 120}])
    discover_controller_map(sample_index, tmp_path / "inventory.json", cmap, tmp_path / "map.md")

    rows = extract_cowgirl_features_v1(windows, sample_index, cmap, tmp_path / "features.jsonl", tmp_path / "features.npz", tmp_path / "features.md")

    assert rows[0]["feature_quality"]["has_pelvis_features"] is True
    assert rows[0]["feature_quality"]["has_hand_features"] is False
    assert "hands" in rows[0]["missing_controller_groups"]
    assert np.isnan(rows[0]["feature_values"]["left_hand_motion_energy"])


def test_weak_labels_are_prefixed_and_separate_from_manual_labels(tmp_path):
    features = tmp_path / "features.jsonl"
    weak = tmp_path / "weak.jsonl"
    windows = tmp_path / "windows.jsonl"
    dataset = tmp_path / "dataset.npz"
    write_jsonl(
        features,
        [
            {
                "window_id": "w1",
                "sample_id": "s1",
                "source_id": "src",
                "source_scene_file": "scene",
                "technical_atom_id": "Actor",
                "feature_values": {"pelvis_vertical_amplitude": 0.2, "pelvis_forward_back_amplitude": 0.01, "pelvis_lateral_amplitude": 0.01, "pelvis_movement_energy": 0.5},
                "feature_quality": {"has_pelvis_features": True},
            }
        ],
    )
    write_jsonl(windows, [{"window_id": "w1", "labels": []}])

    weak_rows = generate_weak_labels_v1(features, weak, tmp_path / "weak.md")
    summary = build_ml_dataset_v1(features, windows, weak, dataset, tmp_path / "dataset.md")
    with np.load(dataset, allow_pickle=True) as data:
        assert weak_rows[0]["weak_labels"]
        assert all(item["label"].startswith("weak_") for item in weak_rows[0]["weak_labels"])
        assert summary["manual_label_count"] == 0
        assert data["manual_y_multilabel"].shape[1] == 0
        assert data["weak_y_multilabel"].shape[1] > 0


def test_review_queue_limits_duplicate_windows_from_same_sample(tmp_path):
    features = tmp_path / "features.jsonl"
    weak = tmp_path / "weak.jsonl"
    clusters = tmp_path / "clusters.jsonl"
    windows = tmp_path / "windows.jsonl"
    feature_rows = []
    weak_rows = []
    window_rows = []
    for i in range(5):
        wid = f"s1:{i}.000-{i + 2}.000"
        feature_rows.append({"window_id": wid, "sample_id": "s1", "source_scene_file": "ride.json", "technical_atom_id": "Actor", "feature_values": {"pelvis_movement_energy": 1.0 + i}})
        weak_rows.append({"window_id": wid, "weak_labels": [{"label": "weak_high_energy", "score": 1.0}]})
        window_rows.append({"window_id": wid, "start_seconds": i, "end_seconds": i + 2, "duration_seconds": 2})
    write_jsonl(features, feature_rows)
    write_jsonl(weak, weak_rows)
    write_jsonl(clusters, [{"window_id": row["window_id"], "cluster_id": 1} for row in feature_rows])
    write_jsonl(windows, window_rows)

    rows = build_review_queue_v1(features, weak, clusters, windows, tmp_path / "queue.jsonl", tmp_path / "queue.md", max_per_sample=2, max_records=5)

    assert len(rows) == 2
    assert {row["sample_id"] for row in rows} == {"s1"}


def test_ml_dataset_v1_groups_and_readiness_warns_against_random_window_splits(tmp_path):
    features = tmp_path / "features.jsonl"
    windows = tmp_path / "windows.jsonl"
    weak = tmp_path / "weak.jsonl"
    dataset = tmp_path / "dataset.npz"
    report_dir = tmp_path / "reports"
    write_jsonl(features, [{"window_id": "w1", "sample_id": "s1", "source_id": "src", "source_scene_file": "scene", "technical_atom_id": "Actor", "feature_values": {"a": 1.0}, "feature_quality": {"has_pelvis_features": True}}])
    write_jsonl(windows, [{"window_id": "w1", "labels": []}])
    write_jsonl(weak, [{"window_id": "w1", "weak_labels": [{"label": "weak_high_energy", "score": 1.0}]}])

    build_ml_dataset_v1(features, windows, weak, dataset, tmp_path / "dataset.md")
    analyze_ml_v1(dataset, report_dir)

    with np.load(dataset, allow_pickle=True) as data:
        assert "group_scene" in data.files
        assert "group_sample" in data.files
        assert "group_source" in data.files
    assert "Random window splits valid: False" in (report_dir / "ml_readiness_report_v1.md").read_text(encoding="utf-8")

import json
from pathlib import Path

import numpy as np

from vam_timeline_ai.cowgirl.feature_extractor import extract_cowgirl_features_v0
from vam_timeline_ai.datasets.window_dataset import build_movement_windows
from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.ml.clustering import analyze_ml_v0
from vam_timeline_ai.ml.dataset import build_ml_dataset_v0
from vam_timeline_ai.motion.source_inventory import inventory_scene_file
from vam_timeline_ai.semantics.manual_labels import apply_manual_labels


def test_motion_source_record_serialization(tmp_path):
    scene = {
        "atoms": [
            {
                "id": "ActorA",
                "type": "Person",
                "storables": [
                    {
                        "id": "plugin#0_VamTimeline.AtomPlugin",
                        "Animation": {
                            "SerializeVersion": "283",
                            "Clips": [
                                {
                                    "AnimationName": "Anim 1",
                                    "AnimationLength": "2",
                                    "Controllers": [
                                        {
                                            "Controller": "hipControl",
                                            "X": [],
                                            "Y": [],
                                            "Z": [],
                                            "RotX": [],
                                            "RotY": [],
                                            "RotZ": [],
                                            "RotW": [],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
            }
        ]
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene), encoding="utf-8")

    records = inventory_scene_file(path)

    assert len(records) == 1
    assert records[0]["source_type"] == "timeline_controller_motion"
    assert records[0]["technical_atom_id"] == "ActorA"


def test_movement_window_record_generation(tmp_path):
    sample_index = tmp_path / "samples.jsonl"
    out = tmp_path / "windows.jsonl"
    write_jsonl(
        sample_index,
        [
            {
                "sample_id": "sampleA",
                "source_id": "sourceA",
                "source_scene_file": "scene.json",
                "technical_atom_id": "ActorA",
                "duration_seconds": 5.0,
                "fps": 60,
                "bake_status": "ok",
                "baked_npz_path": "sample.npz",
                "warnings": [],
            }
        ],
    )

    rows = build_movement_windows(sample_index, out)

    assert any(row["window_id"].startswith("win_sampleA_0000.000_0002.000_") for row in rows)
    assert len({row["window_id"] for row in rows}) == len(rows)
    assert all(row["semantic_role_guess"] == "unknown" for row in rows)
    assert all(row["labels"] == [] for row in rows)


def test_feature_matrix_from_synthetic_baked_sample(tmp_path):
    npz = tmp_path / "sample.npz"
    times = np.linspace(0, 4, 241, dtype=np.float32)
    positions = np.zeros((len(times), 1, 3), dtype=np.float32)
    positions[:, 0, 1] = np.sin(times * 2)
    rotations = np.zeros((len(times), 1, 4), dtype=np.float32)
    rotations[:, 0, 3] = 1.0
    np.savez_compressed(
        npz,
        times=times,
        positions=positions,
        rotations=rotations,
        velocities=np.zeros_like(positions),
        angular_deltas=rotations,
        controller_names=np.asarray(["pelvisControl"], dtype=object),
        metadata_json="{}",
    )
    sample_index = tmp_path / "samples.jsonl"
    windows = tmp_path / "windows.jsonl"
    features_jsonl = tmp_path / "features.jsonl"
    features_npz = tmp_path / "features.npz"
    report = tmp_path / "report.md"
    write_jsonl(sample_index, [{"sample_id": "sampleA", "bake_status": "ok", "baked_npz_path": str(npz), "fps": 60}])
    write_jsonl(windows, [{"window_id": "sampleA:0.000-2.000", "sample_id": "sampleA", "frame_start": 0, "frame_end": 120}])

    rows = extract_cowgirl_features_v0(windows, sample_index, features_jsonl, features_npz, report)
    data = np.load(features_npz, allow_pickle=True)

    assert rows[0]["feature_quality"]["has_numeric_features"] is True
    assert data["X"].shape[0] == 1
    assert np.isfinite(data["X"]).any()


def test_ml_dataset_builder_with_synthetic_features(tmp_path):
    features = tmp_path / "features.jsonl"
    windows = tmp_path / "windows.jsonl"
    out = tmp_path / "dataset.npz"
    report = tmp_path / "report.md"
    write_jsonl(features, [{"window_id": "w1", "sample_id": "s1", "source_scene_file": "scene", "technical_atom_id": "Actor", "features": {"vertical_amplitude": 1.0}}])
    write_jsonl(windows, [{"window_id": "w1", "labels": ["cowgirl_vertical_bounce"]}])

    summary = build_ml_dataset_v0(features, windows, out, report)
    data = np.load(out, allow_pickle=True)

    assert summary["row_count"] == 1
    assert data["X"].shape[0] == 1
    assert "cowgirl_vertical_bounce" in data["label_names"].tolist()


def test_clustering_handles_too_small_dataset(tmp_path):
    dataset = tmp_path / "dataset.npz"
    out_dir = tmp_path / "reports"
    np.savez_compressed(
        dataset,
        X=np.asarray([[1.0, np.nan]], dtype=np.float32),
        feature_names=np.asarray(["a", "b"], dtype=object),
        window_ids=np.asarray(["w1"], dtype=object),
        sample_ids=np.asarray(["s1"], dtype=object),
        label_names=np.asarray([], dtype=object),
    )

    summary = analyze_ml_v0(dataset, out_dir)

    assert summary["assignments"] == 0
    assert (out_dir / "ml_readiness_report.md").exists()


def test_missing_manual_labels_does_not_use_template(tmp_path):
    windows = tmp_path / "windows.jsonl"
    out = tmp_path / "labeled.jsonl"
    report = tmp_path / "report.md"
    write_jsonl(windows, [{"window_id": "w1", "sample_id": "s1", "labels": []}])

    rows = apply_manual_labels(windows, tmp_path / "manual_labels.yaml", out, report)

    assert rows[0]["labels"] == []
    assert "Template labels were not applied" in report.read_text(encoding="utf-8")

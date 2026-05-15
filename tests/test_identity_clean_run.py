import json
from pathlib import Path

import numpy as np

from vam_timeline_ai.audits.integrity import audit_data_integrity
from vam_timeline_ai.io.artifacts import prepare_clean_run
from vam_timeline_ai.io.identity import (
    make_pair_id,
    make_pair_window_id,
    make_sample_id,
    make_source_id,
    make_window_id,
    sanitize_id_part,
    stable_hash,
)
from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.motion.source_inventory import inventory_scene_file


def test_stable_hash_and_sanitize_are_deterministic():
    assert stable_hash(["a", "b", "c"]) == stable_hash(["a", "b", "c"])
    assert stable_hash(["a", "b", "c"]) != stable_hash(["a", "b", "d"])
    assert sanitize_id_part("plugin#1 / Clip: A") == "plugin_1_Clip_A"


def test_source_sample_window_and_pair_ids_are_unique_for_similar_names():
    s1 = make_source_id("scene.json", "timeline_controller_motion", "Person", "plugin#0", "plugin#0", "Anim", 0)
    s2 = make_source_id("scene.json", "timeline_controller_motion", "Person", "plugin#1", "plugin#1", "Anim", 0)
    assert s1 != s2
    sample1 = make_sample_id(s1, 60, "extract_v2", "Person", "Anim", 0)
    sample2 = make_sample_id(s2, 60, "extract_v2", "Person", "Anim", 0)
    assert sample1 != sample2
    w1 = make_window_id(sample1, 0, 2, 2, 1, 60)
    w2 = make_window_id(sample1, 1, 3, 2, 1, 60)
    assert w1 != w2
    p1 = make_pair_id("scene.json", sample1, sample2, "Anim", "Anim")
    pw1 = make_pair_window_id(p1, w1, w2, 1, 2)
    pw2 = make_pair_window_id(p1, w2, w1, 1, 2)
    assert pw1 != pw2


def test_source_inventory_does_not_infer_roles_from_atom_names(tmp_path):
    scene = {
        "atoms": [
            {
                "id": "man",
                "type": "Person",
                "storables": [
                    {
                        "id": "plugin#0_VamTimeline.AtomPlugin",
                        "Animation": {
                            "SerializeVersion": "283",
                            "Clips": [
                                {
                                    "AnimationName": "Anim",
                                    "AnimationLength": "2",
                                    "Controllers": [{"Controller": "pelvisControl", "X": ["A00000000"]}],
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
    rows = inventory_scene_file(path, raw_root=tmp_path)

    assert rows
    assert rows[0]["technical_atom_id"] == "man"
    assert "semantic_role" not in rows[0]


def test_strict_integrity_rejects_duplicates_and_unknown_references(tmp_path):
    sources = tmp_path / "sources.jsonl"
    samples = tmp_path / "samples.jsonl"
    windows = tmp_path / "windows.jsonl"
    features = tmp_path / "features.jsonl"
    dataset = tmp_path / "dataset.npz"
    npz = tmp_path / "sample.npz"
    np.savez_compressed(npz, times=np.asarray([0, 1], dtype=np.float32), positions=np.zeros((2, 1, 3), dtype=np.float32), rotations=np.zeros((2, 1, 4), dtype=np.float32), controller_names=np.asarray(["pelvisControl"], dtype=object))
    write_jsonl(sources, [{"source_id": "src1"}, {"source_id": "src1"}])
    write_jsonl(samples, [{"sample_id": "s1", "bake_status": "ok", "baked_npz_path": str(npz)}, {"sample_id": "s1", "bake_status": "ok", "baked_npz_path": str(npz)}])
    write_jsonl(windows, [{"window_id": "w1", "sample_id": "s1"}, {"window_id": "w2", "sample_id": "missing"}])
    write_jsonl(features, [{"window_id": "w1", "sample_id": "s1"}, {"window_id": "unknown", "sample_id": "s1"}])
    np.savez_compressed(dataset, X=np.ones((2, 1), dtype=np.float32), window_ids=np.asarray(["w1", "unknown"], dtype=object), sample_ids=np.asarray(["s1", "s1"], dtype=object), group_scene=np.asarray(["scene", "scene"], dtype=object), group_source=np.asarray(["src", "src"], dtype=object), metadata_json="{}")

    summary = audit_data_integrity(sources, samples, windows, features, dataset, tmp_path / "report.md", strict=True)

    assert summary["errors"]
    assert any("Duplicate source IDs" in error for error in summary["errors"])
    assert any("not baked-ok" in error for error in summary["errors"])
    assert any("missing from movement_windows" in error for error in summary["errors"])


def test_artifact_manifest_generation_project_relative(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "features").mkdir(parents=True)
    (data_root / "features" / "cowgirl_window_features_v0.jsonl").write_text("", encoding="utf-8")
    (data_root / "features" / "cowgirl_window_features_v1.jsonl").write_text("", encoding="utf-8")

    manifest = prepare_clean_run(data_root, "clean_test", True, data_root / "runs" / "clean_test" / "run_manifest.json", data_root / "runs" / "clean_test" / "report.md")

    assert Path(manifest["run_root"]).exists()
    assert (data_root / "runs" / "clean_test" / "semantic").exists()
    assert any("v0 and v1" in warning for warning in manifest["warnings"])

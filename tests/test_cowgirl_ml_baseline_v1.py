from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vam_timeline_ai.audits.ml_assisted_review_batch import export_ml_assisted_cowgirl_review_v1
from vam_timeline_ai.ml.grouped_splits import split_cowgirl_ml_dataset_v1
from vam_timeline_ai.ml.human_label_dataset import build_human_reviewed_ml_labels_v1
from vam_timeline_ai.ml.supervised_feature_table import build_cowgirl_ml_feature_table_v1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_human_label_extraction_maps_positive_and_negative(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            {"review_id": "r1", "window_id": "w1", "human_semantic_family": "cowgirl", "error_tags": ["clean_cowgirl_motion"], "notes": "normal cowgirl grinding"},
            {"review_id": "r2", "window_id": "w2", "human_semantic_family": "bj_oral", "error_tags": ["not_cowgirl"], "notes": "bj/oral not cowgirl"},
        ],
    )
    out = tmp_path / "labels.jsonl"
    summary = build_human_reviewed_ml_labels_v1(tmp_path, ledger, out, tmp_path / "report.md")
    rows = out.read_text(encoding="utf-8").splitlines()
    assert summary["cowgirl_label_counts"]["true"] == 1
    assert summary["cowgirl_label_counts"]["false"] == 1
    assert len(rows) == 2


def test_feature_table_joins_by_window_id(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, [{"window_id": "w1", "label_cowgirl_candidate": "true", "label_clean_motion": "false", "label_generation_safe": "unknown"}])
    candidate = tmp_path / "cowgirl.jsonl"
    _write_jsonl(candidate, [{"window_id": "w1", "source_scene_file": "s1.json", "technical_actor_id": "Person", "category": "cowgirl_clean_motion_generation_safe", "semantic_family": "cowgirl", "motion_score": 0.9}])
    rel = tmp_path / "rel.jsonl"
    _write_jsonl(rel, [{"window_id": "w1", "feature_values": {"local_motion_energy": 0.5}}])
    empty = tmp_path / "empty.jsonl"
    _write_jsonl(empty, [])
    summary = build_cowgirl_ml_feature_table_v1(tmp_path, labels, rel, empty, empty, empty, empty, empty, empty, candidate, tmp_path / "features.npz", tmp_path / "meta.jsonl", tmp_path / "report.md")
    with np.load(tmp_path / "features.npz", allow_pickle=True) as data:
        assert data["X"].shape[0] == 1
    assert summary["rows"] == 1


def test_grouped_split_prevents_same_scene_leakage(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "features.npz", X=np.zeros((4, 2)), y=np.asarray([[1], [0], [1], [0]], dtype=np.int8), label_names=np.asarray(["label_cowgirl_candidate"], dtype=object))
    _write_jsonl(
        tmp_path / "meta.jsonl",
        [
            {"source_scene_file": "a.json"},
            {"source_scene_file": "a.json"},
            {"source_scene_file": "b.json"},
            {"source_scene_file": "c.json"},
        ],
    )
    summary = split_cowgirl_ml_dataset_v1(tmp_path / "features.npz", tmp_path / "meta.jsonl", tmp_path / "splits", "source_scene_file", 1)
    assert not summary["leakage_warnings"]


def test_ml_assisted_review_excludes_reviewed_windows(tmp_path: Path) -> None:
    scores = tmp_path / "scores.jsonl"
    _write_jsonl(
        scores,
        [
            {"window_id": "w1", "source_scene_file": "s1", "sample_id": "a", "recommended_review_priority": "high_confidence_cowgirl", "model_cowgirl_probability": 0.9},
            {"window_id": "w2", "source_scene_file": "s2", "sample_id": "b", "recommended_review_priority": "high_confidence_cowgirl", "model_cowgirl_probability": 0.8},
        ],
    )
    reviewed = tmp_path / "reviewed.jsonl"
    _write_jsonl(reviewed, [{"window_id": "w1", "duplicate_status": "unique"}])
    summary = export_ml_assisted_cowgirl_review_v1(tmp_path, scores, reviewed, tmp_path / "review", count=2, build_vam_package=False, build_static_ui=False)
    rows = (tmp_path / "review" / "semantic_review_010.jsonl").read_text(encoding="utf-8")
    assert "w1" not in rows
    assert "w2" in rows
    assert summary["exported_count"] == 1


def test_no_manual_labels_path_written(tmp_path: Path) -> None:
    assert not (tmp_path / "manual_labels.yaml").exists()

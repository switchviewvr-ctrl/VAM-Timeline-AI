"""ML dataset v1 with manual and weak labels kept separate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_ml_dataset_v1(features_path: str | Path, windows_path: str | Path, weak_labels_path: str | Path, out: str | Path, report: str | Path) -> dict[str, Any]:
    features = _load_jsonl(features_path)
    windows = {r["window_id"]: r for r in _load_jsonl(windows_path) if r.get("window_id")}
    weak = {r["window_id"]: r for r in _load_jsonl(weak_labels_path) if r.get("window_id")}
    feature_names = sorted({key for row in features for key in row.get("feature_values", {})})
    rows = [row for row in features if row.get("window_id") in windows]
    X = np.asarray([[row.get("feature_values", {}).get(name, np.nan) for name in feature_names] for row in rows], dtype=np.float32)
    manual_label_names = sorted({label for row in rows for label in windows[row["window_id"]].get("labels", [])})
    weak_label_names = sorted({item["label"] for row in rows for item in weak.get(row["window_id"], {}).get("weak_labels", [])})
    manual_y = _label_matrix(rows, windows, manual_label_names, source="manual")
    weak_y = _label_matrix(rows, weak, weak_label_names, source="weak")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        feature_names=np.asarray(feature_names, dtype=object),
        window_ids=np.asarray([r["window_id"] for r in rows], dtype=object),
        sample_ids=np.asarray([r.get("sample_id", "") for r in rows], dtype=object),
        source_ids=np.asarray([r.get("source_id", "") for r in rows], dtype=object),
        source_scene_files=np.asarray([r.get("source_scene_file", "") for r in rows], dtype=object),
        technical_atom_ids=np.asarray([r.get("technical_atom_id", "") for r in rows], dtype=object),
        manual_y_multilabel=manual_y,
        manual_label_names=np.asarray(manual_label_names, dtype=object),
        weak_y_multilabel=weak_y,
        weak_label_names=np.asarray(weak_label_names, dtype=object),
        has_manual_labels=np.asarray(bool(manual_label_names)),
        has_weak_labels=np.asarray(bool(weak_label_names)),
        group_scene=np.asarray([r.get("source_scene_file", "") for r in rows], dtype=object),
        group_sample=np.asarray([r.get("sample_id", "") for r in rows], dtype=object),
        group_source=np.asarray([r.get("source_id", "") for r in rows], dtype=object),
        feature_quality=np.asarray([_quality_score(r) for r in rows], dtype=np.float32),
        metadata_json=json.dumps({"dataset_version": "cowgirl_ml_dataset_v1", "manual_and_weak_labels_separate": True}, ensure_ascii=False),
    )
    summary = {
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "manual_label_count": len(manual_label_names),
        "weak_label_count": len(weak_label_names),
        "shape": list(X.shape),
    }
    _write_report(summary, report)
    return summary


def _label_matrix(rows: list[dict[str, Any]], label_source: dict[str, dict[str, Any]], labels: list[str], source: str) -> np.ndarray:
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int8)
    index = {label: idx for idx, label in enumerate(labels)}
    for r_idx, row in enumerate(rows):
        item = label_source.get(row["window_id"], {})
        row_labels = item.get("labels", []) if source == "manual" else [x["label"] for x in item.get("weak_labels", [])]
        for label in row_labels:
            if label in index:
                matrix[r_idx, index[label]] = 1
    return matrix


def _quality_score(row: dict[str, Any]) -> float:
    q = row.get("feature_quality", {})
    keys = ["has_pelvis_features", "has_torso_features", "has_hand_features", "has_leg_features", "has_head_features"]
    return float(sum(1 for k in keys if q.get(k)) / len(keys))


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Cowgirl ML Dataset v1",
        "",
        f"- Shape: {summary['shape']}",
        f"- Features: {summary['feature_count']}",
        f"- Manual label classes: {summary['manual_label_count']}",
        f"- Weak label classes: {summary['weak_label_count']}",
        "",
        "Manual labels and weak labels are stored separately.",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

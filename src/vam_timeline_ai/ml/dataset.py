"""Build ML-ready datasets from semantic windows and feature rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.cowgirl.feature_extractor import FEATURE_NAMES


def build_ml_dataset_v0(features_path: str | Path, windows_path: str | Path, out: str | Path, report: str | Path) -> dict[str, Any]:
    features = _load_jsonl(features_path)
    windows = {row["window_id"]: row for row in _load_jsonl(windows_path) if row.get("window_id")}
    rows = [row for row in features if row.get("window_id") in windows]
    X = np.asarray([[row.get("features", {}).get(name, np.nan) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    label_names = sorted({label for row in rows for label in windows[row["window_id"]].get("labels", [])})
    y = np.zeros((len(rows), len(label_names)), dtype=np.int8)
    label_index = {label: idx for idx, label in enumerate(label_names)}
    for r_idx, feature_row in enumerate(rows):
        for label in windows[feature_row["window_id"]].get("labels", []):
            if label in label_index:
                y[r_idx, label_index[label]] = 1
    has_manual_labels = bool(label_names)
    quality_mask = np.isfinite(X).any(axis=1) if len(X) else np.zeros((0,), dtype=bool)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        window_ids=np.asarray([row["window_id"] for row in rows], dtype=object),
        sample_ids=np.asarray([row.get("sample_id", "") for row in rows], dtype=object),
        source_scene_files=np.asarray([row.get("source_scene_file", "") for row in rows], dtype=object),
        technical_atom_ids=np.asarray([row.get("technical_atom_id", "") for row in rows], dtype=object),
        y_multilabel=y,
        label_names=np.asarray(label_names, dtype=object),
        has_manual_labels=np.asarray(has_manual_labels),
        feature_quality_mask=quality_mask,
        metadata_json=json.dumps({"dataset_version": "cowgirl_ml_v0", "row_count": len(rows)}, ensure_ascii=False),
    )
    summary = {
        "row_count": len(rows),
        "feature_count": len(FEATURE_NAMES),
        "label_count": len(label_names),
        "has_manual_labels": has_manual_labels,
        "usable_feature_rows": int(np.sum(quality_mask)),
    }
    _write_report(summary, report)
    return summary


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cowgirl ML Dataset v0",
        "",
        f"- Rows: {summary['row_count']}",
        f"- Features: {summary['feature_count']}",
        f"- Usable feature rows: {summary['usable_feature_rows']}",
        f"- Manual label classes: {summary['label_count']}",
        f"- Has manual labels: {summary['has_manual_labels']}",
        "",
        "No supervised model was trained by this command.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

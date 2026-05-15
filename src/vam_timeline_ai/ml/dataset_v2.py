"""ML dataset v2 with positive, negative, uncertain, and weak labels separated."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def build_ml_dataset_v2(
    features_path: str | Path,
    windows_path: str | Path,
    weak_labels_path: str | Path,
    manual_labels_path: str | Path,
    out: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    features = _load_jsonl(features_path)
    windows = {r["window_id"]: r for r in _load_jsonl(windows_path) if r.get("window_id")}
    weak = {r["window_id"]: r for r in _load_jsonl(weak_labels_path) if r.get("window_id")}
    manual = _load_yaml(Path(manual_labels_path)) if Path(manual_labels_path).exists() and "template" not in Path(manual_labels_path).name.lower() else {}
    manual_windows = manual.get("windows", {}) if isinstance(manual.get("windows", {}), dict) else {}

    rows = [row for row in features if row.get("window_id") in windows]
    feature_names = sorted({key for row in rows for key in row.get("feature_values", {})})
    X = np.asarray([[row.get("feature_values", {}).get(name, np.nan) for name in feature_names] for row in rows], dtype=np.float32)

    manual_label_names = sorted({
        str(label)
        for wid in [r.get("window_id") for r in rows]
        for label in _manual_all_labels(manual_windows.get(wid, windows.get(wid, {})))
        if not str(label).startswith("weak_")
    })
    weak_label_names = sorted({str(item.get("label")) for row in rows for item in weak.get(row["window_id"], {}).get("weak_labels", []) if item.get("label")})

    positive = _manual_matrix(rows, windows, manual_windows, manual_label_names, "labels")
    negative = _manual_matrix(rows, windows, manual_windows, manual_label_names, "negative_labels")
    uncertain = _manual_matrix(rows, windows, manual_windows, manual_label_names, "uncertain_labels")
    weak_y = _weak_matrix(rows, weak, weak_label_names)

    include_for_ml = []
    confidence = []
    movement_quality = []
    semantic_role = []
    focus_actor = []
    for row in rows:
        wid = row["window_id"]
        wrow = windows.get(wid, {})
        entry = manual_windows.get(wid, {})
        include_for_ml.append(bool(entry.get("include_for_ml", wrow.get("include_for_ml", True))))
        confidence.append(_float_or_nan(entry.get("confidence", wrow.get("manual_label_confidence"))))
        movement_quality.append(str(entry.get("movement_quality", wrow.get("manual_movement_quality", "unknown"))))
        semantic_role.append(str(entry.get("semantic_role", wrow.get("semantic_role_guess", "unknown"))))
        focus_actor.append(str(entry.get("focus_actor", wrow.get("manual_focus_actor", "unknown"))))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_version": "cowgirl_ml_dataset_v2",
        "manual_positive_negative_uncertain_separated": True,
        "weak_labels_separate": True,
        "manual_labels_path": str(manual_labels_path),
        "manual_label_classes": len(manual_label_names),
        "weak_label_classes": len(weak_label_names),
    }
    np.savez_compressed(
        out_path,
        X=X,
        feature_names=np.asarray(feature_names, dtype=object),
        window_ids=np.asarray([r["window_id"] for r in rows], dtype=object),
        sample_ids=np.asarray([r.get("sample_id", "") for r in rows], dtype=object),
        source_ids=np.asarray([r.get("source_id", "") for r in rows], dtype=object),
        source_scene_files=np.asarray([r.get("source_scene_file", "") for r in rows], dtype=object),
        technical_atom_ids=np.asarray([r.get("technical_atom_id", "") for r in rows], dtype=object),
        manual_y_positive_multilabel=positive,
        manual_y_negative_multilabel=negative,
        manual_y_uncertain_multilabel=uncertain,
        manual_label_names=np.asarray(manual_label_names, dtype=object),
        weak_y_multilabel=weak_y,
        weak_label_names=np.asarray(weak_label_names, dtype=object),
        include_for_ml=np.asarray(include_for_ml, dtype=bool),
        confidence=np.asarray(confidence, dtype=np.float32),
        movement_quality=np.asarray(movement_quality, dtype=object),
        semantic_role=np.asarray(semantic_role, dtype=object),
        focus_actor=np.asarray(focus_actor, dtype=object),
        group_scene=np.asarray([r.get("source_scene_file", "") for r in rows], dtype=object),
        group_sample=np.asarray([r.get("sample_id", "") for r in rows], dtype=object),
        group_source=np.asarray([r.get("source_id", "") for r in rows], dtype=object),
        feature_quality=np.asarray([_quality_score(r) for r in rows], dtype=np.float32),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    summary = {
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "shape": list(X.shape),
        "manual_label_count": len(manual_label_names),
        "weak_label_count": len(weak_label_names),
        "manual_positive_assignments": int(positive.sum()),
        "manual_negative_assignments": int(negative.sum()),
        "manual_uncertain_assignments": int(uncertain.sum()),
        "include_for_ml_true": int(np.asarray(include_for_ml, dtype=bool).sum()),
    }
    _write_report(summary, report)
    return summary


def _manual_all_labels(entry: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ["labels", "negative_labels", "uncertain_labels"]:
        labels.extend(entry.get(key, []) or [])
    return labels


def _manual_matrix(rows: list[dict[str, Any]], windows: dict[str, dict[str, Any]], manual_windows: dict[str, dict[str, Any]], labels: list[str], field: str) -> np.ndarray:
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int8)
    index = {label: idx for idx, label in enumerate(labels)}
    for r_idx, row in enumerate(rows):
        wid = row["window_id"]
        entry = manual_windows.get(wid, windows.get(wid, {}))
        for label in entry.get(field, []) or []:
            if label in index:
                matrix[r_idx, index[label]] = 1
    return matrix


def _weak_matrix(rows: list[dict[str, Any]], weak: dict[str, dict[str, Any]], labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int8)
    index = {label: idx for idx, label in enumerate(labels)}
    for r_idx, row in enumerate(rows):
        for item in weak.get(row["window_id"], {}).get("weak_labels", []) or []:
            label = item.get("label")
            if label in index:
                matrix[r_idx, index[label]] = 1
    return matrix


def _float_or_nan(value: Any) -> float:
    try:
        if value in {None, "", "manual"}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _quality_score(row: dict[str, Any]) -> float:
    q = row.get("feature_quality", {})
    keys = ["has_pelvis_features", "has_torso_features", "has_hand_features", "has_leg_features", "has_head_features"]
    return float(sum(1 for k in keys if q.get(k)) / len(keys))


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Cowgirl ML Dataset v2",
        "",
        f"- Shape: {summary['shape']}",
        f"- Features: {summary['feature_count']}",
        f"- Manual label classes: {summary['manual_label_count']}",
        f"- Weak label classes: {summary['weak_label_count']}",
        f"- Manual positive assignments: {summary['manual_positive_assignments']}",
        f"- Manual negative assignments: {summary['manual_negative_assignments']}",
        f"- Manual uncertain assignments: {summary['manual_uncertain_assignments']}",
        f"- Include for ML true rows: {summary['include_for_ml_true']}",
        "",
        "Positive, negative, uncertain, and weak labels are stored separately.",
        "Weak labels are not training targets.",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

"""ML dataset v3 with manual, weak, and silver labels separated."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vam_timeline_ai.io.json_utils import load_jsonl


def build_ml_dataset_v3(
    features_path: str | Path,
    windows_path: str | Path,
    weak_labels_path: str | Path,
    manual_labels_path: str | Path,
    silver_labels_path: str | Path,
    out: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    features = load_jsonl(features_path)
    windows = {r["window_id"]: r for r in load_jsonl(windows_path) if r.get("window_id")}
    weak = {r["window_id"]: r for r in load_jsonl(weak_labels_path) if r.get("window_id")}
    silver = _load_silver_by_window(silver_labels_path)
    manual = _load_manual(Path(manual_labels_path))
    manual_windows = manual.get("windows", {}) if isinstance(manual.get("windows", {}), dict) else {}

    rows = [row for row in features if row.get("window_id") in windows]
    feature_names = sorted({key for row in rows for key in row.get("feature_values", {})})
    X = np.asarray([[row.get("feature_values", {}).get(name, np.nan) for name in feature_names] for row in rows], dtype=np.float32)

    manual_label_names = sorted(
        {
            str(label)
            for wid in [r.get("window_id") for r in rows]
            for label in _manual_all_labels(manual_windows.get(wid, windows.get(wid, {})))
            if label and not str(label).startswith("weak_")
        }
    )
    weak_label_names = sorted(
        {
            str(item.get("label"))
            for row in rows
            for item in weak.get(row["window_id"], {}).get("weak_labels", [])
            if item.get("label")
        }
    )
    silver_label_names = sorted(
        {
            str(label)
            for row in rows
            for label in _silver_all_labels(silver.get(row["window_id"], {}))
            if label and not str(label).startswith("weak_")
        }
    )

    manual_pos = _manual_matrix(rows, windows, manual_windows, manual_label_names, "labels")
    manual_neg = _manual_matrix(rows, windows, manual_windows, manual_label_names, "negative_labels")
    manual_unc = _manual_matrix(rows, windows, manual_windows, manual_label_names, "uncertain_labels")
    weak_y = _weak_matrix(rows, weak, weak_label_names)
    silver_pos = _silver_matrix(rows, silver, silver_label_names, ["positive_labels", "role_candidates", "contact_candidates"])
    silver_neg = _silver_matrix(rows, silver, silver_label_names, ["negative_labels"])
    silver_conf = _silver_confidence(rows, silver, silver_label_names)

    include_for_ml = []
    conf_manual = []
    conf_silver = []
    movement_quality = []
    semantic_role = []
    focus_actor = []
    for row in rows:
        wid = row["window_id"]
        wrow = windows.get(wid, {})
        entry = manual_windows.get(wid, {})
        include_for_ml.append(bool(entry.get("include_for_ml", wrow.get("include_for_ml", True))))
        conf_manual.append(_float_or_nan(entry.get("confidence", wrow.get("manual_label_confidence"))))
        conf_silver.append(max((float(v) for v in (silver.get(wid, {}).get("confidence_by_label", {}) or {}).values()), default=float("nan")))
        movement_quality.append(str(entry.get("movement_quality", wrow.get("manual_movement_quality", "unknown"))))
        semantic_role.append(str(entry.get("semantic_role", wrow.get("semantic_role_guess", "unknown"))))
        focus_actor.append(str(entry.get("focus_actor", wrow.get("manual_focus_actor", "unknown"))))

    metadata = {
        "dataset_version": "cowgirl_ml_dataset_v3",
        "manual_labels_separate": True,
        "silver_labels_separate": True,
        "weak_labels_separate": True,
        "silver_is_human_ground_truth": False,
        "manual_labels_path": str(manual_labels_path),
        "silver_labels_path": str(silver_labels_path),
        "warning": "Silver labels are machine-generated proxy labels, not human semantic ground truth.",
    }
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
        manual_y_positive_multilabel=manual_pos,
        manual_y_negative_multilabel=manual_neg,
        manual_y_uncertain_multilabel=manual_unc,
        manual_label_names=np.asarray(manual_label_names, dtype=object),
        silver_y_positive_multilabel=silver_pos,
        silver_y_negative_multilabel=silver_neg,
        silver_label_names=np.asarray(silver_label_names, dtype=object),
        silver_confidence=silver_conf,
        weak_y_multilabel=weak_y,
        weak_label_names=np.asarray(weak_label_names, dtype=object),
        include_for_ml=np.asarray(include_for_ml, dtype=bool),
        confidence_manual=np.asarray(conf_manual, dtype=np.float32),
        confidence_silver=np.asarray(conf_silver, dtype=np.float32),
        confidence=np.asarray(conf_manual, dtype=np.float32),
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
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "manual_label_count": len(manual_label_names),
        "silver_label_count": len(silver_label_names),
        "weak_label_count": len(weak_label_names),
        "manual_positive_assignments": int(manual_pos.sum()),
        "manual_negative_assignments": int(manual_neg.sum()),
        "manual_uncertain_assignments": int(manual_unc.sum()),
        "silver_positive_assignments": int(silver_pos.sum()),
        "silver_negative_assignments": int(silver_neg.sum()),
        "weak_assignments": int(weak_y.sum()),
    }
    _write_report(summary, report)
    return summary


def _manual_all_labels(entry: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ["labels", "negative_labels", "uncertain_labels"]:
        labels.extend(entry.get(key, []) or [])
    return labels


def _silver_all_labels(entry: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ["positive_labels", "negative_labels", "role_candidates", "contact_candidates"]:
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


def _silver_matrix(rows: list[dict[str, Any]], silver: dict[str, dict[str, Any]], labels: list[str], fields: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int8)
    index = {label: idx for idx, label in enumerate(labels)}
    for r_idx, row in enumerate(rows):
        entry = silver.get(row["window_id"], {})
        for field in fields:
            for label in entry.get(field, []) or []:
                if label in index:
                    matrix[r_idx, index[label]] = 1
    return matrix


def _silver_confidence(rows: list[dict[str, Any]], silver: dict[str, dict[str, Any]], labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(labels)), dtype=np.float32)
    index = {label: idx for idx, label in enumerate(labels)}
    for r_idx, row in enumerate(rows):
        confs = silver.get(row["window_id"], {}).get("confidence_by_label", {}) or {}
        for label, value in confs.items():
            if label in index:
                matrix[r_idx, index[label]] = float(value)
    return matrix


def _load_manual(path: Path) -> dict[str, Any]:
    if not path.exists() or "template" in path.name.lower():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_silver_by_window(path: str | Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        wid = row.get("window_id")
        if not wid:
            continue
        entry = grouped.setdefault(
            str(wid),
            {
                "positive_labels": set(),
                "negative_labels": set(),
                "uncertain_labels": set(),
                "role_candidates": set(),
                "contact_candidates": set(),
                "confidence_by_label": {},
            },
        )
        for key in ["positive_labels", "negative_labels", "uncertain_labels", "role_candidates", "contact_candidates"]:
            entry[key].update(row.get(key, []) or [])
        for label, value in (row.get("confidence_by_label", {}) or {}).items():
            try:
                conf = float(value)
            except Exception:
                conf = 0.0
            entry["confidence_by_label"][label] = max(float(entry["confidence_by_label"].get(label, 0.0)), conf)
    return {
        wid: {
            "positive_labels": sorted(entry["positive_labels"]),
            "negative_labels": sorted(entry["negative_labels"]),
            "uncertain_labels": sorted(entry["uncertain_labels"]),
            "role_candidates": sorted(entry["role_candidates"]),
            "contact_candidates": sorted(entry["contact_candidates"]),
            "confidence_by_label": {k: round(float(v), 5) for k, v in sorted(entry["confidence_by_label"].items())},
        }
        for wid, entry in grouped.items()
    }


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
        "# Cowgirl ML Dataset v3",
        "",
        f"- Shape: {summary['shape']}",
        f"- Features: {summary['feature_count']}",
        f"- Manual label classes: {summary['manual_label_count']}",
        f"- Silver label classes: {summary['silver_label_count']}",
        f"- Weak label classes: {summary['weak_label_count']}",
        f"- Manual positive assignments: {summary['manual_positive_assignments']}",
        f"- Manual negative assignments: {summary['manual_negative_assignments']}",
        f"- Manual uncertain assignments: {summary['manual_uncertain_assignments']}",
        f"- Silver positive assignments: {summary['silver_positive_assignments']}",
        f"- Silver negative assignments: {summary['silver_negative_assignments']}",
        f"- Weak assignments: {summary['weak_assignments']}",
        "",
        "Manual, silver, and weak labels are separate arrays. Silver labels are machine-generated proxy labels, not human semantic ground truth.",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

"""ML dataset v4 using aggregated silver v2 labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vam_timeline_ai.io.json_utils import load_jsonl


def build_ml_dataset_v4(
    features_path: str | Path,
    windows_path: str | Path,
    weak_labels_path: str | Path,
    manual_labels_path: str | Path,
    silver_window_labels_path: str | Path,
    silver_pair_labels_path: str | Path,
    out: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    features = load_jsonl(features_path)
    windows = {r["window_id"]: r for r in load_jsonl(windows_path) if r.get("window_id")}
    weak = {r["window_id"]: r for r in load_jsonl(weak_labels_path) if r.get("window_id")}
    manual = _load_manual(Path(manual_labels_path))
    manual_windows = manual.get("windows", {}) if isinstance(manual.get("windows", {}), dict) else {}
    silver_window = {r["window_id"]: r for r in load_jsonl(silver_window_labels_path) if r.get("window_id")}
    silver_pair = _pair_labels_by_window(load_jsonl(silver_pair_labels_path))

    rows = [row for row in features if row.get("window_id") in windows]
    feature_names = sorted({key for row in rows for key in row.get("feature_values", {})})
    X = np.asarray([[row.get("feature_values", {}).get(name, np.nan) for name in feature_names] for row in rows], dtype=np.float32)
    manual_label_names = sorted({str(label) for row in rows for label in _manual_all_labels(manual_windows.get(row.get("window_id"), windows.get(row.get("window_id"), {}))) if label and not str(label).startswith("weak_")})
    weak_label_names = sorted({str(item.get("label")) for row in rows for item in weak.get(row["window_id"], {}).get("weak_labels", []) if item.get("label")})
    silver_window_label_names = sorted({str(label) for row in rows for label in _silver_labels(silver_window.get(row["window_id"], {}), include_review=False)})
    silver_pair_label_names = sorted({str(label) for row in rows for label in _silver_labels(silver_pair.get(row["window_id"], {}), include_review=False)})
    silver_label_names = sorted(set(silver_window_label_names) | set(silver_pair_label_names))

    manual_pos = _manual_matrix(rows, windows, manual_windows, manual_label_names, "labels")
    manual_neg = _manual_matrix(rows, windows, manual_windows, manual_label_names, "negative_labels")
    manual_unc = _manual_matrix(rows, windows, manual_windows, manual_label_names, "uncertain_labels")
    weak_y = _weak_matrix(rows, weak, weak_label_names)
    silver_window_y = _silver_matrix(rows, silver_window, silver_window_label_names)
    silver_pair_y = _silver_matrix(rows, silver_pair, silver_pair_label_names)
    silver_window_scores = _silver_scores(rows, silver_window, silver_window_label_names)
    silver_pair_scores = _silver_scores(rows, silver_pair, silver_pair_label_names)

    excluded_reasons = _excluded_reasons(silver_window, silver_pair, silver_label_names, silver_pair_label_names)
    default_mask = np.asarray([label not in excluded_reasons for label in silver_label_names], dtype=bool)
    metadata = {
        "dataset_version": "cowgirl_ml_dataset_v4",
        "manual_weak_silver_v2_separated": True,
        "silver_source": "silver_machine_v2",
        "silver_is_human_ground_truth": False,
        "default_training_uses_silver_v2_only": True,
        "warning": "Silver v2 labels are aggregated machine labels, not human semantic ground truth.",
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
        weak_y_multilabel=weak_y,
        weak_label_names=np.asarray(weak_label_names, dtype=object),
        silver_v2_window_y_multilabel=silver_window_y,
        silver_v2_pair_y_multilabel=silver_pair_y,
        silver_v2_window_scores=silver_window_scores,
        silver_v2_pair_scores=silver_pair_scores,
        silver_window_label_names=np.asarray(silver_window_label_names, dtype=object),
        silver_pair_label_names=np.asarray(silver_pair_label_names, dtype=object),
        silver_label_names=np.asarray(silver_label_names, dtype=object),
        default_trainable_silver_label_mask=default_mask,
        excluded_silver_label_names=np.asarray(sorted(excluded_reasons), dtype=object),
        exclusion_reasons=np.asarray([excluded_reasons[label] for label in sorted(excluded_reasons)], dtype=object),
        group_scene=np.asarray([r.get("source_scene_file", "") for r in rows], dtype=object),
        group_sample=np.asarray([r.get("sample_id", "") for r in rows], dtype=object),
        group_source=np.asarray([r.get("source_id", "") for r in rows], dtype=object),
        feature_quality=np.asarray([_quality_score(r) for r in rows], dtype=np.float32),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    summary = {
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "manual_label_classes": len(manual_label_names),
        "weak_label_classes": len(weak_label_names),
        "silver_window_label_classes": len(silver_window_label_names),
        "silver_pair_label_classes": len(silver_pair_label_names),
        "silver_label_classes": len(silver_label_names),
        "silver_window_assignments": int(silver_window_y.sum()),
        "silver_pair_assignments": int(silver_pair_y.sum()),
        "default_trainable_silver_labels": [label for label, keep in zip(silver_label_names, default_mask.tolist()) if keep],
        "excluded_silver_labels": excluded_reasons,
    }
    _write_report(summary, report)
    return summary


def _pair_labels_by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        labels = _silver_labels(row, include_review=False)
        for wid in row.get("window_ids", []) or []:
            entry = out.setdefault(str(wid), {"positive_labels": set(), "negative_labels": set(), "scores_by_label": {}, "excluded_from_default_training": {}})
            entry["positive_labels"].update(row.get("positive_labels", []) or [])
            entry["negative_labels"].update(row.get("negative_labels", []) or [])
            entry["excluded_from_default_training"].update(row.get("excluded_from_default_training", {}) or {})
            for label, score in (row.get("scores_by_label", {}) or {}).items():
                entry["scores_by_label"][label] = max(float(entry["scores_by_label"].get(label, 0.0)), _float(score))
    return {
        wid: {
            "positive_labels": sorted(entry["positive_labels"]),
            "negative_labels": sorted(entry["negative_labels"]),
            "scores_by_label": entry["scores_by_label"],
            "excluded_from_default_training": entry["excluded_from_default_training"],
        }
        for wid, entry in out.items()
    }


def _silver_labels(entry: dict[str, Any], include_review: bool = False) -> list[str]:
    labels: list[str] = []
    labels.extend(entry.get("positive_labels", []) or [])
    labels.extend(entry.get("negative_labels", []) or [])
    if include_review:
        labels.extend(entry.get("review_only_labels", []) or [])
    return labels


def _excluded_reasons(window: dict[str, dict[str, Any]], pair: dict[str, dict[str, Any]], labels: list[str], pair_label_names: list[str]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for label in pair_label_names:
        reasons[label] = "pair-specific label; excluded from single-window default proxy training"
    for source in [window, pair]:
        for row in source.values():
            for label, reason in (row.get("excluded_from_default_training", {}) or {}).items():
                reasons[str(label)] = str(reason)
    for label in labels:
        if label in {"rider_active", "receiver_passive", "partner_context_static"}:
            reasons[label] = "high-risk role proxy label"
        if label == "contact_unknown":
            reasons[label] = "ambiguous contact_unknown label"
    return reasons


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
            if item.get("label") in index:
                matrix[r_idx, index[item["label"]]] = 1
    return matrix


def _silver_matrix(rows: list[dict[str, Any]], silver: dict[str, dict[str, Any]], labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int8)
    index = {label: idx for idx, label in enumerate(labels)}
    for r_idx, row in enumerate(rows):
        for label in _silver_labels(silver.get(row["window_id"], {}), include_review=False):
            if label in index:
                matrix[r_idx, index[label]] = 1
    return matrix


def _silver_scores(rows: list[dict[str, Any]], silver: dict[str, dict[str, Any]], labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(labels)), dtype=np.float32)
    index = {label: idx for idx, label in enumerate(labels)}
    for r_idx, row in enumerate(rows):
        scores = silver.get(row["window_id"], {}).get("scores_by_label", {}) or {}
        for label, value in scores.items():
            if label in index:
                matrix[r_idx, index[label]] = _float(value)
    return matrix


def _load_manual(path: Path) -> dict[str, Any]:
    if not path.exists() or "template" in path.name.lower():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _quality_score(row: dict[str, Any]) -> float:
    q = row.get("feature_quality", {})
    keys = ["has_pelvis_features", "has_torso_features", "has_hand_features", "has_leg_features", "has_head_features"]
    return float(sum(1 for k in keys if q.get(k)) / len(keys))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Cowgirl ML Dataset v4",
        "",
        f"- Shape: {summary['shape']}",
        f"- Manual label classes: {summary['manual_label_classes']}",
        f"- Weak label classes: {summary['weak_label_classes']}",
        f"- Silver window label classes: {summary['silver_window_label_classes']}",
        f"- Silver pair label classes: {summary['silver_pair_label_classes']}",
        f"- Silver window assignments: {summary['silver_window_assignments']}",
        f"- Silver pair assignments: {summary['silver_pair_assignments']}",
        f"- Default trainable silver labels: {summary['default_trainable_silver_labels'] or 'None'}",
        "",
        "Manual, weak, silver-v2 window labels, and silver-v2 pair labels are stored separately.",
        "Silver v2 labels are still machine-generated proxy labels, not human semantic ground truth.",
        "",
        "## Excluded Silver Labels",
        "",
    ]
    if summary["excluded_silver_labels"]:
        lines.extend(f"- `{label}`: {reason}" for label, reason in summary["excluded_silver_labels"].items())
    else:
        lines.append("- None")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

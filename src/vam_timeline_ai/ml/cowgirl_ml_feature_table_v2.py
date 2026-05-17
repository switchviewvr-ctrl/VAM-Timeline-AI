"""Feature table for Cowgirl ML v2 review-ranker."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ml.cowgirl_ml_label_dataset_v2 import LABEL_KEYS, UNKNOWN


CATEGORICAL_FIELDS = [
    "category",
    "resolved_motion_family",
    "resolved_motion_subtype",
    "resolved_semantic_family",
    "motion_state",
    "pose_family",
    "pose_subtype",
    "primary_driver_controller",
    "primary_motion_center",
    "dominant_axis",
    "motion_pattern",
    "controller_gate_result",
    "transform_distance_gate_result",
    "motion_pattern_gate_result",
    "final_clean_motion_gate",
    "driver_gate_result",
    "cycle_gate_result",
    "anchor_gate_result",
    "pose_preservation_gate_result",
    "partner_alignment_gate_result",
    "break_state_gate_result",
    "anchor_stability_status",
    "contact_support",
    "target_region",
    "old_system_guess",
    "old_motion_subtype",
    "cycle_hipControl_cowgirl_pattern",
    "cycle_pelvisControl_cowgirl_pattern",
    "cycle_headControl_cowgirl_pattern",
    "cycle_chestControl_cowgirl_pattern",
    "cycle_lHandControl_cowgirl_pattern",
    "cycle_rHandControl_cowgirl_pattern",
    "controller_completeness_status",
]

IMPORTANT_CONTROLLERS = [
    "hipControl",
    "pelvisControl",
    "headControl",
    "chestControl",
    "abdomenControl",
    "abdomen2Control",
    "lHandControl",
    "rHandControl",
    "lFootControl",
    "rFootControl",
    "lKneeControl",
    "rKneeControl",
    "lThighControl",
    "rThighControl",
]


def build_cowgirl_ml_feature_table_v2(
    labels: str | Path,
    pose_resolved: str | Path,
    cycle_features: str | Path,
    motion_resolved: str | Path,
    candidates: str | Path,
    manual_gt: str | Path,
    out_npz: str | Path,
    out_meta: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    label_rows = load_jsonl(labels)
    candidate_rows = load_jsonl(candidates)
    pose_by_window = _by_window(load_jsonl(pose_resolved))
    motion_by_window = _by_window(load_jsonl(motion_resolved))
    cycle_by_window = _by_window(load_jsonl(cycle_features))
    labels_by_window = _labels_by_window(label_rows)
    manual_gt_rows = load_jsonl(manual_gt)

    rows: list[dict[str, Any]] = []
    y_rows: list[list[int]] = []
    for candidate in candidate_rows:
        row = dict(candidate)
        wid = str(row.get("window_id") or "")
        _merge_prefixed(row, pose_by_window.get(wid, {}), "pose")
        _merge_prefixed(row, motion_by_window.get(wid, {}), "motion")
        _add_cycle_features(row, cycle_by_window.get(wid, {}))
        _add_controller_completeness(row, cycle_by_window.get(wid, {}))
        _add_manual_gt_hint_features(row, manual_gt_rows)
        label = labels_by_window.get(wid) or _match_label_by_scene_time(row, label_rows)
        if label:
            row["human_label_source_kind"] = label.get("source_kind")
            row["human_review_id"] = label.get("review_id")
        for key in LABEL_KEYS:
            row[key] = (label or {}).get(key, UNKNOWN)
        rows.append(row)
        y_rows.append([_label_value(row.get(key)) for key in LABEL_KEYS])

    feature_names, X = vectorize_records(rows)
    y = np.asarray(y_rows, dtype=np.int8)
    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X.astype(np.float32),
        y=y,
        label_names=np.asarray(LABEL_KEYS, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        metadata_json=json.dumps({"schema": "cowgirl_ml_feature_table_v2", "row_count": len(rows)}, ensure_ascii=False),
    )
    meta_rows = [_metadata_row(row, idx) for idx, row in enumerate(rows)]
    write_jsonl(out_meta, meta_rows)
    summary = {
        "status": "ok",
        "schema": "cowgirl_ml_feature_table_v2",
        "rows": len(rows),
        "features": len(feature_names),
        "shape": [int(X.shape[0]), int(X.shape[1]) if X.ndim == 2 else 0],
        "label_counts": {key: dict(Counter(row.get(key, UNKNOWN) for row in rows)) for key in LABEL_KEYS},
        "labeled_rows": sum(1 for row in rows if any(row.get(k) in {"true", "false"} for k in LABEL_KEYS)),
        "manual_gt_reference_rows": len(manual_gt_rows),
        "feature_groups": [
            "pose_first",
            "motion_cycle",
            "driver",
            "anchor_gate",
            "controller_completeness",
            "existing_category_as_feature",
        ],
        "feature_names": feature_names,
    }
    _write_report(report, summary)
    return summary


def vectorize_records(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    numeric = sorted({k for row in rows for k in _numeric_features(row)})
    cats = sorted({f"{field}={_cat(row.get(field))}" for row in rows for field in CATEGORICAL_FIELDS if _cat(row.get(field))})
    names = numeric + cats
    return names, vectorize_with_feature_names(rows, names)


def vectorize_with_feature_names(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        nums = _numeric_features(row)
        cats = {f"{field}={_cat(row.get(field))}" for field in CATEGORICAL_FIELDS if _cat(row.get(field))}
        for j, name in enumerate(feature_names):
            if name in nums:
                X[i, j] = nums[name]
            elif name in cats:
                X[i, j] = 1.0
    return X


def _by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("window_id")): row for row in rows if row.get("window_id")}


def _labels_by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        wid = str(row.get("window_id") or "")
        if wid and wid not in out:
            out[wid] = row
    return out


def _match_label_by_scene_time(row: dict[str, Any], labels: list[dict[str, Any]]) -> dict[str, Any] | None:
    scene = str(row.get("source_scene_file") or "")
    actor = str(row.get("technical_actor_id") or row.get("technical_atom_id") or "")
    start = _float(row.get("start_seconds"))
    end = _float(row.get("end_seconds"))
    best = None
    best_overlap = 0.0
    for label in labels:
        if label.get("window_id"):
            continue
        if scene and str(label.get("source_scene_file") or "") not in {"", scene}:
            continue
        if actor and str(label.get("technical_actor_id") or "") not in {"", actor}:
            continue
        ls = _float(label.get("start_seconds"))
        le = _float(label.get("end_seconds"))
        if start is None or end is None or ls is None or le is None:
            continue
        overlap = max(0.0, min(end, le) - max(start, ls))
        if overlap > best_overlap:
            best = label
            best_overlap = overlap
    return best if best_overlap > 0 else None


def _merge_prefixed(row: dict[str, Any], extra: dict[str, Any], prefix: str) -> None:
    for key, value in extra.items():
        if key in {"window_id", "sample_id", "source_id"}:
            continue
        if key not in row:
            row[key] = value
        elif row.get(key) != value:
            row.setdefault(f"{prefix}_{key}", value)


def _add_cycle_features(row: dict[str, Any], cycle: dict[str, Any]) -> None:
    metrics = cycle.get("controller_metrics") if isinstance(cycle.get("controller_metrics"), dict) else {}
    for controller in IMPORTANT_CONTROLLERS:
        cm = metrics.get(controller) if isinstance(metrics.get(controller), dict) else {}
        prefix = f"cycle_{controller}"
        for key in ["max_displacement_range", "total_path_length", "cyclicity_score", "transition_score", "pose_hold_score", "estimated_cycle_count", "estimated_frequency_hz"]:
            row[f"{prefix}_{key}"] = _float(cm.get(key), 0.0)
        row[f"{prefix}_present"] = 1.0 if cm else 0.0
        row[f"{prefix}_has_real_motion"] = 1.0 if cm.get("has_real_motion") else 0.0
        row[f"{prefix}_net_displacement_distance"] = _float(cm.get("net_displacement_distance"), 0.0)
        row[f"{prefix}_net_to_path_ratio"] = _float(cm.get("net_to_path_ratio"), 0.0)
        row[f"{prefix}_moving_step_count"] = _float(cm.get("moving_step_count"), 0.0)
        row[f"{prefix}_active_axis_count"] = _float(cm.get("active_axis_count"), 0.0)
        row[f"{prefix}_dominant_axis"] = cm.get("dominant_axis") or "missing"
        pattern = cm.get("cowgirl_motion_pattern") if isinstance(cm.get("cowgirl_motion_pattern"), dict) else {}
        row[f"{prefix}_cowgirl_pattern"] = pattern.get("pattern") or "missing"
        row[f"{prefix}_valid_clean_cowgirl_pattern"] = 1.0 if pattern.get("valid_clean_cowgirl_pattern") else 0.0
        row[f"{prefix}_valid_soft_cowgirl_pattern"] = 1.0 if pattern.get("valid_soft_cowgirl_pattern") else 0.0
        row[f"{prefix}_cowgirl_pattern_net_to_path_ratio"] = _float(pattern.get("net_to_path_ratio"), 0.0)
        axis_metrics = cm.get("axis_metrics") if isinstance(cm.get("axis_metrics"), dict) else {}
        for axis in ["x", "y", "z"]:
            am = axis_metrics.get(axis) if isinstance(axis_metrics.get(axis), dict) else {}
            row[f"{prefix}_{axis}_range"] = _float(am.get("displacement_range"), 0.0)
            row[f"{prefix}_{axis}_cycles"] = _float(am.get("estimated_cycle_count"), 0.0)
            row[f"{prefix}_{axis}_cyclicity"] = _float(am.get("cyclicity_score"), 0.0)
            row[f"{prefix}_{axis}_transition"] = _float(am.get("transition_score"), 0.0)


def _add_controller_completeness(row: dict[str, Any], cycle: dict[str, Any]) -> None:
    metrics = cycle.get("controller_metrics") if isinstance(cycle.get("controller_metrics"), dict) else {}
    summary = cycle.get("controller_completeness_summary") if isinstance(cycle.get("controller_completeness_summary"), dict) else {}
    present = set(metrics)
    row["has_hipControl"] = "hipControl" in present
    row["has_pelvisControl"] = "pelvisControl" in present
    row["has_lHandControl"] = "lHandControl" in present
    row["has_rHandControl"] = "rHandControl" in present
    row["has_hand_controls"] = row["has_lHandControl"] and row["has_rHandControl"]
    row["missing_hand_controllers"] = not row["has_hand_controls"]
    row["has_foot_controls"] = "lFootControl" in present and "rFootControl" in present
    row["missing_core_controllers"] = not (row["has_hipControl"] and row["has_pelvisControl"] and row["has_foot_controls"])
    row["controller_completeness_status"] = summary.get("status") or "unknown"
    row["controller_count"] = _float(summary.get("controller_count"), 0.0)
    row["only_feet_controllers"] = 1.0 if summary.get("only_feet") else 0.0
    row["only_upper_body_hands_head_controllers"] = 1.0 if summary.get("only_upper_body_hands_head") else 0.0
    row["has_real_hip_motion"] = 1.0 if summary.get("has_real_hip_motion") else 0.0
    row["has_real_pelvis_motion"] = 1.0 if summary.get("has_real_pelvis_motion") else 0.0


def _add_manual_gt_hint_features(row: dict[str, Any], manual_gt: list[dict[str, Any]]) -> None:
    hint = row.get("manual_gt_reference_hint") if isinstance(row.get("manual_gt_reference_hint"), dict) else {}
    row["manual_gt_matching_family_examples"] = _float(hint.get("matching_family_examples"), 0.0)
    row["manual_gt_matching_subtype_examples"] = _float(hint.get("matching_subtype_examples"), 0.0)
    row["manual_gt_available_reference_count"] = len(manual_gt)


def _numeric_features(row: dict[str, Any]) -> dict[str, float]:
    skip = set(LABEL_KEYS) | {
        "review_id",
        "human_review_id",
        "window_id",
        "sample_id",
        "source_id",
        "source_scene_file",
        "source_scene_path",
        "technical_actor_id",
        "technical_atom_id",
        "start_seconds",
        "end_seconds",
        "candidate_id",
        "explanation",
        "why_not_cowgirl",
        "human_label_source_kind",
    }
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            val = float(value)
            if math.isfinite(val):
                out[key] = val
    return out


def _metadata_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    keys = [
        "window_id",
        "sample_id",
        "source_id",
        "source_scene_file",
        "technical_actor_id",
        "technical_atom_id",
        "start_seconds",
        "end_seconds",
        "category",
        "resolved_motion_family",
        "resolved_motion_subtype",
        "motion_state",
        "primary_driver_controller",
        "cycle_count",
        "cyclicity_score",
        "transition_score",
        "final_clean_motion_gate",
        "anchor_stability_status",
        "contact_support",
        "human_label_source_kind",
        "human_review_id",
    ]
    out = {"row_index": idx}
    for key in keys:
        out[key] = row.get(key)
    for key in LABEL_KEYS:
        out[key] = row.get(key, UNKNOWN)
    return out


def _label_value(value: Any) -> int:
    text = str(value or UNKNOWN).lower()
    if text == "true":
        return 1
    if text == "false":
        return 0
    return -1


def _cat(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value if str(v))
    if isinstance(value, dict):
        return ""
    return str(value).strip()


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {None, ""}:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cowgirl ML Feature Table v2",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Features: `{summary['features']}`",
        f"- Shape: `{summary['shape']}`",
        f"- Labeled rows: `{summary['labeled_rows']}`",
        f"- Feature groups: `{summary['feature_groups']}`",
        "",
        "## Label Counts",
        "",
    ]
    for key, counts in summary["label_counts"].items():
        lines.append(f"- `{key}`: `{counts}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

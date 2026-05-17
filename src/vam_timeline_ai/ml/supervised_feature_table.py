"""Build feature matrices for Cowgirl review-assist ML."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


LABEL_KEYS = ["label_cowgirl_candidate", "label_clean_motion", "label_generation_safe"]
CATEGORICAL_FIELDS = [
    "pose_family",
    "pose_subtype",
    "motion_subtype",
    "phase",
    "interaction_family",
    "contact_support",
    "clean_motion_gate",
    "trajectory_shape_classification",
    "dominant_motion_plane",
    "support_context",
    "facing_context",
    "torso_lean_direction",
]


def build_cowgirl_ml_feature_table_v1(
    run_dir: str | Path,
    labels: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    pose_features: str | Path,
    pose_semantics: str | Path,
    partner_relative_features: str | Path,
    interaction_semantics: str | Path,
    semantic_actions: str | Path,
    candidate_db: str | Path,
    out_npz: str | Path,
    out_meta: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    label_rows = load_jsonl(labels)
    sources = _load_feature_sources(
        relative_features,
        trajectory_features,
        pose_features,
        pose_semantics,
        partner_relative_features,
        interaction_semantics,
        semantic_actions,
        candidate_db,
    )
    candidate_rows = sources["candidate_db"]
    feature_records = _merged_records(candidate_rows, sources)
    matched = _match_labels_to_records(label_rows, feature_records)
    rows = [r for r in matched if any(r.get(k) in {"true", "false"} for k in LABEL_KEYS)]
    if not rows:
        rows = []
    feature_names, X = vectorize_records(rows)
    y = np.asarray([[_label_value(r.get(k)) for k in LABEL_KEYS] for r in rows], dtype=np.int8)
    target = Path(out_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        X=X.astype(np.float32),
        y=y,
        label_names=np.asarray(LABEL_KEYS, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        metadata_json=json.dumps({"schema": "cowgirl_ml_feature_table_v1", "row_count": len(rows)}, ensure_ascii=False),
    )
    metadata = [_metadata_row(r, idx) for idx, r in enumerate(rows)]
    write_jsonl(out_meta, metadata)
    summary = {
        "status": "ok",
        "schema": "cowgirl_ml_feature_table_v1",
        "rows": len(rows),
        "features": len(feature_names),
        "shape": [int(X.shape[0]), int(X.shape[1]) if X.ndim == 2 else 0],
        "label_counts": {k: dict(Counter(r.get(k, "unknown") for r in rows)) for k in LABEL_KEYS},
        "matched_label_rows": len(rows),
        "input_label_rows": len(label_rows),
        "candidate_rows": len(candidate_rows),
        "fallbacks": _fallbacks(semantic_actions, candidate_db),
        "feature_names": feature_names,
    }
    _write_report(report, summary)
    return summary


def build_all_candidate_feature_matrix(run_dir: str | Path, feature_names: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    run = Path(run_dir)
    sources = _load_feature_sources(
        run / "relative_motion" / "relative_motion_features.jsonl",
        run / "relative_motion" / "trajectory_shape_features.jsonl",
        run / "pose_semantics" / "pose_features_v0.jsonl",
        run / "pose_semantics" / "pose_semantics_v0.jsonl",
        run / "interaction_semantics" / "partner_relative_features_v0.jsonl",
        run / "interaction_semantics" / "interaction_semantics_v0.jsonl",
        _latest_existing(run / "semantic_actions", ["semantic_actions_v3.jsonl", "semantic_actions_v2.jsonl", "semantic_actions_v1.jsonl", "semantic_actions_v0.jsonl"]),
        _latest_existing(run / "datasets", ["cowgirl_candidate_db_v8.jsonl", "cowgirl_candidate_db_v7.jsonl", "cowgirl_candidate_db_v6.jsonl", "cowgirl_candidate_db_v5.jsonl"]),
    )
    records = _merged_records(sources["candidate_db"], sources)
    X = vectorize_with_feature_names(records, feature_names)
    return X, records


def vectorize_records(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    numeric_names = sorted({name for row in rows for name in _numeric_features(row)})
    cat_names = sorted({f"{field}={_cat_value(row.get(field))}" for row in rows for field in CATEGORICAL_FIELDS if _cat_value(row.get(field))})
    feature_names = numeric_names + cat_names
    return feature_names, vectorize_with_feature_names(rows, feature_names)


def vectorize_with_feature_names(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        nums = _numeric_features(row)
        cats = {f"{field}={_cat_value(row.get(field))}" for field in CATEGORICAL_FIELDS if _cat_value(row.get(field))}
        for j, name in enumerate(feature_names):
            if name in nums:
                X[i, j] = nums[name]
            elif name in cats:
                X[i, j] = 1.0
    return X


def _load_feature_sources(
    relative_features: str | Path,
    trajectory_features: str | Path,
    pose_features: str | Path,
    pose_semantics: str | Path,
    partner_relative_features: str | Path,
    interaction_semantics: str | Path,
    semantic_actions: str | Path,
    candidate_db: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "relative": load_jsonl(relative_features),
        "trajectory": load_jsonl(trajectory_features),
        "pose_features": load_jsonl(pose_features),
        "pose_semantics": load_jsonl(pose_semantics),
        "partner_relative": load_jsonl(partner_relative_features),
        "interaction_semantics": load_jsonl(interaction_semantics),
        "semantic_actions": load_jsonl(semantic_actions),
        "candidate_db": load_jsonl(candidate_db),
    }


def _merged_records(base_rows: list[dict[str, Any]], sources: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    indices = {name: _by_window(rows) for name, rows in sources.items() if name != "candidate_db"}
    merged = []
    for row in base_rows:
        out = dict(row)
        wid = str(row.get("window_id") or "")
        for name, index in indices.items():
            extra = index.get(wid)
            if extra:
                _merge_prefixed(out, extra, name)
        _parse_window_time(out)
        merged.append(out)
    return merged


def _by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("window_id")): r for r in rows if r.get("window_id")}


def _merge_prefixed(target: dict[str, Any], extra: dict[str, Any], prefix: str) -> None:
    for key, value in extra.items():
        if key in {"window_id", "sample_id", "source_id", "source_scene_file", "technical_atom_id", "technical_actor_id"}:
            target.setdefault(key, value)
        elif key == "feature_values" and isinstance(value, dict):
            for fk, fv in value.items():
                target.setdefault(fk, fv)
        elif key not in target:
            target[key] = value
        else:
            target.setdefault(f"{prefix}_{key}", value)


def _match_labels_to_records(label_rows: list[dict[str, Any]], feature_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_window = {str(r.get("window_id")): r for r in feature_records if r.get("window_id")}
    matched = []
    for label in label_rows:
        rec = None
        wid = str(label.get("window_id") or "")
        if wid:
            rec = by_window.get(wid)
        if rec is None:
            rec = _match_by_scene_actor_time(label, feature_records)
        if rec is None:
            continue
        out = dict(rec)
        for key in LABEL_KEYS:
            out[key] = label.get(key, "unknown")
        out["human_review_id"] = label.get("review_id") or ""
        out["human_notes"] = label.get("human_notes") or ""
        out["human_error_tags"] = label.get("error_tags") or []
        matched.append(out)
    return matched


def _match_by_scene_actor_time(label: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    scene = str(label.get("source_scene_file") or "")
    actor = str(label.get("technical_actor_id") or label.get("technical_atom_id") or "")
    start = _float(label.get("start_seconds"))
    end = _float(label.get("end_seconds"))
    best = None
    best_overlap = 0.0
    for rec in records:
        if scene and scene != str(rec.get("source_scene_file") or ""):
            continue
        if actor and actor != str(rec.get("technical_actor_id") or rec.get("technical_atom_id") or ""):
            continue
        rs = _float(rec.get("start_seconds"))
        re = _float(rec.get("end_seconds"))
        if start is None or end is None or rs is None or re is None:
            continue
        overlap = max(0.0, min(end, re) - max(start, rs))
        if overlap > best_overlap:
            best_overlap = overlap
            best = rec
    return best if best_overlap > 0 else None


def _numeric_features(row: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in row.items():
        if key in LABEL_KEYS or key in {
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
            "generation_safe",
            "semantic_actions_generation_safe",
            "candidate_db_generation_safe",
        }:
            continue
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            val = float(value)
            if math.isfinite(val):
                out[key] = val
    return out


def _cat_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple, set)):
        return "|".join(sorted(str(v) for v in value if str(v)))
    return str(value)


def _metadata_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "row_index": idx,
        "window_id": row.get("window_id") or "",
        "sample_id": row.get("sample_id") or "",
        "source_scene_file": row.get("source_scene_file") or "",
        "technical_actor_id": row.get("technical_actor_id") or row.get("technical_atom_id") or "",
        "start_seconds": row.get("start_seconds"),
        "end_seconds": row.get("end_seconds"),
        "labels": {k: row.get(k, "unknown") for k in LABEL_KEYS},
        "category": row.get("category") or "",
        "semantic_family": row.get("semantic_family") or "",
        "feature_available": True,
        "human_review_id": row.get("human_review_id") or "",
    }


def _label_value(value: Any) -> int:
    if value == "true":
        return 1
    if value == "false":
        return 0
    return -1


def _parse_window_time(row: dict[str, Any]) -> None:
    if row.get("start_seconds") not in {None, ""} and row.get("end_seconds") not in {None, ""}:
        return
    wid = str(row.get("window_id") or "")
    match = re.search(r"_(\d+\.\d{3})_(\d+\.\d{3})_", wid)
    if match:
        row.setdefault("start_seconds", float(match.group(1)))
        row.setdefault("end_seconds", float(match.group(2)))
        row.setdefault("duration_seconds", round(float(match.group(2)) - float(match.group(1)), 6))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_existing(folder: Path, names: list[str]) -> Path:
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return folder / names[-1]


def _fallbacks(semantic_actions: str | Path, candidate_db: str | Path) -> list[str]:
    fallbacks = []
    if "semantic_actions_v2" not in str(semantic_actions):
        fallbacks.append(f"semantic_actions fallback used: {semantic_actions}")
    if "cowgirl_candidate_db_v7" not in str(candidate_db):
        fallbacks.append(f"candidate DB fallback used: {candidate_db}")
    return fallbacks


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Cowgirl ML Feature Table v1",
        "",
        f"- Rows: {summary['rows']}",
        f"- Features: {summary['features']}",
        f"- Shape: `{summary['shape']}`",
        f"- Input labels: {summary['input_label_rows']}",
        f"- Matched labels: {summary['matched_label_rows']}",
        f"- Candidate rows: {summary['candidate_rows']}",
        f"- Label counts: `{summary['label_counts']}`",
        "",
        "The table excludes review IDs, filenames, and human labels from features. Scene/sample metadata is retained only for grouped splitting and diagnostics.",
    ]
    if summary["fallbacks"]:
        lines.extend(["", "## Fallbacks", ""])
        lines.extend(f"- {f}" for f in summary["fallbacks"])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

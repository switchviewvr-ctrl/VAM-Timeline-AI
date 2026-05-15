"""Controller/anatomical plausibility audit for relative motion windows.

This audit is intentionally about export/generation safety, not semantic truth.
A window can be a correct Cowgirl semantic hit while still having invalid foot,
hand, or torso controller placement for generation-template reuse.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


HIP_PARTS = ("hip", "pelvis", "abdomen")
CHEST_PARTS = ("chest", "abdomen")
HEAD_PARTS = ("head", "neck")
LEFT_FOOT_PARTS = ("left_foot",)
RIGHT_FOOT_PARTS = ("right_foot",)
LEFT_KNEE_PARTS = ("left_knee",)
RIGHT_KNEE_PARTS = ("right_knee",)
LEFT_THIGH_PARTS = ("left_thigh",)
RIGHT_THIGH_PARTS = ("right_thigh",)
LEFT_HAND_PARTS = ("left_hand",)
RIGHT_HAND_PARTS = ("right_hand",)
LEFT_ELBOW_PARTS = ("left_elbow",)
RIGHT_ELBOW_PARTS = ("right_elbow",)


DEFAULT_THRESHOLDS = {
    "foot_to_hip_distance_max": 2.25,
    "foot_to_knee_distance_max": 1.35,
    "foot_to_thigh_distance_mean": 1.65,
    "knee_to_hip_distance_max": 1.65,
    "hand_to_chest_distance_max": 2.10,
    "hand_to_head_distance_max": 2.20,
    "hand_to_hip_distance_max": 2.25,
    "hand_to_elbow_distance_max": 1.15,
    "head_to_chest_distance_max": 1.20,
    "chest_to_hip_distance_max": 1.35,
}


def audit_controller_validity(
    run_dir: str | Path,
    relative_index: str | Path,
    sample_index: str | Path,
    controller_map: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
    pose_anchor_completeness: str | Path | None = None,
    controller_orientation_validity: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Audit controller plausibility for every relative motion window.

    ``sample_index`` and ``controller_map`` are accepted for command symmetry
    and future richer checks.  Current plausibility is derived from relative
    window NPZ files and robust run-local distance thresholds.
    """
    run = Path(run_dir)
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    relative_rows = load_jsonl(relative_index)
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")} if pose_anchor_completeness else {}
    orientations = {r.get("window_id"): r for r in load_jsonl(controller_orientation_validity) if r.get("window_id")} if controller_orientation_validity else {}
    raw_rows = []
    for row in relative_rows:
        sample = samples.get(row.get("sample_id"), {})
        raw = _raw_controller_metrics(row, sample, run)
        raw["pose_anchor_completeness"] = anchors.get(row.get("window_id"), {})
        raw["controller_orientation_validity"] = orientations.get(row.get("window_id"), {})
        raw_rows.append(raw)
    thresholds = _derive_thresholds(raw_rows)
    rows = [classify_controller_validity(row, thresholds) for row in raw_rows]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, thresholds, report)
    return rows


def classify_controller_validity(raw: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if raw.get("metric_status") != "ok":
        out = dict(raw)
        out.update(
            {
                "controller_validity_status": "unknown",
                "controller_validity_score": 0.35,
                "generation_pose_valid": "unknown",
                "export_pose_valid": "unknown",
                "controller_outlier_count": 0,
                "outlier_bodyparts": [],
                "left_foot_outlier_score": 0.0,
                "right_foot_outlier_score": 0.0,
                "foot_controller_outlier": False,
                "left_hand_outlier_score": 0.0,
                "right_hand_outlier_score": 0.0,
                "hand_controller_outlier": False,
                "leg_chain_plausibility_score": 0.0,
                "arm_chain_plausibility_score": 0.0,
                "torso_chain_plausibility_score": 0.0,
                "orientation_validity_status": (raw.get("controller_orientation_validity") or {}).get("orientation_validity_status", "unknown"),
                "orientation_validity_score": (raw.get("controller_orientation_validity") or {}).get("orientation_validity_score", 0.35),
                "controller_rotation_invalid": bool((raw.get("controller_orientation_validity") or {}).get("controller_rotation_invalid")),
                "controller_twist_invalid": bool((raw.get("controller_orientation_validity") or {}).get("controller_twist_invalid")),
                "twisted_controller_names": (raw.get("controller_orientation_validity") or {}).get("twisted_controller_names", []),
                "foot_rotation_outlier": bool((raw.get("controller_orientation_validity") or {}).get("foot_rotation_outlier")),
                "generation_pose_invalid_reason": ["controller_validity_unknown"],
                "warnings": _dedupe([*raw.get("warnings", []), "Controller validity could not be computed."]),
            }
        )
        return out

    left_foot_score = _max_score(
        _outlier_score(raw.get("left_foot_to_hip_distance_max"), thresholds["foot_to_hip_distance_max"]),
        _outlier_score(raw.get("left_foot_to_knee_distance_max"), thresholds["foot_to_knee_distance_max"]),
        _outlier_score(raw.get("left_foot_to_thigh_distance_mean"), thresholds["foot_to_thigh_distance_mean"]),
    )
    right_foot_score = _max_score(
        _outlier_score(raw.get("right_foot_to_hip_distance_max"), thresholds["foot_to_hip_distance_max"]),
        _outlier_score(raw.get("right_foot_to_knee_distance_max"), thresholds["foot_to_knee_distance_max"]),
        _outlier_score(raw.get("right_foot_to_thigh_distance_mean"), thresholds["foot_to_thigh_distance_mean"]),
    )
    left_hand_score = _max_score(
        _outlier_score(raw.get("left_hand_to_chest_distance_max"), thresholds["hand_to_chest_distance_max"]),
        _outlier_score(raw.get("left_hand_to_head_distance_max"), thresholds["hand_to_head_distance_max"]),
        _outlier_score(raw.get("left_hand_to_hip_distance_max"), thresholds["hand_to_hip_distance_max"]),
    )
    right_hand_score = _max_score(
        _outlier_score(raw.get("right_hand_to_chest_distance_max"), thresholds["hand_to_chest_distance_max"]),
        _outlier_score(raw.get("right_hand_to_head_distance_max"), thresholds["hand_to_head_distance_max"]),
        _outlier_score(raw.get("right_hand_to_hip_distance_max"), thresholds["hand_to_hip_distance_max"]),
    )
    torso_score = _max_score(
        _outlier_score(raw.get("head_to_chest_distance_max"), thresholds["head_to_chest_distance_max"]),
        _outlier_score(raw.get("chest_to_hip_distance_max"), thresholds["chest_to_hip_distance_max"]),
    )
    knee_score = _max_score(
        _outlier_score(raw.get("left_knee_to_hip_distance_max"), thresholds["knee_to_hip_distance_max"]),
        _outlier_score(raw.get("right_knee_to_hip_distance_max"), thresholds["knee_to_hip_distance_max"]),
    )
    foot_outlier = left_foot_score >= 0.15 or right_foot_score >= 0.15
    hand_outlier = left_hand_score >= 0.20 or right_hand_score >= 0.20
    torso_outlier = torso_score >= 0.20
    knee_outlier = knee_score >= 0.20
    outlier_bodyparts = []
    if left_foot_score >= 0.15:
        outlier_bodyparts.append("left_foot")
    if right_foot_score >= 0.15:
        outlier_bodyparts.append("right_foot")
    if left_hand_score >= 0.20:
        outlier_bodyparts.append("left_hand")
    if right_hand_score >= 0.20:
        outlier_bodyparts.append("right_hand")
    if torso_outlier:
        outlier_bodyparts.append("head_or_torso")
    if knee_outlier:
        outlier_bodyparts.append("knee_or_leg_chain")
    controller_outlier_count = len(outlier_bodyparts)
    anchor = raw.get("pose_anchor_completeness") or {}
    orientation = raw.get("controller_orientation_validity") or {}
    missing_foot = bool(anchor.get("missing_foot_controllers"))
    missing_knee = bool(anchor.get("missing_knee_controllers"))
    anchor_incomplete = bool(anchor.get("pose_anchor_incomplete"))
    orientation_status = str(orientation.get("orientation_validity_status") or "unknown")
    orientation_score = _finite_float(orientation.get("orientation_validity_score"))
    if orientation_score is None:
        orientation_score = 0.35 if orientation else 1.0
    controller_rotation_invalid = bool(orientation.get("controller_rotation_invalid") or orientation_status == "invalid")
    controller_twist_invalid = bool(orientation.get("controller_twist_invalid") or controller_rotation_invalid)
    twisted_controller_names = list(orientation.get("twisted_controller_names") or [])
    foot_rotation_outlier = bool(orientation.get("foot_rotation_outlier"))
    max_outlier = max(left_foot_score, right_foot_score, left_hand_score, right_hand_score, torso_score, knee_score)
    anchor_penalty = 0.35 if missing_foot else 0.22 if missing_knee else 0.12 if anchor_incomplete else 0.0
    orientation_penalty = 0.45 if controller_rotation_invalid else 0.16 if orientation_status == "warning" else 0.0
    controller_validity_score = float(np.clip(1.0 - 0.22 * controller_outlier_count - 0.45 * max_outlier - anchor_penalty - orientation_penalty, 0.0, 1.0))
    if controller_outlier_count == 0 and raw.get("allowed_body_controller_count", 0) >= 2:
        status = "valid"
    elif foot_outlier or torso_outlier or max_outlier >= 0.55:
        status = "invalid"
    elif controller_outlier_count:
        status = "warning"
    else:
        status = "unknown"
    if anchor_incomplete and status == "valid":
        status = "warning"
    if controller_rotation_invalid:
        status = "invalid"
    elif orientation_status == "warning" and status == "valid":
        status = "warning"
    generation_valid: bool | str = status == "valid" and not anchor_incomplete and not missing_foot and not missing_knee and not controller_rotation_invalid
    export_valid: bool | str = status in {"valid", "warning"}
    warnings = list(raw.get("warnings", []))
    if foot_outlier:
        warnings.append("Foot controller outlier; semantic motion may be correct but generation pose is unsafe.")
    if hand_outlier:
        warnings.append("Hand controller outlier; inspect arm/controller placement.")
    if torso_outlier:
        warnings.append("Head/torso chain outlier; inspect pose validity.")
    if status == "invalid":
        warnings.append("Controller plausibility is invalid for generation-template use.")
    if missing_foot:
        warnings.append("Required foot anchor controllers are missing; generation pose is unsafe.")
    if missing_knee:
        warnings.append("Required knee anchor controllers are missing or incomplete.")
    if controller_rotation_invalid:
        warnings.append("Controller rotation/orientation invalidity blocks generation-template use but does not automatically make semantics wrong.")
    elif orientation_status == "warning":
        warnings.append("Controller orientation has warning-level twist; inspect before generation use.")
    if foot_rotation_outlier:
        warnings.append("Foot controller rotation outlier detected.")
    invalid_reasons = []
    if foot_outlier:
        invalid_reasons.append("foot_controller_outlier")
    if hand_outlier:
        invalid_reasons.append("hand_controller_outlier")
    if torso_outlier:
        invalid_reasons.append("torso_or_head_controller_outlier")
    if missing_foot:
        invalid_reasons.append("missing_foot_controllers")
    if missing_knee:
        invalid_reasons.append("missing_knee_controllers")
    if anchor_incomplete:
        invalid_reasons.append("pose_anchor_incomplete")
    if controller_rotation_invalid:
        invalid_reasons.append("controller_orientation_invalid")
    if controller_twist_invalid:
        invalid_reasons.append("controller_twist_invalid")
    if foot_rotation_outlier:
        invalid_reasons.append("foot_rotation_outlier")

    out = dict(raw)
    out.update(
        {
            "controller_validity_status": status,
            "controller_validity_score": round(controller_validity_score, 6),
            "generation_pose_valid": generation_valid,
            "export_pose_valid": export_valid,
            "controller_outlier_count": controller_outlier_count,
            "outlier_bodyparts": outlier_bodyparts,
            "left_foot_outlier_score": round(float(left_foot_score), 6),
            "right_foot_outlier_score": round(float(right_foot_score), 6),
            "foot_controller_outlier": bool(foot_outlier),
            "left_hand_outlier_score": round(float(left_hand_score), 6),
            "right_hand_outlier_score": round(float(right_hand_score), 6),
            "hand_controller_outlier": bool(hand_outlier),
            "leg_chain_plausibility_score": round(float(np.clip(1.0 - max(left_foot_score, right_foot_score, knee_score), 0.0, 1.0)), 6),
            "arm_chain_plausibility_score": round(float(np.clip(1.0 - max(left_hand_score, right_hand_score), 0.0, 1.0)), 6),
            "torso_chain_plausibility_score": round(float(np.clip(1.0 - torso_score, 0.0, 1.0)), 6),
            "missing_foot_controllers": missing_foot,
            "missing_knee_controllers": missing_knee,
            "pose_anchor_incomplete": anchor_incomplete,
            "pose_anchor_completeness_score": anchor.get("pose_anchor_completeness_score"),
            "generation_pose_anchor_safe": anchor.get("generation_pose_anchor_safe"),
            "missing_required_anchor_controllers": anchor.get("missing_required_anchor_controllers", []),
            "pose_anchor_completeness": anchor,
            "orientation_validity_status": orientation_status,
            "orientation_validity_score": round(float(orientation_score), 6),
            "controller_rotation_invalid": bool(controller_rotation_invalid),
            "controller_twist_invalid": bool(controller_twist_invalid),
            "twisted_controller_names": twisted_controller_names,
            "foot_rotation_outlier": bool(foot_rotation_outlier),
            "controller_orientation_validity": orientation,
            "generation_pose_invalid_reason": _dedupe(invalid_reasons),
            "warnings": _dedupe(warnings),
            "is_human_ground_truth": False,
            "is_training_label": False,
        }
    )
    return out


def controller_validity_for_arrays(
    positions: np.ndarray,
    bodyparts: list[str],
    controller_names: list[str] | None = None,
    scale: float | None = None,
    row: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convenience API for tests and synthetic audits."""
    row = row or {}
    raw = _metrics_from_arrays(np.asarray(positions, dtype=np.float32), bodyparts, controller_names or bodyparts, float(scale or 1.0), row)
    return classify_controller_validity(raw, thresholds)


def _raw_controller_metrics(relative_row: dict[str, Any], sample: dict[str, Any], run: Path) -> dict[str, Any]:
    path = relative_row.get("relative_npz_path")
    warnings = []
    if not path:
        return _unknown_row(relative_row, sample, "relative_npz_path missing")
    npz_path = Path(str(path))
    if not npz_path.is_absolute():
        project_root = run.parents[2] if len(run.parents) >= 3 else Path.cwd()
        candidate = project_root / npz_path if str(path).startswith("data") else Path.cwd() / npz_path
        npz_path = candidate if candidate.exists() else Path.cwd() / npz_path
    if not npz_path.exists():
        return _unknown_row(relative_row, sample, f"relative NPZ missing: {path}")
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            baseline = np.asarray(data.get("baseline_positions"), dtype=np.float32)
            delta = np.asarray(data.get("position_delta"), dtype=np.float32)
            bodyparts = [str(x) for x in data.get("bodyparts", [])]
            names = [str(x) for x in data.get("controller_names", [])]
            if delta.ndim != 3 or baseline.ndim != 2:
                return _unknown_row(relative_row, sample, "relative arrays have unexpected shape")
            positions = baseline[None, :, :] + delta
    except Exception as exc:
        return _unknown_row(relative_row, sample, f"could not load relative NPZ: {exc}")
    scale = float((relative_row.get("baseline_pose_summary") or {}).get("body_scale") or 1.0)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
        warnings.append("Body scale unavailable; controller distances use unit normalization.")
    raw = _metrics_from_arrays(positions, bodyparts, names, scale, relative_row)
    raw["warnings"] = _dedupe([*raw.get("warnings", []), *warnings])
    return raw


def _metrics_from_arrays(positions: np.ndarray, bodyparts: list[str], names: list[str], scale: float, row: dict[str, Any]) -> dict[str, Any]:
    slots = _slots(bodyparts)
    norm = max(float(scale), 1e-6)
    metrics: dict[str, Any] = {
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_id": row.get("source_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_atom_id": row.get("technical_atom_id"),
        "controller_names": list(names),
        "bodyparts": list(bodyparts),
        "allowed_body_controller_count": len(bodyparts),
        "body_scale": round(float(norm), 6),
        "metric_status": "ok",
        "warnings": [],
    }
    metric_specs = {
        "left_foot_to_hip": (LEFT_FOOT_PARTS, HIP_PARTS),
        "right_foot_to_hip": (RIGHT_FOOT_PARTS, HIP_PARTS),
        "left_foot_to_knee": (LEFT_FOOT_PARTS, LEFT_KNEE_PARTS),
        "right_foot_to_knee": (RIGHT_FOOT_PARTS, RIGHT_KNEE_PARTS),
        "left_foot_to_thigh": (LEFT_FOOT_PARTS, LEFT_THIGH_PARTS),
        "right_foot_to_thigh": (RIGHT_FOOT_PARTS, RIGHT_THIGH_PARTS),
        "left_knee_to_hip": (LEFT_KNEE_PARTS, HIP_PARTS),
        "right_knee_to_hip": (RIGHT_KNEE_PARTS, HIP_PARTS),
        "left_hand_to_chest": (LEFT_HAND_PARTS, CHEST_PARTS),
        "right_hand_to_chest": (RIGHT_HAND_PARTS, CHEST_PARTS),
        "left_hand_to_head": (LEFT_HAND_PARTS, HEAD_PARTS),
        "right_hand_to_head": (RIGHT_HAND_PARTS, HEAD_PARTS),
        "left_hand_to_hip": (LEFT_HAND_PARTS, HIP_PARTS),
        "right_hand_to_hip": (RIGHT_HAND_PARTS, HIP_PARTS),
        "left_hand_to_elbow": (LEFT_HAND_PARTS, LEFT_ELBOW_PARTS),
        "right_hand_to_elbow": (RIGHT_HAND_PARTS, RIGHT_ELBOW_PARTS),
        "head_to_chest": (HEAD_PARTS, CHEST_PARTS),
        "chest_to_hip": (CHEST_PARTS, HIP_PARTS),
    }
    for prefix, (a_parts, b_parts) in metric_specs.items():
        mean, maxv = _distance_mean_max(positions, slots, a_parts, b_parts, norm)
        metrics[f"{prefix}_distance_mean"] = _json_float(mean)
        metrics[f"{prefix}_distance_max"] = _json_float(maxv)
    # Friendly aliases requested by the operator.
    metrics["foot_to_hip_distance_mean"] = _json_float(_nanmean([metrics.get("left_foot_to_hip_distance_mean"), metrics.get("right_foot_to_hip_distance_mean")]))
    metrics["foot_to_hip_distance_max"] = _json_float(_nanmax([metrics.get("left_foot_to_hip_distance_max"), metrics.get("right_foot_to_hip_distance_max")]))
    metrics["foot_to_knee_distance_mean"] = _json_float(_nanmean([metrics.get("left_foot_to_knee_distance_mean"), metrics.get("right_foot_to_knee_distance_mean")]))
    metrics["foot_to_knee_distance_max"] = _json_float(_nanmax([metrics.get("left_foot_to_knee_distance_max"), metrics.get("right_foot_to_knee_distance_max")]))
    metrics["foot_to_thigh_distance_mean"] = _json_float(_nanmean([metrics.get("left_foot_to_thigh_distance_mean"), metrics.get("right_foot_to_thigh_distance_mean")]))
    metrics["knee_to_hip_distance_mean"] = _json_float(_nanmean([metrics.get("left_knee_to_hip_distance_mean"), metrics.get("right_knee_to_hip_distance_mean")]))
    metrics["knee_to_hip_distance_max"] = _json_float(_nanmax([metrics.get("left_knee_to_hip_distance_max"), metrics.get("right_knee_to_hip_distance_max")]))
    metrics["hand_to_chest_distance_mean"] = _json_float(_nanmean([metrics.get("left_hand_to_chest_distance_mean"), metrics.get("right_hand_to_chest_distance_mean")]))
    metrics["hand_to_chest_distance_max"] = _json_float(_nanmax([metrics.get("left_hand_to_chest_distance_max"), metrics.get("right_hand_to_chest_distance_max")]))
    metrics["hand_to_head_distance_mean"] = _json_float(_nanmean([metrics.get("left_hand_to_head_distance_mean"), metrics.get("right_hand_to_head_distance_mean")]))
    metrics["hand_to_head_distance_max"] = _json_float(_nanmax([metrics.get("left_hand_to_head_distance_max"), metrics.get("right_hand_to_head_distance_max")]))
    metrics["hand_to_hip_distance_mean"] = _json_float(_nanmean([metrics.get("left_hand_to_hip_distance_mean"), metrics.get("right_hand_to_hip_distance_mean")]))
    metrics["hand_to_hip_distance_max"] = _json_float(_nanmax([metrics.get("left_hand_to_hip_distance_max"), metrics.get("right_hand_to_hip_distance_max")]))
    metrics["hand_to_elbow_distance_mean"] = _json_float(_nanmean([metrics.get("left_hand_to_elbow_distance_mean"), metrics.get("right_hand_to_elbow_distance_mean")]))
    metrics["hand_to_elbow_distance_max"] = _json_float(_nanmax([metrics.get("left_hand_to_elbow_distance_max"), metrics.get("right_hand_to_elbow_distance_max")]))
    return metrics


def _derive_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    metric_names = list(DEFAULT_THRESHOLDS)
    for metric in metric_names:
        values = [_finite_float(row.get(metric)) for row in rows if row.get("metric_status") == "ok"]
        values = [v for v in values if v is not None and v > 1e-6]
        if len(values) < 8:
            continue
        arr = np.asarray(values, dtype=np.float32)
        med = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - med)))
        p95 = float(np.nanpercentile(arr, 95))
        robust = med + 4.0 * max(mad, 1e-4)
        # Keep thresholds conservative but data-aware.  The lower-bound defaults
        # prevent a quiet dataset from making ordinary limbs look invalid.
        thresholds[metric] = round(float(max(DEFAULT_THRESHOLDS[metric], min(max(robust, p95), float(np.nanpercentile(arr, 99))))), 6)
    return thresholds


def _unknown_row(relative_row: dict[str, Any], sample: dict[str, Any], warning: str) -> dict[str, Any]:
    return {
        "window_id": relative_row.get("window_id"),
        "sample_id": relative_row.get("sample_id") or sample.get("sample_id"),
        "source_id": relative_row.get("source_id") or sample.get("source_id"),
        "source_scene_file": relative_row.get("source_scene_file") or sample.get("source_scene_file"),
        "technical_atom_id": relative_row.get("technical_atom_id") or sample.get("technical_atom_id"),
        "controller_names": relative_row.get("controllers", []),
        "bodyparts": relative_row.get("bodyparts", []),
        "allowed_body_controller_count": len(relative_row.get("bodyparts", []) or []),
        "metric_status": "unknown",
        "warnings": [warning],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _slots(bodyparts: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, raw in enumerate(bodyparts):
        part = _normalize_part(raw)
        out.setdefault(part, []).append(idx)
    return out


def _normalize_part(part: Any) -> str:
    text = str(part or "").strip().lower()
    aliases = {
        "lfoot": "left_foot",
        "rfoot": "right_foot",
        "lknee": "left_knee",
        "rknee": "right_knee",
        "lhand": "left_hand",
        "rhand": "right_hand",
        "lelbow": "left_elbow",
        "relbow": "right_elbow",
        "lthigh": "left_thigh",
        "rthigh": "right_thigh",
    }
    return aliases.get(text, text)


def _distance_mean_max(positions: np.ndarray, slots: dict[str, list[int]], a_parts: tuple[str, ...], b_parts: tuple[str, ...], scale: float) -> tuple[float, float]:
    a_idx = _first_slot(slots, a_parts)
    b_idx = _first_slot(slots, b_parts)
    if a_idx is None or b_idx is None or positions.size == 0:
        return float("nan"), float("nan")
    delta = positions[:, a_idx, :] - positions[:, b_idx, :]
    dist = np.linalg.norm(delta, axis=1) / max(scale, 1e-6)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(finite)), float(np.nanmax(finite))


def _first_slot(slots: dict[str, list[int]], parts: tuple[str, ...]) -> int | None:
    for part in parts:
        values = slots.get(part)
        if values:
            return values[0]
    return None


def _outlier_score(value: Any, threshold: float) -> float:
    val = _finite_float(value)
    if val is None or threshold <= 1e-6:
        return 0.0
    return max(0.0, (val / threshold) - 1.0)


def _max_score(*values: float) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return max(finite) if finite else 0.0


def _finite_float(value: Any) -> float | None:
    try:
        val = float(value)
        if np.isfinite(val):
            return val
    except Exception:
        pass
    return None


def _json_float(value: Any) -> float | None:
    val = _finite_float(value)
    return round(float(val), 6) if val is not None else None


def _nanmean(values: list[Any]) -> float:
    finite = [_finite_float(v) for v in values]
    finite = [v for v in finite if v is not None]
    return float(np.nanmean(finite)) if finite else float("nan")


def _nanmax(values: list[Any]) -> float:
    finite = [_finite_float(v) for v in values]
    finite = [v for v in finite if v is not None]
    return float(np.nanmax(finite)) if finite else float("nan")


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _write_report(rows: list[dict[str, Any]], thresholds: dict[str, float], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(r.get("controller_validity_status") for r in rows)
    foot_count = sum(1 for r in rows if r.get("foot_controller_outlier"))
    hand_count = sum(1 for r in rows if r.get("hand_controller_outlier"))
    missing_foot = sum(1 for r in rows if r.get("missing_foot_controllers"))
    missing_knee = sum(1 for r in rows if r.get("missing_knee_controllers"))
    orientation_invalid = sum(1 for r in rows if r.get("controller_rotation_invalid"))
    foot_rotation = sum(1 for r in rows if r.get("foot_rotation_outlier"))
    invalid = [r for r in rows if r.get("controller_validity_status") == "invalid"]
    scenes = Counter(r.get("source_scene_file") for r in invalid)
    samples = Counter(r.get("sample_id") for r in invalid)
    lines = [
        "# Controller Validity / Anatomical Plausibility Report",
        "",
        "This is an audit of export/generation pose safety. It does not change semantic labels.",
        "",
        f"- Windows audited: {len(rows)}",
        f"- Foot controller outliers: {foot_count}",
        f"- Missing foot anchors: {missing_foot}",
        f"- Missing knee anchors: {missing_knee}",
        f"- Orientation-invalid controllers: {orientation_invalid}",
        f"- Foot rotation outliers: {foot_rotation}",
        f"- Hand controller outliers: {hand_count}",
        "",
        "## Controller Validity Status",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in status_counts.most_common()) if status_counts else lines.append("- None")
    lines.extend(["", "## Recommended Thresholds", ""])
    for key, value in sorted(thresholds.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Review-003 / Review-005-Like Foot Outliers", ""])
    examples = sorted([r for r in rows if r.get("foot_controller_outlier")], key=lambda r: max(float(r.get("left_foot_outlier_score") or 0.0), float(r.get("right_foot_outlier_score") or 0.0)), reverse=True)
    for row in examples[:20]:
        lines.append(
            f"- `{row.get('window_id')}` status=`{row.get('controller_validity_status')}` "
            f"left={row.get('left_foot_outlier_score')} right={row.get('right_foot_outlier_score')} "
            f"scene=`{row.get('source_scene_file')}`"
        )
    if not examples:
        lines.append("- None")
    lines.extend(["", "## Frequent Invalid Scenes", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in scenes.most_common(20)) if scenes else lines.append("- None")
    lines.extend(["", "## Frequent Invalid Samples", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in samples.most_common(20)) if samples else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

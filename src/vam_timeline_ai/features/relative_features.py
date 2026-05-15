"""Feature extraction from safe relative/local motion windows."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


RELATIVE_FEATURE_NAMES = [
    "relative_pelvis_vertical_amplitude",
    "relative_pelvis_forward_back_amplitude",
    "relative_pelvis_lateral_amplitude",
    "local_path_length",
    "local_motion_energy",
    "local_velocity_mean",
    "local_velocity_max",
    "local_rhythm_regularity",
    "local_circularity",
    "local_grind_score",
    "local_bounce_score",
    "torso_relative_to_pelvis_motion",
    "hands_relative_to_chest_pelvis_head",
    "head_relative_to_chest_motion",
    "limb_motion_relative_energy",
    "root_world_motion_removed",
    "safe_for_learning",
]


def extract_relative_motion_features(
    relative_index: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    index_rows = load_jsonl(relative_index)
    rows: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    for irow in index_rows:
        feature_row = feature_row_from_relative_index(irow)
        rows.append(feature_row)
        matrix.append([float(feature_row["feature_values"].get(name, np.nan)) for name in RELATIVE_FEATURE_NAMES])
    write_jsonl(out_jsonl, rows)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        X=np.asarray(matrix, dtype=np.float32),
        feature_names=np.asarray(RELATIVE_FEATURE_NAMES, dtype=object),
        window_ids=np.asarray([r.get("window_id") for r in rows], dtype=object),
        sample_ids=np.asarray([r.get("sample_id") for r in rows], dtype=object),
    )
    _write_report(rows, report)
    return rows


def feature_row_from_relative_index(index_row: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, float]
    quality: dict[str, Any]
    if index_row.get("relative_npz_path") and Path(str(index_row["relative_npz_path"])).exists():
        with np.load(str(index_row["relative_npz_path"]), allow_pickle=True) as data:
            positions = np.asarray(data["normalized_position_delta"], dtype=np.float32)
            velocities = np.asarray(data["velocity_local"], dtype=np.float32)
            names = [str(x) for x in data["controller_names"].tolist()]
            bodyparts = [str(x) for x in data["bodyparts"].tolist()]
            times = np.asarray(data["times"], dtype=np.float32)
        values, quality = relative_features_from_arrays(positions, velocities, bodyparts, times)
    else:
        values = {name: 0.0 for name in RELATIVE_FEATURE_NAMES}
        quality = {"has_relative_npz": False, "rejection_reason": "missing_relative_npz"}
    values["root_world_motion_removed"] = 1.0 if index_row.get("root_world_motion_removed") else 0.0
    values["safe_for_learning"] = 1.0 if index_row.get("safe_for_learning") else 0.0
    quality.update(
        {
            "safe_for_learning": bool(index_row.get("safe_for_learning")),
            "teleport_risk": index_row.get("teleport_risk"),
            "unsafe_reasons": index_row.get("unsafe_reasons", []),
        }
    )
    return {
        "window_id": index_row.get("window_id"),
        "sample_id": index_row.get("sample_id"),
        "source_id": index_row.get("source_id"),
        "source_scene_file": index_row.get("source_scene_file"),
        "technical_atom_id": index_row.get("technical_atom_id"),
        "duration_seconds": index_row.get("duration_seconds"),
        "relative_npz_path": index_row.get("relative_npz_path"),
        "feature_version": "relative_motion_features_v1",
        "feature_values": {k: _round(v) for k, v in values.items()},
        "feature_quality": quality,
        "controllers_used": index_row.get("controllers", []),
        "bodyparts": index_row.get("bodyparts", []),
        "warnings": index_row.get("warnings", []),
        "is_human_ground_truth": False,
    }


def relative_features_from_arrays(
    positions: np.ndarray,
    velocities: np.ndarray | None,
    bodyparts: list[str],
    times: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    values = {name: 0.0 for name in RELATIVE_FEATURE_NAMES}
    if positions.size == 0 or not bodyparts:
        return values, {"has_relative_npz": True, "has_pelvis_features": False}
    pelvis_idx = _first_index(bodyparts, {"pelvis", "hip", "abdomen"}) or 0
    pelvis = np.asarray(positions[:, pelvis_idx, :], dtype=np.float32)
    spans = np.nanmax(pelvis, axis=0) - np.nanmin(pelvis, axis=0)
    # Axis names are audit assumptions: y vertical, z forward/back, x lateral.
    values["relative_pelvis_lateral_amplitude"] = float(abs(spans[0]))
    values["relative_pelvis_vertical_amplitude"] = float(abs(spans[1]))
    values["relative_pelvis_forward_back_amplitude"] = float(abs(spans[2]))
    diffs = np.diff(pelvis, axis=0) if len(pelvis) > 1 else np.zeros((0, 3), dtype=np.float32)
    step = np.linalg.norm(diffs, axis=1) if len(diffs) else np.asarray([], dtype=np.float32)
    values["local_path_length"] = float(np.nansum(step)) if len(step) else 0.0
    if velocities is None or not velocities.size:
        dt = _median_dt(times)
        velocities = np.gradient(positions, max(dt, 1e-6), axis=0) if len(positions) > 1 else np.zeros_like(positions)
    pv = np.asarray(velocities[:, pelvis_idx, :], dtype=np.float32)
    speeds = np.linalg.norm(pv, axis=1)
    values["local_motion_energy"] = float(np.nanmean(speeds**2)) if len(speeds) else 0.0
    values["local_velocity_mean"] = float(np.nanmean(speeds)) if len(speeds) else 0.0
    values["local_velocity_max"] = float(np.nanmax(speeds)) if len(speeds) else 0.0
    values["local_rhythm_regularity"] = _rhythm_regularity(speeds)
    horizontal_min = min(values["relative_pelvis_lateral_amplitude"], values["relative_pelvis_forward_back_amplitude"])
    horizontal_max = max(values["relative_pelvis_lateral_amplitude"], values["relative_pelvis_forward_back_amplitude"], 1e-6)
    values["local_circularity"] = float(horizontal_min / horizontal_max)
    horizontal = values["relative_pelvis_lateral_amplitude"] + values["relative_pelvis_forward_back_amplitude"]
    total = horizontal + values["relative_pelvis_vertical_amplitude"] + 1e-6
    values["local_grind_score"] = float(values["local_circularity"] * horizontal / total)
    values["local_bounce_score"] = float(values["relative_pelvis_vertical_amplitude"] / total)
    values["torso_relative_to_pelvis_motion"] = _relative_group_motion(positions, bodyparts, {"abdomen", "chest"}, pelvis_idx)
    values["hands_relative_to_chest_pelvis_head"] = _hand_relative_motion(positions, bodyparts)
    values["head_relative_to_chest_motion"] = _head_relative_motion(positions, bodyparts)
    limb_indices = [i for i, p in enumerate(bodyparts) if p in {"left_hand", "right_hand", "left_elbow", "right_elbow", "left_knee", "right_knee", "left_foot", "right_foot", "left_thigh", "right_thigh"}]
    if limb_indices:
        limb_speed = np.linalg.norm(velocities[:, limb_indices, :], axis=2)
        values["limb_motion_relative_energy"] = float(np.nanmean(limb_speed**2))
    return values, {
        "has_relative_npz": True,
        "has_pelvis_features": pelvis_idx is not None,
        "pelvis_controller_bodypart": bodyparts[pelvis_idx],
        "axis_note": "Relative axes retain baked coordinate orientation; interpretation remains audited.",
    }


def _relative_group_motion(positions: np.ndarray, bodyparts: list[str], group: set[str], pelvis_idx: int) -> float:
    indices = [i for i, p in enumerate(bodyparts) if p in group]
    if not indices:
        return 0.0
    pelvis = positions[:, pelvis_idx : pelvis_idx + 1, :]
    rel = positions[:, indices, :] - pelvis
    return _mean_path_length(rel)


def _hand_relative_motion(positions: np.ndarray, bodyparts: list[str]) -> float:
    hand_indices = [i for i, p in enumerate(bodyparts) if p in {"left_hand", "right_hand"}]
    target_indices = [i for i, p in enumerate(bodyparts) if p in {"chest", "pelvis", "hip", "head"}]
    if not hand_indices or not target_indices:
        return 0.0
    vals = []
    for h in hand_indices:
        for t in target_indices:
            vals.append(_mean_path_length((positions[:, h : h + 1, :] - positions[:, t : t + 1, :])))
    return float(np.nanmean(vals)) if vals else 0.0


def _head_relative_motion(positions: np.ndarray, bodyparts: list[str]) -> float:
    head = _first_index(bodyparts, {"head"})
    chest = _first_index(bodyparts, {"chest", "abdomen"})
    if head is None or chest is None:
        return 0.0
    return _mean_path_length(positions[:, head : head + 1, :] - positions[:, chest : chest + 1, :])


def _mean_path_length(values: np.ndarray) -> float:
    if values.size == 0 or values.shape[0] < 2:
        return 0.0
    diffs = np.diff(values, axis=0)
    return float(np.nanmean(np.nansum(np.linalg.norm(diffs, axis=2), axis=0)))


def _first_index(items: list[str], choices: set[str]) -> int | None:
    for idx, item in enumerate(items):
        if item in choices:
            return idx
    return None


def _median_dt(times: np.ndarray | None) -> float:
    if times is None or len(times) < 2:
        return 1.0 / 60.0
    diffs = np.diff(times.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 1.0 / 60.0


def _rhythm_regularity(values: np.ndarray) -> float:
    values = values[np.isfinite(values)] if len(values) else values
    if len(values) < 4:
        return 0.0
    mean = float(np.mean(values))
    if abs(mean) < 1e-8:
        return 0.0
    return float(np.clip(1.0 - (np.std(values) / (abs(mean) + 1e-6)), 0.0, 1.0))


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe = sum(1 for r in rows if r.get("feature_values", {}).get("safe_for_learning"))
    reject = Counter(reason for r in rows for reason in (r.get("feature_quality", {}).get("unsafe_reasons") or []))
    high_old_reject = [r for r in rows if not r.get("feature_values", {}).get("safe_for_learning")]
    lines = [
        "# Relative Motion Feature Report",
        "",
        "Features are computed from window-local body-controller deltas, not raw world placement.",
        "",
        f"- Rows: {len(rows)}",
        f"- Safe for learning: {safe}",
        f"- Unsafe/rejected: {len(rows) - safe}",
        "",
        "## Unsafe Reasons",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in reject.most_common()) if reject else lines.append("- None")
    lines.extend(
        [
            "",
            "## Old Absolute vs Relative Behavior",
            "",
            "The old feature file is not modified here. This report flags windows whose relative representation is unsafe even if old absolute/world-space features looked energetic.",
            "",
        ]
    )
    for row in high_old_reject[:15]:
        lines.append(f"- `{row.get('window_id')}` unsafe={row.get('feature_quality', {}).get('unsafe_reasons')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return round(f, 6) if np.isfinite(f) else f


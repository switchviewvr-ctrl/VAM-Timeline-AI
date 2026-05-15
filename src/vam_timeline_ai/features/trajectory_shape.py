"""Trajectory-shape analysis for relative pelvis/hip motion."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


TRAJECTORY_FEATURE_NAMES = [
    "trajectory_path_length",
    "trajectory_displacement_start_to_end",
    "closed_loop_ratio",
    "ellipse_fit_score",
    "oval_path_score",
    "circularity_score_v2",
    "linearity_score",
    "jitter_score",
    "path_aspect_ratio",
    "path_area_2d",
    "normalized_path_area",
    "rhythm_repeat_score",
    "cycle_count_estimate",
    "grind_pattern_score",
    "bounce_pattern_score",
    "forward_back_rock_pattern_score",
    "transition_path_score",
]


def analyze_trajectory_shapes(
    relative_index: str | Path,
    relative_features: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    index = {r.get("window_id"): r for r in load_jsonl(relative_index) if r.get("window_id")}
    rel_features = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    rows: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    for wid, irow in index.items():
        row = trajectory_row_from_relative_index(irow, rel_features.get(wid, {}))
        rows.append(row)
        matrix.append([float(row["feature_values"].get(name, np.nan)) for name in TRAJECTORY_FEATURE_NAMES])
    write_jsonl(out_jsonl, rows)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        X=np.asarray(matrix, dtype=np.float32),
        feature_names=np.asarray(TRAJECTORY_FEATURE_NAMES, dtype=object),
        window_ids=np.asarray([r.get("window_id") for r in rows], dtype=object),
    )
    _write_report(rows, report)
    return rows


def trajectory_row_from_relative_index(index_row: dict[str, Any], relative_feature_row: dict[str, Any] | None = None) -> dict[str, Any]:
    if index_row.get("relative_npz_path") and Path(str(index_row["relative_npz_path"])).exists():
        with np.load(str(index_row["relative_npz_path"]), allow_pickle=True) as data:
            positions = np.asarray(data["normalized_position_delta"], dtype=np.float32)
            bodyparts = [str(x) for x in data["bodyparts"].tolist()]
            times = np.asarray(data["times"], dtype=np.float32)
        pelvis_idx = _first_index(bodyparts, {"pelvis", "hip", "abdomen"})
        points = positions[:, pelvis_idx if pelvis_idx is not None else 0, :] if positions.size and bodyparts else np.zeros((0, 3), dtype=np.float32)
        values, quality = trajectory_shape_for_points(points, times=times, safe_for_learning=bool(index_row.get("safe_for_learning")))
    else:
        values, quality = _empty_values(), {"trajectory_shape_classification": "unknown", "safe_for_learning": False, "reason": "missing_relative_npz"}
    return {
        "window_id": index_row.get("window_id"),
        "sample_id": index_row.get("sample_id"),
        "source_id": index_row.get("source_id"),
        "source_scene_file": index_row.get("source_scene_file"),
        "technical_atom_id": index_row.get("technical_atom_id"),
        "duration_seconds": index_row.get("duration_seconds"),
        "feature_version": "trajectory_shape_features_v1",
        "feature_values": {k: _round(v) for k, v in values.items()},
        "trajectory_shape_classification": quality.get("trajectory_shape_classification"),
        "dominant_motion_plane": quality.get("dominant_motion_plane"),
        "dominant_axis_1": quality.get("dominant_axis_1"),
        "dominant_axis_2": quality.get("dominant_axis_2"),
        "safe_for_learning": bool(index_row.get("safe_for_learning")),
        "relative_feature_quality": (relative_feature_row or {}).get("feature_quality", {}),
        "warnings": quality.get("warnings", []),
        "is_human_ground_truth": False,
    }


def trajectory_shape_for_points(points: np.ndarray, times: np.ndarray | None = None, safe_for_learning: bool = True) -> tuple[dict[str, float], dict[str, Any]]:
    values = _empty_values()
    warnings: list[str] = []
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return values, {"trajectory_shape_classification": "unknown", "dominant_motion_plane": "unknown", "warnings": ["not_enough_points"]}
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) < 3:
        return values, {"trajectory_shape_classification": "unknown", "dominant_motion_plane": "unknown", "warnings": ["not_enough_finite_points"]}
    diffs = np.diff(pts, axis=0)
    steps = np.linalg.norm(diffs, axis=1)
    path = float(np.sum(steps))
    displacement = float(np.linalg.norm(pts[-1] - pts[0]))
    spans = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
    axes = np.argsort(spans)[::-1][:2]
    axes = [int(a) for a in axes]
    plane = _plane_name(axes)
    centered = pts - np.nanmean(pts, axis=0, keepdims=True)
    plane_pts = centered[:, axes]
    area = float(abs(_shoelace_area(plane_pts)))
    max_span = float(max(spans[axes[0]], spans[axes[1]], 1e-6))
    min_span = float(max(min(spans[axes[0]], spans[axes[1]]), 0.0))
    normalized_area = float(np.clip(area / (max_span * max_span + 1e-6), 0.0, 1.0))
    circularity = float(np.clip(min_span / max_span, 0.0, 1.0))
    aspect = float(max_span / max(min_span, 1e-6))
    closed = float(np.clip(1.0 - displacement / (path + 1e-6), 0.0, 1.0))
    line = _linearity(plane_pts)
    jitter = _jitter_score(steps, path, max_span, safe_for_learning)
    rhythm, cycles = _repeat_score(plane_pts[:, 0], times)
    vertical_span = float(spans[1])
    forward_span = float(spans[2])
    lateral_span = float(spans[0])
    total_span = float(vertical_span + forward_span + lateral_span + 1e-6)
    ellipse_fit = float(np.clip(0.35 * closed + 0.30 * normalized_area + 0.25 * circularity + 0.10 * rhythm, 0.0, 1.0))
    oval = float(np.clip(0.35 * ellipse_fit + 0.30 * closed + 0.20 * normalized_area + 0.15 * (1.0 - min(line, 1.0)), 0.0, 1.0))
    bounce = float(np.clip((vertical_span / total_span) * rhythm * (1.0 - normalized_area * 0.35), 0.0, 1.0))
    forward = float(np.clip((forward_span / total_span) * rhythm * (1.0 - circularity * 0.35), 0.0, 1.0))
    grind = float(np.clip(0.45 * oval + 0.25 * circularity + 0.20 * normalized_area + 0.10 * rhythm - 0.35 * jitter, 0.0, 1.0))
    transition = float(np.clip((displacement / (path + 1e-6)) * (1.0 - rhythm) + 0.25 * line, 0.0, 1.0))
    values.update(
        {
            "trajectory_path_length": path,
            "trajectory_displacement_start_to_end": displacement,
            "closed_loop_ratio": closed,
            "ellipse_fit_score": ellipse_fit,
            "oval_path_score": oval,
            "circularity_score_v2": circularity,
            "linearity_score": line,
            "jitter_score": jitter,
            "path_aspect_ratio": aspect,
            "path_area_2d": area,
            "normalized_path_area": normalized_area,
            "rhythm_repeat_score": rhythm,
            "cycle_count_estimate": cycles,
            "grind_pattern_score": grind,
            "bounce_pattern_score": bounce,
            "forward_back_rock_pattern_score": forward,
            "transition_path_score": transition,
        }
    )
    classification = _classify(values, safe_for_learning)
    if not safe_for_learning:
        warnings.append("relative window is not safe_for_learning; shape is audit evidence only")
    return values, {
        "trajectory_shape_classification": classification,
        "dominant_motion_plane": plane,
        "dominant_axis_1": _axis_name(axes[0]),
        "dominant_axis_2": _axis_name(axes[1]),
        "warnings": warnings,
    }


def _empty_values() -> dict[str, float]:
    return {name: 0.0 for name in TRAJECTORY_FEATURE_NAMES}


def _shoelace_area(points_2d: np.ndarray) -> float:
    if len(points_2d) < 3:
        return 0.0
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _linearity(points_2d: np.ndarray) -> float:
    if len(points_2d) < 3:
        return 1.0
    _, s, _ = np.linalg.svd(points_2d - np.mean(points_2d, axis=0), full_matrices=False)
    if len(s) < 2 or s[0] <= 1e-8:
        return 1.0
    return float(np.clip(1.0 - s[1] / s[0], 0.0, 1.0))


def _jitter_score(steps: np.ndarray, path: float, max_span: float, safe_for_learning: bool) -> float:
    if len(steps) < 3:
        return 1.0
    amp = max(max_span, 1e-6)
    micro = float(np.clip(1.0 - amp / 0.03, 0.0, 1.0))
    variability = float(np.clip(np.std(steps) / (np.mean(steps) + 1e-6), 0.0, 1.0))
    path_excess = float(np.clip(path / (amp + 1e-6) / 20.0, 0.0, 1.0))
    unsafe = 0.25 if not safe_for_learning else 0.0
    return float(np.clip(0.45 * micro + 0.35 * variability + 0.20 * path_excess + unsafe, 0.0, 1.0))


def _repeat_score(signal: np.ndarray, times: np.ndarray | None) -> tuple[float, float]:
    if len(signal) < 10:
        return 0.0, 0.0
    centered = signal - np.mean(signal)
    if np.std(centered) < 1e-6:
        return 0.0, 0.0
    signs = np.sign(centered)
    crossings = int(np.sum(signs[1:] * signs[:-1] < 0))
    cycles = crossings / 2.0
    regularity = min(1.0, cycles / 2.0)
    ac = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    ac = ac / max(float(ac[0]), 1e-6)
    peak = float(np.max(ac[2: len(ac) // 2])) if len(ac) > 6 else 0.0
    return float(np.clip(0.55 * regularity + 0.45 * max(peak, 0.0), 0.0, 1.0)), float(cycles)


def _classify(values: dict[str, float], safe_for_learning: bool) -> str:
    if not safe_for_learning or values["jitter_score"] >= 0.65 or values["trajectory_path_length"] < 0.015:
        return "jitter/static"
    if values["grind_pattern_score"] >= 0.58 and values["oval_path_score"] >= 0.45:
        return "circular_grind" if values["circularity_score_v2"] >= 0.72 else "oval_grind"
    if values["bounce_pattern_score"] >= 0.45 and values["bounce_pattern_score"] >= values["forward_back_rock_pattern_score"]:
        return "vertical_bounce"
    if values["forward_back_rock_pattern_score"] >= 0.38:
        return "forward_back_rock"
    if values["transition_path_score"] >= 0.55:
        return "transition"
    return "unknown"


def _axis_name(idx: int) -> str:
    return {0: "lateral_x", 1: "vertical_y", 2: "forward_back_z"}.get(int(idx), "unknown")


def _plane_name(axes: list[int]) -> str:
    names = {_axis_name(a) for a in axes}
    if names == {"lateral_x", "forward_back_z"}:
        return "horizontal_local_xz"
    if names == {"vertical_y", "forward_back_z"}:
        return "side_vertical_local_yz"
    if names == {"lateral_x", "vertical_y"}:
        return "front_vertical_local_xy"
    return "_".join(sorted(names))


def _first_index(items: list[str], choices: set[str]) -> int | None:
    for idx, item in enumerate(items):
        if item in choices:
            return idx
    return None


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    shape_counts = Counter(r.get("trajectory_shape_classification") for r in rows)
    safe_counts = Counter("safe" if r.get("safe_for_learning") else "unsafe" for r in rows)
    lines = [
        "# Trajectory Shape Report",
        "",
        "Trajectory features use relative pelvis/hip paths. They are audit evidence, not semantic truth.",
        "",
        f"- Rows: {len(rows)}",
        "",
        "## Shape Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in shape_counts.most_common())
    lines.extend(["", "## Safety Counts", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in safe_counts.most_common())
    lines.extend(["", "## Top Oval/Grind Candidates", ""])
    for row in sorted(rows, key=lambda r: float(r.get("feature_values", {}).get("grind_pattern_score") or 0.0), reverse=True)[:20]:
        fv = row.get("feature_values", {})
        lines.append(f"- `{row.get('window_id')}` shape={row.get('trajectory_shape_classification')} grind={fv.get('grind_pattern_score')} oval={fv.get('oval_path_score')} closed={fv.get('closed_loop_ratio')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return round(f, 6) if np.isfinite(f) else f

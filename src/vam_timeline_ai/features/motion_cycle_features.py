"""Cycle-aware motion features for semantic review.

This module is analysis-only. It reads existing baked/relative window arrays and
does not create labels, train models, or generate Timeline animations.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


AXES = ("x", "y", "z")


def extract_motion_cycle_features_v1(run_dir: str | Path, out_jsonl: str | Path, report: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    relative_rows = load_jsonl(run / "relative_motion" / "relative_motion_features.jsonl")
    window_rows = {str(r.get("window_id")): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl")}
    sample_rows = {str(r.get("sample_id")): r for r in load_jsonl(run / "baked" / "motion_sample_index.jsonl")}
    rows: list[dict[str, Any]] = []
    missing_npz = 0
    loaded_npz = 0

    for rel in relative_rows:
        wid = str(rel.get("window_id") or "")
        window = window_rows.get(wid) or {}
        sample = sample_rows.get(str(rel.get("sample_id") or window.get("sample_id") or "")) or {}
        npz_path = _resolve_npz_path(run, rel.get("relative_npz_path"))
        if not npz_path or not npz_path.exists():
            missing_npz += 1
            rows.append(_fallback_row(rel, window, sample, "relative window NPZ missing"))
            continue
        try:
            row = _features_from_npz(npz_path, rel, window, sample)
            loaded_npz += 1
        except Exception as exc:  # noqa: BLE001
            row = _fallback_row(rel, window, sample, f"could not read relative NPZ: {exc}")
        rows.append(row)

    write_jsonl(out_jsonl, rows)
    _write_report(Path(report), rows, loaded_npz, missing_npz)
    return {
        "status": "ok",
        "records": len(rows),
        "loaded_npz": loaded_npz,
        "missing_npz": missing_npz,
        "out_jsonl": str(out_jsonl),
        "report": str(report),
    }


def compute_signal_cycle_metrics(signal: Any, times: Any | None = None, epsilon: float | None = None) -> dict[str, Any]:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return _empty_axis_metrics("insufficient_samples")
    t = np.asarray(times, dtype=np.float64).reshape(-1) if times is not None else np.arange(arr.size, dtype=np.float64)
    if t.size != arr.size:
        t = np.arange(arr.size, dtype=np.float64)
    smooth = _smooth(arr)
    amp = float(np.nanmax(smooth) - np.nanmin(smooth))
    eps = float(epsilon if epsilon is not None else max(amp * 0.08, 1e-4))
    centered = smooth - float(np.nanmean(smooth))
    velocity = np.gradient(smooth, t) if arr.size > 2 else np.zeros_like(smooth)
    v_eps = max(float(np.nanmax(np.abs(velocity))) * 0.08, eps * 0.25, 1e-5)
    sign_changes = _sign_changes(velocity, v_eps)
    peaks, troughs = _peaks_troughs(smooth, eps)
    path = float(np.nansum(np.abs(np.diff(smooth))))
    net = float(abs(smooth[-1] - smooth[0]))
    monotonicity = float(net / max(path, 1e-6))
    return_error = float(abs(smooth[-1] - smooth[0]) / max(amp, 1e-6))
    cycle_count = float(min(peaks, troughs))
    if cycle_count == 0 and sign_changes >= 2 and return_error < 0.75:
        cycle_count = 0.5
    duration = float(max(t[-1] - t[0], 1e-6))
    frequency = float(cycle_count / duration)
    pose_hold = 1.0 if amp < max(eps * 2.0, 0.005) and path < max(eps * 6.0, 0.02) else 0.0
    cyclicity = _clamp((cycle_count / 1.5) * (1.0 - min(return_error, 1.0)) * (1.0 - min(monotonicity, 1.0) * 0.65))
    if pose_hold >= 1.0:
        cyclicity = 0.0
    transition = _clamp(monotonicity * (1.0 - min(cyclicity, 0.9)) + (0.25 if cycle_count < 1.0 and amp >= eps * 4 else 0.0))
    return {
        "displacement_range": round(amp, 6),
        "velocity_zero_crossings": int(sign_changes),
        "sign_change_count": int(sign_changes),
        "peak_count": int(peaks),
        "trough_count": int(troughs),
        "estimated_cycle_count": round(cycle_count, 3),
        "estimated_frequency_hz": round(frequency, 4),
        "cyclicity_score": round(float(cyclicity), 4),
        "monotonicity_score": round(float(monotonicity), 4),
        "pose_hold_score": round(float(pose_hold), 4),
        "transition_score": round(float(transition), 4),
        "return_error": round(float(return_error), 4),
        "sample_count": int(arr.size),
    }


def _features_from_npz(npz_path: Path, rel: dict[str, Any], window: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        delta = np.asarray(data["position_delta"], dtype=np.float64)
        velocity = np.asarray(data["velocity_local"], dtype=np.float64) if "velocity_local" in data.files else np.gradient(delta, axis=0)
        controller_names = [str(x) for x in data["controller_names"].tolist()]
        bodyparts = [str(x) for x in data["bodyparts"].tolist()] if "bodyparts" in data.files else [_part_from_controller(n) for n in controller_names]
    controller_metrics: dict[str, Any] = {}
    for idx, name in enumerate(controller_names):
        metrics = _controller_metrics(name, bodyparts[idx] if idx < len(bodyparts) else _part_from_controller(name), delta[:, idx, :], velocity[:, idx, :], times)
        controller_metrics[name] = metrics
    driver_summary = _driver_summary(controller_metrics)
    anchor_summary = _anchor_summary(controller_metrics)
    completeness_summary = _controller_completeness_summary(controller_metrics)
    return {
        "schema": "motion_cycle_features_v1",
        "window_id": rel.get("window_id") or window.get("window_id"),
        "sample_id": rel.get("sample_id") or window.get("sample_id"),
        "source_id": rel.get("source_id") or window.get("source_id") or sample.get("source_id"),
        "source_scene_file": rel.get("source_scene_file") or window.get("source_scene_file") or sample.get("source_scene_file"),
        "technical_atom_id": rel.get("technical_atom_id") or window.get("technical_atom_id") or sample.get("technical_atom_id"),
        "start_seconds": window.get("start_seconds"),
        "end_seconds": window.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds"),
        "relative_npz_path": str(npz_path),
        "controller_metrics": controller_metrics,
        "driver_summary": driver_summary,
        "anchor_summary": anchor_summary,
        "controller_completeness_summary": completeness_summary,
        "has_hip_control": "hipControl" in controller_metrics,
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }


def _controller_metrics(name: str, bodypart: str, pos: np.ndarray, vel: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    axis_metrics = {}
    ranges = np.nanmax(pos, axis=0) - np.nanmin(pos, axis=0)
    dominant_idx = int(np.nanargmax(np.abs(ranges))) if ranges.size else 0
    for axis_idx, axis in enumerate(AXES):
        axis_metrics[axis] = compute_signal_cycle_metrics(pos[:, axis_idx], times)
    speed_path = float(np.nansum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    net_distance = float(np.linalg.norm(pos[-1] - pos[0])) if len(pos) else 0.0
    max_range = float(np.nanmax(np.abs(ranges))) if ranges.size else 0.0
    moving_steps = int(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1) > 1e-4)) if len(pos) > 1 else 0
    real_motion = bool(max_range >= 0.025 and speed_path >= 0.05 and moving_steps >= 3)
    return {
        "controller_name": name,
        "bodypart": bodypart,
        "axis_metrics": axis_metrics,
        "axis_displacement_ranges": {axis: round(float(abs(ranges[idx])), 6) for idx, axis in enumerate(AXES)},
        "active_axis_count": int(sum(float(abs(r)) >= 0.025 for r in ranges)),
        "cowgirl_motion_pattern": _cowgirl_motion_pattern(axis_metrics, ranges, speed_path, net_distance),
        "dominant_axis": AXES[dominant_idx],
        "dominant_axis_range": round(float(abs(ranges[dominant_idx])), 6),
        "total_path_length": round(speed_path, 6),
        "net_displacement_distance": round(net_distance, 6),
        "net_to_path_ratio": round(float(net_distance / max(speed_path, 1e-6)), 6),
        "moving_step_count": moving_steps,
        "has_real_motion": real_motion,
        "max_displacement_range": round(max_range, 6),
        "cyclicity_score": round(float(max(m["cyclicity_score"] for m in axis_metrics.values())), 4),
        "transition_score": round(float(max(m["transition_score"] for m in axis_metrics.values())), 4),
        "pose_hold_score": round(float(min(1.0, sum(m["pose_hold_score"] for m in axis_metrics.values()) / 3.0)), 4),
        "estimated_cycle_count": round(float(max(m["estimated_cycle_count"] for m in axis_metrics.values())), 3),
        "estimated_frequency_hz": round(float(max(m["estimated_frequency_hz"] for m in axis_metrics.values())), 4),
    }


def _driver_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    preferred = ["hipControl", "pelvisControl", "headControl", "chestControl", "lHandControl", "rHandControl"]
    summary = {}
    for name in preferred:
        if name in metrics:
            m = metrics[name]
            summary[name] = {
                "dominant_axis": m["dominant_axis"],
                "dominant_axis_range": m["dominant_axis_range"],
                "cycle_count": m["estimated_cycle_count"],
                "frequency_hz": m["estimated_frequency_hz"],
                "cyclicity_score": m["cyclicity_score"],
                "transition_score": m["transition_score"],
                "pose_hold_score": m["pose_hold_score"],
                "total_path_length": m["total_path_length"],
            }
    moving = sorted(metrics.values(), key=lambda m: float(m.get("total_path_length") or 0.0), reverse=True)[:5]
    return {
        "key_controllers": summary,
        "top_moving_controllers": [m["controller_name"] for m in moving],
        "global_motion_suspect": _global_motion_suspect(metrics),
    }


def _anchor_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    anchor_names = ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"]
    anchors = {name: metrics[name] for name in anchor_names if name in metrics}
    ranges = {name: m["max_displacement_range"] for name, m in anchors.items()}
    return {
        "anchor_ranges": ranges,
        "feet_stable": all(float(ranges.get(name, 0.0)) <= 0.08 for name in ["lFootControl", "rFootControl"] if name in ranges),
        "knees_stable": all(float(ranges.get(name, 0.0)) <= 0.16 for name in ["lKneeControl", "rKneeControl"] if name in ranges),
        "hands_stable": all(float(ranges.get(name, 0.0)) <= 0.16 for name in ["lHandControl", "rHandControl"] if name in ranges),
        "possible_locomotion": _possible_locomotion(metrics),
    }


def _controller_completeness_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether the window has enough real controllers to classify motion.

    Empty Timeline windows often contain only feet, only hands/head/chest, or a
    few non-driver controllers. Those can have keyframes, but they are not
    biomechanically classifiable Cowgirl motion because the hip/pelvis driver is
    absent or static.
    """

    names = set(metrics)
    required_core = ["hipControl", "pelvisControl", "chestControl", "headControl"]
    lower_anchors = ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl"]
    hands = ["lHandControl", "rHandControl"]
    thighs = ["lThighControl", "rThighControl"]
    present_core = [name for name in required_core if name in names]
    present_lower = [name for name in lower_anchors if name in names]
    present_hands = [name for name in hands if name in names]
    present_thighs = [name for name in thighs if name in names]
    present_major = sorted(name for name in names if name.endswith("Control"))
    has_hip_or_pelvis = bool({"hipControl", "pelvisControl"} & names)
    has_real_hip_motion = _metric_has_real_motion(metrics.get("hipControl"))
    has_real_pelvis_motion = _metric_has_real_motion(metrics.get("pelvisControl"))
    only_feet = bool(present_major) and all("FootControl" in name for name in present_major)
    only_upper = bool(present_major) and all(name in {"chestControl", "headControl", "lHandControl", "rHandControl"} for name in present_major)
    missing = [name for name in required_core + lower_anchors if name not in names]
    status = "usable"
    reasons: list[str] = []
    if not present_major:
        status = "empty"
        reasons.append("no controller tracks found")
    if only_feet:
        status = "only_feet"
        reasons.append("only foot controllers are present")
    if only_upper:
        status = "only_upper_body_hands_head"
        reasons.append("only chest/head/hand controllers are present")
    if not has_hip_or_pelvis:
        status = "missing_hip_and_pelvis"
        reasons.append("hipControl and pelvisControl are both missing")
    elif "hipControl" not in names:
        reasons.append("hipControl missing")
    if has_hip_or_pelvis and not (has_real_hip_motion or has_real_pelvis_motion):
        reasons.append("hip/pelvis controllers have no real transform-distance motion")
    return {
        "status": status,
        "controller_count": len(present_major),
        "present_major_controllers": present_major,
        "present_core_controllers": present_core,
        "present_lower_anchor_controllers": present_lower,
        "present_hand_controllers": present_hands,
        "present_thigh_controllers": present_thighs,
        "missing_expected_controllers": missing,
        "has_hip_control": "hipControl" in names,
        "has_pelvis_control": "pelvisControl" in names,
        "has_hip_or_pelvis": has_hip_or_pelvis,
        "has_real_hip_motion": has_real_hip_motion,
        "has_real_pelvis_motion": has_real_pelvis_motion,
        "only_feet": only_feet,
        "only_upper_body_hands_head": only_upper,
        "reasons": reasons,
    }


def _cowgirl_motion_pattern(axis_metrics: dict[str, Any], ranges: np.ndarray, path: float, net: float) -> dict[str, Any]:
    axis_ranges = {axis: float(abs(ranges[idx])) for idx, axis in enumerate(AXES)}
    active_axes = [axis for axis, value in axis_ranges.items() if value >= 0.025]
    clean_axes = [
        axis
        for axis, metric in axis_metrics.items()
        if float(metric.get("estimated_cycle_count") or 0.0) >= 3.0
        and float(metric.get("estimated_frequency_hz") or 0.0) >= 1.5
        and float(metric.get("cyclicity_score") or 0.0) >= 0.35
    ]
    soft_axes = [
        axis
        for axis, metric in axis_metrics.items()
        if float(metric.get("estimated_cycle_count") or 0.0) >= 2.0
        and float(metric.get("estimated_frequency_hz") or 0.0) >= 1.0
        and float(metric.get("cyclicity_score") or 0.0) >= 0.3
    ]
    net_ratio = float(net / max(path, 1e-6))
    pattern = "none"
    valid_clean = False
    valid_soft = False
    if "y" in clean_axes and axis_ranges["y"] >= max(axis_ranges["x"], axis_ranges["z"]) * 0.7:
        pattern = "vertical_bounce"
        valid_clean = True
    elif clean_axes and len(active_axes) >= 2 and net_ratio <= 0.65:
        pattern = "oval_or_loop"
        valid_clean = True
    elif clean_axes and any(axis in clean_axes for axis in ["x", "z"]):
        pattern = "forward_back_or_lateral_cycle"
        valid_clean = True
    elif soft_axes:
        pattern = "slow_or_partial_cycle"
        valid_soft = True
    elif path < 0.05 or max(axis_ranges.values(), default=0.0) < 0.025:
        pattern = "no_real_motion"
    elif net_ratio >= 0.75:
        pattern = "monotonic_transition"
    return {
        "pattern": pattern,
        "valid_clean_cowgirl_pattern": valid_clean,
        "valid_soft_cowgirl_pattern": valid_soft,
        "clean_cycle_axes": clean_axes,
        "soft_cycle_axes": soft_axes,
        "active_axes": active_axes,
        "net_to_path_ratio": round(net_ratio, 6),
    }


def _global_motion_suspect(metrics: dict[str, Any]) -> bool:
    """Detect empty/global-translation-like clips, not articulated animation.

    If many major body controllers move a large amount with nearly identical
    start-to-end direction and high net/path ratio, this is usually a whole-body
    displacement or Timeline/import artifact rather than semantic body motion.
    """

    major = [
        "hipControl",
        "pelvisControl",
        "chestControl",
        "headControl",
        "lHandControl",
        "rHandControl",
        "lFootControl",
        "rFootControl",
    ]
    vectors = []
    for name in major:
        metric = metrics.get(name)
        if not metric:
            continue
        if float(metric.get("max_displacement_range") or 0.0) < 0.08:
            continue
        if float(metric.get("net_to_path_ratio") or 0.0) < 0.65:
            continue
        # Direction is approximated from dominant axis and net/path. The full
        # vector is not retained in the JSON, but dominant same-axis monotonic
        # motion across many controllers is enough to flag suspicious clips.
        vectors.append(str(metric.get("dominant_axis") or ""))
    if len(vectors) < 5:
        return False
    most_common = Counter(vectors).most_common(1)[0][1]
    return most_common >= 5


def _possible_locomotion(metrics: dict[str, Any]) -> bool:
    hand = max(float((metrics.get(n) or {}).get("max_displacement_range") or 0.0) for n in ["lHandControl", "rHandControl"])
    foot = max(float((metrics.get(n) or {}).get("max_displacement_range") or 0.0) for n in ["lFootControl", "rFootControl"])
    return hand > 0.22 or foot > 0.18


def _metric_has_real_motion(metric: dict[str, Any] | None) -> bool:
    if not metric:
        return False
    if metric.get("has_real_motion") is not None:
        return bool(metric.get("has_real_motion"))
    return float(metric.get("max_displacement_range") or 0.0) >= 0.025 and float(metric.get("total_path_length") or 0.0) >= 0.05


def _fallback_row(rel: dict[str, Any], window: dict[str, Any], sample: dict[str, Any], warning: str) -> dict[str, Any]:
    return {
        "schema": "motion_cycle_features_v1",
        "window_id": rel.get("window_id") or window.get("window_id"),
        "sample_id": rel.get("sample_id") or window.get("sample_id"),
        "source_scene_file": rel.get("source_scene_file") or window.get("source_scene_file") or sample.get("source_scene_file"),
        "technical_atom_id": rel.get("technical_atom_id") or window.get("technical_atom_id") or sample.get("technical_atom_id"),
        "controller_metrics": {},
        "driver_summary": {},
        "anchor_summary": {},
        "controller_completeness_summary": {
            "status": "empty",
            "controller_count": 0,
            "present_major_controllers": [],
            "has_hip_control": False,
            "has_pelvis_control": False,
            "has_hip_or_pelvis": False,
            "has_real_hip_motion": False,
            "has_real_pelvis_motion": False,
            "reasons": [warning],
        },
        "has_hip_control": False,
        "warnings": [warning],
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }


def _empty_axis_metrics(reason: str) -> dict[str, Any]:
    return {
        "displacement_range": 0.0,
        "velocity_zero_crossings": 0,
        "sign_change_count": 0,
        "peak_count": 0,
        "trough_count": 0,
        "estimated_cycle_count": 0.0,
        "estimated_frequency_hz": 0.0,
        "cyclicity_score": 0.0,
        "monotonicity_score": 1.0,
        "pose_hold_score": 1.0,
        "transition_score": 0.0,
        "return_error": 1.0,
        "sample_count": 0,
        "warning": reason,
    }


def _smooth(arr: np.ndarray) -> np.ndarray:
    if arr.size < 7:
        return arr.astype(np.float64)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    padded = np.pad(arr, (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _sign_changes(values: np.ndarray, eps: float) -> int:
    signs = np.zeros(values.shape, dtype=np.int8)
    signs[values > eps] = 1
    signs[values < -eps] = -1
    nz = signs[signs != 0]
    if nz.size < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def _peaks_troughs(values: np.ndarray, eps: float) -> tuple[int, int]:
    if values.size < 5:
        return 0, 0
    diff = np.diff(values)
    signs = np.zeros(diff.shape, dtype=np.int8)
    signs[diff > eps * 0.25] = 1
    signs[diff < -eps * 0.25] = -1
    nz_idx = np.where(signs != 0)[0]
    if nz_idx.size < 2:
        return 0, 0
    nz = signs[nz_idx]
    peak_count = 0
    trough_count = 0
    for a, b in zip(nz[:-1], nz[1:]):
        if a > 0 and b < 0:
            peak_count += 1
        elif a < 0 and b > 0:
            trough_count += 1
    return peak_count, trough_count


def _resolve_npz_path(run: Path, value: Any) -> Path | None:
    text = str(value or "")
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, run / path, run.parent.parent.parent / path]
    return next((c for c in candidates if c.exists()), candidates[0])


def _part_from_controller(name: str) -> str:
    lowered = name.lower()
    for token in ["hip", "pelvis", "chest", "head", "hand", "foot", "knee", "thigh"]:
        if token in lowered:
            return token
    return "unknown"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _write_report(path: Path, rows: list[dict[str, Any]], loaded_npz: int, missing_npz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    states = Counter("has_hip" if r.get("has_hip_control") else "missing_hip" for r in rows)
    completeness = Counter(str((r.get("controller_completeness_summary") or {}).get("status") or "unknown") for r in rows)
    lines = [
        "# Motion Cycle Features V1",
        "",
        "Cycle-aware analysis only. No labels, ML training, or Timeline generation.",
        "",
        f"- Records: {len(rows)}",
        f"- Relative NPZ loaded: {loaded_npz}",
        f"- Relative NPZ missing: {missing_npz}",
        f"- HipControl availability: `{dict(states)}`",
        f"- Controller completeness statuses: `{dict(completeness)}`",
        "- Jitter handling: smoothed lightly; tiny sign changes below epsilon ignored.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

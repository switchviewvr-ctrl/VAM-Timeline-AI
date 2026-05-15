"""Relative/local motion representation for baked Timeline windows.

This module deliberately treats raw baked controller coordinates as source
scene data.  It strips Person/root/world-like tracks, keeps only mapped body
controllers, and writes window-local deltas for semantic analysis.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_json, load_jsonl, safe_id_for_path, write_jsonl
from vam_timeline_ai.motion.controller_mapping import map_controller_name
from vam_timeline_ai.motion.coordinate_spaces import classify_controller_track


STATIC_QUALITY = {"static_or_empty", "static_or_micro_motion", "root_only_motion", "controller_only_whole_person_motion"}


@dataclass(frozen=True)
class RelativeMotionFrame:
    time: float
    controller_name: str
    bodypart: str
    position_delta_from_window_start: tuple[float, float, float]
    rotation_delta_from_window_start: tuple[float, ...]
    position_delta_from_local_baseline: tuple[float, float, float]
    normalized_position_delta: tuple[float, float, float]
    velocity_local: tuple[float, float, float]
    angular_velocity_local: tuple[float, ...]
    parent_bodypart_reference: str | None
    confidence: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RelativeMotionWindow:
    window_id: str
    source_id: str
    sample_id: str
    source_scene_file: str
    technical_atom_id: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    controllers: tuple[str, ...]
    baseline_pose_summary: dict[str, Any]
    coordinate_space_assumptions: dict[str, Any]
    stripped_tracks: tuple[dict[str, Any], ...]
    teleport_risk: str
    safe_for_learning: bool
    safe_for_export_template: bool
    warnings: tuple[str, ...]


def build_relative_motion_windows(
    run_dir: str | Path,
    sample_index: str | Path,
    windows: str | Path,
    controller_map: str | Path,
    body_quality: str | Path,
    out_dir: str | Path,
    index_out: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    out = Path(out_dir)
    npz_dir = out / "windows"
    npz_dir.mkdir(parents=True, exist_ok=True)
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    window_rows = load_jsonl(windows)
    mappings = (load_json(controller_map).get("controller_mappings") if Path(controller_map).exists() else {}) or {}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rows: list[dict[str, Any]] = []
    sample_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    reason_counts: Counter[str] = Counter()
    stripped_counts: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    teleport_counts: Counter[str] = Counter()
    for wrow in window_rows:
        sample = samples.get(wrow.get("sample_id"))
        if not sample or sample.get("bake_status") != "ok":
            row = _unsafe_index_row(wrow, sample or {}, "missing_or_unbaked_sample")
            rows.append(row)
            reason_counts.update(row["unsafe_reasons"])
            continue
        try:
            positions, rotations, times, names = _load_sample_arrays(sample, run, sample_cache)
            row = build_relative_motion_window_row(wrow, sample, positions, rotations, times, names, mappings, body.get(wrow.get("window_id"), {}), npz_dir)
        except Exception as exc:  # keep audit generation robust on malformed samples
            row = _unsafe_index_row(wrow, sample, f"relative_conversion_failed: {exc}")
        rows.append(row)
        reason_counts.update(row.get("unsafe_reasons", []))
        stripped_counts.update(str(t.get("transform_type") or "unknown") for t in row.get("stripped_tracks", []))
        coverage.update(row.get("bodyparts", []))
        teleport_counts[str(row.get("teleport_risk") or "unknown")] += 1
    write_jsonl(index_out, rows)
    _write_report(rows, reason_counts, stripped_counts, coverage, teleport_counts, report)
    return rows


def build_relative_motion_window_row(
    wrow: dict[str, Any],
    sample: dict[str, Any],
    positions: np.ndarray,
    rotations: np.ndarray,
    times: np.ndarray,
    controller_names: list[str],
    controller_mappings: dict[str, Any] | None,
    body_quality: dict[str, Any] | None,
    npz_dir: str | Path | None = None,
) -> dict[str, Any]:
    body_quality = body_quality or {}
    start = max(0, min(int(wrow.get("frame_start") or 0), max(len(times) - 1, 0)))
    end = max(start + 1, min(int(wrow.get("frame_end") or len(times)), len(times)))
    pos = np.asarray(positions[start:end], dtype=np.float32)
    rot = np.asarray(rotations[start:end], dtype=np.float32) if rotations.size else np.zeros((pos.shape[0], pos.shape[1], 0), dtype=np.float32)
    rel_times = np.asarray(times[start:end], dtype=np.float32)
    if len(rel_times):
        rel_times = rel_times - rel_times[0]
    classifications = []
    safe_indices: list[int] = []
    stripped: list[dict[str, Any]] = []
    for idx, name in enumerate(controller_names):
        mapping = (controller_mappings or {}).get(name) or map_controller_name(name)
        c = classify_controller_track(name, mapping)
        classifications.append(c)
        if c.allowed_body_controller:
            safe_indices.append(idx)
        else:
            stripped.append(
                {
                    "controller_name": name,
                    "bodypart": c.bodypart,
                    "coordinate_space": c.coordinate_space,
                    "transform_type": c.transform_type,
                    "teleport_risk": c.teleport_risk,
                    "warnings": list(c.warnings),
                }
            )
    safe_names = [controller_names[i] for i in safe_indices]
    bodyparts = [classifications[i].bodypart for i in safe_indices]
    warnings: list[str] = []
    unsafe_reasons: list[str] = []
    if safe_indices:
        safe_pos = pos[:, safe_indices, :]
        safe_rot = rot[:, safe_indices, :] if rot.ndim == 3 and rot.shape[1] >= len(controller_names) else np.zeros((pos.shape[0], len(safe_indices), 0), dtype=np.float32)
    else:
        safe_pos = np.zeros((pos.shape[0], 0, 3), dtype=np.float32)
        safe_rot = np.zeros((pos.shape[0], 0, 0), dtype=np.float32)
    baseline = safe_pos[0:1] if safe_pos.size else safe_pos
    position_delta = safe_pos - baseline if safe_pos.size else safe_pos
    rotation_delta = safe_rot - safe_rot[0:1] if safe_rot.size else safe_rot
    scale, scale_status = estimate_body_scale(safe_pos, bodyparts)
    normalized = position_delta / max(scale, 1e-6) if position_delta.size else position_delta
    dt = _median_dt(rel_times)
    velocity = _gradient(normalized, dt)
    angular_velocity = _gradient(rotation_delta, dt) if rotation_delta.size else rotation_delta
    if not safe_names:
        unsafe_reasons.append("no_allowed_body_controller_tracks")
    quality = str(body_quality.get("body_motion_quality") or "unknown")
    if quality in STATIC_QUALITY or body_quality.get("static_or_micro_motion"):
        unsafe_reasons.append(quality if quality != "unknown" else "static_or_micro_motion")
    if body_quality.get("minimal_head_motion_only"):
        unsafe_reasons.append("minimal_head_motion_only")
    if body_quality.get("minimal_hand_jitter_only"):
        unsafe_reasons.append("minimal_hand_jitter_only")
    moving_count = _moving_controller_count(position_delta)
    if moving_count == 0 and safe_names:
        unsafe_reasons.append("relative_motion_static")
    if scale_status == "unknown":
        warnings.append("Body scale unavailable; normalized deltas use unit scale.")
    teleport_risk = _teleport_risk(stripped, safe_names)
    if teleport_risk == "high":
        unsafe_reasons.append("high_teleport_risk")
    safe_for_learning = bool(safe_names) and not unsafe_reasons and teleport_risk in {"low", "medium"}
    safe_for_export_template = safe_for_learning and scale_status != "unknown" and teleport_risk == "low"
    rel_path: str | None = None
    if npz_dir is not None and safe_names:
        Path(npz_dir).mkdir(parents=True, exist_ok=True)
        target = Path(npz_dir) / f"{safe_id_for_path(str(wrow.get('window_id')))}.npz"
        np.savez(
            target,
            times=rel_times.astype(np.float32),
            controller_names=np.asarray(safe_names, dtype=object),
            bodyparts=np.asarray(bodyparts, dtype=object),
            position_delta=position_delta.astype(np.float32),
            normalized_position_delta=normalized.astype(np.float32),
            velocity_local=velocity.astype(np.float32),
            rotation_delta=rotation_delta.astype(np.float32),
            angular_velocity_local=angular_velocity.astype(np.float32),
            baseline_positions=safe_pos[0].astype(np.float32) if safe_pos.size else np.zeros((0, 3), dtype=np.float32),
        )
        rel_path = str(target)
    return {
        "window_id": wrow.get("window_id"),
        "source_id": wrow.get("source_id") or sample.get("source_id"),
        "sample_id": wrow.get("sample_id") or sample.get("sample_id"),
        "source_scene_file": wrow.get("source_scene_file") or sample.get("source_scene_file"),
        "source_scene_path": wrow.get("source_scene_path") or sample.get("source_scene_path"),
        "technical_atom_id": wrow.get("technical_atom_id") or sample.get("technical_atom_id"),
        "start_seconds": float(wrow.get("start_seconds") or 0.0),
        "end_seconds": float(wrow.get("end_seconds") or 0.0),
        "duration_seconds": float(wrow.get("duration_seconds") or 0.0),
        "frame_start": int(wrow.get("frame_start") or start),
        "frame_end": int(wrow.get("frame_end") or end),
        "controllers": safe_names,
        "bodyparts": bodyparts,
        "relative_npz_path": rel_path,
        "baseline_pose_summary": {
            "body_scale": round(float(scale), 6),
            "body_scale_status": scale_status,
            "baseline_controller_count": len(safe_names),
        },
        "coordinate_space_assumptions": {
            "input_controller_positions": "source_scene_coordinates",
            "relative_representation": "delta_from_window_start_normalized_by_body_scale_or_unit",
            "allowed_body_controller_count": len(safe_names),
            "stripped_track_count": len(stripped),
        },
        "stripped_tracks": stripped,
        "stripped_track_count": len(stripped),
        "stripped_world_transform_count": sum(1 for t in stripped if t.get("coordinate_space") == "world_absolute"),
        "stripped_atom_root_count": sum(1 for t in stripped if t.get("transform_type") == "person_atom_transform"),
        "teleport_risk": teleport_risk,
        "safe_for_learning": safe_for_learning,
        "safe_for_export_template": safe_for_export_template,
        "moving_controller_count_relative": moving_count,
        "root_world_motion_removed": bool(stripped),
        "unsafe_reasons": _dedupe(unsafe_reasons),
        "warnings": _dedupe(warnings),
    }


def estimate_body_scale(positions: np.ndarray, bodyparts: list[str]) -> tuple[float, str]:
    if not positions.size or not bodyparts:
        return 1.0, "unknown"
    baseline = np.asarray(positions[0], dtype=np.float32)
    anchors = {"hip", "pelvis", "abdomen", "chest", "head", "left_hand", "right_hand", "left_knee", "right_knee", "left_foot", "right_foot"}
    points = [baseline[idx] for idx, part in enumerate(bodyparts) if part in anchors]
    distances = []
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            d = float(np.linalg.norm(a - b))
            if np.isfinite(d) and d > 1e-4:
                distances.append(d)
    if distances:
        return float(np.clip(np.median(distances), 0.1, 3.0)), "estimated_from_controller_baseline"
    spans = np.nanmax(positions, axis=(0, 1)) - np.nanmin(positions, axis=(0, 1))
    span = float(np.linalg.norm(spans))
    if np.isfinite(span) and span > 1e-4:
        return float(np.clip(span, 0.1, 3.0)), "estimated_from_motion_span"
    return 1.0, "unknown"


def _load_sample_arrays(
    sample: dict[str, Any],
    run_dir: Path,
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sid = str(sample.get("sample_id"))
    if sid in cache:
        return cache[sid]
    path = Path(str(sample.get("baked_npz_path") or ""))
    if not path.is_absolute():
        project_root = run_dir.parents[2] if len(run_dir.parents) > 2 else Path.cwd()
        path = project_root / path if str(path).startswith("data") else run_dir / path
    with np.load(path, allow_pickle=True) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        rotations = np.asarray(data["rotations"], dtype=np.float32)
        times = np.asarray(data["times"], dtype=np.float32)
        names = [str(x) for x in data["controller_names"].tolist()]
    cache[sid] = (positions, rotations, times, names)
    return cache[sid]


def _unsafe_index_row(wrow: dict[str, Any], sample: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "window_id": wrow.get("window_id"),
        "source_id": wrow.get("source_id") or sample.get("source_id"),
        "sample_id": wrow.get("sample_id") or sample.get("sample_id"),
        "source_scene_file": wrow.get("source_scene_file") or sample.get("source_scene_file"),
        "technical_atom_id": wrow.get("technical_atom_id") or sample.get("technical_atom_id"),
        "start_seconds": float(wrow.get("start_seconds") or 0.0),
        "end_seconds": float(wrow.get("end_seconds") or 0.0),
        "duration_seconds": float(wrow.get("duration_seconds") or 0.0),
        "controllers": [],
        "bodyparts": [],
        "relative_npz_path": None,
        "baseline_pose_summary": {"body_scale": 1.0, "body_scale_status": "unknown"},
        "coordinate_space_assumptions": {},
        "stripped_tracks": [],
        "stripped_track_count": 0,
        "stripped_world_transform_count": 0,
        "stripped_atom_root_count": 0,
        "teleport_risk": "unknown",
        "safe_for_learning": False,
        "safe_for_export_template": False,
        "moving_controller_count_relative": 0,
        "root_world_motion_removed": False,
        "unsafe_reasons": [reason],
        "warnings": [reason],
    }


def _median_dt(times: np.ndarray) -> float:
    if len(times) < 2:
        return 1.0 / 60.0
    diffs = np.diff(times.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 1.0 / 60.0


def _gradient(values: np.ndarray, dt: float) -> np.ndarray:
    if values.size == 0:
        return values
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values.astype(np.float32), max(dt, 1e-6), axis=0).astype(np.float32)


def _moving_controller_count(position_delta: np.ndarray, threshold: float = 0.015) -> int:
    if not position_delta.size:
        return 0
    ranges = np.nanmax(position_delta, axis=0) - np.nanmin(position_delta, axis=0)
    return int(np.sum(np.linalg.norm(ranges, axis=1) >= threshold))


def _teleport_risk(stripped: list[dict[str, Any]], safe_names: list[str]) -> str:
    if not safe_names:
        return "high" if stripped else "unknown"
    if any(t.get("transform_type") == "person_atom_transform" for t in stripped):
        return "medium"
    if any(t.get("coordinate_space") == "world_absolute" for t in stripped):
        return "medium"
    return "low"


def _write_report(
    rows: list[dict[str, Any]],
    reason_counts: Counter[str],
    stripped_counts: Counter[str],
    coverage: Counter[str],
    teleport_counts: Counter[str],
    report: str | Path,
) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe = [r for r in rows if r.get("safe_for_learning")]
    lines = [
        "# Relative Motion Report",
        "",
        "Raw source-scene coordinates are converted to window-local body-controller deltas. Person/root/world tracks are stripped.",
        "",
        f"- Total windows: {len(rows)}",
        f"- Safe for learning: {len(safe)}",
        f"- Unsafe: {len(rows) - len(safe)}",
        f"- Stripped track records: {sum(r.get('stripped_track_count', 0) for r in rows)}",
        "",
        "## Unsafe Reasons",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in reason_counts.most_common()) if reason_counts else lines.append("- None")
    lines.extend(["", "## Stripped Track Types", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in stripped_counts.most_common()) if stripped_counts else lines.append("- None")
    lines.extend(["", "## Controller Bodypart Coverage", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in coverage.most_common()) if coverage else lines.append("- None")
    lines.extend(["", "## Teleport Risk", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in teleport_counts.most_common()) if teleport_counts else lines.append("- None")
    lines.extend(["", "## Safe Examples", ""])
    for row in safe[:10]:
        lines.append(f"- `{row.get('window_id')}` scene=`{row.get('source_scene_file')}` controllers={len(row.get('controllers', []))}")
    lines.extend(["", "## Rejected Examples", ""])
    for row in [r for r in rows if not r.get("safe_for_learning")][:10]:
        lines.append(f"- `{row.get('window_id')}` reasons={row.get('unsafe_reasons')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out

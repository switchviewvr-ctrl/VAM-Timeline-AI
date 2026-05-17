"""Bake generated relative flow onto a baseline pose for Timeline targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_json


def bake_relative_flow_to_timeline_targets(
    relative_flow: str | Path | dict[str, Any],
    baseline_pose: str | Path | dict[str, Any],
    *,
    include_baseline_keyframe: bool = True,
    include_rotation_tracks: bool = True,
) -> dict[str, Any]:
    flow = load_json(relative_flow) if not isinstance(relative_flow, dict) else relative_flow
    baseline = load_json(baseline_pose) if not isinstance(baseline_pose, dict) else baseline_pose
    baseline_map = {row.get("controller_name"): row for row in baseline.get("controller_poses", []) or []}
    baked_tracks: list[dict[str, Any]] = []
    missing: list[str] = []

    for track in flow.get("controller_tracks", []) or []:
        name = str(track.get("controller_name") or "")
        base = baseline_map.get(name)
        if not base:
            missing.append(name)
            continue
        times = np.asarray(track.get("times") or [], dtype=np.float32)
        deltas = _deltas(track)
        if deltas is None or len(deltas) == 0:
            missing.append(name)
            continue
        count = min(len(times), len(deltas)) if len(times) else len(deltas)
        deltas = deltas[:count]
        if len(times):
            times = times[:count]
        else:
            times = np.arange(count, dtype=np.float32) / float(flow.get("fps") or 60.0)
        base_pos = np.asarray(base.get("baseline_position") or [0.0, 0.0, 0.0], dtype=np.float32)
        rebased = deltas - deltas[0:1] if include_baseline_keyframe else deltas
        positions = base_pos.reshape(1, 3) + rebased
        if include_baseline_keyframe:
            times[0] = 0.0
            positions[0, :] = base_pos
        base_rot = _baseline_rotation_for_controller(name, base.get("baseline_rotation")) if include_rotation_tracks else None
        rotations = None
        if base_rot is not None:
            rotations = np.repeat(base_rot.reshape(1, 4), len(times), axis=0)
        baked_tracks.append({
            "controller_name": name,
            "bodypart": track.get("bodypart"),
            "role": track.get("role"),
            "times": [round(float(t), 6) for t in times.tolist()],
            "positions": _round_path(positions),
            "rotations": _round_path(rotations) if rotations is not None else None,
            "baseline_position": [round(float(v), 6) for v in base_pos.tolist()],
            "baseline_rotation": [round(float(v), 6) for v in base_rot.tolist()] if base_rot is not None else None,
            "baseline_keyframe_included": bool(include_baseline_keyframe),
            "rotation_source": baseline.get("rotation_source") or "synthetic_approximate_fallback" if base_rot is not None else "omitted",
        })

    return {
        "schema": "timeline_target_bake_v1",
        "source_flow_id": flow.get("flow_id"),
        "source_flow_schema": flow.get("schema"),
        "baseline_pose_id": baseline.get("baseline_id"),
        "baseline_style": baseline.get("baseline_style") or baseline.get("style"),
        "generated_baseline_pose": bool(baseline.get("generated_baseline")),
        "include_baseline_keyframe": bool(include_baseline_keyframe),
        "include_rotation_tracks": bool(include_rotation_tracks),
        "duration_seconds": float(flow.get("duration_seconds") or _duration(baked_tracks)),
        "fps": float(flow.get("fps") or 60.0),
        "controller_tracks": baked_tracks,
        "missing_baseline_controllers": missing,
        "source_world_coords_used": False,
        "clip_stitching_used": False,
        "person_root_tracks_included": False,
    }


def _deltas(track: dict[str, Any]) -> np.ndarray | None:
    values = track.get("position_deltas_applied")
    if values is None:
        values = track.get("position_deltas")
    if values is None and track.get("retargeted_positions") is not None and track.get("baseline_position") is not None:
        values = np.asarray(track.get("retargeted_positions"), dtype=np.float32) - np.asarray(track.get("baseline_position"), dtype=np.float32).reshape(1, 3)
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def _rotation(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (4,):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return arr / norm


def _baseline_rotation_for_controller(name: str, values: Any) -> np.ndarray | None:
    parsed = _rotation(values)
    if parsed is not None:
        return parsed
    slight_forward = [-0.087156, 0.0, 0.0, 0.996195]
    head_forward = [-0.043619, 0.0, 0.0, 0.999048]
    if name in {"abdomenControl", "abdomen2Control", "chestControl", "lHandControl", "rHandControl"}:
        return _rotation(slight_forward)
    if name == "headControl":
        return _rotation(head_forward)
    return _rotation([0.0, 0.0, 0.0, 1.0])


def _duration(tracks: list[dict[str, Any]]) -> float:
    value = 0.0
    for track in tracks:
        if track.get("times"):
            value = max(value, float(max(track["times"])))
    return value


def _round_path(path: np.ndarray | None) -> list[list[float]] | None:
    if path is None:
        return None
    return [[round(float(v), 6) for v in row] for row in path.tolist()]

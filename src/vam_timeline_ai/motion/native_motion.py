"""Native MotionAnimationMaster / *Animation.steps[] technical extraction."""

from __future__ import annotations

from typing import Any

import numpy as np

from vam_timeline_ai.io.identity import make_sample_id
from vam_timeline_ai.io.json_utils import as_bool, as_float
from vam_timeline_ai.motion.quaternion_utils import ensure_quat_continuity, quat_normalize


NATIVE_ANIMATION_TO_CONTROLLER = {
    "hipAnimation": "hipControl",
    "pelvisAnimation": "pelvisControl",
    "chestAnimation": "chestControl",
    "headAnimation": "headControl",
    "rHandAnimation": "rHandControl",
    "lHandAnimation": "lHandControl",
    "rFootAnimation": "rFootControl",
    "lFootAnimation": "lFootControl",
    "rElbowAnimation": "rElbowControl",
    "lElbowAnimation": "lElbowControl",
    "rKneeAnimation": "rKneeControl",
    "lKneeAnimation": "lKneeControl",
}


def bake_native_source(scene_data: dict[str, Any], source: dict[str, Any], fps: float = 60.0) -> dict[str, Any]:
    from vam_timeline_ai.motion.baker import (
        compute_angular_delta_array,
        compute_velocities,
        interpolate_positions,
        make_time_grid,
        phase_from_times,
        resample_quaternions,
    )

    atom = _find_atom(scene_data, source.get("technical_atom_id"))
    if atom is None:
        raise ValueError(f"atom not found: {source.get('technical_atom_id')}")
    tracks = []
    warnings: list[str] = []
    for storable in atom.get("storables", []) or []:
        if not isinstance(storable, dict):
            continue
        sid = str(storable.get("id", ""))
        if sid in NATIVE_ANIMATION_TO_CONTROLLER and isinstance(storable.get("steps"), list):
            track, track_warnings = parse_native_track(storable, NATIVE_ANIMATION_TO_CONTROLLER[sid])
            warnings.extend([f"{track['controller_name']}: {w}" for w in track_warnings])
            if len(track["times"]):
                tracks.append(track)
    if not tracks:
        raise ValueError("no native animation tracks found")

    duration = float(source.get("duration_seconds") or max(float(t["times"][-1]) for t in tracks))
    times = make_time_grid(duration, fps=fps, loop=False)
    controller_names = [t["controller_name"] for t in tracks]
    positions = np.zeros((len(times), len(tracks), 3), dtype=np.float32)
    rotations = np.zeros((len(times), len(tracks), 4), dtype=np.float32)
    for idx, track in enumerate(tracks):
        positions[:, idx, :] = interpolate_positions(track["times"], track["positions"], times)
        rotations[:, idx, :] = resample_quaternions(track["times"], track["rotations"], times)
        if float(track["times"][-1]) + 1e-3 < duration:
            warnings.append(f"{track['controller_name']}: native track ends before duration")
    velocities = compute_velocities(positions, times)
    angular = compute_angular_delta_array(rotations)
    sample_id = make_sample_id(str(source.get("source_id") or "unknown_source"), fps=fps, extraction_version="extract_v2", technical_atom_id=source.get("technical_atom_id"), clip_name=source.get("clip_name") or "native_motion_animation", clip_index=source.get("clip_index"))
    return {
        "sample_id": sample_id,
        "duration_seconds": duration,
        "times": times,
        "phase": phase_from_times(times, duration),
        "positions": positions,
        "rotations": rotations,
        "velocities": velocities,
        "angular_deltas": angular,
        "controller_names": controller_names,
        "metadata": {"native_tracks": [_track_metadata(t) for t in tracks], "source": source},
        "warnings": warnings,
    }


def parse_native_track(storable: dict[str, Any], controller_name: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    parsed = []
    for step in storable.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        t = as_float(step.get("timeStep"))
        if t is None:
            continue
        pos = step.get("position") if isinstance(step.get("position"), dict) else {}
        rot = step.get("rotation") if isinstance(step.get("rotation"), dict) else {}
        parsed.append(
            (
                float(t),
                as_bool(step.get("positionOn"), True),
                as_bool(step.get("rotationOn"), True),
                np.asarray([as_float(pos.get("x"), 0.0) or 0.0, as_float(pos.get("y"), 0.0) or 0.0, as_float(pos.get("z"), 0.0) or 0.0], dtype=np.float32),
                np.asarray([as_float(rot.get("x"), 0.0) or 0.0, as_float(rot.get("y"), 0.0) or 0.0, as_float(rot.get("z"), 0.0) or 0.0, as_float(rot.get("w"), 1.0) or 1.0], dtype=np.float32),
            )
        )
    parsed.sort(key=lambda item: item[0])
    if not parsed:
        return {"controller_name": controller_name, "times": np.zeros((0,), dtype=np.float32), "positions": np.zeros((0, 3), dtype=np.float32), "rotations": np.zeros((0, 4), dtype=np.float32)}, warnings
    times = np.asarray([p[0] for p in parsed], dtype=np.float32)
    positions = np.stack([p[3] for p in parsed]).astype(np.float32)
    rotations = ensure_quat_continuity(quat_normalize(np.stack([p[4] for p in parsed]).astype(np.float32)))
    if float(times[0]) > 1e-5:
        warnings.append("native track had no t=0 step; copied first key")
        times = np.concatenate(([0.0], times)).astype(np.float32)
        positions = np.vstack([positions[0], positions]).astype(np.float32)
        rotations = ensure_quat_continuity(np.vstack([rotations[0], rotations]).astype(np.float32))
    return {
        "controller_name": controller_name,
        "source_storable_id": storable.get("id"),
        "times": times,
        "positions": positions,
        "rotations": rotations,
        "position_on_ratio": float(np.mean([p[1] for p in parsed])),
        "rotation_on_ratio": float(np.mean([p[2] for p in parsed])),
        "original_sample_count": len(parsed),
    }, warnings


def _track_metadata(track: dict[str, Any]) -> dict[str, Any]:
    times = track["times"]
    return {
        "controller_name": track["controller_name"],
        "source_storable_id": track.get("source_storable_id"),
        "original_sample_count": track.get("original_sample_count", 0),
        "first_time": float(times[0]) if len(times) else None,
        "last_time": float(times[-1]) if len(times) else None,
        "position_on_ratio": track.get("position_on_ratio"),
        "rotation_on_ratio": track.get("rotation_on_ratio"),
    }


def _find_atom(scene_data: dict[str, Any], atom_id: str | None) -> dict[str, Any] | None:
    for atom in scene_data.get("atoms", []) or []:
        if isinstance(atom, dict) and atom.get("id") == atom_id:
            return atom
    return None

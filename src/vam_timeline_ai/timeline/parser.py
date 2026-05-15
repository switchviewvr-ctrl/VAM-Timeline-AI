"""Technical Timeline parser and baker."""

from __future__ import annotations

from typing import Any

import numpy as np

from vam_timeline_ai.io.identity import make_sample_id
from vam_timeline_ai.io.json_utils import as_bool, as_float, as_int, safe_id_for_path
from vam_timeline_ai.motion.quaternion_utils import ensure_quat_continuity, quat_normalize
from vam_timeline_ai.timeline.bezier import BezierCurve, CurveTypeValues
from vam_timeline_ai.timeline.codec import TimelineKeyframe, decode_keyframe_sequence


POSITION_AXES = ("X", "Y", "Z")
ROTATION_AXES = ("RotX", "RotY", "RotZ", "RotW")
ALL_AXES = POSITION_AXES + ROTATION_AXES


def looks_like_external_timeline_export(data: Any) -> bool:
    return isinstance(data, dict) and "SerializeVersion" in data and "Clips" in data and "AtomType" in data


def bake_timeline_source(scene_data: dict[str, Any], source: dict[str, Any], fps: float = 60.0) -> dict[str, Any]:
    from vam_timeline_ai.motion.baker import (
        compute_angular_delta_array,
        compute_velocities,
        make_time_grid,
        phase_from_times,
    )

    plugin = find_timeline_plugin(scene_data, source.get("technical_atom_id"), source.get("storable_id"))
    clip = find_timeline_clip(plugin, source.get("clip_name"), source.get("clip_index"))
    if plugin is None or clip is None:
        raise ValueError("Timeline plugin or clip not found")
    animation = plugin.get("Animation") if isinstance(plugin.get("Animation"), dict) else {}
    version = as_int(animation.get("SerializeVersion"), 0) or 0
    duration = float(source.get("duration_seconds") or as_float(clip.get("AnimationLength"), 0.0) or 0.0)
    if duration <= 0:
        raise ValueError("Timeline clip duration is missing or zero")
    loop = as_bool(clip.get("Loop"), False)
    times = make_time_grid(duration, fps=fps, loop=False)
    controllers = [c for c in clip.get("Controllers", []) or [] if isinstance(c, dict)]
    if not controllers:
        raise ValueError("Timeline clip has no controllers")
    parsed = [parse_timeline_controller_curves(c, version) for c in controllers]
    controller_names = [c["controller_name"] for c in parsed]
    positions = np.zeros((len(times), len(parsed), 3), dtype=np.float32)
    rotations = np.zeros((len(times), len(parsed), 4), dtype=np.float32)
    warnings: list[str] = []
    curve_stats: dict[str, Any] = {}
    for c_idx, controller in enumerate(parsed):
        controller_stats: dict[str, Any] = {}
        for axis_idx, axis in enumerate(POSITION_AXES):
            curve, stats = make_curve(controller["curves"].get(axis, []), duration, loop, default_value=0.0)
            positions[:, c_idx, axis_idx] = curve.evaluate_many(times)
            controller_stats[axis] = stats
            warnings.extend([f"{controller['controller_name']}.{axis}: {w}" for w in stats["warnings"]])
        rot_values = np.zeros((len(times), 4), dtype=np.float32)
        for axis_idx, axis in enumerate(ROTATION_AXES):
            default = 1.0 if axis == "RotW" else 0.0
            curve, stats = make_curve(controller["curves"].get(axis, []), duration, loop, default_value=default)
            rot_values[:, axis_idx] = curve.evaluate_many(times)
            controller_stats[axis] = stats
            warnings.extend([f"{controller['controller_name']}.{axis}: {w}" for w in stats["warnings"]])
        rotations[:, c_idx, :] = ensure_quat_continuity(quat_normalize(rot_values))
        curve_stats[controller["controller_name"]] = controller_stats
    velocities = compute_velocities(positions, times)
    angular = compute_angular_delta_array(rotations)
    sample_id = make_sample_id(str(source.get("source_id") or "unknown_source"), fps=fps, extraction_version="extract_v2", technical_atom_id=source.get("technical_atom_id"), clip_name=source.get("clip_name"), clip_index=source.get("clip_index"))
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
        "metadata": {
            "source": source,
            "SerializeVersion": version,
            "curve_stats": curve_stats,
            "loop": loop,
        },
        "warnings": warnings,
    }


def find_timeline_plugin(scene_data: dict[str, Any], atom_id: str | None, plugin_id: str | None) -> dict[str, Any] | None:
    for atom in scene_data.get("atoms", []) or []:
        if not isinstance(atom, dict) or atom.get("id") != atom_id:
            continue
        for storable in atom.get("storables", []) or []:
            if isinstance(storable, dict) and storable.get("id") == plugin_id:
                return storable
    return None


def find_timeline_clip(plugin: dict[str, Any] | None, clip_name: str | None, clip_index: int | None = None) -> dict[str, Any] | None:
    if plugin is None:
        return None
    animation = plugin.get("Animation") if isinstance(plugin.get("Animation"), dict) else {}
    clips = [c for c in animation.get("Clips", []) or [] if isinstance(c, dict)]
    if clip_index is not None and 0 <= int(clip_index) < len(clips):
        if clip_name is None or clips[int(clip_index)].get("AnimationName") == clip_name:
            return clips[int(clip_index)]
    for clip in clips:
        if clip.get("AnimationName") == clip_name:
            return clip
    return None


def parse_timeline_controller_curves(controller: dict[str, Any], version: int) -> dict[str, Any]:
    return {
        "controller_name": str(controller.get("Controller") or "unknownControl"),
        "targets_position": as_bool(controller.get("TargetsPosition"), True),
        "targets_rotation": as_bool(controller.get("TargetsRotation"), True),
        "control_position": as_bool(controller.get("ControlPosition"), True),
        "control_rotation": as_bool(controller.get("ControlRotation"), True),
        "curves": {axis: decode_curve_keys(controller.get(axis), version) for axis in ALL_AXES},
    }


def decode_curve_keys(value: Any, version: int) -> list[TimelineKeyframe]:
    if isinstance(value, list):
        return decode_keyframe_sequence(value, version=version)
    if isinstance(value, dict) and isinstance(value.get("keys"), list):
        return decode_keyframe_sequence(value.get("keys"), version=version)
    return []


def make_curve(keys: list[TimelineKeyframe], duration: float, loop: bool, default_value: float = 0.0) -> tuple[BezierCurve, dict[str, Any]]:
    curve_keys = list(keys)
    if not curve_keys:
        curve_keys = [TimelineKeyframe(0.0, default_value, CurveTypeValues.Linear), TimelineKeyframe(duration, default_value, CurveTypeValues.Linear)]
    curve = BezierCurve(curve_keys, loop=loop)
    added = curve.add_edge_frames_if_missing(duration, default_curve_type=CurveTypeValues.Linear)
    warnings = curve.compute_curves()
    return curve, {
        "key_count": len(keys),
        "computed_key_count": len(curve.keys),
        "missing_edge_frames_added": added,
        "warnings": warnings,
    }


def make_timeline_sample_id(scene_file: str | None, atom_id: str | None, plugin_id: str | None, clip_name: str | None) -> str:
    scene_prefix = safe_id_for_path(str(scene_file or "scene").replace(".json", "")).split("_")[0]
    plugin = safe_id_for_path(str(plugin_id or "plugin").split("_")[0]).replace("_", "")
    return f"{scene_prefix}_{safe_id_for_path(atom_id)}_{plugin}_{safe_id_for_path(clip_name)}"

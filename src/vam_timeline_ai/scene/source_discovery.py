"""Lightweight technical source discovery for raw scans."""

from __future__ import annotations

from typing import Any


TIMELINE_PLUGIN_MARKERS = ("VamTimeline.AtomPlugin", "AcidBubbles.Timeline")


def discover_sources(data: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "has_motion_animation_master": False,
        "has_native_motion_tracks": False,
        "native_track_count": 0,
        "native_track_names": [],
        "has_timeline": False,
        "timeline_plugin_count": 0,
        "timeline_controller_clip_count": 0,
        "timeline_floatparam_clip_count": 0,
        "timeline_trigger_only_clip_count": 0,
        "source_types": [],
        "technical_tags": [],
    }

    if not isinstance(data, dict):
        return summary

    atoms = data.get("atoms")
    if isinstance(atoms, list):
        _discover_scene_sources(atoms, summary)
    else:
        _discover_external_timeline_sources(data, summary)

    source_types: list[str] = []
    if summary["has_native_motion_tracks"]:
        source_types.append("vam_native_motion_animation")
    if summary["timeline_controller_clip_count"]:
        source_types.append("timeline_controller_motion")
    if summary["timeline_floatparam_clip_count"]:
        source_types.append("timeline_floatparams")
    if summary["timeline_trigger_only_clip_count"]:
        source_types.append("timeline_trigger_only")
    summary["source_types"] = source_types
    return summary


def _discover_scene_sources(atoms: list[Any], summary: dict[str, Any]) -> None:
    seen_timeline_plugins: set[str] = set()
    native_track_names: set[str] = set()

    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        for storable in atom.get("storables", []) or []:
            if not isinstance(storable, dict):
                continue
            sid = str(storable.get("id", ""))
            if sid == "MotionAnimationMaster" or "MotionAnimationMaster" in sid:
                summary["has_motion_animation_master"] = True

            steps = storable.get("steps")
            if sid.endswith("Animation") and isinstance(steps, list) and steps:
                native_track_names.add(sid)

            if _is_timeline_plugin(sid, storable):
                seen_timeline_plugins.add(sid or f"timeline_plugin_{len(seen_timeline_plugins) + 1}")
                _count_timeline_clips(storable.get("Animation"), summary)

    summary["native_track_names"] = sorted(native_track_names)
    summary["native_track_count"] = len(native_track_names)
    summary["has_native_motion_tracks"] = bool(native_track_names)
    summary["timeline_plugin_count"] = len(seen_timeline_plugins)
    summary["has_timeline"] = bool(seen_timeline_plugins)


def _discover_external_timeline_sources(data: dict[str, Any], summary: dict[str, Any]) -> None:
    if "SerializeVersion" in data and "Clips" in data:
        summary["has_timeline"] = True
        summary["timeline_plugin_count"] = 1
        _count_clip_list(data.get("Clips"), summary)


def _is_timeline_plugin(storable_id: str, storable: dict[str, Any]) -> bool:
    if any(marker in storable_id for marker in TIMELINE_PLUGIN_MARKERS):
        return True
    animation = storable.get("Animation")
    return isinstance(animation, dict) and isinstance(animation.get("Clips"), list)


def _count_timeline_clips(animation: Any, summary: dict[str, Any]) -> None:
    if isinstance(animation, dict):
        _count_clip_list(animation.get("Clips"), summary)


def _count_clip_list(clips: Any, summary: dict[str, Any]) -> None:
    if not isinstance(clips, list):
        return
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        controllers = clip.get("Controllers") or []
        float_params = clip.get("FloatParams") or []
        triggers = clip.get("Triggers") or []
        has_controllers = isinstance(controllers, list) and len(controllers) > 0
        has_floatparams = isinstance(float_params, list) and len(float_params) > 0
        has_triggers = isinstance(triggers, list) and len(triggers) > 0

        if has_controllers:
            summary["timeline_controller_clip_count"] += 1
        if has_floatparams:
            summary["timeline_floatparam_clip_count"] += 1
        if has_triggers and not has_controllers and not has_floatparams:
            summary["timeline_trigger_only_clip_count"] += 1

"""Build Timeline baselines directly from manual VaM pose captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.generated_motion import is_allowed_generated_controller
from vam_timeline_ai.io.json_utils import as_float, dump_json, load_json


def build_manual_gt_baseline_for_plan(plan: dict[str, Any], *, out_dir: str | Path | None = None) -> dict[str, Any]:
    capture_path = Path(str(plan.get("baseline_source_capture") or ""))
    raw = load_json(capture_path)
    rider = ((raw.get("atoms") or {}).get("rider") or {})
    controllers = rider.get("controllers") or {}
    baseline: dict[str, Any] = {
        "schema_version": "manual_gt_controller_baseline_v1",
        "clip_id": plan.get("clip_id"),
        "capture_id": plan.get("capture_id"),
        "source_capture": str(capture_path),
        "coordinate_space": "manual_capture_atom_local",
        "coordinate_note": "Uses local_position_to_atom from SkeletonPoseCaptureTool when available; falls back to world_position only if local is missing.",
        "review_only": True,
        "controller_baseline": {},
        "missing_controllers": [],
        "rotation_source_counts": {},
        "missing_rotations": [],
        "identity_rotation_fallback_controllers": [],
        "source_world_tracks_included": False,
        "person_root_world_tracks_included": False,
    }
    needed = set(plan.get("driver_controllers") or []) | set(plan.get("follower_controllers") or []) | set(plan.get("static_anchor_controllers") or []) | set(plan.get("explicitly_static_controllers") or [])
    for name in sorted(needed):
        if not is_allowed_generated_controller(name):
            continue
        ctrl = controllers.get(name)
        if not ctrl:
            baseline["missing_controllers"].append(name)
            continue
        position = _vec(ctrl.get("local_position_to_atom")) or _vec(ctrl.get("world_position"))
        local_rotation = _quat(ctrl.get("local_rotation_to_atom_quat"))
        world_rotation = _quat(ctrl.get("world_rotation_quat"))
        rotation = local_rotation or world_rotation
        rotation_source = "local_rotation_to_atom_quat" if local_rotation is not None else "world_rotation_quat" if world_rotation is not None else "identity_missing_rotation_fallback"
        if position is None:
            baseline["missing_controllers"].append(name)
            continue
        if rotation is None:
            baseline["missing_rotations"].append(name)
            baseline["identity_rotation_fallback_controllers"].append(name)
        baseline["rotation_source_counts"][rotation_source] = baseline["rotation_source_counts"].get(rotation_source, 0) + 1
        baseline["controller_baseline"][name] = {
            "controller": name,
            "position": position,
            "rotation_quat": rotation or [0.0, 0.0, 0.0, 1.0],
            "rotation_source": rotation_source,
            "source_position_field": "local_position_to_atom" if _vec(ctrl.get("local_position_to_atom")) is not None else "world_position_fallback",
            "active": ctrl.get("active"),
        }
    if out_dir is not None:
        target = Path(out_dir) / str(plan.get("clip_id") or "clip") / "baseline_summary.json"
        dump_json(target, baseline)
    return baseline


def _vec(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    parsed = [as_float(value[i]) for i in range(3)]
    if any(v is None for v in parsed):
        return None
    return [float(v) for v in parsed if v is not None]


def _quat(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    parsed = [as_float(value[i]) for i in range(4)]
    if any(v is None for v in parsed):
        return None
    return [float(v) for v in parsed if v is not None]

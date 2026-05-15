"""Motion source inventory for VaM scene JSON files.

This is technical inventory only. `technical_atom_id` is a source identifier,
not a semantic role.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.identity import make_source_id
from vam_timeline_ai.io.json_utils import as_bool, as_float, as_int, load_json, write_jsonl
from vam_timeline_ai.motion.native_motion import NATIVE_ANIMATION_TO_CONTROLLER


def build_motion_source_index(raw_dir: str | Path, out: str | Path, report: str | Path, recursive: bool = True) -> list[dict[str, Any]]:
    raw_path = Path(raw_dir)
    rows: list[dict[str, Any]] = []
    for scene_path in _iter_json_files(raw_path, recursive=recursive):
        rows.extend(inventory_scene_file(scene_path, raw_root=raw_path))
    _mark_duplicate_ids(rows, "source_id")
    write_jsonl(out, rows)
    write_source_index_report(rows, report)
    return rows


def inventory_scene_file(scene_path: str | Path, raw_root: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(scene_path)
    rel = _relative_scene_path(p, Path(raw_root) if raw_root else None)
    try:
        data = load_json(p)
    except Exception as exc:  # noqa: BLE001
        return [_error_record(p, rel, str(exc))]
    if not isinstance(data, dict) or not isinstance(data.get("atoms"), list):
        return []

    rows: list[dict[str, Any]] = []
    for atom in data.get("atoms", []) or []:
        if not isinstance(atom, dict):
            continue
        atom_id = str(atom.get("id") or "unknown_atom")
        rows.extend(_inventory_native_sources(p, rel, atom, atom_id))
        rows.extend(_inventory_timeline_sources(p, rel, atom, atom_id))
    return rows


def _inventory_native_sources(scene_path: Path, rel_scene_path: str, atom: dict[str, Any], atom_id: str) -> list[dict[str, Any]]:
    native_tracks = []
    for storable in atom.get("storables", []) or []:
        if not isinstance(storable, dict):
            continue
        sid = str(storable.get("id", ""))
        steps = storable.get("steps")
        if sid in NATIVE_ANIMATION_TO_CONTROLLER and isinstance(steps, list) and steps:
            native_tracks.append((sid, NATIVE_ANIMATION_TO_CONTROLLER[sid], len(steps)))
    if not native_tracks:
        return []
    duration = _native_duration_guess(atom)
    source_id = make_source_id(
        rel_scene_path,
        "native_motion_animation",
        atom_id,
        "native_motion_animation",
        None,
        "native_motion_animation",
        None,
        ",".join(track[0] for track in native_tracks),
    )
    return [
        {
            "source_id": source_id,
            "source_type": "native_motion_animation",
            "source_scene_file": scene_path.name,
            "source_scene_path": str(scene_path),
            "source_scene_relative_path": rel_scene_path,
            "technical_atom_id": atom_id,
            "storable_id": "native_motion_animation",
            "plugin_id": None,
            "clip_name": "native_motion_animation",
            "clip_index": None,
            "duration_seconds": duration,
            "controller_count": len(native_tracks),
            "floatparam_count": 0,
            "trigger_count": 0,
            "controller_names": [track[1] for track in native_tracks],
            "has_position_channels": True,
            "has_rotation_channels": True,
            "has_floatparams": False,
            "has_triggers": False,
            "parse_status": "ok",
            "warnings": [],
        }
    ]


def _inventory_timeline_sources(scene_path: Path, rel_scene_path: str, atom: dict[str, Any], atom_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for storable in atom.get("storables", []) or []:
        if not isinstance(storable, dict):
            continue
        animation = storable.get("Animation") if isinstance(storable.get("Animation"), dict) else None
        if animation is None or not isinstance(animation.get("Clips"), list):
            continue
        storable_id = str(storable.get("id") or "timeline_plugin")
        version = as_int(animation.get("SerializeVersion"), 0) or 0
        for idx, clip in enumerate(animation.get("Clips", []) or []):
            if not isinstance(clip, dict):
                continue
            controllers = [c for c in clip.get("Controllers", []) or [] if isinstance(c, dict)]
            float_params = [fp for fp in clip.get("FloatParams", []) or [] if isinstance(fp, dict)]
            triggers = [t for t in clip.get("Triggers", []) or [] if isinstance(t, dict)]
            source_type = _timeline_source_type(controllers, float_params, triggers)
            controller_names = [str(c.get("Controller") or "unknownControl") for c in controllers]
            clip_name = clip.get("AnimationName")
            source_id = make_source_id(rel_scene_path, source_type, atom_id, storable_id, storable_id, str(clip_name or ""), idx)
            rows.append(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_scene_file": scene_path.name,
                    "source_scene_path": str(scene_path),
                    "source_scene_relative_path": rel_scene_path,
                    "technical_atom_id": atom_id,
                    "storable_id": storable_id,
                    "plugin_id": storable_id,
                    "plugin_label": storable.get("pluginLabel"),
                    "clip_name": clip_name,
                    "clip_index": idx,
                    "duration_seconds": as_float(clip.get("AnimationLength")),
                    "controller_count": len(controllers),
                    "floatparam_count": len(float_params),
                    "trigger_count": len(triggers),
                    "controller_names": controller_names,
                    "has_position_channels": any(_has_any_axis(c, ("X", "Y", "Z")) for c in controllers),
                    "has_rotation_channels": any(_has_any_axis(c, ("RotX", "RotY", "RotZ", "RotW")) for c in controllers),
                    "has_floatparams": bool(float_params),
                    "has_triggers": bool(triggers),
                    "timeline_serialize_version": version,
                    "loop": as_bool(clip.get("Loop"), False),
                    "parse_status": "ok",
                    "warnings": [],
                }
            )
    return rows


def write_source_index_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, int] = {}
    by_scene: dict[str, int] = {}
    duplicate_ids = [source_id for source_id, count in _counts(row.get("source_id") for row in rows).items() if count > 1]
    for row in rows:
        by_type[row.get("source_type", "unknown")] = by_type.get(row.get("source_type", "unknown"), 0) + 1
        by_scene[row.get("source_scene_file", "unknown")] = by_scene.get(row.get("source_scene_file", "unknown"), 0) + 1
    lines = [
        "# Motion Source Index",
        "",
        "This is a technical source inventory. Technical atom IDs are not semantic roles.",
        "",
        f"- Total source records: {len(rows)}",
        f"- Duplicate source IDs: {len(duplicate_ids)}",
        "",
        "## Counts By Source Type",
        "",
    ]
    for key, count in sorted(by_type.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Top Scenes By Source Count", ""])
    for scene, count in sorted(by_scene.items(), key=lambda item: item[1], reverse=True)[:20]:
        lines.append(f"- `{scene}`: {count}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _timeline_source_type(controllers: list[dict[str, Any]], float_params: list[dict[str, Any]], triggers: list[dict[str, Any]]) -> str:
    if controllers:
        return "timeline_controller_motion"
    if float_params:
        return "timeline_floatparam_motion"
    if triggers:
        return "timeline_trigger_only"
    return "unknown_motion_source"


def _has_any_axis(controller: dict[str, Any], axes: tuple[str, ...]) -> bool:
    return any(isinstance(controller.get(axis), list) and len(controller.get(axis)) > 0 for axis in axes)


def _native_duration_guess(atom: dict[str, Any]) -> float | None:
    for storable in atom.get("storables", []) or []:
        if isinstance(storable, dict) and str(storable.get("id", "")) == "MotionAnimationMaster":
            return as_float(storable.get("recordedLength"), as_float(storable.get("stopTimestep")))
    max_time: float | None = None
    for storable in atom.get("storables", []) or []:
        if not isinstance(storable, dict) or not isinstance(storable.get("steps"), list):
            continue
        for step in storable.get("steps", []) or []:
            if isinstance(step, dict):
                t = as_float(step.get("timeStep"))
                if t is not None:
                    max_time = max(max_time or 0.0, t)
    return max_time


def _error_record(scene_path: Path, rel_scene_path: str, error: str) -> dict[str, Any]:
    return {
        "source_id": make_source_id(rel_scene_path, "unknown_motion_source", None, None, None, "parse_error", None),
        "source_type": "unknown_motion_source",
        "source_scene_file": scene_path.name,
        "source_scene_path": str(scene_path),
        "source_scene_relative_path": rel_scene_path,
        "technical_atom_id": None,
        "storable_id": None,
        "parse_status": "error",
        "warnings": [error],
    }


def _relative_scene_path(scene_path: Path, raw_root: Path | None) -> str:
    if raw_root:
        try:
            return str(scene_path.relative_to(raw_root)).replace("\\", "/")
        except ValueError:
            pass
    return scene_path.name


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def _mark_duplicate_ids(rows: list[dict[str, Any]], key: str) -> None:
    counts = _counts(row.get(key) for row in rows)
    for row in rows:
        if counts.get(str(row.get(key)), 0) > 1:
            row.setdefault("warnings", []).append(f"duplicate {key} detected")


def _iter_json_files(raw_dir: Path, recursive: bool) -> list[Path]:
    if not recursive:
        return sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    ignored = {"out_dataset", "out_inspect", "out_km190", "out_voxta", "out_audit", "out_dataset_batch1", "vam_mocap_dataset_compiler", "vam-timeline-master"}
    out: list[Path] = []
    for path in raw_dir.rglob("*.json"):
        rel_parts = path.relative_to(raw_dir).parts[:-1]
        if any(part in ignored for part in rel_parts):
            continue
        out.append(path)
    return sorted(out)

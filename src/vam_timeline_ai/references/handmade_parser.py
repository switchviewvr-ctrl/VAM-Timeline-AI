"""Parse and bake handmade Timeline reference animations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.motion.controller_mapping import map_controller_name
from vam_timeline_ai.timeline.codec import decode_keyframe_sequence


ALLOWED_BODY_CONTROLLERS = {
    "hipcontrol",
    "pelviscontrol",
    "abdomencontrol",
    "chestcontrol",
    "headcontrol",
    "lhandcontrol",
    "rhandcontrol",
    "lelbowcontrol",
    "relbowcontrol",
    "lkneecontrol",
    "rkneecontrol",
    "lfootcontrol",
    "rfootcontrol",
    "lthighcontrol",
    "rthighcontrol",
}
DISALLOWED_ROOT_TOKENS = {"control", "person", "atom", "root", "world", "worldcontrol", "rootcontrol"}


def bake_handmade_manifest(manifest: str | Path, sample_index_out: str | Path, baked_dir: str | Path, fps: float = 60.0) -> dict[str, Any]:
    rows = load_jsonl(manifest)
    baked = Path(baked_dir)
    baked.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for row in rows:
        sample_rows.append(_parse_and_bake(row, baked, fps))
    write_jsonl(sample_index_out, sample_rows)
    return {
        "samples": len(sample_rows),
        "parse_ok": sum(1 for r in sample_rows if r.get("parse_status") == "ok"),
        "bake_ok": sum(1 for r in sample_rows if r.get("bake_status") == "ok"),
    }


def _parse_and_bake(row: dict[str, Any], baked_dir: Path, fps: float) -> dict[str, Any]:
    warnings: list[str] = []
    json_path = Path(str(row.get("json_path") or ""))
    base = {
        "reference_id": row.get("reference_id"),
        "label_family": row.get("label_family"),
        "label_subtype": row.get("label_subtype"),
        "style": row.get("style"),
        "intensity": row.get("intensity"),
        "depth": row.get("depth"),
        "is_transition_or_realign": row.get("is_transition_or_realign"),
        "fps": fps,
    }
    if not json_path.exists():
        return {**base, "parse_status": "failed", "bake_status": "failed", "warnings": ["reference JSON missing"]}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {**base, "parse_status": "failed", "bake_status": "failed", "warnings": [f"JSON parse failed: {exc}"]}

    version = int(str(data.get("SerializeVersion", "283")).split(".")[0] or 283) if isinstance(data, dict) else 283
    clips = data.get("Clips", []) if isinstance(data, dict) else []
    clip = clips[0] if clips else data if isinstance(data, dict) else {}
    duration = float(clip.get("AnimationLength") or clip.get("animation_length") or 0.0)
    controllers = clip.get("Controllers", []) if isinstance(clip, dict) else []
    allowed = []
    disallowed = []
    unknown = []
    for controller in controllers:
        name = str(controller.get("Controller") or controller.get("controller_name") or controller.get("name") or "")
        kind = classify_timeline_target(name)
        if kind == "allowed_body_controller":
            allowed.append(controller)
        elif kind == "disallowed_person_atom_or_root":
            disallowed.append(name)
        else:
            unknown.append(name)
    if not allowed:
        warnings.append("No allowed body controllers found after stripping root/Person/world tracks.")
        return {
            **base,
            "animation_length": duration,
            "loop": bool(clip.get("Loop")),
            "controller_names": [str(c.get("Controller") or "") for c in controllers],
            "allowed_body_controller_names": [],
            "disallowed_root_or_atom_track_count": len(disallowed),
            "unknown_track_count": len(unknown),
            "contains_world_space_pose_data": bool(disallowed),
            "contains_person_atom_transform": bool(disallowed),
            "teleport_risk": "high" if disallowed else "unknown",
            "safe_for_timeline_retargeting": False,
            "frame_count": 0,
            "baked_npz_path": None,
            "parse_status": "ok",
            "bake_status": "failed",
            "warnings": warnings,
        }
    duration = max(duration, _infer_duration(allowed, version), 0.25)
    frame_count = max(2, int(round(duration * fps)) + 1)
    times = np.arange(frame_count, dtype=np.float32) / float(fps)
    positions = np.zeros((frame_count, len(allowed), 3), dtype=np.float32)
    rotations = np.zeros((frame_count, len(allowed), 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    names = []
    for c_idx, controller in enumerate(allowed):
        names.append(str(controller.get("Controller") or ""))
        for axis_idx, axis in enumerate(["X", "Y", "Z"]):
            positions[:, c_idx, axis_idx] = _sample_axis(controller.get(axis, []), version, times)
        for axis_idx, axis in enumerate(["RotX", "RotY", "RotZ", "RotW"]):
            default = 1.0 if axis == "RotW" else 0.0
            rotations[:, c_idx, axis_idx] = _sample_axis(controller.get(axis, []), version, times, default=default)
    npz_path = baked_dir / f"{row.get('reference_id')}.npz"
    np.savez_compressed(
        npz_path,
        times=times,
        positions=positions,
        rotations=rotations,
        controller_names=np.asarray(names, dtype=object),
        metadata_json=json.dumps(row, ensure_ascii=False),
    )
    return {
        **base,
        "animation_length": duration,
        "loop": bool(clip.get("Loop")),
        "controller_names": [str(c.get("Controller") or "") for c in controllers],
        "allowed_body_controller_names": names,
        "disallowed_root_or_atom_track_count": len(disallowed),
        "unknown_track_count": len(unknown),
        "contains_world_space_pose_data": bool(disallowed),
        "contains_person_atom_transform": bool(disallowed),
        "teleport_risk": "medium" if disallowed else "low",
        "safe_for_timeline_retargeting": not disallowed,
        "frame_count": frame_count,
        "baked_npz_path": str(npz_path),
        "parse_status": "ok",
        "bake_status": "ok",
        "warnings": warnings,
    }


def classify_timeline_target(name: str) -> str:
    token = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if token in ALLOWED_BODY_CONTROLLERS:
        return "allowed_body_controller"
    if token in DISALLOWED_ROOT_TOKENS or token.startswith("person") or token.endswith("root") or "eyetarget" in token:
        return "disallowed_person_atom_or_root"
    mapped = map_controller_name(name).get("body_part")
    if mapped and mapped != "unknown" and token.endswith("control"):
        return "allowed_body_controller"
    return "unknown_control"


def _infer_duration(controllers: list[dict[str, Any]], version: int) -> float:
    max_time = 0.0
    for controller in controllers:
        for axis in ["X", "Y", "Z", "RotX", "RotY", "RotZ", "RotW"]:
            try:
                keys = decode_keyframe_sequence(controller.get(axis, []), version=version)
                if keys:
                    max_time = max(max_time, max(k.time for k in keys))
            except Exception:
                continue
    return max_time


def _sample_axis(raw_keys: Any, version: int, times: np.ndarray, default: float = 0.0) -> np.ndarray:
    if not raw_keys:
        return np.full(times.shape, default, dtype=np.float32)
    try:
        keys = sorted(decode_keyframe_sequence(raw_keys, version=version), key=lambda k: k.time)
    except Exception:
        return np.full(times.shape, default, dtype=np.float32)
    if not keys:
        return np.full(times.shape, default, dtype=np.float32)
    key_times = np.asarray([k.time for k in keys], dtype=np.float32)
    key_values = np.asarray([k.value for k in keys], dtype=np.float32)
    if len(keys) == 1:
        return np.full(times.shape, float(key_values[0]), dtype=np.float32)
    return np.interp(times, key_times, key_values, left=key_values[0], right=key_values[-1]).astype(np.float32)

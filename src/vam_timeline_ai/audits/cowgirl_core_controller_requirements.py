"""Core controller requirements for generation-safe Cowgirl candidates.

This audit checks controller availability and core hip/pelvis motion.  It is a
generation-safety gate only; it does not promote audit labels to training truth.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


HIP_PARTS = {"hip", "pelvis"}
TORSO_PARTS = {"abdomen", "chest"}
THIGH_PARTS = {"left_thigh", "right_thigh"}
KNEE_PARTS = {"left_knee", "right_knee"}
FOOT_PARTS = {"left_foot", "right_foot"}
HAND_PARTS = {"left_hand", "right_hand"}


def audit_cowgirl_core_controllers(
    run_dir: str | Path,
    relative_index: str | Path,
    controller_map: str | Path,
    body_quality: str | Path,
    pose_anchor_completeness: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    """Audit required core controllers for generation-safe Cowgirl candidates."""
    run = Path(run_dir)
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    rows = [
        cowgirl_core_controller_requirements_for_window(
            row,
            body.get(row.get("window_id"), {}),
            anchors.get(row.get("window_id"), {}),
            run,
        )
        for row in load_jsonl(relative_index)
    ]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def cowgirl_core_controller_requirements_for_window(
    relative_row: dict[str, Any],
    body_quality: dict[str, Any] | None = None,
    pose_anchor: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    body_quality = body_quality or {}
    pose_anchor = pose_anchor or {}
    controllers = [str(c) for c in (relative_row.get("controllers") or relative_row.get("controller_names") or [])]
    bodyparts = [_normalize_part(x) for x in (relative_row.get("bodyparts") or _parts_from_names(controllers))]
    parts = set(bodyparts)
    hip = "hip" in parts
    pelvis = "pelvis" in parts
    torso = bool(TORSO_PARTS & parts)
    thighs = bool(THIGH_PARTS & parts)
    knees = KNEE_PARTS <= parts
    feet = FOOT_PARTS <= parts
    hands = bool(HAND_PARTS & parts)
    core_present = hip or pelvis
    core_motion = _core_motion_amplitude(relative_row, bodyparts, run_dir)
    meaningful_core_motion = bool(core_motion is not None and core_motion >= 0.015)
    if core_motion is None and core_present:
        meaningful_core_motion = bool(
            (body_quality.get("body_motion_quality") in {"good_body_motion", "partial_body_motion"})
            and not body_quality.get("only_root_or_hip_moves")
            and int(body_quality.get("moving_bodypart_count") or 0) >= 2
        )
    lower_complete = bool(core_present and knees and feet)
    missing = []
    if not core_present:
        missing.append("hipControl_or_pelvisControl")
    if core_present and not meaningful_core_motion:
        missing.append("meaningful_hip_or_pelvis_motion")
    if not torso:
        missing.append("abdomenControl_or_chestControl")
    if not thighs:
        missing.append("thigh_controls")
    if not knees:
        missing.append("knee_controls")
    if not feet:
        missing.append("foot_controls")
    missing_core = bool(not core_present or not meaningful_core_motion)
    if missing_core:
        status = "missing_core"
        gate: bool | str = False
    elif lower_complete and torso:
        status = "complete"
        gate = True
    elif core_present:
        status = "partial"
        gate = False
    else:
        status = "unknown"
        gate = "unknown"
    warnings = []
    if missing_core:
        warnings.append("Missing hip/pelvis controller or meaningful hip/pelvis motion; not generation-safe Cowgirl.")
    if not thighs:
        warnings.append("Thigh controllers are missing; lower-body Cowgirl context is less reliable.")
    if not knees or not feet:
        warnings.append("Knee/foot anchors are incomplete for generation-safe Cowgirl.")
    return {
        "window_id": relative_row.get("window_id"),
        "sample_id": relative_row.get("sample_id"),
        "source_id": relative_row.get("source_id"),
        "source_scene_file": relative_row.get("source_scene_file"),
        "technical_atom_id": relative_row.get("technical_atom_id"),
        "has_hip_control": hip,
        "has_pelvis_control": pelvis,
        "has_abdomen_or_chest_control": torso,
        "has_thigh_controls": thighs,
        "has_knee_controls": knees,
        "has_foot_controls": feet,
        "has_hand_controls": hands,
        "core_pelvis_motion_controller_present": bool(core_present and meaningful_core_motion),
        "core_pelvis_motion_amplitude": round(float(core_motion), 6) if core_motion is not None else None,
        "lower_body_controller_set_complete": lower_complete,
        "cowgirl_core_controller_status": status,
        "missing_core_controllers": _dedupe(missing),
        "generation_safe_core_controller_gate": gate,
        "warnings": _dedupe(warnings),
        "pose_anchor_completeness": pose_anchor,
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def cowgirl_core_controller_requirements_for_parts(
    bodyparts: list[str],
    controllers: list[str] | None = None,
    core_motion_amplitude: float | None = 0.1,
) -> dict[str, Any]:
    row = {
        "window_id": "synthetic",
        "controllers": controllers or bodyparts,
        "bodyparts": bodyparts,
        "core_motion_amplitude": core_motion_amplitude,
    }
    return cowgirl_core_controller_requirements_for_window(row, {}, {}, None)


def _core_motion_amplitude(relative_row: dict[str, Any], bodyparts: list[str], run_dir: str | Path | None) -> float | None:
    explicit = relative_row.get("core_motion_amplitude")
    if explicit is not None:
        return _finite(explicit)
    path = relative_row.get("relative_npz_path")
    if not path:
        return None
    npz_path = Path(str(path))
    if not npz_path.is_absolute() and run_dir is not None:
        run = Path(run_dir)
        project_root = run.parents[2] if len(run.parents) >= 3 else Path.cwd()
        candidate = project_root / npz_path if str(path).startswith("data") else Path.cwd() / npz_path
        npz_path = candidate if candidate.exists() else Path.cwd() / npz_path
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            delta = np.asarray(data.get("position_delta"), dtype=np.float32)
            if delta.ndim != 3:
                return None
            indices = [idx for idx, part in enumerate(bodyparts) if part in HIP_PARTS and idx < delta.shape[1]]
            if not indices:
                return None
            arr = delta[:, indices, :]
            amp = float(np.nanmax(np.linalg.norm(arr - arr[0:1, :, :], axis=2)))
            return amp if np.isfinite(amp) else None
    except Exception:
        return None


def _parts_from_names(names: list[str]) -> list[str]:
    return [_normalize_part(name) for name in names]


def _normalize_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("control", "").replace("_", "").replace(" ", "")
    aliases = {
        "hip": "hip",
        "pelvis": "pelvis",
        "abdomen": "abdomen",
        "chest": "chest",
        "lthigh": "left_thigh",
        "leftthigh": "left_thigh",
        "rthigh": "right_thigh",
        "rightthigh": "right_thigh",
        "lknee": "left_knee",
        "leftknee": "left_knee",
        "rknee": "right_knee",
        "rightknee": "right_knee",
        "lfoot": "left_foot",
        "leftfoot": "left_foot",
        "rfoot": "right_foot",
        "rightfoot": "right_foot",
        "lhand": "left_hand",
        "lefthand": "left_hand",
        "rhand": "right_hand",
        "righthand": "right_hand",
    }
    return aliases.get(text, str(value or "").strip().lower())


def _finite(value: Any) -> float | None:
    try:
        val = float(value)
        return val if np.isfinite(val) else None
    except Exception:
        return None


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(r.get("cowgirl_core_controller_status") for r in rows)
    gate_pass = sum(1 for r in rows if r.get("generation_safe_core_controller_gate") is True)
    missing_core = [r for r in rows if r.get("cowgirl_core_controller_status") == "missing_core"]
    lower_complete = sum(1 for r in rows if r.get("lower_body_controller_set_complete"))
    lines = [
        "# Cowgirl Core Controller Requirements Report",
        "",
        "This audit gates generation-safe Cowgirl candidates by required controller presence and hip/pelvis motion. It is not a training label source.",
        "",
        f"- Windows audited: {len(rows)}",
        f"- Core-controller gate pass: {gate_pass}",
        f"- Missing-core windows: {len(missing_core)}",
        f"- Lower-body controller set complete: {lower_complete}",
        "",
        "## Core Controller Status",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in status_counts.most_common()) if status_counts else lines.append("- None")
    lines.extend(["", "## Missing-Core Examples", ""])
    for row in missing_core[:25]:
        lines.append(
            f"- `{row.get('window_id')}` missing={row.get('missing_core_controllers')} "
            f"scene=`{row.get('source_scene_file')}`"
        )
    if not missing_core:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

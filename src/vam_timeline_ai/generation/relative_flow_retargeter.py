"""Retarget generated relative motion flows onto a baseline pose."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_json


def retarget_motion_flow_v0(
    flow: str | Path,
    baseline_pose: str | Path,
    out_json: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    flow_data = load_json(flow)
    baseline = load_json(baseline_pose)
    baseline_map = {pose.get("controller_name"): pose for pose in baseline.get("controller_poses", []) or []}
    tracks = []
    missing = []
    for source_track in flow_data.get("controller_tracks", []) or []:
        name = source_track.get("controller_name")
        base = baseline_map.get(name)
        if not base:
            missing.append(name)
            continue
        base_pos = np.asarray(base.get("baseline_position") or [0.0, 0.0, 0.0], dtype=float)
        deltas = np.asarray(source_track.get("position_deltas") or [], dtype=float)
        positions = base_pos.reshape(1, 3) + deltas
        tracks.append({
            "controller_name": name,
            "bodypart": source_track.get("bodypart"),
            "role": source_track.get("role"),
            "times": source_track.get("times") or [],
            "baseline_position": _round_vec(base_pos),
            "retargeted_positions": _round_path(positions),
            "position_deltas_applied": source_track.get("position_deltas") or [],
            "rotation_values": None,
            "coordinate_space": "retargeted_to_baseline_pose",
            "baseline_coordinate_space": base.get("coordinate_space"),
            "generation_method": "baseline_position_plus_relative_delta_v0",
            "source_generated_track_method": source_track.get("generation_method"),
            "warnings": [
                "Retargeted from generated relative deltas onto synthetic/current baseline.",
                "Not copied from source scene world coordinates.",
            ],
        })
    safe_candidate = not missing and bool(tracks)
    data = {
        "schema": "retargeted_motion_flow_v0",
        "flow_id": f"retargeted_flow_v0::{uuid.uuid4().hex[:12]}",
        "source_generated_flow": flow_data.get("flow_id"),
        "baseline_pose_id": baseline.get("baseline_id"),
        "baseline_source": baseline.get("source"),
        "controller_tracks": tracks,
        "coordinate_space": "retargeted_to_baseline_pose",
        "person_root_included": False,
        "world_coords_source": "none",
        "source_world_coords_used": False,
        "clip_stitching_used": False,
        "safe_for_review_export_candidate": safe_candidate,
        "safe_for_generation_template_candidate": False,
        "generation_template_candidate": False,
        "missing_baseline_controllers": missing,
        "warnings": [
            "Review/prototype retargeting only; production export is not claimed.",
            "Synthetic baseline coordinates are not source Timeline coordinates.",
        ] + ([f"Missing baseline controllers: {missing}"] if missing else []),
    }
    dump_json(out_json, data)
    _write_npz(out_npz, data)
    _write_report(data, report)
    return data


def retarget_motion_flow_v1(
    flow: str | Path,
    baseline_pose: str | Path,
    out_json: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    data = retarget_motion_flow_v0(flow, baseline_pose, out_json, out_npz, report)
    source = load_json(flow)
    baseline = load_json(baseline_pose)
    data.update({
        "schema": "retargeted_motion_flow_v1",
        "flow_id": f"retargeted_flow_v1::{uuid.uuid4().hex[:12]}",
        "baseline_style": baseline.get("style"),
        "intended_family": baseline.get("intended_family"),
        "anchor_profile": baseline.get("anchor_profile"),
        "coordination_profile": source.get("coordination_profile", {}).get("profile_id") if isinstance(source.get("coordination_profile"), dict) else source.get("coordination_profile"),
        "axis_scales": source.get("axis_scales", {}),
        "safe_for_generation_template_candidate": False,
        "generation_template_candidate": False,
    })
    dump_json(out_json, data)
    _write_report(data, report)
    return data


def _write_npz(out_npz: str | Path, data: dict[str, Any]) -> None:
    target = Path(out_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    tracks = data.get("controller_tracks", []) or []
    times = np.asarray(tracks[0].get("times") if tracks else [], dtype=np.float32)
    positions = np.stack([np.asarray(track.get("retargeted_positions"), dtype=np.float32) for track in tracks], axis=1) if tracks else np.zeros((0, 0, 3), dtype=np.float32)
    np.savez_compressed(
        target,
        times=times,
        retargeted_positions=positions,
        controller_names=np.asarray([track.get("controller_name") for track in tracks]),
        roles=np.asarray([track.get("role") for track in tracks]),
        coordinate_space=np.asarray(["retargeted_to_baseline_pose"]),
        source_world_coords_used=np.asarray([False]),
        clip_stitching_used=np.asarray([False]),
    )


def _write_report(data: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    tracks = data.get("controller_tracks", []) or []
    version = "V1" if str(data.get("schema")) == "retargeted_motion_flow_v1" else "V0"
    lines = [
        f"# Retargeted Motion Flow {version} Report",
        "",
        "Generated relative deltas were applied to a baseline pose. This is a review/prototype retarget, not production Timeline output.",
        "",
        f"- Flow: `{data.get('flow_id')}`",
        f"- Source generated flow: `{data.get('source_generated_flow')}`",
        f"- Baseline pose: `{data.get('baseline_pose_id')}`",
        f"- Coordinate space: `{data.get('coordinate_space')}`",
        f"- Source world coords used: `{data.get('source_world_coords_used')}`",
        f"- Clip stitching used: `{data.get('clip_stitching_used')}`",
        f"- Person/root included: `{data.get('person_root_included')}`",
        f"- Review export candidate: `{data.get('safe_for_review_export_candidate')}`",
        f"- Generation template candidate: `{data.get('safe_for_generation_template_candidate')}`",
        f"- Controllers: `{[track.get('controller_name') for track in tracks]}`",
        "",
        "## Limitations",
        "",
        "- Synthetic baseline pose only.",
        "- No VaM import/retarget solver yet.",
        "- No production generation-template claim.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round_vec(values: np.ndarray) -> list[float]:
    return [round(float(v), 6) for v in values.tolist()]


def _round_path(path: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 6), round(float(y), 6), round(float(z), 6)] for x, y, z in path.tolist()]

"""Synthesize v0 generated relative controller flows from primitive statistics.

This module intentionally does not export Timeline data. It creates new,
parameterized relative curves from primitive group statistics so downstream
retargeting can be designed separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import numpy as np

from vam_timeline_ai.generation.generated_motion import (
    GeneratedControllerTrack,
    GeneratedMotionFlow,
    controller_bodypart,
)
from vam_timeline_ai.generation.body_coordination import get_body_coordination_profile
from vam_timeline_ai.generation.trajectory_synthesizer import path_stats, synthesize_path
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl


def synthesize_motion_flow_v0(
    plan: str | Path,
    primitive_groups: str | Path,
    primitives: str | Path,
    out_json: str | Path,
    out_npz: str | Path,
    report: str | Path,
    duration: float = 4.0,
    fps: float = 60.0,
    seed: int = 42,
) -> dict[str, Any]:
    plan_data = load_json(plan)
    group_data = load_json(primitive_groups)
    primitive_rows = load_jsonl(primitives)
    selected = _select_group(plan_data, group_data)
    subtype = _subtype_from_group(selected, plan_data)
    amp = _amplitudes_from_group(selected, subtype, plan_data)
    cycles = _cycles_from_group(selected, primitive_rows, plan_data, duration)
    times, driver_path = synthesize_path(
        subtype=subtype,
        duration=duration,
        fps=fps,
        cycles=cycles,
        amplitude_forward_back=amp["forward_back"],
        amplitude_lateral=amp["lateral"],
        amplitude_vertical=amp["vertical"],
        seed=seed,
        irregularity=_irregularity_for_plan(plan_data),
    )
    driver_name = _select_driver_controller(selected, primitive_rows)
    group_id = str(selected.get("primitive_set_id") or "unknown_group")
    tracks = [
        GeneratedControllerTrack(
            controller_name=driver_name,
            bodypart=controller_bodypart(driver_name),
            role="driver",
            coordinate_space="relative_body_motion",
            times=_round_list(times),
            position_deltas=_round_path(driver_path),
            rotation_deltas=None,
            generation_method="parametric_trajectory_synthesis_v0",
            source_primitive_group=group_id,
            safety_flags=_track_safety_flags(),
            warnings=["New synthesized relative curve; not copied from source Timeline keyframes."],
        )
    ]
    tracks.extend(_follower_tracks(plan_data, times, driver_path, group_id))
    tracks.extend(_anchor_tracks(times, group_id))

    rhythm_profile = {
        "tempo": _phase_query(plan_data).get("tempo") or "medium",
        "cycles": round(float(cycles), 6),
        "cycle_duration_seconds": round(float(duration) / max(float(cycles), 1e-6), 6),
        "fps": float(fps),
        "source": "primitive_group_statistics_and_prompt_parameters",
    }
    amplitude_profile = {
        "forward_back": round(float(amp["forward_back"]), 6),
        "lateral": round(float(amp["lateral"]), 6),
        "vertical": round(float(amp["vertical"]), 6),
        "units": "normalized_relative_controller_delta",
    }
    flow = GeneratedMotionFlow(
        flow_id=f"generated_flow_v0::{uuid.uuid4().hex[:12]}",
        semantic_plan=plan_data,
        selected_primitive_group=group_id,
        duration_seconds=float(duration),
        fps=float(fps),
        controller_tracks=tracks,
        trajectory_shape=subtype,
        rhythm_profile=rhythm_profile,
        amplitude_profile=amplitude_profile,
        anchor_policy={
            "mode": "stable_zero_delta_anchors_v0",
            "anchor_controllers": [track.controller_name for track in tracks if track.role == "anchor"],
            "anchors_are_static_relative_deltas": True,
            "retargeting_required_before_timeline_export": True,
        },
        warnings=[
            "Generated relative motion flow only; no VaM Timeline export was produced.",
            "No absolute source-scene coordinates, Person/root tracks, or clip stitching are used.",
            "Retargeting, contact solving, and Timeline export safety remain future work.",
        ],
    )
    data = flow.to_dict()
    dump_json(out_json, data)
    _write_npz(out_npz, flow)
    _write_report(data, report, path_stats(driver_path, times))
    return data


def synthesize_motion_flow_v1(
    plan: str | Path,
    primitive_groups: str | Path,
    primitives: str | Path,
    coordination_profile: str,
    out_json: str | Path,
    out_npz: str | Path,
    report: str | Path,
    duration: float = 4.0,
    fps: float = 60.0,
    seed: int = 42,
    tempo: str | None = None,
    vertical_scale: float = 1.25,
    lateral_scale: float = 0.70,
    forward_back_scale: float = 1.0,
    chest_follower_scale: float = 0.35,
) -> dict[str, Any]:
    plan_data = load_json(plan)
    group_data = load_json(primitive_groups)
    primitive_rows = load_jsonl(primitives)
    selected = _select_group(plan_data, group_data)
    profile = get_body_coordination_profile(coordination_profile)
    subtype = profile.subtype or _subtype_from_group(selected, plan_data)
    amp = _amplitudes_from_group(selected, subtype, plan_data)
    amp["lateral"] *= float(lateral_scale)
    amp["vertical"] *= float(vertical_scale)
    amp["forward_back"] *= float(forward_back_scale)
    cycles = _cycles_from_group(selected, primitive_rows, plan_data, duration)
    if tempo:
        tempo_factor = {"slow": 0.82, "medium": 1.0, "fast": 1.18}.get(str(tempo), 1.0)
        cycles *= tempo_factor
    times, driver_path = synthesize_path(
        subtype=subtype,
        duration=duration,
        fps=fps,
        cycles=cycles,
        amplitude_forward_back=amp["forward_back"],
        amplitude_lateral=amp["lateral"],
        amplitude_vertical=amp["vertical"],
        seed=seed,
        irregularity=_irregularity_for_plan(plan_data) * 0.75,
    )
    group_id = str(selected.get("primitive_set_id") or "unknown_group")
    tracks: list[GeneratedControllerTrack] = [
        _make_track(profile.driver_controller, "driver", times, driver_path, group_id, "parametric_cowgirl_driver_v1", "Coordinated Cowgirl pelvis/hip driver.")
    ]
    for name in profile.follower_controllers:
        damp = float(profile.damping.get(name, 0.2))
        phase = float(profile.phase_offsets.get(name, 0.0))
        follower = _phase_shift_path(driver_path, phase) * damp
        if name == "chestControl":
            follower *= float(chest_follower_scale) / max(damp, 1e-6)
            follower[:, 2] += float(amp["forward_back"]) * 0.10
            follower[:, 1] += float(amp["vertical"]) * 0.04
        if name == "headControl":
            follower[:, 0] *= 0.35
            follower[:, 1] *= 0.25
            follower[:, 2] *= 0.35
        tracks.append(_make_track(name, "follower", times, follower, group_id, "damped_phase_follower_v1", f"{name} follows pelvis with damping and phase offset."))
    zeros = np.zeros((len(times), 3), dtype=np.float32)
    for name in profile.anchor_controllers:
        role = "support" if "Hand" in name else "anchor"
        tracks.append(_make_track(name, role, times, zeros, group_id, "stable_pose_anchor_v1", "Stable pose anchor for Cowgirl review baseline."))
    rhythm_profile = {
        "tempo": tempo or _phase_query(plan_data).get("tempo") or "medium",
        "cycles": round(float(cycles), 6),
        "cycle_duration_seconds": round(float(duration) / max(float(cycles), 1e-6), 6),
        "fps": float(fps),
        "coordination_profile": profile.profile_id,
    }
    amplitude_profile = {
        "forward_back": round(float(amp["forward_back"]), 6),
        "lateral": round(float(amp["lateral"]), 6),
        "vertical": round(float(amp["vertical"]), 6),
        "units": "normalized_relative_controller_delta",
    }
    flow = GeneratedMotionFlow(
        flow_id=f"generated_flow_v1::{uuid.uuid4().hex[:12]}",
        semantic_plan=plan_data,
        selected_primitive_group=group_id,
        duration_seconds=float(duration),
        fps=float(fps),
        controller_tracks=tracks,
        trajectory_shape=subtype,
        rhythm_profile=rhythm_profile,
        amplitude_profile=amplitude_profile,
        anchor_policy={
            "mode": "cowgirl_motion_plus_static_anchors_v1",
            "anchor_controllers": [t.controller_name for t in tracks if t.role in {"anchor", "support"}],
            "anchors_are_static_relative_deltas": True,
        },
        warnings=[
            "Generated relative Cowgirl motion flow v1; review-only.",
            "No source-scene coordinates, Person/root tracks, or clip stitching are used.",
        ],
    )
    data = flow.to_dict()
    data.update({
        "schema": "generated_motion_flow_v1",
        "coordination_profile": profile.to_dict(),
        "axis_scales": {
            "vertical_scale": float(vertical_scale),
            "lateral_scale": float(lateral_scale),
            "forward_back_scale": float(forward_back_scale),
            "chest_follower_scale": float(chest_follower_scale),
        },
        "review_findings_addressed": ["reduced_lateral_hula_hoop", "added_body_followers", "cowgirl_anchor_policy"],
    })
    dump_json(out_json, data)
    _write_npz_from_dict(out_npz, data)
    _write_report(data, report, path_stats(driver_path, times))
    return data


def _make_track(name: str, role: str, times: np.ndarray, path: np.ndarray, group_id: str, method: str, warning: str) -> GeneratedControllerTrack:
    return GeneratedControllerTrack(
        controller_name=name,
        bodypart=controller_bodypart(name),
        role=role,
        coordinate_space="relative_body_motion",
        times=_round_list(times),
        position_deltas=_round_path(path.astype(np.float32)),
        rotation_deltas=None,
        generation_method=method,
        source_primitive_group=group_id,
        safety_flags=_track_safety_flags(),
        warnings=[warning, "New synthesized relative deltas; not copied from source Timeline."],
    )


def _phase_shift_path(path: np.ndarray, phase_fraction: float) -> np.ndarray:
    if not len(path):
        return path
    shift = int(round(float(phase_fraction) * len(path) / 6.28318))
    return np.roll(path, shift, axis=0).astype(np.float32)


def _write_npz_from_dict(out_npz: str | Path, data: dict[str, Any]) -> None:
    target = Path(out_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    tracks = data.get("controller_tracks", []) or []
    times = np.asarray(tracks[0].get("times") if tracks else [], dtype=np.float32)
    positions = np.stack([np.asarray(track.get("position_deltas"), dtype=np.float32) for track in tracks], axis=1) if tracks else np.zeros((0, 0, 3), dtype=np.float32)
    np.savez_compressed(
        target,
        times=times,
        position_deltas=positions,
        controller_names=np.asarray([track.get("controller_name") for track in tracks]),
        roles=np.asarray([track.get("role") for track in tracks]),
        coordinate_space=np.asarray(["relative_body_motion"]),
        export_ready=np.asarray([False]),
        clip_stitching_used=np.asarray([False]),
    )


def _phase_query(plan_data: dict[str, Any]) -> dict[str, Any]:
    for phase in plan_data.get("sequence", []) or []:
        query = phase.get("primitive_query", {}) or {}
        if query:
            return query
    return {}


def _select_group(plan_data: dict[str, Any], group_data: dict[str, Any]) -> dict[str, Any]:
    query = _phase_query(plan_data)
    subtype = str(query.get("subtype") or query.get("trajectory_shape") or "").lower()
    wanted = {
        "oval_grind": "cowgirl_oval_grind",
        "grinding": "cowgirl_oval_grind",
        "circular_grind": "cowgirl_circular_grind",
        "vertical_bounce": "cowgirl_vertical_bounce",
        "forward_back_rock": "cowgirl_forward_back_rock",
        "riding": "cowgirl_riding_general",
    }.get(subtype, "cowgirl_riding_general")
    groups = group_data.get("groups", []) or []
    by_id = {str(group.get("primitive_set_id")): group for group in groups}
    selected = by_id.get(wanted)
    if selected and int((selected.get("cluster_summary") or {}).get("count") or 0) > 0:
        return selected
    fallbacks = ["cowgirl_oval_grind", "cowgirl_vertical_bounce", "cowgirl_riding_general", "cowgirl_circular_grind"]
    for key in fallbacks:
        group = by_id.get(key)
        if group and int((group.get("cluster_summary") or {}).get("count") or 0) > 0:
            return group
    return selected or (groups[0] if groups else {"primitive_set_id": "unknown_group", "variation_ranges": {}})


def _subtype_from_group(group: dict[str, Any], plan_data: dict[str, Any]) -> str:
    query = _phase_query(plan_data)
    requested = str(query.get("subtype") or query.get("trajectory_shape") or "").lower()
    if requested in {"oval_grind", "circular_grind", "vertical_bounce", "forward_back_rock"}:
        return requested
    group_id = str(group.get("primitive_set_id") or "")
    if group_id.startswith("cowgirl_"):
        return group_id.replace("cowgirl_", "")
    return "riding_general"


def _range_mid(group: dict[str, Any], key: str, default: float) -> float:
    value = ((group.get("variation_ranges") or {}).get(key) or {})
    lo = value.get("min")
    hi = value.get("max")
    nums = [float(v) for v in [lo, hi] if v is not None]
    if not nums:
        return default
    mid = sum(nums) / len(nums)
    if mid <= 0:
        return default
    return mid


def _amplitudes_from_group(group: dict[str, Any], subtype: str, plan_data: dict[str, Any]) -> dict[str, float]:
    query = _phase_query(plan_data)
    scale = {"small": 0.7, "medium": 1.0, "large": 1.25}.get(str(query.get("amplitude") or "medium"), 1.0)
    fb = _range_mid(group, "forward_back_amplitude", 0.10)
    lat = _range_mid(group, "lateral_amplitude", 0.07)
    vert = _range_mid(group, "vertical_amplitude", 0.035)
    if subtype == "oval_grind":
        fb = max(fb, 0.08)
        lat = max(lat, 0.055)
        vert = min(max(vert, 0.015), 0.08)
    elif subtype == "vertical_bounce":
        vert = max(vert, 0.08)
        fb = max(fb, 0.035)
        lat = min(lat, 0.035)
    elif subtype == "forward_back_rock":
        fb = max(fb, 0.10)
        lat = min(lat, 0.025)
    return {
        "forward_back": _clamp(fb * scale, 0.015, 0.28),
        "lateral": _clamp(lat * scale, 0.0, 0.22),
        "vertical": _clamp(vert * scale, 0.0, 0.22),
    }


def _cycles_from_group(group: dict[str, Any], primitives: list[dict[str, Any]], plan_data: dict[str, Any], duration: float) -> float:
    group_ids = set(str(pid) for pid in group.get("primitives", []) or [])
    cycle_values = []
    for row in primitives:
        if str(row.get("primitive_id")) in group_ids:
            rhythm = row.get("rhythm_profile", {}) or {}
            try:
                value = float(rhythm.get("cycle_count_estimate") or 0.0)
            except Exception:
                value = 0.0
            if value > 0:
                source_duration = float(row.get("duration_seconds") or duration or 1.0)
                cycle_values.append(value / max(source_duration, 1e-6))
    cycles_per_second = sum(cycle_values) / len(cycle_values) if cycle_values else 0.45
    tempo = str(_phase_query(plan_data).get("tempo") or "medium")
    cycles_per_second *= {"slow": 0.72, "medium": 1.0, "fast": 1.3}.get(tempo, 1.0)
    return _clamp(cycles_per_second * float(duration), 1.0, 5.0)


def _irregularity_for_plan(plan_data: dict[str, Any]) -> float:
    intensity = str(_phase_query(plan_data).get("intensity") or "medium")
    return {"low": 0.01, "medium": 0.018, "high": 0.026}.get(intensity, 0.018)


def _select_driver_controller(group: dict[str, Any], primitives: list[dict[str, Any]]) -> str:
    ids = set(str(pid) for pid in group.get("primitives", []) or [])
    counts = {"hipControl": 0, "pelvisControl": 0, "abdomenControl": 0}
    for row in primitives:
        if str(row.get("primitive_id")) not in ids:
            continue
        roles = row.get("controller_role_map", {}) or {}
        for name in roles.get("driver_controllers", []) or []:
            if name in counts:
                counts[name] += 1
    best = max(counts, key=lambda key: counts[key])
    return best if counts[best] > 0 else "hipControl"


def _follower_tracks(plan_data: dict[str, Any], times: np.ndarray, driver_path: np.ndarray, group_id: str) -> list[GeneratedControllerTrack]:
    tracks: list[GeneratedControllerTrack] = []
    phase = (plan_data.get("sequence") or [{}])[0]
    body_params = phase.get("body_parameters", {}) or {}
    if body_params.get("torso_lean") == "forward":
        chest = np.zeros_like(driver_path, dtype=np.float32)
        chest[:, 2] = float(np.max(np.abs(driver_path[:, 2])) if driver_path.size else 0.05) * 0.25
        chest[:, 1] = float(np.max(np.abs(driver_path[:, 1])) if driver_path.size else 0.02) * 0.12
        tracks.append(GeneratedControllerTrack(
            controller_name="chestControl",
            bodypart="chest",
            role="follower",
            coordinate_space="relative_body_motion",
            times=_round_list(times),
            position_deltas=_round_path(chest),
            rotation_deltas=None,
            generation_method="parametric_follower_lean_v0",
            source_primitive_group=group_id,
            safety_flags=_track_safety_flags(),
            warnings=["Forward lean is a small relative follower offset; retargeting must solve final posture."],
        ))
    return tracks


def _anchor_tracks(times: np.ndarray, group_id: str) -> list[GeneratedControllerTrack]:
    zeros = np.zeros((len(times), 3), dtype=np.float32)
    tracks = []
    for name in ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl"]:
        tracks.append(GeneratedControllerTrack(
            controller_name=name,
            bodypart=controller_bodypart(name),
            role="anchor",
            coordinate_space="relative_body_motion",
            times=_round_list(times),
            position_deltas=_round_path(zeros),
            rotation_deltas=None,
            generation_method="stable_zero_delta_anchor_v0",
            source_primitive_group=group_id,
            safety_flags=_track_safety_flags(),
            warnings=["Static relative anchor placeholder; no source-scene pose is copied."],
        ))
    return tracks


def _track_safety_flags() -> dict[str, Any]:
    return {
        "no_world_coordinates": True,
        "relative_deltas_only": True,
        "source_timeline_keyframes_copied": False,
        "person_root_track": False,
    }


def _write_npz(out_npz: str | Path, flow: GeneratedMotionFlow) -> None:
    target = Path(out_npz)
    target.parent.mkdir(parents=True, exist_ok=True)
    tracks = flow.controller_tracks
    times = np.asarray(tracks[0].times if tracks else [], dtype=np.float32)
    positions = np.stack([np.asarray(track.position_deltas, dtype=np.float32) for track in tracks], axis=1) if tracks else np.zeros((0, 0, 3), dtype=np.float32)
    np.savez_compressed(
        target,
        times=times,
        position_deltas=positions,
        controller_names=np.asarray([track.controller_name for track in tracks]),
        roles=np.asarray([track.role for track in tracks]),
        coordinate_space=np.asarray(["relative_body_motion"]),
        export_ready=np.asarray([False]),
        clip_stitching_used=np.asarray([False]),
    )


def _write_report(flow: dict[str, Any], report: str | Path, stats: dict[str, Any]) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    tracks = flow.get("controller_tracks", []) or []
    amp = flow.get("amplitude_profile", {}) or {}
    rhythm = flow.get("rhythm_profile", {}) or {}
    lines = [
        "# Generated Motion Flow V0 Report",
        "",
        "This flow contains synthesized relative controller deltas only. It is not a Timeline export and not clip stitching.",
        "",
        f"- Flow: `{flow.get('flow_id')}`",
        f"- Selected primitive group: `{flow.get('selected_primitive_group')}`",
        f"- Trajectory shape: `{flow.get('trajectory_shape')}`",
        f"- Duration seconds: `{flow.get('duration_seconds')}`",
        f"- FPS: `{flow.get('fps')}`",
        f"- Controller tracks: `{[track.get('controller_name') for track in tracks]}`",
        f"- Amplitude forward/back: `{amp.get('forward_back')}`",
        f"- Amplitude lateral: `{amp.get('lateral')}`",
        f"- Amplitude vertical: `{amp.get('vertical')}`",
        f"- Cycles: `{rhythm.get('cycles')}`",
        f"- Tempo: `{rhythm.get('tempo')}`",
        f"- Driver path stats: `{stats}`",
        "",
        "## Coordinate Safety",
        "",
        f"- Coordinate space: `{flow.get('coordinate_space')}`",
        f"- No world coordinates: `{flow.get('no_world_coordinates')}`",
        f"- No Person/root tracks: `{flow.get('no_person_root_tracks')}`",
        f"- Clip stitching used: `{flow.get('clip_stitching_used')}`",
        f"- Export ready: `{flow.get('export_ready')}`",
        "",
        "## Limitations",
        "",
        "- Retargeting to a VaM character pose is not implemented.",
        "- Contact and anchor constraint solving are placeholders.",
        "- Rotation synthesis is not enabled in v0.",
        "- Timeline export remains a later safety-gated stage.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round_list(values: np.ndarray) -> list[float]:
    return [round(float(v), 6) for v in values.tolist()]


def _round_path(path: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 6), round(float(y), 6), round(float(z), 6)] for x, y, z in path.tolist()]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))

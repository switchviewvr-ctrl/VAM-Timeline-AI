"""Minimal controlled motion synthesis on top of manual pose baselines."""

from __future__ import annotations

from typing import Any
import math

import numpy as np


STATIC_EPSILON = 1e-9


def synthesize_manual_gt_motion(plan: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    duration = float(plan.get("duration_seconds") or 4.0)
    keyframe_rate = float(plan.get("keyframe_rate") or plan.get("fps") or 60)
    fps = float(plan.get("fps") or keyframe_rate)
    count = int(round(duration * keyframe_rate)) + 1
    times = np.linspace(0.0, duration, count, dtype=np.float32)
    curve = str(plan.get("motion_curve_type") or "")
    base_map = baseline.get("controller_baseline") or {}
    driver = set(plan.get("driver_controllers") or [])
    followers = set(plan.get("follower_controllers") or [])
    static = set(plan.get("static_anchor_controllers") or []) | set(plan.get("explicitly_static_controllers") or [])
    controllers = sorted(set(base_map) | driver | followers | static)
    tracks: list[dict[str, Any]] = []
    for name in controllers:
        if name not in base_map:
            continue
        base = np.asarray(base_map[name]["position"], dtype=np.float32)
        rotation = np.asarray(base_map[name].get("rotation_quat") or [0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        role = _role(name, driver, followers, static)
        positions = np.repeat(base.reshape(1, 3), len(times), axis=0)
        rotations = np.repeat(rotation.reshape(1, 4), len(times), axis=0)
        positions = _apply_motion(name, role, curve, positions, times, plan)
        tracks.append(
            {
                "controller_name": name,
                "role": role,
                "times": [round(float(v), 6) for v in times.tolist()],
                "positions": [[round(float(x), 6) for x in row] for row in positions.tolist()],
                "rotations": [[round(float(x), 6) for x in row] for row in rotations.tolist()],
                "baseline_position": [round(float(x), 6) for x in base.tolist()],
                "baseline_rotation_quat": [round(float(x), 6) for x in rotation.tolist()],
                "rotation_source": base_map[name].get("rotation_source"),
                "motion_range": round(_motion_range(positions), 6),
                "rotation_range": round(_motion_range(rotations), 6),
            }
        )
    return {
        "schema_version": "manual_gt_synthesized_clip_v1",
        "clip_id": plan.get("clip_id"),
        "capture_id": plan.get("capture_id"),
        "family": plan.get("family"),
        "subtype": plan.get("subtype"),
        "motion_example_name": plan.get("motion_example_name"),
        "duration_seconds": duration,
        "fps": fps,
        "keyframe_rate": keyframe_rate,
        "coordinate_space": baseline.get("coordinate_space"),
        "review_only": True,
        "driver_controllers": plan.get("driver_controllers") or [],
        "follower_controllers": plan.get("follower_controllers") or [],
        "static_anchor_controllers": plan.get("static_anchor_controllers") or [],
        "explicitly_static_controllers": plan.get("explicitly_static_controllers") or [],
        "forbidden_motion_axes": plan.get("forbidden_motion_axes") or [],
        "controller_tracks": tracks,
        "amplitude_profile_key": plan.get("amplitude_profile_key"),
        "amplitude_profile": plan.get("amplitude_profile") or {},
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
        "person_root_world_tracks_included": False,
        "include_rotations": True,
    }


def _role(name: str, driver: set[str], followers: set[str], static: set[str]) -> str:
    if name in driver:
        return "driver"
    if name in followers:
        return "follower"
    if name in static:
        return "static_anchor"
    return "static"


def _apply_motion(name: str, role: str, curve: str, positions: np.ndarray, times: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    if role == "static_anchor" or role == "static":
        return positions
    scale = float(plan.get("amplitude_scale") or 1.0)
    profile = plan.get("amplitude_profile") or {}
    phase = _phase(times, cycles=1.0)
    delayed = _phase(times, cycles=1.0, delay=0.18)
    if curve in {"oval_grind", "slow_grind", "small_grind"}:
        if role == "driver" and name in {"hipControl", "pelvisControl"}:
            amp_x = 0.035 * scale * _profile_float(profile, "hip_lateral_scale", 1.0)
            amp_z = 0.045 * scale * _profile_float(profile, "hip_forward_back_scale", 1.0)
            amp_y = 0.010 * scale * _profile_float(profile, "hip_vertical_scale", 1.0)
            positions[:, 0] += amp_x * np.sin(phase)
            positions[:, 2] += amp_z * np.sin(phase + math.pi / 2.0)
            positions[:, 1] += amp_y * np.sin(phase * 2.0)
        elif role == "follower":
            amount = _follower_amount(name) * scale * _profile_follow_scale(name, profile)
            positions[:, 0] += 0.012 * amount * np.sin(delayed)
            positions[:, 2] += 0.014 * amount * np.sin(delayed + math.pi / 2.0)
            positions[:, 1] += 0.004 * amount * np.sin(delayed * 2.0)
    elif curve == "vertical_bounce":
        if role == "driver" and name in {"hipControl", "pelvisControl"}:
            # Smooth non-negative lift around captured baseline.
            lift = (1.0 - np.cos(phase)) * 0.5
            positions[:, 1] += 0.060 * scale * _profile_float(profile, "hip_vertical_scale", 1.0) * lift
            positions[:, 2] += 0.014 * scale * _profile_float(profile, "hip_forward_back_scale", 1.0) * np.sin(phase)
            positions[:, 0] += 0.006 * scale * _profile_float(profile, "hip_lateral_scale", 0.0) * np.sin(phase + math.pi / 2.0)
        elif role == "follower":
            amount = _follower_amount(name) * scale * _profile_follow_scale(name, profile)
            positions[:, 1] += 0.018 * amount * (1.0 - np.cos(delayed)) * 0.5
            positions[:, 2] += 0.005 * amount * np.sin(delayed)
    elif curve == "head_chest_forward_back":
        if role == "driver" and name == "headControl":
            positions[:, 2] += 0.070 * scale * _profile_float(profile, "head_forward_back_scale", 1.0) * np.sin(phase)
            positions[:, 1] += 0.018 * scale * _profile_float(profile, "head_vertical_scale", 1.0) * np.sin(phase + math.pi / 2.0)
        elif role == "follower" and name == "chestControl":
            follow = _profile_float(profile, "chest_follow_scale", 1.0)
            positions[:, 2] += 0.030 * scale * follow * np.sin(delayed)
            positions[:, 1] += 0.006 * scale * follow * np.sin(delayed + math.pi / 2.0)
    elif curve == "single_hand_forward_back":
        if role == "driver" and name.endswith("HandControl"):
            positions[:, 2] += 0.070 * scale * _profile_float(profile, "active_hand_forward_back_scale", 1.0) * np.sin(phase)
            positions[:, 1] += 0.006 * scale * _profile_float(profile, "active_hand_vertical_scale", 1.0) * np.sin(phase + math.pi / 2.0)
    elif curve in {"receiver_response_z", "standing_receiver_response_z"}:
        if role == "driver" and name in {"hipControl", "pelvisControl"}:
            response = _profile_float(profile, "hip_response_scale", 1.0) if name == "hipControl" else _profile_float(profile, "pelvis_response_scale", 1.0)
            positions[:, 2] += 0.038 * scale * response * np.sin(phase)
            positions[:, 1] += 0.008 * scale * response * np.sin(phase + math.pi / 2.0)
        elif role == "follower" and name in {"pelvisControl", "chestControl"}:
            amount = _profile_float(profile, "pelvis_response_scale", 0.65) if name == "pelvisControl" else _profile_float(profile, "chest_response_scale", 1.0)
            positions[:, 2] += 0.018 * amount * scale * np.sin(delayed)
            positions[:, 1] += 0.004 * amount * scale * np.sin(delayed + math.pi / 2.0)
    elif curve == "pelvis_counter_lift":
        if role == "driver" and name in {"hipControl", "pelvisControl"}:
            counter = _profile_float(profile, "pelvis_counter_scale", 1.0)
            positions[:, 1] += 0.040 * scale * counter * (1.0 - np.cos(phase)) * 0.5
            positions[:, 2] += 0.010 * scale * counter * np.sin(phase)
        elif role == "follower" and name in {"lKneeControl", "rKneeControl", "lFootControl", "rFootControl"}:
            side = -1.0 if name.startswith("l") else 1.0
            leg = _profile_float(profile, "leg_reactive_scale", 1.0)
            positions[:, 1] += 0.014 * scale * leg * np.sin(delayed + side * 0.25)
            positions[:, 2] += 0.008 * scale * leg * np.sin(delayed)
        elif role == "follower" and name == "pelvisControl":
            counter = _profile_float(profile, "pelvis_counter_scale", 1.0)
            positions[:, 1] += 0.018 * scale * counter * (1.0 - np.cos(delayed)) * 0.5
            positions[:, 2] += 0.004 * scale * counter * np.sin(delayed)
    return positions


def _phase(times: np.ndarray, *, cycles: float, delay: float = 0.0) -> np.ndarray:
    duration = max(float(times[-1]), 1e-6)
    return ((times / duration) * cycles * 2.0 * math.pi) - delay


def _follower_amount(name: str) -> float:
    if name == "pelvisControl":
        return 0.32
    if name == "abdomenControl":
        return 0.55
    if name == "chestControl":
        return 0.35
    if name == "headControl":
        return 0.18
    if name in {"lThighControl", "rThighControl"}:
        return 0.22
    return 0.15


def _profile_float(profile: dict[str, Any], key: str, default: float) -> float:
    value = profile.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _profile_follow_scale(name: str, profile: dict[str, Any]) -> float:
    mapping = {
        "pelvisControl": "pelvis_follow_scale",
        "abdomenControl": "abdomen_follow_scale",
        "chestControl": "chest_follow_scale",
        "headControl": "head_follow_scale",
        "lThighControl": "thigh_follow_scale",
        "rThighControl": "thigh_follow_scale",
    }
    key = mapping.get(name)
    return _profile_float(profile, key, 1.0) if key else 1.0


def _motion_range(positions: np.ndarray) -> float:
    if len(positions) == 0:
        return 0.0
    span = np.max(positions, axis=0) - np.min(positions, axis=0)
    return float(np.max(np.abs(span)))

"""Richer Cowgirl/Riding feature extraction v1.

These features are numeric measurements and weak proxies. They are not final
semantic labels and do not infer actor roles from names.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.identity import make_feature_record_id
from vam_timeline_ai.io.json_utils import load_json, write_jsonl


PELVIS_FEATURES = [
    "pelvis_vertical_amplitude",
    "pelvis_forward_back_amplitude",
    "pelvis_lateral_amplitude",
    "pelvis_total_position_range",
    "pelvis_movement_energy",
    "pelvis_mean_speed",
    "pelvis_max_speed",
    "pelvis_speed_std",
    "pelvis_acceleration_peak_count",
    "pelvis_pause_ratio",
    "pelvis_tempo_proxy",
    "pelvis_rhythm_regularity_proxy",
    "pelvis_bounce_score_proxy",
    "pelvis_rock_score_proxy",
    "pelvis_lateral_sway_score_proxy",
    "pelvis_circularity_score_proxy",
    "pelvis_grind_score_proxy",
]
TORSO_FEATURES = [
    "torso_lean_forward_proxy",
    "torso_lean_back_proxy",
    "torso_upright_stability_proxy",
    "torso_motion_energy",
    "torso_vs_pelvis_motion_ratio",
    "torso_countermotion_proxy",
    "chest_pelvis_distance_mean",
    "chest_pelvis_distance_std",
    "head_chest_distance_mean",
    "posture_change_count_proxy",
]
HAND_FEATURES = [
    "left_hand_motion_energy",
    "right_hand_motion_energy",
    "hand_symmetry_motion_score",
    "hand_asymmetry_motion_score",
    "left_hand_to_chest_distance_mean",
    "right_hand_to_chest_distance_mean",
    "left_hand_to_pelvis_distance_mean",
    "right_hand_to_pelvis_distance_mean",
    "left_hand_to_head_distance_mean",
    "right_hand_to_head_distance_mean",
    "hands_near_own_body_proxy",
    "hands_static_support_proxy",
    "hand_support_transition_count_proxy",
]
LEG_FEATURES = [
    "knee_motion_energy_left",
    "knee_motion_energy_right",
    "foot_motion_energy_left",
    "foot_motion_energy_right",
    "stance_width_proxy",
    "left_right_weight_shift_proxy",
    "foot_stability_proxy",
    "kneeling_or_squat_proxy_uncertain",
]
HEAD_FEATURES = [
    "head_motion_energy",
    "head_vertical_range",
    "head_turn_proxy_from_rotation_if_available",
    "head_down_proxy_uncertain",
    "head_up_proxy_uncertain",
    "head_relative_to_chest_motion",
    "head_stability_proxy",
]
RHYTHM_FEATURES = [
    "slow_motion_score_proxy",
    "fast_motion_score_proxy",
    "steady_rhythm_score_proxy",
    "irregular_rhythm_score_proxy",
    "pause_hold_score_proxy",
    "adjustment_transition_score_proxy",
    "intensity_score_proxy",
    "tempo_increase_proxy",
    "tempo_decrease_proxy",
    "depth_increase_proxy",
    "depth_decrease_proxy",
]
FEATURE_NAMES = PELVIS_FEATURES + TORSO_FEATURES + HAND_FEATURES + LEG_FEATURES + HEAD_FEATURES + RHYTHM_FEATURES


def extract_cowgirl_features_v1(
    windows_path: str | Path,
    sample_index_path: str | Path,
    controller_map_path: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    windows = _load_jsonl(windows_path)
    samples = {row["sample_id"]: row for row in _load_jsonl(sample_index_path) if row.get("sample_id")}
    mappings = load_json(controller_map_path).get("controller_mappings", {})
    cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    X: list[list[float]] = []
    quality_matrix: list[list[int]] = []
    for window in windows:
        row = _extract_one(window, samples.get(window.get("sample_id")), mappings, cache)
        rows.append(row)
        X.append([row["feature_values"].get(name, np.nan) for name in FEATURE_NAMES])
        quality_matrix.append([
            int(row["feature_quality"].get("has_pelvis_features", False)),
            int(row["feature_quality"].get("has_torso_features", False)),
            int(row["feature_quality"].get("has_hand_features", False)),
            int(row["feature_quality"].get("has_leg_features", False)),
            int(row["feature_quality"].get("has_head_features", False)),
        ])
    write_jsonl(out_jsonl, rows)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=np.asarray(X, dtype=np.float32) if X else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        window_ids=np.asarray([r["window_id"] for r in rows], dtype=object),
        sample_ids=np.asarray([r.get("sample_id", "") for r in rows], dtype=object),
        source_ids=np.asarray([r.get("source_id", "") for r in rows], dtype=object),
        source_scene_files=np.asarray([r.get("source_scene_file", "") for r in rows], dtype=object),
        technical_atom_ids=np.asarray([r.get("technical_atom_id", "") for r in rows], dtype=object),
        feature_quality_matrix=np.asarray(quality_matrix, dtype=np.int8),
        metadata_json=json.dumps({"feature_version": "cowgirl_features_v1", "feature_count": len(FEATURE_NAMES)}, ensure_ascii=False),
    )
    _write_report(rows, report)
    return rows


def _extract_one(window: dict[str, Any], sample: dict[str, Any] | None, mappings: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    values = {name: np.nan for name in FEATURE_NAMES}
    warnings = list(window.get("warnings", []))
    controllers_used: dict[str, list[str]] = {}
    missing_groups: list[str] = []
    quality = {
        "has_any_numeric_features": False,
        "has_pelvis_features": False,
        "has_torso_features": False,
        "has_hand_features": False,
        "has_leg_features": False,
        "has_head_features": False,
        "root_mapping_confidence": "none",
    }
    if sample is None or sample.get("bake_status") != "ok":
        warnings.append("no baked sample available")
    else:
        try:
            arrays = _load_arrays(sample, cache)
            positions = arrays["positions"]
            rotations = arrays["rotations"]
            times = arrays["times"]
            names = arrays["controller_names"]
            parts = _bodypart_indices(names, mappings)
            start = max(0, min(int(window.get("frame_start") or 0), positions.shape[0] - 1))
            end = max(start + 1, min(int(window.get("frame_end") or positions.shape[0]), positions.shape[0]))
            pos = positions[start:end]
            rot = rotations[start:end]
            t = times[start:end]
            root_idx, root_part = _first_part(parts, ["pelvis", "hip", "root", "abdomen"])
            if root_idx is None:
                missing_groups.append("pelvis")
                warnings.append("pelvis/hip/root/abdomen controller missing")
            else:
                root = pos[:, root_idx, :]
                root_stats = _motion_stats(root, t)
                values.update(_pelvis_features(root, t, root_stats))
                controllers_used["pelvis"] = [names[root_idx]]
                quality["has_pelvis_features"] = True
                quality["root_mapping_confidence"] = mappings.get(names[root_idx], {}).get("mapping_confidence", "unknown")
            _torso_features(values, quality, missing_groups, controllers_used, pos, t, names, parts, root_idx)
            _hand_features(values, quality, missing_groups, controllers_used, pos, t, names, parts, root_idx)
            _leg_features(values, quality, missing_groups, controllers_used, pos, t, names, parts, root_idx)
            _head_features(values, quality, missing_groups, controllers_used, pos, rot, t, names, parts)
            _rhythm_features(values)
            quality["has_any_numeric_features"] = bool(np.isfinite(list(values.values())).any())
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"feature v1 extraction failed: {exc}")
    return {
        "feature_record_id": make_feature_record_id(str(window.get("window_id")), "cowgirl_features_v1"),
        "window_id": window.get("window_id"),
        "sample_id": window.get("sample_id"),
        "source_id": window.get("source_id"),
        "source_scene_file": window.get("source_scene_file"),
        "technical_atom_id": window.get("technical_atom_id"),
        "feature_version": "cowgirl_features_v1",
        "feature_values": values,
        "feature_quality": quality,
        "controllers_used": controllers_used,
        "missing_controller_groups": sorted(set(missing_groups)),
        "warnings": warnings,
    }


def _load_arrays(sample: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    key = sample["sample_id"]
    if key in cache:
        return cache[key]
    data = np.load(sample["baked_npz_path"], allow_pickle=True)
    arrays = {
        "positions": data["positions"],
        "rotations": data["rotations"],
        "times": data["times"],
        "controller_names": [str(v) for v in data["controller_names"].tolist()],
    }
    data.close()
    if len(cache) > 8:
        cache.clear()
    cache[key] = arrays
    return arrays


def _bodypart_indices(names: list[str], mappings: dict[str, Any]) -> dict[str, list[int]]:
    parts: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        part = mappings.get(name, {}).get("body_part", "unknown")
        parts.setdefault(part, []).append(idx)
    return parts


def _first_part(parts: dict[str, list[int]], preferred: list[str]) -> tuple[int | None, str | None]:
    for part in preferred:
        if parts.get(part):
            return parts[part][0], part
    return None, None


def _motion_stats(pos: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    if len(pos) < 2:
        return {"speed": np.zeros((0,)), "energy": np.nan, "mean_speed": np.nan, "max_speed": np.nan, "speed_std": np.nan}
    dt = np.diff(t.astype(np.float64))
    dt = np.where(dt <= 0, 1.0 / 60.0, dt)
    vel = np.diff(pos.astype(np.float64), axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    return {
        "speed": speed,
        "energy": float(np.mean(speed**2)) if len(speed) else np.nan,
        "mean_speed": float(np.mean(speed)) if len(speed) else np.nan,
        "max_speed": float(np.max(speed)) if len(speed) else np.nan,
        "speed_std": float(np.std(speed)) if len(speed) else np.nan,
    }


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _pelvis_features(root: np.ndarray, t: np.ndarray, stats: dict[str, Any]) -> dict[str, float]:
    span = np.ptp(root, axis=0) if len(root) else np.asarray([np.nan, np.nan, np.nan])
    total_range = float(np.linalg.norm(span))
    speed = stats["speed"]
    accel = np.diff(speed) if len(speed) > 1 else np.zeros((0,))
    mean_speed = stats["mean_speed"]
    pause_ratio = float(np.mean(speed < max(0.03, mean_speed * 0.15))) if len(speed) and np.isfinite(mean_speed) else np.nan
    tempo = _tempo_proxy(root[:, 1], t)
    regularity = float(1.0 / (1.0 + stats["speed_std"] / (mean_speed + 1e-6))) if np.isfinite(mean_speed) else np.nan
    axis_sum = float(np.sum(span) + 1e-6)
    circularity = float(min(span[0], span[2]) / (max(span[0], span[2]) + 1e-6)) if np.isfinite(span[0]) else np.nan
    return {
        "pelvis_vertical_amplitude": float(span[1]),
        "pelvis_forward_back_amplitude": float(span[2]),
        "pelvis_lateral_amplitude": float(span[0]),
        "pelvis_total_position_range": total_range,
        "pelvis_movement_energy": stats["energy"],
        "pelvis_mean_speed": stats["mean_speed"],
        "pelvis_max_speed": stats["max_speed"],
        "pelvis_speed_std": stats["speed_std"],
        "pelvis_acceleration_peak_count": float(np.sum(accel > (np.mean(accel) + np.std(accel)))) if len(accel) else 0.0,
        "pelvis_pause_ratio": pause_ratio,
        "pelvis_tempo_proxy": tempo,
        "pelvis_rhythm_regularity_proxy": regularity,
        "pelvis_bounce_score_proxy": float(span[1] / axis_sum),
        "pelvis_rock_score_proxy": float(span[2] / axis_sum),
        "pelvis_lateral_sway_score_proxy": float(span[0] / axis_sum),
        "pelvis_circularity_score_proxy": circularity,
        "pelvis_grind_score_proxy": float(circularity * (1.0 - min(1.0, span[1] / (span[0] + span[2] + 1e-6)))) if np.isfinite(circularity) else np.nan,
    }


def _torso_features(values, quality, missing, used, pos, t, names, parts, root_idx):
    chest_idx, _ = _first_part(parts, ["chest", "abdomen"])
    head_idx, _ = _first_part(parts, ["head"])
    if chest_idx is None or root_idx is None:
        missing.append("torso")
        return
    chest = pos[:, chest_idx, :]
    root = pos[:, root_idx, :]
    rel = chest - root
    chest_stats = _motion_stats(chest, t)
    root_stats = _motion_stats(root, t)
    values["torso_lean_forward_proxy"] = float(np.mean(rel[:, 2] > np.median(rel[:, 2]))) if len(rel) else np.nan
    values["torso_lean_back_proxy"] = float(np.mean(rel[:, 2] < np.median(rel[:, 2]))) if len(rel) else np.nan
    values["torso_upright_stability_proxy"] = float(1.0 / (1.0 + np.std(rel[:, 0]) + np.std(rel[:, 2]))) if len(rel) else np.nan
    values["torso_motion_energy"] = chest_stats["energy"]
    values["torso_vs_pelvis_motion_ratio"] = float(chest_stats["energy"] / (root_stats["energy"] + 1e-6)) if np.isfinite(chest_stats["energy"]) else np.nan
    if len(chest) > 2:
        v1 = np.diff(chest, axis=0).reshape(len(chest) - 1, 3)
        v2 = np.diff(root, axis=0).reshape(len(root) - 1, 3)
        dot = np.sum(v1 * v2, axis=1)
        values["torso_countermotion_proxy"] = float(np.mean(dot < 0))
    dist = np.linalg.norm(rel, axis=1)
    values["chest_pelvis_distance_mean"] = float(np.mean(dist))
    values["chest_pelvis_distance_std"] = float(np.std(dist))
    if head_idx is not None:
        values["head_chest_distance_mean"] = float(np.mean(np.linalg.norm(pos[:, head_idx, :] - chest, axis=1)))
    values["posture_change_count_proxy"] = float(np.sum(np.linalg.norm(np.diff(rel, axis=0), axis=1) > (np.mean(np.linalg.norm(np.diff(rel, axis=0), axis=1)) + np.std(np.linalg.norm(np.diff(rel, axis=0), axis=1))))) if len(rel) > 2 else 0.0
    used["torso"] = [names[chest_idx]] + ([names[head_idx]] if head_idx is not None else [])
    quality["has_torso_features"] = True


def _hand_features(values, quality, missing, used, pos, t, names, parts, root_idx):
    lh = parts.get("left_hand", [None])[0]
    rh = parts.get("right_hand", [None])[0]
    chest = parts.get("chest", [None])[0]
    head = parts.get("head", [None])[0]
    if lh is None and rh is None:
        missing.append("hands")
        return
    energies = []
    for key, idx in [("left", lh), ("right", rh)]:
        if idx is not None:
            energy = _motion_stats(pos[:, idx, :], t)["energy"]
            values[f"{key}_hand_motion_energy"] = energy
            energies.append(energy)
    if len(energies) == 2:
        values["hand_symmetry_motion_score"] = float(1.0 - abs(energies[0] - energies[1]) / (max(energies) + 1e-6))
        values["hand_asymmetry_motion_score"] = float(1.0 - values["hand_symmetry_motion_score"])
    distances = []
    for side, idx in [("left", lh), ("right", rh)]:
        if idx is None:
            continue
        for target_name, target_idx in [("chest", chest), ("pelvis", root_idx), ("head", head)]:
            if target_idx is not None:
                d = np.linalg.norm(pos[:, idx, :] - pos[:, target_idx, :], axis=1)
                values[f"{side}_hand_to_{target_name}_distance_mean"] = float(np.mean(d))
                distances.append(float(np.mean(d)))
    distance_mean = _finite_mean(distances)
    energy_mean = _finite_mean(energies)
    values["hands_near_own_body_proxy"] = float(1.0 / (1.0 + distance_mean)) if np.isfinite(distance_mean) else np.nan
    values["hands_static_support_proxy"] = float(1.0 / (1.0 + energy_mean)) if np.isfinite(energy_mean) else np.nan
    values["hand_support_transition_count_proxy"] = float(np.sum(np.abs(np.diff(distances)) > 0.1)) if len(distances) > 1 else 0.0
    used["hands"] = [names[i] for i in [lh, rh] if i is not None]
    quality["has_hand_features"] = True


def _leg_features(values, quality, missing, used, pos, t, names, parts, root_idx):
    lk, rk = parts.get("left_knee", [None])[0], parts.get("right_knee", [None])[0]
    lf, rf = parts.get("left_foot", [None])[0], parts.get("right_foot", [None])[0]
    if all(i is None for i in [lk, rk, lf, rf]):
        missing.append("legs")
        return
    for key, idx in [("left", lk), ("right", rk)]:
        if idx is not None:
            values[f"knee_motion_energy_{key}"] = _motion_stats(pos[:, idx, :], t)["energy"]
    for key, idx in [("left", lf), ("right", rf)]:
        if idx is not None:
            values[f"foot_motion_energy_{key}"] = _motion_stats(pos[:, idx, :], t)["energy"]
    if lk is not None and rk is not None:
        values["stance_width_proxy"] = float(np.mean(np.linalg.norm(pos[:, lk, :] - pos[:, rk, :], axis=1)))
        values["left_right_weight_shift_proxy"] = float(np.std((pos[:, lk, 1] - pos[:, rk, 1])))
    if lf is not None and rf is not None:
        foot_energy = _finite_mean([values.get("foot_motion_energy_left", np.nan), values.get("foot_motion_energy_right", np.nan)])
        values["foot_stability_proxy"] = float(1.0 / (1.0 + foot_energy)) if np.isfinite(foot_energy) else np.nan
    if root_idx is not None and lk is not None and rk is not None:
        knee_height = np.mean([np.mean(pos[:, lk, 1]), np.mean(pos[:, rk, 1])])
        root_height = np.mean(pos[:, root_idx, 1])
        values["kneeling_or_squat_proxy_uncertain"] = float(1.0 / (1.0 + abs(root_height - knee_height)))
    used["legs"] = [names[i] for i in [lk, rk, lf, rf] if i is not None]
    quality["has_leg_features"] = True


def _head_features(values, quality, missing, used, pos, rot, t, names, parts):
    head = parts.get("head", [None])[0]
    chest = parts.get("chest", [None])[0]
    if head is None:
        missing.append("head_gaze")
        return
    hpos = pos[:, head, :]
    stats = _motion_stats(hpos, t)
    values["head_motion_energy"] = stats["energy"]
    values["head_vertical_range"] = float(np.ptp(hpos[:, 1]))
    if rot.size and rot.shape[1] > head:
        values["head_turn_proxy_from_rotation_if_available"] = float(np.linalg.norm(np.ptp(rot[:, head, :3], axis=0)))
    if chest is not None:
        rel = hpos - pos[:, chest, :]
        values["head_down_proxy_uncertain"] = float(np.mean(rel[:, 1] < np.median(rel[:, 1])))
        values["head_up_proxy_uncertain"] = float(np.mean(rel[:, 1] > np.median(rel[:, 1])))
        values["head_relative_to_chest_motion"] = float(np.mean(np.linalg.norm(np.diff(rel, axis=0), axis=1))) if len(rel) > 1 else 0.0
    values["head_stability_proxy"] = float(1.0 / (1.0 + (stats["mean_speed"] if np.isfinite(stats["mean_speed"]) else 0.0)))
    used["head_gaze"] = [names[head]]
    quality["has_head_features"] = True


def _rhythm_features(values):
    mean_speed = values.get("pelvis_mean_speed", np.nan)
    max_speed = values.get("pelvis_max_speed", np.nan)
    pause = values.get("pelvis_pause_ratio", np.nan)
    regularity = values.get("pelvis_rhythm_regularity_proxy", np.nan)
    tempo = values.get("pelvis_tempo_proxy", np.nan)
    vertical = values.get("pelvis_vertical_amplitude", np.nan)
    forward = values.get("pelvis_forward_back_amplitude", np.nan)
    values["slow_motion_score_proxy"] = float(1.0 / (1.0 + mean_speed)) if np.isfinite(mean_speed) else np.nan
    values["fast_motion_score_proxy"] = float(min(1.0, mean_speed / 2.0)) if np.isfinite(mean_speed) else np.nan
    values["steady_rhythm_score_proxy"] = regularity
    values["irregular_rhythm_score_proxy"] = float(1.0 - regularity) if np.isfinite(regularity) else np.nan
    values["pause_hold_score_proxy"] = pause
    values["adjustment_transition_score_proxy"] = float((1.0 - regularity) * (1.0 - pause)) if np.isfinite(regularity) and np.isfinite(pause) else np.nan
    values["intensity_score_proxy"] = float(min(1.0, max_speed / 3.0)) if np.isfinite(max_speed) else np.nan
    values["tempo_increase_proxy"] = float(min(1.0, tempo / 180.0)) if np.isfinite(tempo) else np.nan
    values["tempo_decrease_proxy"] = np.nan
    values["depth_increase_proxy"] = float(min(1.0, max(vertical, forward) / 0.5)) if np.isfinite(vertical) and np.isfinite(forward) else np.nan
    values["depth_decrease_proxy"] = np.nan


def _tempo_proxy(y: np.ndarray, times: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    centered = y - np.mean(y)
    signs = np.sign(centered)
    crossings = np.sum((signs[1:] * signs[:-1]) < 0)
    duration = float(times[-1] - times[0]) if len(times) else 0.0
    if duration <= 0:
        return 0.0
    return float((crossings / 2.0 / duration) * 60.0)


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    confidence = Counter()
    missing = Counter()
    for row in rows:
        q = row["feature_quality"]
        for key in ["has_any_numeric_features", "has_pelvis_features", "has_torso_features", "has_hand_features", "has_leg_features", "has_head_features"]:
            if q.get(key):
                counts[key] += 1
        confidence[q.get("root_mapping_confidence", "none")] += 1
        for group in row.get("missing_controller_groups", []):
            missing[group] += 1
    lines = [
        "# Cowgirl Feature Report v1",
        "",
        f"- Total windows: {len(rows)}",
        f"- Rows with any numeric features: {counts['has_any_numeric_features']}",
        f"- Rows with pelvis/hip features: {counts['has_pelvis_features']}",
        f"- Rows with torso features: {counts['has_torso_features']}",
        f"- Rows with hand features: {counts['has_hand_features']}",
        f"- Rows with leg features: {counts['has_leg_features']}",
        f"- Rows with head features: {counts['has_head_features']}",
        "",
        "## Mapping Confidence",
        "",
    ]
    for key, value in confidence.most_common():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Missing Feature Groups", ""])
    for key, value in missing.most_common():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Notes", "", "Lean, gaze, squat, and support features are numeric proxies, not semantic truth."])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

"""Extended actor/partner relational features for semantic motion analysis.

This is analysis-only. It reads baked controller transforms and writes
partner-relative/contact/axis features. It does not train ML, create labels, or
generate Timeline animations.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import math

import numpy as np

from vam_timeline_ai.io.json_utils import load_json, load_jsonl, write_jsonl


BODY_TARGETS = {
    "pelvis": ["pelvis", "hip", "root", "abdomen"],
    "hip": ["hip", "pelvis", "abdomen"],
    "chest": ["chest", "abdomen", "abdomen2"],
    "head": ["head", "neck"],
    "lHand": ["left_hand", "lhand", "hand_l"],
    "rHand": ["right_hand", "rhand", "hand_r"],
    "lFoot": ["left_foot", "lfoot", "foot_l"],
    "rFoot": ["right_foot", "rfoot", "foot_r"],
    "lKnee": ["left_knee", "lknee", "knee_l"],
    "rKnee": ["right_knee", "rknee", "knee_r"],
    "lThigh": ["left_thigh", "lthigh", "thigh_l"],
    "rThigh": ["right_thigh", "rthigh", "thigh_r"],
}


def extract_relational_semantic_features_v1(
    run_dir: str | Path,
    pair_windows: str | Path,
    pair_features: str | Path,
    sample_index: str | Path,
    controller_map: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    samples = {str(r.get("sample_id")): r for r in load_jsonl(sample_index) if r.get("sample_id") and r.get("bake_status") == "ok"}
    mappings = load_json(controller_map).get("controller_mappings", {})
    pair_feature_rows = {str(r.get("pair_window_id")): r for r in load_jsonl(pair_features) if r.get("pair_window_id")}
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    errors = 0
    for pair in load_jsonl(pair_windows):
        feature = pair_feature_rows.get(str(pair.get("pair_window_id") or ""), {})
        try:
            rows.extend(_rows_for_pair(pair, feature, samples, mappings, cache))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            rows.append(_error_row(pair, str(exc)))
    write_jsonl(out_jsonl, rows)
    _write_report(Path(report), rows, run, errors)
    return {
        "status": "ok",
        "rows": len(rows),
        "pair_errors": errors,
        "out_jsonl": str(out_jsonl),
        "report": str(report),
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }


def _rows_for_pair(pair: dict[str, Any], feature: dict[str, Any], samples: dict[str, dict[str, Any]], mappings: dict[str, Any], cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    a = _slice(samples[str(pair["sample_id_a"])], pair, "a", cache)
    b = _slice(samples[str(pair["sample_id_b"])], pair, "b", cache)
    pa = _parts(a["names"], mappings)
    pb = _parts(b["names"], mappings)
    active = str((feature.get("feature_quality") or {}).get("active_actor_candidate") or "unknown")
    rows: list[dict[str, Any]] = []
    if active in {"a", "unknown"}:
        rows.append(_feature_row(pair, feature, "a", a, pa, b, pb))
    if active in {"b", "unknown"}:
        rows.append(_feature_row(pair, feature, "b", b, pb, a, pa))
    return rows


def _feature_row(pair: dict[str, Any], feature: dict[str, Any], actor_slot: str, actor: dict[str, Any], actor_parts: dict[str, list[int]], partner: dict[str, Any], partner_parts: dict[str, list[int]]) -> dict[str, Any]:
    partner_slot = "b" if actor_slot == "a" else "a"
    actor_points = _points(actor, actor_parts)
    partner_points = _points(partner, partner_parts)
    n = min(len(actor["times"]), len(partner["times"]))
    actor_frame = _local_frame(actor_points)
    partner_frame = _local_frame(partner_points)
    actor_pelvis = actor_points.get("pelvis")
    actor_hip = actor_points.get("hip") if actor_points.get("hip") is not None else actor_pelvis
    partner_pelvis = partner_points.get("pelvis")
    partner_chest = partner_points.get("chest")
    partner_head = partner_points.get("head")
    warnings: list[str] = []
    if actor_pelvis is None or partner_pelvis is None:
        warnings.append("missing actor or partner pelvis/hip target")
    pelvis_delta = _delta(actor_hip, partner_pelvis, n)
    pelvis_local = _project_series(pelvis_delta, partner_frame, n)
    head_to_partner_pelvis = _distance_series(actor_points.get("head"), partner_pelvis, n)
    chest_to_partner_pelvis = _distance_series(actor_points.get("chest"), partner_pelvis, n)
    hip_motion_world = _axis_motion(actor_hip, n, _world_axes())
    hip_motion_partner = _axis_motion(actor_hip, n, partner_frame)
    pelvis_motion_partner = _axis_motion(actor_pelvis, n, partner_frame)
    hand_targets = _hand_target_summary(actor_points, partner_points, n)
    own_hand_targets = _own_hand_summary(actor_points, n)
    relation = _relation_summary(pelvis_delta, pelvis_local, actor_frame, partner_frame, n)
    contact_axis = _normalize_series(partner_pelvis - actor_hip) if actor_hip is not None and partner_pelvis is not None else None
    contact_motion = _motion_along_dynamic_axis(actor_hip, contact_axis, n)
    return {
        "schema": "relational_semantic_features_v1",
        "pair_window_id": pair.get("pair_window_id"),
        "pair_id": pair.get("pair_id"),
        "window_id": pair.get(f"window_id_{actor_slot}"),
        "partner_window_id": pair.get(f"window_id_{partner_slot}"),
        "sample_id": pair.get(f"sample_id_{actor_slot}"),
        "partner_sample_id": pair.get(f"sample_id_{partner_slot}"),
        "source_scene_file": pair.get("source_scene_file"),
        "technical_atom_id": pair.get(f"technical_atom_id_{actor_slot}"),
        "partner_atom_id": pair.get(f"technical_atom_id_{partner_slot}"),
        "start_seconds": pair.get("start_seconds"),
        "end_seconds": pair.get("end_seconds"),
        "duration_seconds": pair.get("duration_seconds"),
        "actor_slot": actor_slot,
        "partner_slot": partner_slot,
        "actor_reference_points_present": sorted(k for k, v in actor_points.items() if v is not None),
        "partner_reference_points_present": sorted(k for k, v in partner_points.items() if v is not None),
        "partner_local_frame_quality": partner_frame.get("quality", "unknown"),
        "actor_local_frame_quality": actor_frame.get("quality", "unknown"),
        "world_axes": {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]},
        "partner_pelvis_target": _point_stats(partner_pelvis),
        "partner_chest_target": _point_stats(partner_chest),
        "partner_head_target": _point_stats(partner_head),
        "actor_pelvis_to_partner_pelvis_world": _series_stats(pelvis_delta),
        "actor_pelvis_to_partner_pelvis_partner_local": _axis_stats(pelvis_local),
        "actor_pelvis_partner_alignment_distance_mean": _mean_norm(pelvis_delta),
        "actor_pelvis_partner_alignment_distance_min": _min_norm(pelvis_delta),
        "actor_pelvis_partner_alignment_score": _alignment_score(_mean_norm(pelvis_delta)),
        "actor_above_partner_score": _fraction_positive(pelvis_delta[:, 1]) if pelvis_delta is not None else 0.0,
        "actor_forward_of_partner_local_mean": _axis_mean(pelvis_local, "forward"),
        "actor_lateral_to_partner_local_mean": _axis_mean(pelvis_local, "right"),
        "actor_vertical_to_partner_local_mean": _axis_mean(pelvis_local, "up"),
        "facing_context_hint": relation["facing_context_hint"],
        "partner_relation_hints": relation["partner_relation_hints"],
        "head_to_partner_pelvis_distance_mean": _mean_or_none(head_to_partner_pelvis),
        "chest_to_partner_pelvis_distance_mean": _mean_or_none(chest_to_partner_pelvis),
        "head_to_partner_pelvis_target_score": _distance_score(_mean_or_none(head_to_partner_pelvis), close=0.55),
        "chest_to_partner_pelvis_target_score": _distance_score(_mean_or_none(chest_to_partner_pelvis), close=0.75),
        "hand_partner_targets": hand_targets,
        "hand_own_body_targets": own_hand_targets,
        "best_lHand_partner_target": hand_targets.get("lHand", {}).get("best_target", "unknown"),
        "best_rHand_partner_target": hand_targets.get("rHand", {}).get("best_target", "unknown"),
        "best_lHand_own_target": own_hand_targets.get("lHand", {}).get("best_target", "unknown"),
        "best_rHand_own_target": own_hand_targets.get("rHand", {}).get("best_target", "unknown"),
        "support_contact_hints": _support_hints(hand_targets, own_hand_targets),
        "hip_motion_world_axes": hip_motion_world,
        "hip_motion_partner_axes": hip_motion_partner,
        "pelvis_motion_partner_axes": pelvis_motion_partner,
        "hip_motion_contact_axis": contact_motion,
        "interaction_zone": {
            "technical_target": "partner_pelvis_local_contact_zone",
            "actor_anchor_bodypart": "hip_or_pelvis",
            "partner_target_bodypart": "partner_pelvis",
            "distance_mean": _mean_norm(pelvis_delta),
            "alignment_score": _alignment_score(_mean_norm(pelvis_delta)),
        },
        "warnings": warnings + list(pair.get("warnings") or []),
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }


def _slice(sample: dict[str, Any], pair_window: dict[str, Any], side: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arrays = _load_arrays(sample, cache)
    start = int(pair_window.get(f"frame_start_{side}") or 0)
    end = int(pair_window.get(f"frame_end_{side}") or arrays["pos"].shape[0])
    start = max(0, min(start, arrays["pos"].shape[0] - 1))
    end = max(start + 1, min(end, arrays["pos"].shape[0]))
    return {
        "pos": arrays["pos"][start:end],
        "rot": arrays.get("rot", np.zeros((end - start, arrays["pos"].shape[1], 4), dtype=np.float32))[start:end],
        "times": arrays["times"][start:end],
        "names": arrays["names"],
    }


def _load_arrays(sample: dict[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(sample.get("sample_id"))
    if key in cache:
        return cache[key]
    with np.load(sample["baked_npz_path"], allow_pickle=True) as data:
        arrays = {
            "pos": np.asarray(data["positions"], dtype=np.float32),
            "rot": np.asarray(data["rotations"], dtype=np.float32) if "rotations" in data.files else None,
            "times": np.asarray(data["times"], dtype=np.float32),
            "names": [str(x) for x in data["controller_names"].tolist()],
        }
    if len(cache) > 10:
        cache.clear()
    cache[key] = arrays
    return arrays


def _parts(names: list[str], mappings: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        body = str(mappings.get(name, {}).get("body_part") or _infer_part(name))
        out.setdefault(body, []).append(idx)
        lowered = name.lower()
        for alias in BODY_TARGETS:
            if alias.lower().replace("l", "left_", 1) in lowered:
                out.setdefault(alias, []).append(idx)
    return out


def _infer_part(name: str) -> str:
    lowered = name.lower()
    for body, tokens in BODY_TARGETS.items():
        if any(token.lower() in lowered for token in tokens):
            return body
    return "unknown"


def _points(actor: dict[str, Any], parts: dict[str, list[int]]) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for target, aliases in BODY_TARGETS.items():
        idx = _first(parts, [target] + aliases)
        out[target] = actor["pos"][:, idx, :] if idx is not None else None
    if out.get("pelvis") is None and out.get("hip") is not None:
        out["pelvis"] = out["hip"]
    if out.get("hip") is None and out.get("pelvis") is not None:
        out["hip"] = out["pelvis"]
    return out


def _first(parts: dict[str, list[int]], aliases: list[str]) -> int | None:
    for alias in aliases:
        if parts.get(alias):
            return parts[alias][0]
    return None


def _local_frame(points: dict[str, np.ndarray | None]) -> dict[str, Any]:
    pelvis = points.get("pelvis")
    chest = points.get("chest")
    head = points.get("head")
    left = _first_point(points, ["lThigh", "lFoot", "lHand"])
    right = _first_point(points, ["rThigh", "rFoot", "rHand"])
    torso_seed = chest if chest is not None else head
    n = min(len(x) for x in [p for p in [pelvis, torso_seed, left, right] if p is not None]) if pelvis is not None else 0
    if pelvis is None or n == 0:
        return {"up": _tile([0, 1, 0], 1), "right": _tile([1, 0, 0], 1), "forward": _tile([0, 0, 1], 1), "quality": "fallback_world"}
    torso_target = chest if chest is not None else head
    if torso_target is None:
        up = _tile([0, 1, 0], n)
        quality = "missing_torso_axis"
    else:
        up = _normalize_series(torso_target[:n] - pelvis[:n])
        quality = "body_geometry"
    if left is not None and right is not None:
        right_axis = _normalize_series(right[:n] - left[:n])
    else:
        right_axis = _tile([1, 0, 0], n)
        quality = quality + "_missing_lateral_axis"
    forward = _normalize_series(np.cross(right_axis, up))
    right_axis = _normalize_series(np.cross(up, forward))
    return {"up": up, "right": right_axis, "forward": forward, "quality": quality}


def _world_axes() -> dict[str, np.ndarray]:
    return {"right": _tile([1, 0, 0], 1), "up": _tile([0, 1, 0], 1), "forward": _tile([0, 0, 1], 1), "quality": "world"}


def _delta(a: np.ndarray | None, b: np.ndarray | None, n: int) -> np.ndarray | None:
    if a is None or b is None:
        return None
    n = min(n, len(a), len(b))
    return a[:n] - b[:n]


def _project_series(delta: np.ndarray | None, frame: dict[str, Any], n: int) -> dict[str, np.ndarray] | None:
    if delta is None:
        return None
    n = min(n, len(delta))
    axes = {axis: _axis_for_length(frame[axis], n) for axis in ["right", "up", "forward"]}
    return {axis: np.sum(delta[:n] * axes[axis], axis=1) for axis in ["right", "up", "forward"]}


def _distance_series(a: np.ndarray | None, b: np.ndarray | None, n: int) -> np.ndarray | None:
    d = _delta(a, b, n)
    return np.linalg.norm(d, axis=1) if d is not None else None


def _axis_motion(point: np.ndarray | None, n: int, frame: dict[str, Any]) -> dict[str, Any]:
    if point is None or len(point) < 2:
        return {"available": False}
    n = min(n, len(point))
    disp = point[:n] - point[:1]
    proj = _project_series(disp, frame, n)
    return {"available": True, **{axis: _one_axis_motion(vals) for axis, vals in (proj or {}).items()}}


def _one_axis_motion(vals: np.ndarray) -> dict[str, float]:
    if vals.size < 2:
        return {"range": 0.0, "path": 0.0, "net": 0.0}
    diffs = np.diff(vals)
    path = float(np.sum(np.abs(diffs)))
    net = float(abs(vals[-1] - vals[0]))
    return {"range": round(float(np.max(vals) - np.min(vals)), 6), "path": round(path, 6), "net": round(net, 6), "net_to_path": round(net / max(path, 1e-6), 6)}


def _motion_along_dynamic_axis(point: np.ndarray | None, axis: np.ndarray | None, n: int) -> dict[str, float] | dict[str, bool]:
    if point is None or axis is None or len(point) < 2:
        return {"available": False}
    n = min(n, len(point), len(axis))
    disp = point[:n] - point[:1]
    vals = np.sum(disp * axis[:n], axis=1)
    return {"available": True, **_one_axis_motion(vals)}


def _first_point(points: dict[str, np.ndarray | None], names: list[str]) -> np.ndarray | None:
    for name in names:
        value = points.get(name)
        if value is not None:
            return value
    return None


def _axis_for_length(axis: np.ndarray, n: int) -> np.ndarray:
    if len(axis) == n:
        return axis
    if len(axis) == 1:
        return np.tile(axis[:1], (n, 1))
    if len(axis) > n:
        return axis[:n]
    return np.tile(axis[-1:], (n, 1))


def _hand_target_summary(actor_points: dict[str, np.ndarray | None], partner_points: dict[str, np.ndarray | None], n: int) -> dict[str, Any]:
    targets = ["head", "chest", "pelvis", "hip", "lThigh", "rThigh", "lKnee", "rKnee", "lFoot", "rFoot"]
    out: dict[str, Any] = {}
    for hand in ["lHand", "rHand"]:
        h = actor_points.get(hand)
        distances = {}
        for target in targets:
            d = _distance_series(h, partner_points.get(target), n)
            if d is not None:
                distances[target] = _series_distance_stats(d)
        out[hand] = _best_target(distances)
    return out


def _own_hand_summary(actor_points: dict[str, np.ndarray | None], n: int) -> dict[str, Any]:
    targets = ["head", "chest", "pelvis", "hip", "lThigh", "rThigh", "lKnee", "rKnee", "lFoot", "rFoot"]
    out: dict[str, Any] = {}
    for hand in ["lHand", "rHand"]:
        h = actor_points.get(hand)
        distances = {}
        for target in targets:
            if target == hand:
                continue
            d = _distance_series(h, actor_points.get(target), n)
            if d is not None:
                distances[target] = _series_distance_stats(d)
        out[hand] = _best_target(distances)
    return out


def _best_target(distances: dict[str, Any]) -> dict[str, Any]:
    if not distances:
        return {"best_target": "missing", "distances": {}}
    best = min(distances, key=lambda k: float(distances[k].get("mean") or 999.0))
    return {"best_target": best, "best_distance_mean": distances[best]["mean"], "best_target_score": _distance_score(distances[best]["mean"]), "distances": distances}


def _support_hints(partner_targets: dict[str, Any], own_targets: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    for hand, summary in partner_targets.items():
        best = summary.get("best_target")
        score = float(summary.get("best_target_score") or 0.0)
        if score < 0.35:
            continue
        if best in {"chest", "pelvis", "hip", "lThigh", "rThigh", "lKnee", "rKnee"}:
            hints.append(f"{hand}_near_partner_{best}")
    for hand, summary in own_targets.items():
        best = summary.get("best_target")
        score = float(summary.get("best_target_score") or 0.0)
        if score >= 0.45 and best in {"chest", "pelvis", "hip", "lThigh", "rThigh"}:
            hints.append(f"{hand}_near_self_{best}")
    return sorted(set(hints))


def _relation_summary(delta: np.ndarray | None, local: dict[str, np.ndarray] | None, actor_frame: dict[str, Any], partner_frame: dict[str, Any], n: int) -> dict[str, Any]:
    hints: list[str] = []
    facing = "unknown"
    if local:
        up = float(np.nanmean(local["up"]))
        forward = float(np.nanmean(local["forward"]))
        lateral = float(np.nanmean(np.abs(local["right"])))
        if up > 0.1:
            hints.append("actor_above_partner")
        if abs(forward) < max(0.35, lateral * 1.4):
            hints.append("pelvis_aligned_or_near")
        elif forward > 0:
            hints.append("actor_in_front_of_partner_local")
        else:
            hints.append("actor_behind_partner_local")
    if len(actor_frame["forward"]) and len(partner_frame["forward"]):
        n2 = min(n, len(actor_frame["forward"]), len(partner_frame["forward"]))
        dot = np.sum(actor_frame["forward"][:n2] * partner_frame["forward"][:n2], axis=1)
        avg = float(np.nanmean(dot))
        if avg < -0.35:
            facing = "front_to_partner_or_opposed"
        elif avg > 0.35:
            facing = "back_to_partner_or_same_direction"
        else:
            facing = "side_or_unknown"
    return {"partner_relation_hints": sorted(set(hints)), "facing_context_hint": facing}


def _point_stats(point: np.ndarray | None) -> dict[str, Any]:
    if point is None or len(point) == 0:
        return {"available": False}
    return {"available": True, "mean_world": [round(float(x), 6) for x in np.nanmean(point, axis=0)]}


def _series_stats(delta: np.ndarray | None) -> dict[str, Any]:
    if delta is None or len(delta) == 0:
        return {"available": False}
    return {"available": True, "mean": [round(float(x), 6) for x in np.nanmean(delta, axis=0)], "std": [round(float(x), 6) for x in np.nanstd(delta, axis=0)]}


def _axis_stats(projected: dict[str, np.ndarray] | None) -> dict[str, Any]:
    if not projected:
        return {"available": False}
    return {"available": True, **{axis: {"mean": round(float(np.nanmean(vals)), 6), "std": round(float(np.nanstd(vals)), 6)} for axis, vals in projected.items()}}


def _series_distance_stats(d: np.ndarray) -> dict[str, float]:
    return {"mean": round(float(np.nanmean(d)), 6), "min": round(float(np.nanmin(d)), 6), "std": round(float(np.nanstd(d)), 6)}


def _normalize_series(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1)
    norms = np.where(norms < 1e-6, 1.0, norms)
    out = v / norms[:, None]
    out[~np.isfinite(out)] = 0.0
    return out


def _tile(vec: list[float], n: int) -> np.ndarray:
    return np.tile(np.asarray(vec, dtype=np.float32), (max(int(n), 1), 1))


def _mean_norm(delta: np.ndarray | None) -> float | None:
    if delta is None:
        return None
    return round(float(np.nanmean(np.linalg.norm(delta, axis=1))), 6)


def _min_norm(delta: np.ndarray | None) -> float | None:
    if delta is None:
        return None
    return round(float(np.nanmin(np.linalg.norm(delta, axis=1))), 6)


def _mean_or_none(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    return round(float(np.nanmean(values)), 6)


def _axis_mean(projected: dict[str, np.ndarray] | None, axis: str) -> float | None:
    if not projected or axis not in projected:
        return None
    return round(float(np.nanmean(projected[axis])), 6)


def _fraction_positive(values: np.ndarray) -> float:
    return round(float(np.nanmean(values > 0)), 6) if values.size else 0.0


def _alignment_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - abs(float(distance) - 0.35) / 1.25)), 6)


def _distance_score(distance: float | None, close: float = 0.65) -> float:
    if distance is None:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - float(distance) / max(close, 1e-6))), 6)


def _error_row(pair: dict[str, Any], warning: str) -> dict[str, Any]:
    return {
        "schema": "relational_semantic_features_v1",
        "pair_window_id": pair.get("pair_window_id"),
        "source_scene_file": pair.get("source_scene_file"),
        "window_id": pair.get("window_id_a"),
        "partner_window_id": pair.get("window_id_b"),
        "warnings": [warning],
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }


def _write_report(path: Path, rows: list[dict[str, Any]], run: Path, errors: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    relation_hints = Counter(h for row in rows for h in row.get("partner_relation_hints", []))
    facing = Counter(str(row.get("facing_context_hint") or "unknown") for row in rows)
    support = Counter(h for row in rows for h in row.get("support_contact_hints", []))
    best_partner_hand = Counter(row.get("best_lHand_partner_target", "unknown") for row in rows)
    best_partner_hand.update(row.get("best_rHand_partner_target", "unknown") for row in rows)
    lines = [
        "# Relational Semantic Features V1",
        "",
        "Extended actor/partner relation, contact/support, interaction-zone, and axis decomposition features.",
        "",
        f"- Run: `{run}`",
        f"- Rows: {len(rows)}",
        f"- Pair extraction errors: {errors}",
        "- ML training performed: false",
        "- Timeline generation performed: false",
        "",
        "## Partner Relation Hints",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in relation_hints.most_common(20)) or lines.append("- None")
    lines.extend(["", "## Facing Context Hints", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in facing.most_common())
    lines.extend(["", "## Support / Contact Hints", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in support.most_common(20)) or lines.append("- None")
    lines.extend(["", "## Best Partner Hand Targets", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in best_partner_hand.most_common(20))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

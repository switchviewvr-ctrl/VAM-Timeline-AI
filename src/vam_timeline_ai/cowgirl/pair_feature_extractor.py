"""Pair/context feature extraction for Cowgirl/Riding review."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.identity import make_feature_record_id
from vam_timeline_ai.io.json_utils import load_json, load_jsonl, write_jsonl


FEATURE_NAMES = [
    "a_motion_energy",
    "b_motion_energy",
    "a_pelvis_motion_energy",
    "b_pelvis_motion_energy",
    "a_hand_motion_energy",
    "b_hand_motion_energy",
    "activity_ratio_a_over_b",
    "activity_ratio_b_over_a",
    "pelvis_to_pelvis_distance_mean",
    "pelvis_to_pelvis_distance_std",
    "pelvis_vertical_offset_a_minus_b_mean",
    "pelvis_forward_offset_a_minus_b_mean_uncertain_axis",
    "chest_to_chest_distance_mean",
    "head_to_head_distance_mean",
    "a_pelvis_above_b_pelvis_score_proxy",
    "b_pelvis_above_a_pelvis_score_proxy",
    "a_left_hand_to_b_chest_distance_mean",
    "a_right_hand_to_b_chest_distance_mean",
    "a_left_hand_to_b_head_distance_mean",
    "a_right_hand_to_b_head_distance_mean",
    "a_left_hand_to_b_pelvis_distance_mean",
    "a_right_hand_to_b_pelvis_distance_mean",
    "a_hands_near_b_chest_proxy",
    "a_hands_near_b_shoulders_or_head_proxy_uncertain",
    "a_hands_near_b_pelvis_proxy",
    "a_static_hand_support_on_b_candidate_proxy",
    "b_left_hand_to_a_chest_distance_mean",
    "b_right_hand_to_a_chest_distance_mean",
    "b_left_hand_to_a_head_distance_mean",
    "b_right_hand_to_a_head_distance_mean",
    "b_left_hand_to_a_pelvis_distance_mean",
    "b_right_hand_to_a_pelvis_distance_mean",
    "b_hands_near_a_chest_proxy",
    "b_hands_near_a_shoulders_or_head_proxy_uncertain",
    "b_hands_near_a_pelvis_proxy",
    "b_static_hand_support_on_a_candidate_proxy",
    "a_vs_b_pelvis_speed_correlation",
    "a_vs_b_motion_correlation",
    "receiver_static_context_proxy_a_active",
    "receiver_static_context_proxy_b_active",
]


def extract_pair_features_v0(
    pair_windows: str | Path,
    sample_index: str | Path,
    controller_map: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    rows = load_jsonl(pair_windows)
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    mappings = load_json(controller_map).get("controller_mappings", {})
    cache: dict[str, Any] = {}
    out_rows = []
    X = []
    for row in rows:
        rec = _extract_one(row, samples, mappings, cache)
        out_rows.append(rec)
        X.append([rec["feature_values"].get(name, np.nan) for name in FEATURE_NAMES])
    write_jsonl(out_jsonl, out_rows)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=np.asarray(X, dtype=np.float32) if X else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        pair_window_ids=np.asarray([r.get("pair_window_id", "") for r in out_rows], dtype=object),
        window_ids_a=np.asarray([r.get("window_id_a", "") for r in out_rows], dtype=object),
        window_ids_b=np.asarray([r.get("window_id_b", "") for r in out_rows], dtype=object),
        sample_ids_a=np.asarray([r.get("sample_id_a", "") for r in out_rows], dtype=object),
        sample_ids_b=np.asarray([r.get("sample_id_b", "") for r in out_rows], dtype=object),
        metadata_json=json.dumps({"feature_version": "cowgirl_pair_features_v0", "roles_not_assigned": True}, ensure_ascii=False),
    )
    _write_report(out_rows, report)
    return out_rows


def _extract_one(row: dict[str, Any], samples: dict[str, dict[str, Any]], mappings: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    values = {name: np.nan for name in FEATURE_NAMES}
    warnings = list(row.get("warnings", []))
    missing: list[str] = []
    used: dict[str, list[str]] = {}
    quality = {
        "has_activity_contrast_features": False,
        "has_pelvis_relative_features": False,
        "has_hand_to_partner_features": False,
        "has_rhythm_relative_features": False,
        "active_actor_candidate": "unknown",
        "active_actor_confidence": 0.0,
    }
    try:
        a = _slice(samples[str(row["sample_id_a"])], row, "a", cache)
        b = _slice(samples[str(row["sample_id_b"])], row, "b", cache)
        pa = _parts(a["names"], mappings)
        pb = _parts(b["names"], mappings)
        root_a = _first(pa, ["pelvis", "hip", "root", "abdomen"])
        root_b = _first(pb, ["pelvis", "hip", "root", "abdomen"])
        values["a_motion_energy"] = _total_energy(a["pos"], a["times"])
        values["b_motion_energy"] = _total_energy(b["pos"], b["times"])
        values["a_pelvis_motion_energy"] = _energy(a["pos"][:, root_a, :], a["times"]) if root_a is not None else np.nan
        values["b_pelvis_motion_energy"] = _energy(b["pos"][:, root_b, :], b["times"]) if root_b is not None else np.nan
        values["a_hand_motion_energy"] = _hand_energy(a, pa)
        values["b_hand_motion_energy"] = _hand_energy(b, pb)
        values["activity_ratio_a_over_b"] = _ratio(values["a_motion_energy"], values["b_motion_energy"])
        values["activity_ratio_b_over_a"] = _ratio(values["b_motion_energy"], values["a_motion_energy"])
        quality["has_activity_contrast_features"] = np.isfinite(values["a_motion_energy"]) and np.isfinite(values["b_motion_energy"])
        candidate, confidence = _active_candidate(values["a_motion_energy"], values["b_motion_energy"])
        quality["active_actor_candidate"] = candidate
        quality["active_actor_confidence"] = confidence
        if root_a is not None and root_b is not None:
            n = min(len(a["pos"]), len(b["pos"]))
            rel = a["pos"][:n, root_a, :] - b["pos"][:n, root_b, :]
            dist = np.linalg.norm(rel, axis=1)
            values["pelvis_to_pelvis_distance_mean"] = float(np.mean(dist))
            values["pelvis_to_pelvis_distance_std"] = float(np.std(dist))
            values["pelvis_vertical_offset_a_minus_b_mean"] = float(np.mean(rel[:, 1]))
            values["pelvis_forward_offset_a_minus_b_mean_uncertain_axis"] = float(np.mean(rel[:, 2]))
            values["a_pelvis_above_b_pelvis_score_proxy"] = float(np.mean(rel[:, 1] > 0))
            values["b_pelvis_above_a_pelvis_score_proxy"] = float(np.mean(rel[:, 1] < 0))
            quality["has_pelvis_relative_features"] = True
            used["a_pelvis"] = [a["names"][root_a]]
            used["b_pelvis"] = [b["names"][root_b]]
            values["a_vs_b_pelvis_speed_correlation"] = _speed_corr(a["pos"][:n, root_a, :], b["pos"][:n, root_b, :], a["times"][:n])
        else:
            missing.append("pelvis_relative")
        _same_part_distance(values, "chest_to_chest_distance_mean", a, pa, "chest", b, pb, "chest")
        _same_part_distance(values, "head_to_head_distance_mean", a, pa, "head", b, pb, "head")
        _hand_to_other(values, "a", a, pa, "b", b, pb)
        _hand_to_other(values, "b", b, pb, "a", a, pa)
        quality["has_hand_to_partner_features"] = any(np.isfinite(values.get(k, np.nan)) for k in values if "_hand_to_" in k or "hands_near" in k)
        values["a_vs_b_motion_correlation"] = _motion_corr(a["pos"], b["pos"], a["times"])
        values["receiver_static_context_proxy_a_active"] = _receiver_static_proxy(values["a_motion_energy"], values["b_motion_energy"])
        values["receiver_static_context_proxy_b_active"] = _receiver_static_proxy(values["b_motion_energy"], values["a_motion_energy"])
        quality["has_rhythm_relative_features"] = np.isfinite(values["a_vs_b_motion_correlation"])
        if not quality["has_hand_to_partner_features"]:
            missing.append("hand_to_partner")
        warnings.append("active_actor_candidate is motion-based only and is not a semantic role label")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pair feature extraction failed: {exc}")
    return {
        "feature_record_id": make_feature_record_id(str(row.get("pair_window_id")), "cowgirl_pair_features_v0"),
        "pair_window_id": row.get("pair_window_id"),
        "pair_id": row.get("pair_id"),
        "source_scene_file": row.get("source_scene_file"),
        "window_id_a": row.get("window_id_a"),
        "window_id_b": row.get("window_id_b"),
        "sample_id_a": row.get("sample_id_a"),
        "sample_id_b": row.get("sample_id_b"),
        "technical_atom_id_a": row.get("technical_atom_id_a"),
        "technical_atom_id_b": row.get("technical_atom_id_b"),
        "feature_version": "cowgirl_pair_features_v0",
        "feature_values": values,
        "feature_quality": quality,
        "controllers_used": used,
        "missing_controller_groups": sorted(set(missing)),
        "warnings": warnings,
    }


def _slice(sample: dict[str, Any], pair_window: dict[str, Any], side: str, cache: dict[str, Any]) -> dict[str, Any]:
    arrays = _load_arrays(sample, cache)
    start = int(pair_window.get(f"frame_start_{side}") or 0)
    end = int(pair_window.get(f"frame_end_{side}") or arrays["pos"].shape[0])
    start = max(0, min(start, arrays["pos"].shape[0] - 1))
    end = max(start + 1, min(end, arrays["pos"].shape[0]))
    return {"pos": arrays["pos"][start:end], "times": arrays["times"][start:end], "names": arrays["names"]}


def _load_arrays(sample: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    key = str(sample.get("sample_id"))
    if key in cache:
        return cache[key]
    with np.load(sample["baked_npz_path"], allow_pickle=True) as data:
        arrays = {
            "pos": np.asarray(data["positions"], dtype=np.float32),
            "times": np.asarray(data["times"], dtype=np.float32),
            "names": [str(x) for x in data["controller_names"].tolist()],
        }
    if len(cache) > 8:
        cache.clear()
    cache[key] = arrays
    return arrays


def _parts(names: list[str], mappings: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        out.setdefault(mappings.get(name, {}).get("body_part", "unknown"), []).append(idx)
    return out


def _first(parts: dict[str, list[int]], names: list[str]) -> int | None:
    for name in names:
        if parts.get(name):
            return parts[name][0]
    return None


def _energy(pos: np.ndarray, times: np.ndarray) -> float:
    if len(pos) < 2:
        return np.nan
    dt = np.diff(times.astype(np.float64))
    dt = np.where(dt <= 0, 1.0 / 60.0, dt)
    speed = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0) / dt[:, None], axis=1)
    return float(np.mean(speed**2)) if speed.size else np.nan


def _total_energy(pos: np.ndarray, times: np.ndarray) -> float:
    if pos.size == 0:
        return np.nan
    vals = [_energy(pos[:, idx, :], times) for idx in range(pos.shape[1])]
    return _finite_mean(vals)


def _hand_energy(actor: dict[str, Any], parts: dict[str, list[int]]) -> float:
    vals = []
    for part in ["left_hand", "right_hand"]:
        if parts.get(part):
            vals.append(_energy(actor["pos"][:, parts[part][0], :], actor["times"]))
    return _finite_mean(vals)


def _hand_to_other(values: dict[str, float], actor_prefix: str, actor: dict[str, Any], actor_parts: dict[str, list[int]], other_prefix: str, other: dict[str, Any], other_parts: dict[str, list[int]]) -> None:
    target_parts = {"chest": "chest", "head": "head", "pelvis": ["pelvis", "hip", "root", "abdomen"]}
    hand_distance_groups = {"chest": [], "head": [], "pelvis": []}
    for side, hand_part in [("left", "left_hand"), ("right", "right_hand")]:
        if not actor_parts.get(hand_part):
            continue
        hand = actor["pos"][:, actor_parts[hand_part][0], :]
        for target_name, target_spec in target_parts.items():
            target_idx = _first(other_parts, target_spec if isinstance(target_spec, list) else [target_spec])
            if target_idx is None:
                continue
            n = min(len(hand), len(other["pos"]))
            d = np.linalg.norm(hand[:n] - other["pos"][:n, target_idx, :], axis=1)
            key = f"{actor_prefix}_{side}_hand_to_{other_prefix}_{target_name}_distance_mean"
            values[key] = float(np.mean(d))
            hand_distance_groups[target_name].append(float(np.mean(d)))
    for target_name, vals in hand_distance_groups.items():
        if vals:
            values[f"{actor_prefix}_hands_near_{other_prefix}_{target_name}_proxy"] = float(1.0 / (1.0 + _finite_mean(vals)))
    head_chest = hand_distance_groups["head"] + hand_distance_groups["chest"]
    if head_chest:
        values[f"{actor_prefix}_hands_near_{other_prefix}_shoulders_or_head_proxy_uncertain"] = float(1.0 / (1.0 + _finite_mean(head_chest)))
    hand_energy = _hand_energy(actor, actor_parts)
    chest_near = values.get(f"{actor_prefix}_hands_near_{other_prefix}_chest_proxy", np.nan)
    values[f"{actor_prefix}_static_hand_support_on_{other_prefix}_candidate_proxy"] = float(chest_near / (1.0 + hand_energy)) if np.isfinite(chest_near) and np.isfinite(hand_energy) else np.nan


def _same_part_distance(values: dict[str, float], key: str, a: dict[str, Any], pa: dict[str, list[int]], apart: str, b: dict[str, Any], pb: dict[str, list[int]], bpart: str) -> None:
    ia = _first(pa, [apart])
    ib = _first(pb, [bpart])
    if ia is None or ib is None:
        return
    n = min(len(a["pos"]), len(b["pos"]))
    values[key] = float(np.mean(np.linalg.norm(a["pos"][:n, ia, :] - b["pos"][:n, ib, :], axis=1)))


def _ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(a / (b + 1e-6))


def _active_candidate(a: float, b: float) -> tuple[str, float]:
    if not np.isfinite(a) or not np.isfinite(b):
        return "unknown", 0.0
    ratio = max(a, b) / (min(a, b) + 1e-6)
    if ratio < 1.5:
        return "unknown", float(min(1.0, (ratio - 1.0) / 0.5))
    return ("a" if a > b else "b"), float(min(1.0, (ratio - 1.0) / 4.0))


def _speed_corr(a: np.ndarray, b: np.ndarray, times: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    dt = np.diff(times.astype(np.float64))
    dt = np.where(dt <= 0, 1.0 / 60.0, dt)
    sa = np.linalg.norm(np.diff(a.astype(np.float64), axis=0) / dt[:, None], axis=1)
    sb = np.linalg.norm(np.diff(b.astype(np.float64), axis=0) / dt[:, None], axis=1)
    if np.std(sa) < 1e-8 or np.std(sb) < 1e-8:
        return np.nan
    return float(np.corrcoef(sa, sb)[0, 1])


def _motion_corr(a: np.ndarray, b: np.ndarray, times: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return np.nan
    ea = np.linalg.norm(np.diff(a[:n].reshape(n, -1), axis=0), axis=1)
    eb = np.linalg.norm(np.diff(b[:n].reshape(n, -1), axis=0), axis=1)
    if np.std(ea) < 1e-8 or np.std(eb) < 1e-8:
        return np.nan
    return float(np.corrcoef(ea, eb)[0, 1])


def _receiver_static_proxy(active: float, receiver: float) -> float:
    if not np.isfinite(active) or not np.isfinite(receiver):
        return np.nan
    return float((active / (active + receiver + 1e-6)) * (1.0 / (1.0 + receiver)))


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    q = [r.get("feature_quality", {}) for r in rows]
    active = Counter(item.get("active_actor_candidate", "unknown") for item in q)
    missing = Counter(group for row in rows for group in row.get("missing_controller_groups", []))
    lines = [
        "# Cowgirl Pair Feature Report v0",
        "",
        "Pair features are context proxies. They do not assign semantic rider/receiver roles.",
        "",
        f"- Total pair windows: {len(rows)}",
        f"- Pair feature rows: {len(rows)}",
        f"- Rows with hand-to-partner features: {sum(1 for item in q if item.get('has_hand_to_partner_features'))}",
        f"- Rows with pelvis-relative features: {sum(1 for item in q if item.get('has_pelvis_relative_features'))}",
        f"- Rows with activity contrast features: {sum(1 for item in q if item.get('has_activity_contrast_features'))}",
        "",
        "## Motion-Based Active Actor Candidates",
        "",
    ]
    for key, count in active.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Missing Controller Groups", ""])
    for key, count in missing.most_common():
        lines.append(f"- `{key}`: {count}")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

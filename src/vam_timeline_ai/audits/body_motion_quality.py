"""Audit whether baked windows contain real body-controller motion.

This module is deliberately conservative.  Root/Person/world movement can be
useful diagnostic context, but it must not be treated as valid generated VaM
body animation output.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_json, load_jsonl, write_jsonl


ROOT_PARTS = {"root", "hip", "pelvis"}
TORSO_PARTS = {"abdomen", "chest", "neck", "head"}
HAND_PARTS = {"left_hand", "right_hand", "left_elbow", "right_elbow"}
LEG_PARTS = {"left_knee", "right_knee", "left_foot", "right_foot", "left_thigh", "right_thigh"}
BODY_PARTS = ROOT_PARTS | TORSO_PARTS | HAND_PARTS | LEG_PARTS
WHOLE_PERSON_NAMES = {"control", "rootcontrol", "person", "atom", "root", "world", "worldcontrol"}


def audit_body_motion_quality(
    run_dir: str | Path,
    sample_index: str | Path,
    features: str | Path,
    controller_map: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    feature_rows = load_jsonl(features)
    cmap = _load_controller_map(controller_map)
    rows = [body_motion_quality_for_feature(row, samples.get(row.get("sample_id"), {}), cmap) for row in feature_rows]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def body_motion_quality_for_feature(row: dict[str, Any], sample: dict[str, Any] | None = None, controller_map: dict[str, Any] | None = None) -> dict[str, Any]:
    sample = sample or {}
    controller_map = controller_map or {}
    values = row.get("feature_values", {}) or {}
    quality = row.get("feature_quality", {}) or {}
    controller_names = [str(n) for n in sample.get("controller_names", []) or []]
    mapped_parts = _mapped_parts(controller_names, controller_map)

    root_energy = _max_f(values, ["pelvis_movement_energy", "movement_energy"])
    torso_energy = _max_f(values, ["torso_motion_energy", "head_motion_energy"])
    hand_energy = _sum_f(values, ["left_hand_motion_energy", "right_hand_motion_energy"])
    leg_energy = _sum_f(values, ["knee_motion_energy_left", "knee_motion_energy_right", "foot_motion_energy_left", "foot_motion_energy_right"])
    limb_energy = max(hand_energy + leg_energy, torso_energy)
    total_energy = max(root_energy + limb_energy, 0.0)
    energy_eps = _adaptive_eps(values)

    moving_bodyparts = set()
    if root_energy > energy_eps:
        moving_bodyparts.update(sorted(ROOT_PARTS & set(mapped_parts)) or ["hip"])
    if torso_energy > energy_eps:
        moving_bodyparts.update(sorted(TORSO_PARTS & set(mapped_parts)) or ["torso"])
    if hand_energy > energy_eps:
        moving_bodyparts.update(sorted(HAND_PARTS & set(mapped_parts)) or ["hands"])
    if leg_energy > energy_eps:
        moving_bodyparts.update(sorted(LEG_PARTS & set(mapped_parts)) or ["legs"])

    moving_controller_count = _moving_controller_count(sample, energy_eps)
    if moving_controller_count == 0:
        moving_controller_count = max(0, int(quality.get("moving_controller_count") or len(moving_bodyparts) or 0))

    only_whole_names = bool(controller_names) and all(_whole_person_name(n) for n in controller_names)
    only_root_like_parts = bool(mapped_parts) and set(mapped_parts).issubset(ROOT_PARTS | {"unknown"})
    root_to_limb_ratio = root_energy / max(limb_energy, 1e-9)
    root_dominant = root_energy > energy_eps and root_to_limb_ratio >= 5.0
    has_limb = (hand_energy + leg_energy + torso_energy) > energy_eps
    has_multiple = len(moving_bodyparts) >= 2 and has_limb

    if total_energy <= energy_eps and moving_controller_count == 0:
        body_quality = "static_or_empty"
    elif only_whole_names or (moving_controller_count <= 1 and root_dominant and not has_limb):
        body_quality = "controller_only_whole_person_motion"
    elif root_dominant and (only_root_like_parts or not has_limb):
        body_quality = "root_only_motion"
    elif has_multiple and len(moving_bodyparts) >= 3:
        body_quality = "good_body_motion"
    elif has_limb:
        body_quality = "partial_body_motion"
    else:
        body_quality = "unknown"

    warnings: list[str] = []
    if body_quality in {"controller_only_whole_person_motion", "root_only_motion"}:
        warnings.append("Root/whole-person motion is not valid final VaM animation output.")
    if values.get("head_motion_energy") is not None and _f(values.get("head_motion_energy")) > max(root_energy, hand_energy, leg_energy) * 2.5:
        warnings.append("Head-dominant motion may be non-Cowgirl domain or isolated gesture.")
    if not mapped_parts:
        warnings.append("No controller/bodypart mapping available.")

    return {
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_id": row.get("source_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_atom_id": row.get("technical_atom_id"),
        "controller_names": controller_names,
        "mapped_body_parts": sorted(set(mapped_parts)),
        "has_limb_controller_motion": bool(has_limb),
        "has_multiple_bodypart_motion": bool(has_multiple),
        "root_or_whole_actor_motion_dominant": bool(root_dominant or only_whole_names),
        "only_one_controller_moves": bool(moving_controller_count == 1),
        "only_root_or_hip_moves": bool(root_dominant and not has_limb),
        "limb_motion_energy": _json_float(limb_energy),
        "torso_motion_energy": _json_float(torso_energy),
        "hand_motion_energy": _json_float(hand_energy),
        "leg_motion_energy": _json_float(leg_energy),
        "root_motion_energy": _json_float(root_energy),
        "root_to_limb_energy_ratio": _json_float(root_to_limb_ratio if np.isfinite(root_to_limb_ratio) else 0.0),
        "moving_controller_count": int(moving_controller_count),
        "moving_bodypart_count": int(len(moving_bodyparts)),
        "body_motion_quality": body_quality,
        "warnings": warnings,
    }


def _load_controller_map(path: str | Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    data = load_json(path)
    mappings = data.get("controller_mappings", data if isinstance(data, dict) else {})
    return mappings if isinstance(mappings, dict) else {}


def _mapped_parts(names: list[str], cmap: dict[str, Any]) -> list[str]:
    parts = []
    for name in names:
        item = cmap.get(name) or cmap.get(name.strip())
        part = item.get("body_part") if isinstance(item, dict) else None
        if part:
            parts.append(str(part))
        elif _whole_person_name(name):
            parts.append("root")
    return parts


def _moving_controller_count(sample: dict[str, Any], eps: float) -> int:
    path = sample.get("baked_npz_path")
    if not path:
        return 0
    p = Path(str(path))
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        return 0
    try:
        with np.load(p, allow_pickle=True) as data:
            pos = np.asarray(data["positions"], dtype=np.float32)
            if pos.ndim != 3 or pos.shape[0] < 2:
                return 0
            ranges = np.nanmax(pos, axis=0) - np.nanmin(pos, axis=0)
            magnitudes = np.linalg.norm(ranges, axis=1)
            return int(np.sum(magnitudes > max(eps, 1e-5)))
    except Exception:
        return 0


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    quality_counts = Counter(r["body_motion_quality"] for r in rows)
    scene_counts = Counter(r.get("source_scene_file") for r in rows if r.get("body_motion_quality") in {"controller_only_whole_person_motion", "root_only_motion"})
    lines = [
        "# Body Motion Quality Report",
        "",
        "This audit separates real body-controller motion from root/whole-person/controller-only movement. Root/world/Person motion may be diagnostic context, but it is not valid final Timeline animation output.",
        "",
        "## Counts",
        "",
    ]
    for key, count in quality_counts.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Root/Controller-Only Hotspots", ""])
    for scene, count in scene_counts.most_common(20):
        lines.append(f"- `{scene}`: {count}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _whole_person_name(name: str) -> bool:
    token = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return token in WHOLE_PERSON_NAMES or token.endswith("root") or token.startswith("person")


def _adaptive_eps(values: dict[str, Any]) -> float:
    candidates = [_f(values.get(k)) for k in ("pelvis_movement_energy", "torso_motion_energy", "head_motion_energy", "left_hand_motion_energy", "right_hand_motion_energy")]
    finite = [v for v in candidates if np.isfinite(v)]
    return max(1e-6, float(np.nanmedian(finite)) * 0.02) if finite else 1e-6


def _max_f(values: dict[str, Any], keys: list[str]) -> float:
    vals = [_f(values.get(k)) for k in keys]
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else 0.0


def _sum_f(values: dict[str, Any], keys: list[str]) -> float:
    vals = [_f(values.get(k)) for k in keys]
    return float(sum(v for v in vals if np.isfinite(v)))


def _f(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None

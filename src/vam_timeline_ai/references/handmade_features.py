"""Feature extraction for handmade reference animations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.motion.controller_mapping import map_controller_name


def extract_handmade_reference_features(
    manifest: str | Path,
    sample_index: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    manifest_by_id = {r.get("reference_id"): r for r in load_jsonl(manifest)}
    sample_rows = load_jsonl(sample_index)
    rows = []
    feature_names = [
        "pelvis_movement_energy",
        "pelvis_vertical_amplitude",
        "pelvis_forward_back_amplitude",
        "pelvis_lateral_amplitude",
        "head_motion_energy",
        "hand_motion_energy",
        "leg_motion_energy",
        "controller_count",
        "moving_controller_count",
    ]
    matrix = []
    for sample in sample_rows:
        row = _features_for_sample(sample, manifest_by_id.get(sample.get("reference_id"), {}))
        rows.append(row)
        matrix.append([_num((row.get("feature_values") or {}).get(name)) for name in feature_names])
    write_jsonl(out_jsonl, rows)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=np.asarray(matrix, dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
        reference_ids=np.asarray([r.get("reference_id") for r in rows], dtype=object),
        metadata_json=json.dumps({"feature_version": "handmade_reference_features_v1"}, ensure_ascii=False),
    )
    _write_report(rows, report)
    return rows


def _features_for_sample(sample: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    warnings = list(sample.get("warnings", []) or [])
    feature_values = {k: None for k in ("pelvis_movement_energy", "pelvis_vertical_amplitude", "pelvis_forward_back_amplitude", "pelvis_lateral_amplitude", "head_motion_energy", "hand_motion_energy", "leg_motion_energy")}
    controller_names = sample.get("allowed_body_controller_names", []) or []
    body_parts = [map_controller_name(name).get("body_part", "unknown") for name in controller_names]
    primary = "unknown"
    moving_count = 0
    path = sample.get("baked_npz_path")
    if sample.get("bake_status") == "ok" and path and Path(path).exists():
        try:
            with np.load(path, allow_pickle=True) as data:
                positions = np.asarray(data["positions"], dtype=np.float32)
                names = [str(x) for x in data["controller_names"].tolist()]
            energies = _controller_energies(positions, names)
            moving_count = sum(1 for v in energies.values() if v > 1e-6)
            family_energy = Counter()
            for name, energy in energies.items():
                family_energy[_part_family(map_controller_name(name).get("body_part", "unknown"))] += float(energy)
            primary = family_energy.most_common(1)[0][0] if family_energy else "unknown"
            hip_names = [n for n in names if map_controller_name(n).get("body_part") in {"hip", "pelvis"}]
            if hip_names:
                hip_idx = names.index(hip_names[0])
                hip = positions[:, hip_idx, :]
                rng = np.nanmax(hip, axis=0) - np.nanmin(hip, axis=0)
                feature_values.update(
                    {
                        "pelvis_movement_energy": float(np.mean(np.linalg.norm(np.diff(hip, axis=0), axis=1))) if len(hip) > 1 else 0.0,
                        "pelvis_vertical_amplitude": float(rng[1]),
                        "pelvis_forward_back_amplitude": float(rng[2]),
                        "pelvis_lateral_amplitude": float(rng[0]),
                    }
                )
            feature_values["head_motion_energy"] = float(family_energy.get("head", 0.0))
            feature_values["hand_motion_energy"] = float(family_energy.get("hand", 0.0))
            feature_values["leg_motion_energy"] = float(family_energy.get("leg", 0.0))
        except Exception as exc:
            warnings.append(f"feature extraction failed: {exc}")
    else:
        warnings.append("no baked NPZ available")
    has_hip = any(part in {"hip", "pelvis"} for part in body_parts)
    has_head = any(part == "head" for part in body_parts)
    has_hand = any(part in {"left_hand", "right_hand", "left_elbow", "right_elbow"} for part in body_parts)
    has_leg = any(part in {"left_knee", "right_knee", "left_foot", "right_foot", "left_thigh", "right_thigh"} for part in body_parts)
    return {
        "reference_id": sample.get("reference_id"),
        "label_family": sample.get("label_family"),
        "label_subtype": sample.get("label_subtype"),
        "style": sample.get("style"),
        "intensity": sample.get("intensity"),
        "depth": sample.get("depth"),
        "is_transition_or_realign": bool(sample.get("is_transition_or_realign")),
        "controller_names": controller_names,
        "body_parts": sorted(set(body_parts)),
        "primary_controller_family": primary,
        "has_hip_motion": bool(has_hip),
        "has_thigh_motion": bool(has_leg),
        "has_head_motion": bool(has_head),
        "has_hand_motion": bool(has_hand),
        "has_only_head_motion": bool(has_head and not has_hip and not has_hand and not has_leg),
        "has_only_hip_motion": bool(has_hip and not has_head and not has_hand and not has_leg),
        "has_hip_and_thigh_motion": bool(has_hip and has_leg),
        "loop_length_seconds": sample.get("animation_length"),
        "is_loop": bool(sample.get("loop")),
        "teleport_risk": sample.get("teleport_risk"),
        "safe_for_timeline_retargeting": bool(sample.get("safe_for_timeline_retargeting")),
        "feature_values": {**feature_values, "controller_count": len(controller_names), "moving_controller_count": moving_count},
        "warnings": warnings,
    }


def _controller_energies(positions: np.ndarray, names: list[str]) -> dict[str, float]:
    if positions.ndim != 3 or positions.shape[0] < 2:
        return {name: 0.0 for name in names}
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    return {name: float(np.mean(speeds[:, idx])) for idx, name in enumerate(names)}


def _part_family(part: str) -> str:
    if part in {"hip", "pelvis"}:
        return "hip"
    if part in {"head", "neck"}:
        return "head"
    if "hand" in part or "elbow" in part:
        return "hand"
    if part in {"abdomen", "chest"}:
        return "torso"
    if any(token in part for token in ("knee", "foot", "thigh")):
        return "leg"
    return "unknown"


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    family_counts = Counter(r.get("label_family") for r in rows)
    safe_counts = Counter(str(r.get("safe_for_timeline_retargeting")) for r in rows)
    lines = [
        "# Handmade Reference Feature Report",
        "",
        f"- References: {len(rows)}",
        f"- Safe for retargeting true/false: {dict(safe_counts)}",
        "",
        "## Families",
        "",
    ]
    for family, count in family_counts.most_common():
        lines.append(f"- `{family}`: {count}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan

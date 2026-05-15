"""Relative and trajectory features for handmade reference animations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.features.relative_features import RELATIVE_FEATURE_NAMES, relative_features_from_arrays
from vam_timeline_ai.features.trajectory_shape import TRAJECTORY_FEATURE_NAMES, trajectory_shape_for_points
from vam_timeline_ai.io.json_utils import load_json, load_jsonl, write_jsonl
from vam_timeline_ai.motion.controller_mapping import map_controller_name
from vam_timeline_ai.motion.coordinate_spaces import classify_controller_track
from vam_timeline_ai.motion.relative_motion import estimate_body_scale


def build_handmade_relative_reference_features(
    handmade_sample_index: str | Path,
    controller_map: str | Path,
    out_jsonl: str | Path,
    out_npz: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    rows = load_jsonl(handmade_sample_index)
    mappings = (load_json(controller_map).get("controller_mappings") if Path(controller_map).exists() else {}) or {}
    out_base = Path(out_jsonl).parent
    trajectory_path = out_base / "handmade_trajectory_shape_features.jsonl"
    trajectory_report = out_base / "handmade_trajectory_shape_report.md"
    rel_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    for sample in rows:
        rel, traj = _handmade_reference_rows(sample, mappings)
        rel_rows.append(rel)
        traj_rows.append(traj)
        matrix.append([float(rel["feature_values"].get(name, np.nan)) for name in RELATIVE_FEATURE_NAMES])
    write_jsonl(out_jsonl, rel_rows)
    write_jsonl(trajectory_path, traj_rows)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        X=np.asarray(matrix, dtype=np.float32),
        feature_names=np.asarray(RELATIVE_FEATURE_NAMES, dtype=object),
        reference_ids=np.asarray([r.get("reference_id") for r in rel_rows], dtype=object),
        label_families=np.asarray([r.get("label_family") for r in rel_rows], dtype=object),
    )
    _write_report(rel_rows, traj_rows, report)
    _write_trajectory_report(traj_rows, trajectory_report)
    return rel_rows


def _handmade_reference_rows(sample: dict[str, Any], mappings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(str(sample.get("baked_npz_path") or ""))
    if not path.is_absolute():
        path = Path.cwd() / path
    safe = False
    warnings: list[str] = []
    bodyparts: list[str] = []
    safe_names: list[str] = []
    rel_values = {name: 0.0 for name in RELATIVE_FEATURE_NAMES}
    traj_values = {name: 0.0 for name in TRAJECTORY_FEATURE_NAMES}
    traj_quality: dict[str, Any] = {"trajectory_shape_classification": "unknown", "dominant_motion_plane": "unknown", "warnings": []}
    if sample.get("bake_status") == "ok" and path.exists():
        with np.load(path, allow_pickle=True) as data:
            positions = np.asarray(data["positions"], dtype=np.float32)
            rotations = np.asarray(data["rotations"], dtype=np.float32)
            times = np.asarray(data["times"], dtype=np.float32)
            names = [str(x) for x in data["controller_names"].tolist()]
        safe_indices = []
        stripped = 0
        for idx, name in enumerate(names):
            mapping = mappings.get(name) or map_controller_name(name)
            c = classify_controller_track(name, mapping)
            if c.allowed_body_controller:
                safe_indices.append(idx)
                safe_names.append(name)
                bodyparts.append(c.bodypart)
            else:
                stripped += 1
        if safe_indices:
            pos = positions[:, safe_indices, :]
            baseline = pos[0:1]
            delta = pos - baseline
            scale, scale_status = estimate_body_scale(pos, bodyparts)
            norm = delta / max(scale, 1e-6)
            dt = _median_dt(times)
            velocity = np.gradient(norm, max(dt, 1e-6), axis=0) if len(norm) > 1 else np.zeros_like(norm)
            rel_values, rel_quality = relative_features_from_arrays(norm, velocity, bodyparts, times)
            pelvis_idx = _first_index(bodyparts, {"pelvis", "hip", "abdomen"}) or 0
            traj_values, traj_quality = trajectory_shape_for_points(norm[:, pelvis_idx, :], times=times, safe_for_learning=True)
            safe = True
            warnings.extend(rel_quality.get("warnings", []))
            if scale_status == "unknown":
                warnings.append("Body scale unavailable; unit scale used.")
        else:
            warnings.append("No allowed body controllers after stripping root/world tracks.")
    else:
        warnings.append("Handmade sample missing baked NPZ.")
    rel_values["root_world_motion_removed"] = 1.0 if sample.get("disallowed_root_or_atom_track_count") else 0.0
    rel_values["safe_for_learning"] = 1.0 if safe else 0.0
    common = {
        "reference_id": sample.get("reference_id"),
        "label_family": sample.get("label_family"),
        "label_subtype": sample.get("label_subtype"),
        "style": sample.get("style"),
        "intensity": sample.get("intensity"),
        "depth": sample.get("depth"),
        "is_transition_or_realign": bool(sample.get("is_transition_or_realign")),
        "duration_seconds": sample.get("animation_length"),
        "controller_names": sample.get("controller_names", []),
        "allowed_body_controller_names": safe_names or sample.get("allowed_body_controller_names", []),
        "bodyparts": bodyparts,
        "teleport_risk": sample.get("teleport_risk"),
        "safe_for_learning": safe,
        "safe_for_timeline_retargeting": bool(sample.get("safe_for_timeline_retargeting")),
        "is_human_ground_truth": False,
        "reference_label_source": "handmade_filename_metadata",
        "warnings": warnings,
    }
    rel = dict(common)
    rel.update({"feature_version": "handmade_relative_features_v1", "feature_values": {k: _round(v) for k, v in rel_values.items()}})
    traj = dict(common)
    traj.update(
        {
            "feature_version": "handmade_trajectory_shape_features_v1",
            "feature_values": {k: _round(v) for k, v in traj_values.items()},
            "trajectory_shape_classification": traj_quality.get("trajectory_shape_classification"),
            "dominant_motion_plane": traj_quality.get("dominant_motion_plane"),
            "dominant_axis_1": traj_quality.get("dominant_axis_1"),
            "dominant_axis_2": traj_quality.get("dominant_axis_2"),
        }
    )
    return rel, traj


def _write_report(rows: list[dict[str, Any]], traj_rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    families = Counter(r.get("label_family") for r in rows)
    safe = Counter("safe" if r.get("safe_for_learning") else "unsafe" for r in rows)
    lines = [
        "# Handmade Relative Feature Report",
        "",
        "Handmade reference motions are converted to the same relative feature space used for wild windows.",
        "",
        f"- References: {len(rows)}",
        f"- Safe for learning/reference matching: {safe.get('safe', 0)}",
        "",
        "## Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common())
    lines.extend(["", "## Notes", "", "- Filename-derived labels are valid only for these handmade references, not for wild data."])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_trajectory_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    shapes = Counter(r.get("trajectory_shape_classification") for r in rows)
    family_shapes: Counter[str] = Counter(f"{r.get('label_family')}::{r.get('trajectory_shape_classification')}" for r in rows)
    lines = [
        "# Handmade Trajectory Shape Report",
        "",
        "Trajectory shapes are computed from relative body-controller paths.",
        "",
        "## Shape Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in shapes.most_common())
    lines.extend(["", "## Family x Shape", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in family_shapes.most_common())
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median_dt(times: np.ndarray) -> float:
    if len(times) < 2:
        return 1.0 / 60.0
    diffs = np.diff(times.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 1.0 / 60.0


def _first_index(items: list[str], choices: set[str]) -> int | None:
    for idx, item in enumerate(items):
        if item in choices:
            return idx
    return None


def _round(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return round(f, 6) if np.isfinite(f) else f


"""Controller orientation/twist validity audit.

This audit is about review/export/generation safety, not semantic truth.  A
window can be a correct Cowgirl semantic hit while still having twisted foot or
hand controller rotations that make the exported pose unusable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


FOOT_PARTS = {"left_foot", "right_foot", "foot", "lfoot", "rfoot"}
KNEE_PARTS = {"left_knee", "right_knee", "knee", "lknee", "rknee"}
HAND_PARTS = {"left_hand", "right_hand", "hand", "lhand", "rhand"}
HEAD_PARTS = {"head", "neck"}
TORSO_PARTS = {"chest", "abdomen", "torso", "hip", "pelvis"}

DEFAULT_THRESHOLDS = {
    "warning_rotation_delta": 1.50,
    "invalid_rotation_delta": 3.00,
    "warning_rotation_jump": 1.00,
    "invalid_rotation_jump": 2.00,
    "static_offset_warning": 1.80,
    "static_offset_invalid": 2.60,
}


def audit_controller_orientation_validity(
    run_dir: str | Path,
    relative_index: str | Path,
    sample_index: str | Path,
    controller_map: str | Path,
    pose_anchor_completeness: str | Path | None,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    """Audit orientation validity for every relative motion window.

    ``sample_index``/``controller_map`` are accepted for command symmetry and
    future richer body-model checks. Current checks are intentionally simple and
    relative to each window's baseline rotation.
    """
    run = Path(run_dir)
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")} if pose_anchor_completeness else {}
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(relative_index):
        raw = _raw_orientation_metrics(row, samples.get(row.get("sample_id"), {}), run)
        raw["pose_anchor_completeness"] = anchors.get(row.get("window_id"), {})
        rows.append(classify_controller_orientation_validity(raw))
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def controller_orientation_validity_for_arrays(
    rotations: np.ndarray,
    bodyparts: list[str],
    controller_names: list[str] | None = None,
    row: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convenience API for tests and synthetic audits."""
    raw = _metrics_from_rotation_delta(
        np.asarray(rotations, dtype=np.float32),
        bodyparts,
        controller_names or bodyparts,
        row or {},
    )
    return classify_controller_orientation_validity(raw, thresholds)


def classify_controller_orientation_validity(raw: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if raw.get("metric_status") != "ok":
        out = dict(raw)
        out.update(
            {
                "orientation_validity_status": "unknown",
                "orientation_validity_score": 0.35,
                "controller_rotation_outlier_count": 0,
                "twisted_controller_names": [],
                "foot_rotation_outlier": False,
                "knee_rotation_outlier": False,
                "hand_rotation_outlier": False,
                "head_rotation_outlier": False,
                "torso_rotation_outlier": False,
                "orientation_jump_count": 0,
                "unnatural_static_rotation_offset_proxy": False,
                "controller_rotation_invalid": False,
                "controller_twist_invalid": False,
                "generation_pose_valid": "unknown",
                "warnings": _dedupe([*raw.get("warnings", []), "Controller orientation validity could not be computed."]),
                "is_human_ground_truth": False,
                "is_training_label": False,
            }
        )
        return out

    max_by = {str(k): float(v) for k, v in (raw.get("rotation_delta_max_by_controller") or {}).items()}
    mean_by = {str(k): float(v) for k, v in (raw.get("rotation_delta_mean_by_controller") or {}).items()}
    jump_by = {str(k): float(v) for k, v in (raw.get("rotation_jump_max_by_controller") or {}).items()}
    part_by_name = {str(k): _normalize_part(v) for k, v in (raw.get("bodypart_by_controller") or {}).items()}

    twisted = []
    invalid = False
    warning = False
    static_offset = False
    jump_count = int(raw.get("orientation_jump_count") or 0)
    for name, max_delta in max_by.items():
        mean_delta = mean_by.get(name, 0.0)
        jump_delta = jump_by.get(name, 0.0)
        if max_delta >= thresholds["invalid_rotation_delta"] or jump_delta >= thresholds["invalid_rotation_jump"] or mean_delta >= thresholds["static_offset_invalid"]:
            invalid = True
            twisted.append(name)
        elif max_delta >= thresholds["warning_rotation_delta"] or jump_delta >= thresholds["warning_rotation_jump"] or mean_delta >= thresholds["static_offset_warning"]:
            warning = True
            twisted.append(name)
        if mean_delta >= thresholds["static_offset_warning"] and jump_delta < thresholds["warning_rotation_jump"]:
            static_offset = True

    foot_outlier = _has_part(twisted, part_by_name, FOOT_PARTS)
    knee_outlier = _has_part(twisted, part_by_name, KNEE_PARTS)
    hand_outlier = _has_part(twisted, part_by_name, HAND_PARTS)
    head_outlier = _has_part(twisted, part_by_name, HEAD_PARTS)
    torso_outlier = _has_part(twisted, part_by_name, TORSO_PARTS)
    if foot_outlier and warning:
        invalid = True
    worst_max = max(max_by.values()) if max_by else 0.0
    worst_jump = max(jump_by.values()) if jump_by else 0.0
    worst_mean = max(mean_by.values()) if mean_by else 0.0
    severity = max(
        worst_max / max(thresholds["invalid_rotation_delta"], 1e-6),
        worst_jump / max(thresholds["invalid_rotation_jump"], 1e-6),
        worst_mean / max(thresholds["static_offset_invalid"], 1e-6),
    )
    score = float(np.clip(1.0 - 0.65 * severity - 0.06 * len(twisted), 0.0, 1.0))
    status = "invalid" if invalid else "warning" if warning else "valid"
    if not max_by:
        status = "unknown"
        score = 0.35
    warnings = list(raw.get("warnings", []))
    if invalid:
        warnings.append("Controller orientation/twist is invalid for generation-template use.")
    elif warning:
        warnings.append("Controller orientation/twist should be inspected before generation use.")
    if foot_outlier:
        warnings.append("Foot controller rotation outlier blocks generation-template use but does not automatically make semantics wrong.")
    if static_offset:
        warnings.append("Large static rotation offset proxy detected; this can look like twisted controllers in VaM.")

    out = dict(raw)
    out.update(
        {
            "orientation_validity_status": status,
            "orientation_validity_score": round(score, 6),
            "controller_rotation_outlier_count": len(twisted),
            "twisted_controller_names": _dedupe(twisted),
            "foot_rotation_outlier": bool(foot_outlier),
            "knee_rotation_outlier": bool(knee_outlier),
            "hand_rotation_outlier": bool(hand_outlier),
            "head_rotation_outlier": bool(head_outlier),
            "torso_rotation_outlier": bool(torso_outlier),
            "orientation_jump_count": jump_count,
            "unnatural_static_rotation_offset_proxy": bool(static_offset),
            "controller_rotation_invalid": bool(invalid),
            "controller_twist_invalid": bool(invalid and bool(twisted)),
            "generation_pose_valid": status == "valid",
            "warnings": _dedupe(warnings),
            "is_human_ground_truth": False,
            "is_training_label": False,
        }
    )
    return out


def _raw_orientation_metrics(relative_row: dict[str, Any], sample: dict[str, Any], run: Path) -> dict[str, Any]:
    path = relative_row.get("relative_npz_path")
    if not path:
        return _unknown_row(relative_row, sample, "relative_npz_path missing")
    npz_path = Path(str(path))
    if not npz_path.is_absolute():
        project_root = run.parents[2] if len(run.parents) >= 3 else Path.cwd()
        candidate = project_root / npz_path if str(path).startswith("data") else Path.cwd() / npz_path
        npz_path = candidate if candidate.exists() else Path.cwd() / npz_path
    if not npz_path.exists():
        return _unknown_row(relative_row, sample, f"relative NPZ missing: {path}")
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "rotation_delta" not in data.files:
                return _unknown_row(relative_row, sample, "rotation_delta missing")
            rotations = np.asarray(data["rotation_delta"], dtype=np.float32)
            bodyparts = [str(x) for x in data.get("bodyparts", [])]
            names = [str(x) for x in data.get("controller_names", [])]
    except Exception as exc:
        return _unknown_row(relative_row, sample, f"could not load relative NPZ: {exc}")
    raw = _metrics_from_rotation_delta(rotations, bodyparts, names, relative_row)
    return raw


def _metrics_from_rotation_delta(rotations: np.ndarray, bodyparts: list[str], names: list[str], row: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    if rotations.ndim != 3 or rotations.shape[0] == 0 or rotations.shape[1] == 0:
        return _unknown_row(row, {}, "rotation arrays have unexpected shape")
    if rotations.shape[-1] < 3:
        return _unknown_row(row, {}, "rotation arrays do not include enough components")
    names = list(names) if names else [f"controller_{idx}" for idx in range(rotations.shape[1])]
    bodyparts = list(bodyparts) if bodyparts else ["unknown"] * rotations.shape[1]
    count = min(len(names), len(bodyparts), rotations.shape[1])
    if count == 0:
        return _unknown_row(row, {}, "no controller names/bodyparts for rotation audit")
    rotations = rotations[:, :count, :]
    names = names[:count]
    bodyparts = bodyparts[:count]
    norms = np.linalg.norm(np.nan_to_num(rotations, nan=0.0), axis=2)
    jumps = np.linalg.norm(np.diff(np.nan_to_num(rotations, nan=0.0), axis=0), axis=2) if rotations.shape[0] > 1 else np.zeros((0, count), dtype=np.float32)
    max_by = {name: _json_float(np.nanmax(norms[:, idx])) for idx, name in enumerate(names)}
    mean_by = {name: _json_float(np.nanmean(norms[:, idx])) for idx, name in enumerate(names)}
    jump_by = {name: _json_float(np.nanmax(jumps[:, idx])) if jumps.size else 0.0 for idx, name in enumerate(names)}
    jump_count = int(np.sum(jumps > DEFAULT_THRESHOLDS["warning_rotation_jump"])) if jumps.size else 0
    return {
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_id": row.get("source_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_atom_id": row.get("technical_atom_id"),
        "controller_names": names,
        "bodyparts": bodyparts,
        "bodypart_by_controller": {name: bodyparts[idx] for idx, name in enumerate(names)},
        "rotation_delta_max_by_controller": max_by,
        "rotation_delta_mean_by_controller": mean_by,
        "rotation_jump_max_by_controller": jump_by,
        "orientation_jump_count": jump_count,
        "metric_status": "ok",
        "warnings": warnings,
    }


def _unknown_row(relative_row: dict[str, Any], sample: dict[str, Any], warning: str) -> dict[str, Any]:
    return {
        "window_id": relative_row.get("window_id"),
        "sample_id": relative_row.get("sample_id") or sample.get("sample_id"),
        "source_id": relative_row.get("source_id") or sample.get("source_id"),
        "source_scene_file": relative_row.get("source_scene_file") or sample.get("source_scene_file"),
        "technical_atom_id": relative_row.get("technical_atom_id") or sample.get("technical_atom_id"),
        "controller_names": relative_row.get("controllers", []),
        "bodyparts": relative_row.get("bodyparts", []),
        "metric_status": "unknown",
        "warnings": [warning],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _normalize_part(part: Any) -> str:
    text = str(part or "").strip().lower()
    aliases = {
        "lfoot": "left_foot",
        "rfoot": "right_foot",
        "lknee": "left_knee",
        "rknee": "right_knee",
        "lhand": "left_hand",
        "rhand": "right_hand",
        "lthigh": "left_thigh",
        "rthigh": "right_thigh",
    }
    return aliases.get(text, text)


def _has_part(names: list[str], part_by_name: dict[str, str], wanted: set[str]) -> bool:
    for name in names:
        part = _normalize_part(part_by_name.get(name, ""))
        text = str(name).lower()
        if part in wanted:
            return True
        if "foot" in text and wanted is FOOT_PARTS:
            return True
        if "knee" in text and wanted is KNEE_PARTS:
            return True
        if "hand" in text and wanted is HAND_PARTS:
            return True
        if "head" in text and wanted is HEAD_PARTS:
            return True
    return False


def _json_float(value: Any) -> float:
    try:
        val = float(value)
        return round(float(val), 6) if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(r.get("orientation_validity_status") for r in rows)
    foot = sum(1 for r in rows if r.get("foot_rotation_outlier"))
    invalid = [r for r in rows if r.get("orientation_validity_status") == "invalid"]
    twisted = Counter(name for r in rows for name in r.get("twisted_controller_names", []) or [])
    max_values = []
    for row in rows:
        for value in (row.get("rotation_delta_max_by_controller") or {}).values():
            try:
                max_values.append(float(value))
            except Exception:
                pass
    arr = np.asarray(max_values, dtype=np.float32) if max_values else np.asarray([], dtype=np.float32)
    lines = [
        "# Controller Orientation Validity Report",
        "",
        "This is an audit of export/generation pose safety. It does not change semantic labels.",
        "",
        f"- Windows audited: {len(rows)}",
        f"- Foot rotation outliers: {foot}",
        f"- Orientation-invalid windows: {len(invalid)}",
        "",
        "## Orientation Validity Status",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in status_counts.most_common()) if status_counts else lines.append("- None")
    if arr.size:
        lines.extend(
            [
                "",
                "## Rotation Delta Distribution",
                "",
                f"- p50 max-delta: {float(np.nanpercentile(arr, 50)):.4f}",
                f"- p90 max-delta: {float(np.nanpercentile(arr, 90)):.4f}",
                f"- p99 max-delta: {float(np.nanpercentile(arr, 99)):.4f}",
            ]
        )
    lines.extend(["", "## Common Twisted Controllers", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in twisted.most_common(20)) if twisted else lines.append("- None")
    lines.extend(["", "## Review-002-Like Orientation Invalid Examples", ""])
    for row in sorted(invalid, key=lambda r: float(r.get("orientation_validity_score") or 0.0))[:20]:
        lines.append(
            f"- `{row.get('window_id')}` score={row.get('orientation_validity_score')} "
            f"twisted={row.get('twisted_controller_names')} scene=`{row.get('source_scene_file')}`"
        )
    if not invalid:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

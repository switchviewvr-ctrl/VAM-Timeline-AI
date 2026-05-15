"""Controller distance plausibility audit for generation safety.

This detects cases where required body controllers exist but sit implausibly
far from related body-chain controllers.  It is an audit signal only: distance
invalidity blocks generation-template use but does not automatically erase a
semantic Cowgirl interpretation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.audits.controller_validity import (
    DEFAULT_THRESHOLDS,
    _derive_thresholds,
    _finite_float,
    _metrics_from_arrays,
    _outlier_score,
    _raw_controller_metrics,
)
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def audit_controller_distance_validity(
    run_dir: str | Path,
    relative_index: str | Path,
    sample_index: str | Path,
    controller_map: str | Path,
    pose_anchor_completeness: str | Path | None,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    """Audit distance plausibility for every relative motion window."""
    run = Path(run_dir)
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")} if pose_anchor_completeness else {}
    raw_rows = []
    for row in load_jsonl(relative_index):
        raw = _raw_controller_metrics(row, samples.get(row.get("sample_id"), {}), run)
        raw["pose_anchor_completeness"] = anchors.get(row.get("window_id"), {})
        raw_rows.append(raw)
    thresholds = _derive_thresholds(raw_rows)
    rows = [classify_controller_distance_validity(row, thresholds) for row in raw_rows]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, thresholds, report)
    return rows


def controller_distance_validity_for_arrays(
    positions: np.ndarray,
    bodyparts: list[str],
    controller_names: list[str] | None = None,
    scale: float | None = None,
    row: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convenience API for tests and synthetic audits."""
    raw = _metrics_from_arrays(np.asarray(positions, dtype=np.float32), bodyparts, controller_names or bodyparts, float(scale or 1.0), row or {})
    return classify_controller_distance_validity(raw, thresholds)


def classify_controller_distance_validity(raw: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if raw.get("metric_status") != "ok":
        return {
            **raw,
            "controller_distance_validity_status": "unknown",
            "controller_distance_validity_score": 0.35,
            "controller_distance_outlier": False,
            "outlier_controller_names": [],
            "foot_distance_outlier": False,
            "knee_distance_outlier": False,
            "hand_distance_outlier": False,
            "head_distance_outlier": False,
            "max_bodypart_distance_ratio": None,
            "warnings": _dedupe([*raw.get("warnings", []), "Controller distance validity could not be computed."]),
            "is_human_ground_truth": False,
            "is_training_label": False,
        }

    scores = {
        "left_foot": max(
            _outlier_score(raw.get("left_foot_to_hip_distance_max"), thresholds["foot_to_hip_distance_max"]),
            _outlier_score(raw.get("left_foot_to_knee_distance_max"), thresholds["foot_to_knee_distance_max"]),
            _outlier_score(raw.get("left_foot_to_thigh_distance_mean"), thresholds["foot_to_thigh_distance_mean"]),
        ),
        "right_foot": max(
            _outlier_score(raw.get("right_foot_to_hip_distance_max"), thresholds["foot_to_hip_distance_max"]),
            _outlier_score(raw.get("right_foot_to_knee_distance_max"), thresholds["foot_to_knee_distance_max"]),
            _outlier_score(raw.get("right_foot_to_thigh_distance_mean"), thresholds["foot_to_thigh_distance_mean"]),
        ),
        "left_knee": _outlier_score(raw.get("left_knee_to_hip_distance_max"), thresholds["knee_to_hip_distance_max"]),
        "right_knee": _outlier_score(raw.get("right_knee_to_hip_distance_max"), thresholds["knee_to_hip_distance_max"]),
        "left_hand": max(
            _outlier_score(raw.get("left_hand_to_chest_distance_max"), thresholds["hand_to_chest_distance_max"]),
            _outlier_score(raw.get("left_hand_to_head_distance_max"), thresholds["hand_to_head_distance_max"]),
            _outlier_score(raw.get("left_hand_to_hip_distance_max"), thresholds["hand_to_hip_distance_max"]),
        ),
        "right_hand": max(
            _outlier_score(raw.get("right_hand_to_chest_distance_max"), thresholds["hand_to_chest_distance_max"]),
            _outlier_score(raw.get("right_hand_to_head_distance_max"), thresholds["hand_to_head_distance_max"]),
            _outlier_score(raw.get("right_hand_to_hip_distance_max"), thresholds["hand_to_hip_distance_max"]),
        ),
        "head_or_torso": max(
            _outlier_score(raw.get("head_to_chest_distance_max"), thresholds["head_to_chest_distance_max"]),
            _outlier_score(raw.get("chest_to_hip_distance_max"), thresholds["chest_to_hip_distance_max"]),
        ),
    }
    outliers = []
    for name, score in scores.items():
        threshold = 0.15 if "foot" in name or "knee" in name else 0.20
        if score >= threshold:
            outliers.append(name)
    foot_outlier = "left_foot" in outliers or "right_foot" in outliers
    knee_outlier = "left_knee" in outliers or "right_knee" in outliers
    hand_outlier = "left_hand" in outliers or "right_hand" in outliers
    head_outlier = "head_or_torso" in outliers
    max_ratio = max([float(v) for v in scores.values() if np.isfinite(v)] or [0.0])
    invalid = foot_outlier or knee_outlier or head_outlier or max_ratio >= 0.55
    warning = bool(outliers)
    status = "invalid" if invalid else "warning" if warning else "valid"
    score = float(np.clip(1.0 - 0.45 * max_ratio - 0.10 * len(outliers), 0.0, 1.0))
    warnings = list(raw.get("warnings", []))
    if invalid:
        warnings.append("Controller distance validity is invalid for generation-template use.")
    elif warning:
        warnings.append("Controller distance validity has warning-level outliers; inspect before generation use.")
    if foot_outlier:
        warnings.append("Foot controller distance outlier detected; feet may appear too far from the body.")
    if knee_outlier:
        warnings.append("Knee/leg-chain distance outlier detected.")
    if hand_outlier:
        warnings.append("Hand controller distance outlier detected.")
    if head_outlier:
        warnings.append("Head/torso distance outlier detected.")

    return {
        **raw,
        "controller_distance_validity_status": status,
        "controller_distance_validity_score": round(score, 6),
        "controller_distance_outlier": bool(outliers),
        "outlier_controller_names": outliers,
        "foot_distance_outlier": bool(foot_outlier),
        "knee_distance_outlier": bool(knee_outlier),
        "hand_distance_outlier": bool(hand_outlier),
        "head_distance_outlier": bool(head_outlier),
        "max_bodypart_distance_ratio": round(float(max_ratio), 6),
        "foot_to_hip_distance_max_normalized": _json_float(raw.get("foot_to_hip_distance_max")),
        "foot_to_knee_distance_max_normalized": _json_float(raw.get("foot_to_knee_distance_max")),
        "hand_to_chest_distance_max_normalized": _json_float(raw.get("hand_to_chest_distance_max")),
        "hand_to_head_distance_max_normalized": _json_float(raw.get("hand_to_head_distance_max")),
        "warnings": _dedupe(warnings),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _json_float(value: Any) -> float | None:
    val = _finite_float(value)
    return round(float(val), 6) if val is not None else None


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _write_report(rows: list[dict[str, Any]], thresholds: dict[str, float], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(r.get("controller_distance_validity_status") for r in rows)
    distance = [r for r in rows if r.get("controller_distance_outlier")]
    foot = sum(1 for r in rows if r.get("foot_distance_outlier"))
    knee = sum(1 for r in rows if r.get("knee_distance_outlier"))
    hand = sum(1 for r in rows if r.get("hand_distance_outlier"))
    head = sum(1 for r in rows if r.get("head_distance_outlier"))
    lines = [
        "# Controller Distance Validity Report",
        "",
        "This audit detects present-but-implausibly-far controller chains. It does not change semantic labels.",
        "",
        f"- Windows audited: {len(rows)}",
        f"- Distance outlier windows: {len(distance)}",
        f"- Foot distance outliers: {foot}",
        f"- Knee distance outliers: {knee}",
        f"- Hand distance outliers: {hand}",
        f"- Head/torso distance outliers: {head}",
        "",
        "## Distance Validity Status",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in status_counts.most_common()) if status_counts else lines.append("- None")
    lines.extend(["", "## Thresholds", ""])
    for key, value in sorted(thresholds.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Review-009-Like Distance Invalid Examples", ""])
    for row in sorted(distance, key=lambda r: float(r.get("max_bodypart_distance_ratio") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` status=`{row.get('controller_distance_validity_status')}` "
            f"ratio={row.get('max_bodypart_distance_ratio')} outliers={row.get('outlier_controller_names')} "
            f"scene=`{row.get('source_scene_file')}`"
        )
    if not distance:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

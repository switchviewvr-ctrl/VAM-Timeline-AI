"""Calibrate broad weak labels into stricter weak_v2 review hints."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


FEATURE_FOR_LABEL = {
    "weak_v2_high_vertical_motion": "pelvis_vertical_amplitude",
    "weak_v2_forward_back_dominant": "pelvis_forward_back_amplitude",
    "weak_v2_lateral_dominant": "pelvis_lateral_amplitude",
    "weak_v2_circular_grind_candidate": "pelvis_circularity_score_proxy",
    "weak_v2_pause_hold_candidate": "pause_hold_score_proxy",
    "weak_v2_irregular_motion_candidate": "irregular_rhythm_score_proxy",
    "weak_v2_fast_motion_candidate": "fast_motion_score_proxy",
    "weak_v2_slow_motion_candidate": "slow_motion_score_proxy",
    "weak_v2_high_hand_motion": "hand_motion_energy_combined",
    "weak_v2_static_hand_support_candidate": "hands_static_support_proxy",
    "weak_v2_torso_active": "torso_motion_energy",
    "weak_v2_head_active": "head_motion_energy",
    "weak_v2_leg_active": "leg_motion_energy_combined",
    "weak_v2_high_intensity": "intensity_score_proxy",
    "weak_v2_low_intensity": "intensity_score_proxy",
}


def calibrate_weak_labels_v2(features_path: str | Path, weak_labels_path: str | Path, out: str | Path, report: str | Path) -> list[dict[str, Any]]:
    feature_rows = load_jsonl(features_path)
    old_weak = load_jsonl(weak_labels_path)
    old_counts = Counter(item["label"] for row in old_weak for item in row.get("weak_labels", []))
    values_by_feature = _feature_distributions(feature_rows)
    thresholds = _thresholds(values_by_feature)
    rows = []
    for row in feature_rows:
        values = _derived_values(row.get("feature_values", {}))
        labels = _labels_for(values, thresholds)
        rows.append(
            {
                "window_id": row.get("window_id"),
                "sample_id": row.get("sample_id"),
                "source_scene_file": row.get("source_scene_file"),
                "technical_atom_id": row.get("technical_atom_id"),
                "weak_labels": labels,
                "calibration_metadata": {"thresholds": thresholds},
                "note": "weak_v2 labels are calibrated review hints, not semantic truth.",
            }
        )
    write_jsonl(out, rows)
    _write_report(feature_rows, old_counts, rows, thresholds, report)
    return rows


def _feature_distributions(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        derived = _derived_values(row.get("feature_values", {}))
        for key, value in derived.items():
            if np.isfinite(value):
                values[key].append(float(value))
    return values


def _derived_values(v: dict[str, Any]) -> dict[str, float]:
    out = {k: _f(v, k) for k in set(FEATURE_FOR_LABEL.values()) if k not in {"hand_motion_energy_combined", "leg_motion_energy_combined"}}
    out["hand_motion_energy_combined"] = _finite_mean([_f(v, "left_hand_motion_energy"), _f(v, "right_hand_motion_energy")])
    out["leg_motion_energy_combined"] = _finite_mean([
        _f(v, "knee_motion_energy_left"),
        _f(v, "knee_motion_energy_right"),
        _f(v, "foot_motion_energy_left"),
        _f(v, "foot_motion_energy_right"),
    ])
    return out


def _thresholds(values_by_feature: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for name, values in values_by_feature.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            thresholds[name] = {"p10": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p90": np.nan}
        else:
            thresholds[name] = {f"p{p}": float(np.percentile(arr, p)) for p in [10, 25, 50, 75, 90]}
    return thresholds


def _labels_for(values: dict[str, float], thresholds: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []

    def add(label: str, value: float, threshold: float, rule: str, invert: bool = False) -> None:
        if not np.isfinite(value) or not np.isfinite(threshold):
            return
        passed = value <= threshold if invert else value >= threshold
        if not passed:
            return
        score = threshold / (value + 1e-6) if invert else value / (threshold + 1e-6)
        labels.append({"label": label, "proxy_score": round(float(min(max(score, 0.0), 3.0)), 4), "feature_value": float(value), "threshold": float(threshold), "rule": rule})

    for label, feature in FEATURE_FOR_LABEL.items():
        value = values.get(feature, np.nan)
        t = thresholds.get(feature, {})
        if label == "weak_v2_low_intensity":
            add(label, value, t.get("p25", np.nan), "bottom quartile intensity proxy", invert=True)
        elif label in {"weak_v2_static_hand_support_candidate", "weak_v2_pause_hold_candidate", "weak_v2_slow_motion_candidate"}:
            add(label, value, t.get("p75", np.nan), "upper quartile calibrated proxy")
        else:
            add(label, value, t.get("p80", t.get("p75", np.nan)), "high percentile calibrated proxy")
    return [item for item in labels if item["label"].startswith("weak_v2_")]


def _write_report(feature_rows: list[dict[str, Any]], old_counts: Counter[str], rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]], report: str | Path) -> None:
    new_counts = Counter(item["label"] for row in rows for item in row.get("weak_labels", []))
    co = Counter()
    for row in rows:
        labels = sorted(item["label"] for item in row.get("weak_labels", []))
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                co[(a, b)] += 1
    broad_old = [label for label, count in old_counts.items() if count / max(len(feature_rows), 1) > 0.5]
    broad_new = [label for label, count in new_counts.items() if count / max(len(feature_rows), 1) > 0.5]
    lines = [
        "# Weak Label Calibration Report v2",
        "",
        "weak_v2 labels are calibrated review hints, not semantic labels or ground truth.",
        "",
        f"- Feature rows: {len(feature_rows)}",
        f"- Old weak label assignments: {sum(old_counts.values())}",
        f"- New weak_v2 assignments: {sum(new_counts.values())}",
        f"- Old labels likely too broad (>50% rows): {broad_old}",
        f"- New labels likely too broad (>50% rows): {broad_new}",
        "",
        "## New Weak Label Counts",
        "",
    ]
    for label, count in new_counts.most_common():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Old Weak Label Counts", ""])
    for label, count in old_counts.most_common():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Feature Thresholds", ""])
    for feature, vals in sorted(thresholds.items()):
        lines.append(f"- `{feature}`: {vals}")
    lines.extend(["", "## Top Co-occurrences", ""])
    for (a, b), count in co.most_common(20):
        lines.append(f"- `{a}` + `{b}`: {count}")
    lines.extend(["", "## Recommended Manual Review Quotas", ""])
    for label, count in new_counts.most_common():
        quota = min(12, max(4, int(round(count / max(sum(new_counts.values()), 1) * 120))))
        lines.append(f"- `{label}`: review about {quota} windows across multiple scenes/samples")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(values: dict[str, Any], name: str) -> float:
    try:
        return float(values.get(name, np.nan))
    except Exception:
        return np.nan


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


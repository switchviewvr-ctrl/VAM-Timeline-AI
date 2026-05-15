"""Generate weak labels from numeric feature thresholds.

Weak labels are review aids, not ground-truth semantic labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import write_jsonl


def generate_weak_labels_v1(features_path: str | Path, out: str | Path, report: str | Path) -> list[dict[str, Any]]:
    rows = []
    for feature_row in _load_jsonl(features_path):
        labels = _labels_for(feature_row.get("feature_values", {}))
        rows.append(
            {
                "window_id": feature_row.get("window_id"),
                "sample_id": feature_row.get("sample_id"),
                "source_scene_file": feature_row.get("source_scene_file"),
                "technical_atom_id": feature_row.get("technical_atom_id"),
                "weak_labels": labels,
                "note": "Weak labels are numeric review proxies, not manual labels or semantic truth.",
            }
        )
    write_jsonl(out, rows)
    _write_report(rows, report)
    return rows


def _labels_for(v: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def add(label: str, score: float, rule: str) -> None:
        if score > 0 and np.isfinite(score):
            out.append({"label": label, "score": round(float(min(score, 1.0)), 4), "rule": rule})

    vertical = _f(v, "pelvis_vertical_amplitude")
    forward = _f(v, "pelvis_forward_back_amplitude")
    lateral = _f(v, "pelvis_lateral_amplitude")
    energy = _f(v, "pelvis_movement_energy")
    pause = _f(v, "pelvis_pause_ratio")
    irregular = _f(v, "irregular_rhythm_score_proxy")
    fast = _f(v, "fast_motion_score_proxy")
    slow = _f(v, "slow_motion_score_proxy")
    hand_energy = _finite_mean([_f(v, "left_hand_motion_energy"), _f(v, "right_hand_motion_energy")])
    torso_energy = _f(v, "torso_motion_energy")
    head_energy = _f(v, "head_motion_energy")
    leg_energy = _finite_mean([_f(v, "knee_motion_energy_left"), _f(v, "knee_motion_energy_right"), _f(v, "foot_motion_energy_left"), _f(v, "foot_motion_energy_right")])

    total = vertical + forward + lateral + 1e-6 if all(np.isfinite(x) for x in [vertical, forward, lateral]) else np.nan
    add("weak_high_vertical_bounce", vertical / total if np.isfinite(total) and vertical > 0.04 else 0.0, "vertical amplitude dominance")
    add("weak_forward_back_dominant", forward / total if np.isfinite(total) and forward > 0.04 else 0.0, "forward/back amplitude dominance")
    add("weak_lateral_sway", lateral / total if np.isfinite(total) and lateral > 0.04 else 0.0, "lateral amplitude present")
    add("weak_circular_grind_candidate", _f(v, "pelvis_circularity_score_proxy"), "pelvis x/z circularity proxy")
    add("weak_pause_or_hold", pause, "pause ratio")
    add("weak_irregular_motion", irregular, "irregular rhythm proxy")
    add("weak_fast_motion", fast, "mean speed proxy")
    add("weak_slow_motion", slow, "inverse speed proxy")
    add("weak_high_hand_motion", min(hand_energy / 0.5, 1.0), "hand motion energy") if np.isfinite(hand_energy) else None
    add("weak_static_hand_support_candidate", _f(v, "hands_static_support_proxy"), "low hand motion proxy")
    add("weak_torso_active", min(torso_energy / 0.5, 1.0), "torso motion energy") if np.isfinite(torso_energy) else None
    add("weak_head_active", min(head_energy / 0.5, 1.0), "head motion energy") if np.isfinite(head_energy) else None
    add("weak_leg_active", min(leg_energy / 0.5, 1.0), "leg motion energy") if np.isfinite(leg_energy) else None
    add("weak_high_energy", min(energy / 1.0, 1.0), "pelvis/root movement energy") if np.isfinite(energy) else None
    return [label for label in out if label["label"].startswith("weak_")]


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


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    counts = Counter(item["label"] for row in rows for item in row.get("weak_labels", []))
    lines = ["# Weak Label Report v1", "", "Weak labels are not semantic truth and must not be mixed with manual labels.", ""]
    for label, count in counts.most_common():
        lines.append(f"- `{label}`: {count}")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

"""Readiness and balance analysis for aggregated silver v2 labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl


def analyze_silver_readiness_v2(
    dataset: str | Path,
    silver_window_labels: str | Path,
    silver_pair_labels: str | Path,
    out: str | Path,
    min_positive: int = 100,
    min_scenes: int = 5,
    min_samples: int = 20,
) -> dict[str, Any]:
    with np.load(dataset, allow_pickle=True) as data:
        labels = [str(x) for x in data["silver_label_names"].tolist()]
        window_labels = [str(x) for x in data["silver_window_label_names"].tolist()]
        pair_labels = [str(x) for x in data["silver_pair_label_names"].tolist()]
        window_y = data["silver_v2_window_y_multilabel"]
        pair_y = data["silver_v2_pair_y_multilabel"]
        scenes = [str(x) for x in data["group_scene"].tolist()]
        samples = [str(x) for x in data["group_sample"].tolist()]
        default_mask = data["default_trainable_silver_label_mask"].astype(bool)
        excluded_names = [str(x) for x in data["excluded_silver_label_names"].tolist()]
        exclusion_reasons = [str(x) for x in data["exclusion_reasons"].tolist()]
    excluded = dict(zip(excluded_names, exclusion_reasons))
    per_label: dict[str, dict[str, Any]] = {}
    labels_trainable_by_default: list[str] = []
    labels_review_only: list[str] = []
    labels_excluded_high_risk: list[str] = []
    for label in labels:
        counts = _label_indices(label, window_labels, pair_labels, window_y, pair_y)
        pos_idx = counts["indices"]
        label_scenes = {scenes[i] for i in pos_idx}
        label_samples = {samples[i] for i in pos_idx}
        available_negatives = len(scenes) - len(pos_idx)
        excluded_reason = excluded.get(label)
        mask_idx = labels.index(label)
        default_allowed = bool(default_mask[mask_idx]) and not excluded_reason
        ready = default_allowed and len(pos_idx) >= min_positive and len(label_scenes) >= min_scenes and len(label_samples) >= min_samples and available_negatives >= min_positive
        if ready:
            labels_trainable_by_default.append(label)
        elif excluded_reason:
            labels_excluded_high_risk.append(label)
        else:
            labels_review_only.append(label)
        per_label[label] = {
            "positives": int(len(pos_idx)),
            "available_negatives": int(available_negatives),
            "scenes": int(len(label_scenes)),
            "samples": int(len(label_samples)),
            "default_allowed": default_allowed,
            "excluded_reason": excluded_reason,
            "exploratory_proxy_training_ready": bool(ready),
            "too_rare": len(pos_idx) < min_positive,
            "scene_concentrated": len(label_scenes) < min_scenes,
            "sample_concentrated": len(label_samples) < min_samples,
        }
    summary = {
        "silver_proxy_training_ready": bool(labels_trainable_by_default),
        "labels_trainable_by_default": labels_trainable_by_default,
        "labels_review_only": labels_review_only,
        "labels_excluded_high_risk": labels_excluded_high_risk,
        "per_label": per_label,
        "window_silver_records": len(load_jsonl(silver_window_labels)),
        "pair_silver_records": len(load_jsonl(silver_pair_labels)),
        "thresholds": {"min_positive": min_positive, "min_scenes": min_scenes, "min_samples": min_samples},
        "warning": "Silver v2 proxy training is not semantic validation.",
    }
    _write_report(summary, out)
    return summary


def _label_indices(label: str, window_labels: list[str], pair_labels: list[str], window_y: np.ndarray, pair_y: np.ndarray) -> dict[str, Any]:
    indices: set[int] = set()
    if label in window_labels:
        idx = window_labels.index(label)
        indices.update(np.where(window_y[:, idx] > 0)[0].tolist())
    if label in pair_labels:
        idx = pair_labels.index(label)
        indices.update(np.where(pair_y[:, idx] > 0)[0].tolist())
    return {"indices": sorted(indices)}


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Silver Readiness Report v2",
        "",
        "Silver v2 labels are aggregated machine labels. This report is for proxy training readiness, not semantic validation.",
        "",
        f"- silver_proxy_training_ready: {summary['silver_proxy_training_ready']}",
        f"- labels_trainable_by_default: {summary['labels_trainable_by_default'] or 'None'}",
        f"- labels_review_only: {summary['labels_review_only'] or 'None'}",
        f"- labels_excluded_high_risk: {summary['labels_excluded_high_risk'] or 'None'}",
        f"- Window silver records: {summary['window_silver_records']}",
        f"- Pair silver records: {summary['pair_silver_records']}",
        "",
        "## Per Label",
        "",
    ]
    for label, stats in summary["per_label"].items():
        lines.append(f"- `{label}`: {stats}")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Use only `labels_trainable_by_default` for the balanced silver proxy baseline. Excluded high-risk labels can be reviewed by humans but should not be default training targets.",
        ]
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")

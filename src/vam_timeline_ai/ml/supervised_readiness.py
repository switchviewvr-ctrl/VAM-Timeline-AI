"""Readiness checks before any supervised semantic baseline."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def analyze_supervised_readiness(
    dataset: str | Path,
    labels: str | Path,
    split_plan: str | Path,
    out: str | Path,
    min_positive: int = 20,
    min_negative: int = 20,
    min_scenes: int = 3,
    min_samples: int = 5,
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    with np.load(dataset, allow_pickle=True) as data:
        label_names = [str(x) for x in data.get("manual_label_names", np.asarray([], dtype=object)).tolist()]
        pos = data.get("manual_y_positive_multilabel", np.zeros((len(data["window_ids"]), 0), dtype=np.int8))
        neg = data.get("manual_y_negative_multilabel", np.zeros((len(data["window_ids"]), 0), dtype=np.int8))
        unc = data.get("manual_y_uncertain_multilabel", np.zeros((len(data["window_ids"]), 0), dtype=np.int8))
        scenes = [str(x) for x in data["group_scene"].tolist()]
        samples = [str(x) for x in data["group_sample"].tolist()]
        include = data.get("include_for_ml", np.ones(len(scenes), dtype=bool)).astype(bool)
        conf = data.get("confidence", np.full(len(scenes), np.nan, dtype=np.float32))
    label_data = _load_yaml(Path(labels)) if Path(labels).exists() and "template" not in Path(labels).name.lower() else {}
    split = _load_json(Path(split_plan)) if Path(split_plan).exists() else {}
    per_label: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for idx, label in enumerate(label_names):
        positive_idx = np.where(pos[:, idx] > 0)[0]
        negative_idx = np.where(neg[:, idx] > 0)[0]
        uncertain_idx = np.where(unc[:, idx] > 0)[0]
        high_conf_positive = [i for i in positive_idx if include[i] and (np.isnan(conf[i]) or conf[i] >= min_confidence)]
        p_scenes = {scenes[i] for i in positive_idx}
        p_samples = {samples[i] for i in positive_idx}
        trainable = (
            len(positive_idx) >= min_positive
            and len(negative_idx) >= min_negative
            and len(p_scenes) >= min_scenes
            and len(p_samples) >= min_samples
            and len(high_conf_positive) >= min_positive
        )
        per_label[label] = {
            "positive": int(len(positive_idx)),
            "negative": int(len(negative_idx)),
            "uncertain": int(len(uncertain_idx)),
            "positive_scenes": int(len(p_scenes)),
            "positive_samples": int(len(p_samples)),
            "high_confidence_include_for_ml_positive": int(len(high_conf_positive)),
            "eligible_for_supervised_training": bool(trainable),
        }
        if trainable:
            eligible.append(label)
    summary = {
        "has_real_manual_labels": bool(label_names and int(pos.sum() + neg.sum() + unc.sum()) > 0),
        "manual_label_classes": len(label_names),
        "positive_assignments": int(pos.sum()),
        "negative_assignments": int(neg.sum()),
        "uncertain_assignments": int(unc.sum()),
        "unique_scenes": len(set(scenes)),
        "unique_samples": len(set(samples)),
        "split_plan_grouped": split.get("random_window_split_allowed") is False,
        "split_plan_can_train": bool(split.get("can_plan_supervised_split", False)),
        "eligible_labels": eligible,
        "per_label": per_label,
        "thresholds": {
            "min_positive": min_positive,
            "min_negative": min_negative,
            "min_scenes": min_scenes,
            "min_samples": min_samples,
            "min_confidence": min_confidence,
        },
        "manual_yaml_window_entries": len((label_data.get("windows", {}) or {}) if isinstance(label_data.get("windows", {}), dict) else {}),
    }
    _write_report(summary, out)
    return summary


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Supervised Readiness Report",
        "",
        f"- Real manual labels present: {summary['has_real_manual_labels']}",
        f"- Manual label classes: {summary['manual_label_classes']}",
        f"- Positive assignments: {summary['positive_assignments']}",
        f"- Negative assignments: {summary['negative_assignments']}",
        f"- Uncertain assignments: {summary['uncertain_assignments']}",
        f"- Unique scenes: {summary['unique_scenes']}",
        f"- Unique samples: {summary['unique_samples']}",
        f"- Split plan is grouped: {summary['split_plan_grouped']}",
        f"- Split plan can train: {summary['split_plan_can_train']}",
        f"- Eligible labels: {summary['eligible_labels'] or 'None'}",
        "",
        "## Per Label",
        "",
    ]
    if summary["per_label"]:
        for label, stats in summary["per_label"].items():
            lines.append(f"- `{label}`: {stats}")
    else:
        lines.append("- No real manual label classes found.")
    lines.extend([
        "",
        "## Recommendation",
        "",
        "Do not train supervised semantic classifiers unless at least one class is eligible and grouped train/validation/test coverage is possible.",
        "Prioritize manual labels for sparse classes, negative/control examples, and pair/contact windows.",
    ])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

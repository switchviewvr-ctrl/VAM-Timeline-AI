"""Leakage-aware ML readiness reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def analyze_ml_v1(dataset: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    scenes = [str(x) for x in data["group_scene"].tolist()]
    samples = [str(x) for x in data["group_sample"].tolist()]
    sources = [str(x) for x in data["group_source"].tolist()]
    manual_labels = [str(x) for x in data["manual_label_names"].tolist()]
    weak_labels = [str(x) for x in data["weak_label_names"].tolist()]
    weak_y = np.asarray(data["weak_y_multilabel"], dtype=np.int8)
    manual_y = np.asarray(data["manual_y_multilabel"], dtype=np.int8)
    _write_readiness(out / "ml_readiness_report_v1.md", X, scenes, samples, sources, manual_labels, weak_labels, manual_y, weak_y)
    _write_missingness(out / "feature_missingness_report_v1.md", X, feature_names)
    _write_distribution(out / "scene_sample_distribution_report_v1.md", scenes, samples)
    _write_weak_distribution(out / "weak_label_distribution_report_v1.md", weak_labels, weak_y)
    return {"rows": int(X.shape[0]), "features": int(X.shape[1]), "manual_labels": len(manual_labels), "weak_labels": len(weak_labels)}


def _write_readiness(path: Path, X, scenes, samples, sources, manual_labels, weak_labels, manual_y, weak_y) -> None:
    usable = int(np.sum(np.isfinite(X).any(axis=1)))
    scene_counts = Counter(scenes)
    sample_counts = Counter(samples)
    manual_positive = int(np.sum(manual_y)) if manual_y.size else 0
    lines = [
        "# ML Readiness Report v1",
        "",
        f"- Windows: {len(X)}",
        f"- Numeric-feature windows: {usable}",
        f"- Unique scenes: {len(scene_counts)}",
        f"- Unique samples: {len(sample_counts)}",
        f"- Real manual label classes: {len(manual_labels)}",
        f"- Real manual label assignments: {manual_positive}",
        f"- Weak label classes: {len(weak_labels)}",
        f"- Dataset likely highly correlated: True",
        f"- Random window splits valid: False",
        f"- Can clustering be explored: {usable >= 100}",
        f"- Can supervised semantic ML start: {len(manual_labels) >= 2 and manual_positive >= 100}",
        "",
        "## Why Random Window Splits Are Invalid",
        "",
        "Overlapping windows from the same sample and scene are highly correlated. Future train/test splits must group by scene, sample, or source.",
        "",
        "## Dominant Scenes",
        "",
    ]
    for scene, count in scene_counts.most_common(12):
        lines.append(f"- `{scene}`: {count} windows")
    lines.extend(["", "## What To Label Next", "", "Label representative review-queue windows across clusters and scenes before supervised ML."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_missingness(path: Path, X, feature_names) -> None:
    lines = ["# Feature Missingness Report v1", ""]
    for idx, name in sorted(enumerate(feature_names), key=lambda item: np.mean(~np.isfinite(X[:, item[0]])), reverse=True):
        lines.append(f"- `{name}`: {np.mean(~np.isfinite(X[:, idx])):.1%} missing")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_distribution(path: Path, scenes, samples) -> None:
    lines = ["# Scene/Sample Distribution Report v1", "", "## Scenes", ""]
    for name, count in Counter(scenes).most_common(30):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Samples", ""])
    for name, count in Counter(samples).most_common(30):
        lines.append(f"- `{name}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_weak_distribution(path: Path, weak_labels, weak_y) -> None:
    lines = ["# Weak Label Distribution Report v1", "", "Weak labels are not ground truth.", ""]
    if weak_y.size:
        counts = weak_y.sum(axis=0)
        for label, count in sorted(zip(weak_labels, counts, strict=False), key=lambda item: int(item[1]), reverse=True):
            lines.append(f"- `{label}`: {int(count)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

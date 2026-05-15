"""Readiness analysis for silver-supervised proxy experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl


def analyze_silver_readiness(
    dataset: str | Path,
    silver_labels: str | Path,
    out: str | Path,
    min_positive: int = 100,
    min_scenes: int = 5,
    min_samples: int = 20,
) -> dict[str, Any]:
    with np.load(dataset, allow_pickle=True) as data:
        label_names = [str(x) for x in data.get("silver_label_names", np.asarray([], dtype=object)).tolist()]
        silver_pos = data.get("silver_y_positive_multilabel", np.zeros((len(data["window_ids"]), 0), dtype=np.int8))
        silver_neg = data.get("silver_y_negative_multilabel", np.zeros((len(data["window_ids"]), 0), dtype=np.int8))
        scenes = [str(x) for x in data["group_scene"].tolist()]
        samples = [str(x) for x in data["group_sample"].tolist()]
        confidence = data.get("silver_confidence", np.zeros_like(silver_pos, dtype=np.float32))
    rows = load_jsonl(silver_labels)
    per_label: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for idx, label in enumerate(label_names):
        pos_idx = np.where(silver_pos[:, idx] > 0)[0]
        neg_idx = np.where(silver_neg[:, idx] > 0)[0]
        label_scenes = {scenes[i] for i in pos_idx}
        label_samples = {samples[i] for i in pos_idx}
        conf_values = confidence[pos_idx, idx] if len(pos_idx) else np.asarray([], dtype=np.float32)
        ready = len(pos_idx) >= min_positive and len(label_scenes) >= min_scenes and len(label_samples) >= min_samples
        per_label[label] = {
            "silver_positive": int(len(pos_idx)),
            "silver_negative": int(len(neg_idx)),
            "positive_scenes": int(len(label_scenes)),
            "positive_samples": int(len(label_samples)),
            "mean_confidence": float(np.nanmean(conf_values)) if len(conf_values) else None,
            "eligible_for_proxy_training": bool(ready),
            "likely_rule_artifact": True,
        }
        if ready:
            eligible.append(label)
    scene_counts = Counter(scenes)
    sample_counts = Counter(samples)
    summary = {
        "silver_records": len(rows),
        "silver_label_classes": len(label_names),
        "eligible_labels": eligible,
        "per_label": per_label,
        "thresholds": {"min_positive": min_positive, "min_scenes": min_scenes, "min_samples": min_samples},
        "dominant_scenes": scene_counts.most_common(10),
        "dominant_samples": sample_counts.most_common(10),
        "can_train_proxy_baseline": bool(eligible),
        "is_true_semantic_validation": False,
    }
    _write_report(summary, out)
    return summary


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Silver Readiness Report",
        "",
        "Silver labels are machine-generated from numeric feature rules. Training on them is weak supervision / rule-proxy imitation, not human semantic validation.",
        "",
        f"- Silver records: {summary['silver_records']}",
        f"- Silver label classes: {summary['silver_label_classes']}",
        f"- Eligible proxy labels: {summary['eligible_labels'] or 'None'}",
        f"- True semantic validation: {summary['is_true_semantic_validation']}",
        "",
        "## Per Label",
        "",
    ]
    if summary["per_label"]:
        for label, stats in summary["per_label"].items():
            lines.append(f"- `{label}`: {stats}")
    else:
        lines.append("- No silver label classes found.")
    lines.extend(["", "## Dominant Scenes", ""])
    lines.extend(f"- `{scene}`: {count}" for scene, count in summary["dominant_scenes"])
    lines.extend(["", "## Dominant Samples", ""])
    lines.extend(f"- `{sample}`: {count}" for sample, count in summary["dominant_samples"])
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use silver labels to prioritize review and test whether feature representations are learnable. Do not report silver-model metrics as semantic accuracy.",
        ]
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")

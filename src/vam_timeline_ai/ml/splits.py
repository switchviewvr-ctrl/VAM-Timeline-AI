"""Leakage-safe split planning for future supervised ML."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vam_timeline_ai.io.json_utils import dump_json


def plan_ml_splits_v1(dataset: str | Path, labels: str | Path, out: str | Path, report: str | Path) -> dict[str, Any]:
    with np.load(dataset, allow_pickle=True) as data:
        window_ids = [str(x) for x in data["window_ids"].tolist()]
        scenes = [str(x) for x in data["group_scene"].tolist()]
        samples = [str(x) for x in data["group_sample"].tolist()]
        sources = [str(x) for x in data["group_source"].tolist()]
    label_data = _load_yaml(Path(labels)) if Path(labels).exists() and "template" not in Path(labels).name.lower() else {}
    manual_by_window = label_data.get("windows", {}) if isinstance(label_data.get("windows", {}), dict) else {}
    label_to_groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"scene": set(), "sample": set(), "source": set()})
    lookup = {wid: idx for idx, wid in enumerate(window_ids)}
    for wid, entry in manual_by_window.items():
        idx = lookup.get(wid)
        if idx is None:
            continue
        for label in entry.get("labels", []) or []:
            label_to_groups[label]["scene"].add(scenes[idx])
            label_to_groups[label]["sample"].add(samples[idx])
            label_to_groups[label]["source"].add(sources[idx])
    plan = {
        "row_count": len(window_ids),
        "unique_scenes": len(set(scenes)),
        "unique_samples": len(set(samples)),
        "unique_sources": len(set(sources)),
        "random_window_split_allowed": False,
        "recommended_strategy": "grouped split by scene first; fallback grouped by sample/source only for exploratory analysis",
        "scene_group_sizes": dict(Counter(scenes).most_common()),
        "sample_group_sizes_top": dict(Counter(samples).most_common(50)),
        "manual_label_group_coverage": {
            label: {key: len(value) for key, value in groups.items()} for label, groups in label_to_groups.items()
        },
        "can_plan_supervised_split": bool(label_to_groups) and all(len(groups["scene"]) >= 2 and len(groups["sample"]) >= 3 for groups in label_to_groups.values()),
        "warnings": [
            "Random window split is invalid because overlapping windows leak motion.",
            "Do not train supervised semantic classifiers until real manual labels cover multiple scenes and samples with negative/control examples.",
        ],
    }
    dump_json(out, plan)
    _write_report(plan, report)
    return plan


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_report(plan: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Split Plan v1",
        "",
        f"- Rows: {plan['row_count']}",
        f"- Unique scenes: {plan['unique_scenes']}",
        f"- Unique samples: {plan['unique_samples']}",
        f"- Unique sources: {plan['unique_sources']}",
        f"- Random window split allowed: {plan['random_window_split_allowed']}",
        f"- Can plan supervised split now: {plan['can_plan_supervised_split']}",
        f"- Recommended strategy: {plan['recommended_strategy']}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {w}" for w in plan["warnings"])
    lines.extend(["", "## Manual Label Group Coverage", ""])
    if plan["manual_label_group_coverage"]:
        for label, coverage in plan["manual_label_group_coverage"].items():
            lines.append(f"- `{label}`: {coverage}")
    else:
        lines.append("- No real manual labels found.")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")


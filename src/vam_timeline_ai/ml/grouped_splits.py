"""Leakage-aware grouped splits for Cowgirl ML v1."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl


def split_cowgirl_ml_dataset_v1(
    feature_table: str | Path,
    metadata: str | Path,
    out_dir: str | Path,
    group_by: str = "source_scene_file",
    seed: int = 42,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_jsonl(metadata)
    with np.load(feature_table, allow_pickle=True) as data:
        y = data["y"]
        labels = [str(x) for x in data["label_names"].tolist()]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(meta):
        group = str(row.get(group_by) or row.get("sample_id") or row.get("window_id") or f"row_{i}")
        groups[group].append(i)
    group_names = list(groups)
    random.Random(seed).shuffle(group_names)
    if len(group_names) < 3:
        split_map = {"train": group_names, "val": [], "test": []}
        blocked = True
        warnings = ["Fewer than 3 groups; held-out metrics are not reliable."]
    else:
        n = len(group_names)
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.15)) if n >= 8 else 0
        split_map = {
            "test": group_names[:n_test],
            "val": group_names[n_test : n_test + n_val],
            "train": group_names[n_test + n_val :],
        }
        blocked = False
        warnings = []
    indices = {name: sorted(i for group in group_list for i in groups[group]) for name, group_list in split_map.items()}
    for name, idxs in indices.items():
        (out / f"{name}_indices.json").write_text(json.dumps(idxs, indent=2), encoding="utf-8")
    summary = {
        "schema": "cowgirl_grouped_split_v1",
        "status": "limited" if blocked else "ok",
        "group_by": group_by,
        "seed": seed,
        "row_count": len(meta),
        "group_count": len(groups),
        "split_counts": {k: len(v) for k, v in indices.items()},
        "split_group_counts": {k: len(v) for k, v in split_map.items()},
        "label_counts_by_split": _label_counts(y, labels, indices),
        "scene_counts_by_split": _group_counts(meta, indices, "source_scene_file"),
        "leakage_warnings": _leakage_warnings(meta, indices, group_by),
        "warnings": warnings,
        "random_window_split_allowed": False,
    }
    dump_json(out / "split_summary.json", summary)
    _write_report(out / "split_report.md", summary)
    return summary


def _label_counts(y: np.ndarray, labels: list[str], indices: dict[str, list[int]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, idxs in indices.items():
        out[split] = {}
        for j, label in enumerate(labels):
            vals = [int(y[i, j]) for i in idxs if int(y[i, j]) in {0, 1}]
            out[split][label] = dict(Counter(vals))
    return out


def _group_counts(meta: list[dict[str, Any]], indices: dict[str, list[int]], key: str) -> dict[str, int]:
    return {split: len({str(meta[i].get(key) or "") for i in idxs if meta[i].get(key)}) for split, idxs in indices.items()}


def _leakage_warnings(meta: list[dict[str, Any]], indices: dict[str, list[int]], group_by: str) -> list[str]:
    seen: dict[str, str] = {}
    warnings = []
    for split, idxs in indices.items():
        for i in idxs:
            group = str(meta[i].get(group_by) or "")
            if not group:
                continue
            if group in seen and seen[group] != split:
                warnings.append(f"group `{group}` appears in both `{seen[group]}` and `{split}`")
            seen[group] = split
    return warnings


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Cowgirl ML Grouped Split v1",
        "",
        f"- Status: {summary['status']}",
        f"- Group by: `{summary['group_by']}`",
        f"- Rows: {summary['row_count']}",
        f"- Groups: {summary['group_count']}",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Split group counts: `{summary['split_group_counts']}`",
        f"- Random window split allowed: `{summary['random_window_split_allowed']}`",
        f"- Leakage warnings: `{summary['leakage_warnings']}`",
        "",
        "## Label Counts",
        "",
        f"`{summary['label_counts_by_split']}`",
    ]
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in summary["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

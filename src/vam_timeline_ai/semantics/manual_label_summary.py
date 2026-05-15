"""Summarize real manual labels without using weak-label hints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl


def summarize_manual_labels(labels: str | Path, windows: str | Path, pair_windows: str | Path, out: str | Path) -> dict[str, Any]:
    data = _load_yaml(Path(labels)) if Path(labels).exists() and "template" not in Path(labels).name.lower() else {}
    window_lookup = {r.get("window_id"): r for r in load_jsonl(windows)}
    pair_lookup = {r.get("pair_window_id"): r for r in load_jsonl(pair_windows)}
    window_entries = data.get("windows", {}) if isinstance(data.get("windows", {}), dict) else {}
    pair_entries = data.get("pair_windows", {}) if isinstance(data.get("pair_windows", {}), dict) else {}
    label_counts = Counter()
    negative_counts = Counter()
    uncertain_counts = Counter()
    role_counts = Counter()
    quality_counts = Counter()
    confidence_bins = Counter()
    include_for_ml_count = 0
    contact_counts = Counter()
    scene_counts = Counter()
    sample_counts = Counter()
    source_counts = Counter()
    empty_ignored = 0
    for wid, entry in window_entries.items():
        if _is_empty(entry):
            empty_ignored += 1
            continue
        for label in entry.get("labels", []) or []:
            label_counts[label] += 1
        for label in entry.get("negative_labels", []) or []:
            negative_counts[label] += 1
        for label in entry.get("uncertain_labels", []) or []:
            uncertain_counts[label] += 1
        role_counts[entry.get("semantic_role", "unknown")] += 1
        quality_counts[entry.get("movement_quality", "unknown")] += 1
        confidence_bins[_confidence_bin(entry.get("confidence"))] += 1
        if entry.get("include_for_ml") is True:
            include_for_ml_count += 1
        w = window_lookup.get(wid, {})
        scene_counts[w.get("source_scene_file", "unknown")] += 1
        sample_counts[w.get("sample_id", str(wid).split(":")[0])] += 1
        source_counts[w.get("source_id", "unknown")] += 1
    for _, entry in pair_entries.items():
        if _is_empty(entry):
            empty_ignored += 1
            continue
        for label in entry.get("contact_labels", []) or []:
            contact_counts[label] += 1
        for label in entry.get("pair_labels", []) or []:
            label_counts[label] += 1
    can_start = _can_start_supervised(label_counts, negative_counts, scene_counts, sample_counts)
    summary = {
        "total_labeled_windows": len(window_entries),
        "total_labeled_pair_windows": len(pair_entries),
        "empty_entries_ignored": empty_ignored,
        "labels_by_class": dict(label_counts),
        "negative_labels_by_class": dict(negative_counts),
        "uncertain_labels_by_class": dict(uncertain_counts),
        "roles_by_class": dict(role_counts),
        "movement_quality_distribution": dict(quality_counts),
        "confidence_distribution": dict(confidence_bins),
        "include_for_ml_true_count": include_for_ml_count,
        "contacts_by_class": dict(contact_counts),
        "labels_by_scene": dict(scene_counts),
        "labels_by_sample": dict(sample_counts),
        "labels_by_source": dict(source_counts),
        "classes_with_enough_examples": [label for label, count in label_counts.items() if count >= 20 and negative_counts.get(label, 0) >= 20],
        "sparse_classes": [label for label, count in label_counts.items() if count < 20],
        "supervised_ml_can_start": can_start,
    }
    _write_report(summary, out)
    return summary


def _can_start_supervised(labels: Counter[str], negatives: Counter[str], scenes: Counter[str], samples: Counter[str]) -> bool:
    return bool(sum(labels.values()) >= 100 and len(labels) >= 2 and len(scenes) >= 3 and len(samples) >= 10 and sum(negatives.values()) >= 20)


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Manual Label Summary",
        "",
        f"- Total labeled windows: {summary['total_labeled_windows']}",
        f"- Total labeled pair windows: {summary['total_labeled_pair_windows']}",
        f"- Empty entries ignored: {summary['empty_entries_ignored']}",
        f"- Include for ML true count: {summary['include_for_ml_true_count']}",
        f"- Supervised ML can start: {summary['supervised_ml_can_start']}",
        "",
        "## Labels By Class",
        "",
    ]
    for key, count in Counter(summary["labels_by_class"]).most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Negative Labels", ""])
    for key, count in Counter(summary["negative_labels_by_class"]).most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Roles", ""])
    for key, count in Counter(summary["roles_by_class"]).most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Contact Labels", ""])
    for key, count in Counter(summary["contacts_by_class"]).most_common():
        lines.append(f"- `{key}`: {count}")
    if not summary["contacts_by_class"]:
        lines.append("- None")
    lines.extend(["", "## Movement Quality", ""])
    for key, count in Counter(summary["movement_quality_distribution"]).most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Confidence", ""])
    for key, count in Counter(summary["confidence_distribution"]).most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Labels By Scene", ""])
    for key, count in Counter(summary["labels_by_scene"]).most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Trainability", ""])
    lines.append(f"- Classes with enough examples by simple threshold: {summary['classes_with_enough_examples'] or 'None'}")
    lines.append(f"- Sparse classes: {summary['sparse_classes'] or 'None'}")
    lines.append("- Supervised ML needs positives, negatives, multiple scenes/samples, and grouped evaluation.")
    lines.extend(["", "## Leakage Risk", "", "Random window splits are invalid. Use grouped scene/sample/source splits."])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _confidence_bin(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "missing"
    if v >= 0.8:
        return "0.8-1.0"
    if v >= 0.6:
        return "0.6-0.8"
    if v > 0:
        return "0.0-0.6"
    return "0.0"


def _is_empty(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return not bool(entry)
    labels = []
    for key in ["labels", "negative_labels", "uncertain_labels", "pair_labels", "contact_labels"]:
        labels.extend(entry.get(key, []) or [])
    return not labels and not str(entry.get("notes", "") or "").strip() and entry.get("confidence") in {0, 0.0, "0", "0.0", None, ""}

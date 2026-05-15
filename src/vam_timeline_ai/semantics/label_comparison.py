"""Compare machine/silver labels with real manual labels when available."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl


def compare_machine_labels_to_manual(manual_labels: str | Path, silver_labels: str | Path, out: str | Path) -> dict[str, Any]:
    manual_path = Path(manual_labels)
    silver_rows = load_jsonl(silver_labels)
    if not manual_path.exists() or "template" in manual_path.name.lower():
        summary = {
            "status": "no_manual_labels",
            "manual_windows": 0,
            "silver_windows": len(silver_rows),
            "overlap_windows": 0,
            "agreements": 0,
            "conflicts": 0,
        }
        _write_report(summary, [], [], out)
        return summary
    manual = _load_yaml(manual_path)
    manual_windows = manual.get("windows", {}) if isinstance(manual.get("windows", {}), dict) else {}
    silver_by_window = {row.get("window_id"): row for row in silver_rows if row.get("window_id")}
    agreements: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for window_id, entry in manual_windows.items():
        silver = silver_by_window.get(window_id)
        if not silver:
            continue
        manual_pos = set(entry.get("labels", []) or [])
        manual_neg = set(entry.get("negative_labels", []) or [])
        silver_pos = set(silver.get("positive_labels", []) or []) | set(silver.get("role_candidates", []) or []) | set(silver.get("contact_candidates", []) or [])
        silver_neg = set(silver.get("negative_labels", []) or [])
        for label in sorted(manual_pos & silver_pos):
            agreements.append({"window_id": window_id, "label": label, "type": "positive_agreement"})
        for label in sorted(manual_neg & silver_neg):
            agreements.append({"window_id": window_id, "label": label, "type": "negative_agreement"})
        for label in sorted(manual_neg & silver_pos):
            conflicts.append({"window_id": window_id, "label": label, "type": "manual_negative_vs_silver_positive"})
        for label in sorted(manual_pos & silver_neg):
            conflicts.append({"window_id": window_id, "label": label, "type": "manual_positive_vs_silver_negative"})
    summary = {
        "status": "compared",
        "manual_windows": len(manual_windows),
        "silver_windows": len(silver_rows),
        "overlap_windows": len(set(manual_windows) & set(silver_by_window)),
        "agreements": len(agreements),
        "conflicts": len(conflicts),
    }
    _write_report(summary, agreements, conflicts, out)
    return summary


def _write_report(summary: dict[str, Any], agreements: list[dict[str, Any]], conflicts: list[dict[str, Any]], out: str | Path) -> None:
    lines = [
        "# Machine vs Manual Label Comparison",
        "",
        f"- Status: {summary['status']}",
        f"- Manual window entries: {summary['manual_windows']}",
        f"- Silver window records: {summary['silver_windows']}",
        f"- Overlapping windows: {summary['overlap_windows']}",
        f"- Agreements: {summary['agreements']}",
        f"- Conflicts: {summary['conflicts']}",
        "",
    ]
    if summary["status"] == "no_manual_labels":
        lines.extend(
            [
                "No real manual labels were found, so semantic agreement cannot be measured yet.",
                "This is expected before a human edits and merges a review batch.",
            ]
        )
    else:
        lines.extend(["## Agreement Counts", ""])
        agreement_counts = Counter(item["label"] for item in agreements)
        lines.extend(f"- `{label}`: {count}" for label, count in agreement_counts.most_common() or [("None", 0)])
        lines.extend(["", "## Conflicts", ""])
        if conflicts:
            for item in conflicts[:100]:
                lines.append(f"- `{item['window_id']}` `{item['label']}`: {item['type']}")
        else:
            lines.append("- No direct positive/negative conflicts found.")
        lines.extend(["", "Machine/silver labels remain non-ground-truth even when they agree with manual labels."])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

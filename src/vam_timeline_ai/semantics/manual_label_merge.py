"""Safely merge edited review-batch labels into manual_labels.yaml."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.semantics.edited_label_batch import _is_empty_manual_entry, _weak_labels


def merge_manual_label_batch(base: str | Path, batch: str | Path, out: str | Path, backup: bool = True, report: str | Path | None = None) -> dict[str, Any]:
    base_path = Path(base)
    batch_path = Path(batch)
    out_path = Path(out)
    if not batch_path.exists():
        summary = {
            "status": "error",
            "base": str(base_path),
            "batch": str(batch_path),
            "out": str(out_path),
            "backup": None,
            "merged_windows": 0,
            "merged_pair_windows": 0,
            "skipped_empty": 0,
            "skipped_weak_labels": 0,
            "conflicts": [],
            "errors": [f"batch file does not exist: {batch_path}"],
        }
        if report:
            _write_report(summary, report)
        return summary
    base_data = _load_yaml(base_path) if base_path.exists() else {}
    batch_data = _load_yaml(batch_path)
    if backup and out_path.exists():
        backup_path = out_path.with_suffix(out_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        backup_path = None
    merged, merge_info = _merge(base_data, batch_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = {
        "status": "ok",
        "base": str(base_path),
        "batch": str(batch_path),
        "out": str(out_path),
        "backup": str(backup_path) if backup_path else None,
        "merged_windows": merge_info["merged_windows"],
        "merged_pair_windows": merge_info["merged_pair_windows"],
        "total_windows_after_merge": len(merged.get("windows", {})),
        "total_pair_windows_after_merge": len(merged.get("pair_windows", {})),
        "skipped_empty": merge_info["skipped_empty"],
        "skipped_weak_labels": merge_info["skipped_weak_labels"],
        "conflicts": merge_info["conflicts"],
        "errors": [],
    }
    if report:
        _write_report(summary, report)
    return summary


def _merge(base: dict[str, Any], batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(base or {})
    for section in ["scenes", "actors", "samples", "windows", "pair_windows"]:
        out.setdefault(section, {})
    info = {"merged_windows": 0, "merged_pair_windows": 0, "skipped_empty": 0, "skipped_weak_labels": 0, "conflicts": []}
    for section in ["scenes", "actors", "samples"]:
        for key, value in (batch.get(section, {}) or {}).items():
            if _is_empty_manual_entry(value):
                info["skipped_empty"] += 1
                continue
            existing = out[section].get(key, {})
            if existing and existing != value:
                info["conflicts"].append(f"{section}:{key}")
            out[section][key] = _deep_merge(existing, value)
    for section in ["windows", "pair_windows"]:
        for key, value in (batch.get(section, {}) or {}).items():
            if _is_empty_manual_entry(value):
                info["skipped_empty"] += 1
                continue
            if _weak_labels(value):
                info["skipped_weak_labels"] += 1
                continue
            existing = out[section].get(key, {})
            if existing and existing != value:
                info["conflicts"].append(f"{section}:{key}")
            out[section][key] = _deep_merge(existing, value)
            if section == "windows":
                info["merged_windows"] += 1
            else:
                info["merged_pair_windows"] += 1
    return out, info


def _deep_merge(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for key, value in b.items():
            if _is_empty_manual_entry(value):
                continue
            out[key] = _deep_merge(out.get(key), value)
        return out
    return b


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Manual Label Merge Report",
        "",
        f"- Status: {summary.get('status')}",
        f"- Base: `{summary['base']}`",
        f"- Batch: `{summary['batch']}`",
        f"- Output: `{summary['out']}`",
        f"- Backup: `{summary['backup']}`",
        f"- Newly merged window entries: {summary['merged_windows']}",
        f"- Newly merged pair-window entries: {summary['merged_pair_windows']}",
        f"- Total window entries after merge: {summary.get('total_windows_after_merge', 0)}",
        f"- Total pair-window entries after merge: {summary.get('total_pair_windows_after_merge', 0)}",
        f"- Empty/default entries ignored: {summary.get('skipped_empty', 0)}",
        f"- Entries skipped for weak labels: {summary.get('skipped_weak_labels', 0)}",
        f"- Conflicts reported: {len(summary.get('conflicts', []))}",
        "",
        "Empty/default stub entries were ignored. Weak labels were not imported as manual labels.",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {e}" for e in summary.get("errors", [])] or ["- None"])
    lines.extend(["", "## Conflicts", ""])
    lines.extend([f"- {c}" for c in summary.get("conflicts", [])] or ["- None"])
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

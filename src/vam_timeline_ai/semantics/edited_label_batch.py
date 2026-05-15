"""Inspect edited review-batch labels before merging them."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl


LABEL_FIELDS = ("labels", "negative_labels", "uncertain_labels", "pair_labels", "contact_labels")


def inspect_edited_label_batch(
    stub: str | Path,
    edited: str | Path,
    windows: str | Path,
    pair_windows: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    stub_path = Path(stub)
    edited_path = Path(edited)
    window_ids = {str(r.get("window_id")) for r in load_jsonl(windows) if r.get("window_id")}
    pair_window_ids = {str(r.get("pair_window_id")) for r in load_jsonl(pair_windows) if r.get("pair_window_id")}

    errors: list[str] = []
    warnings: list[str] = []
    if not stub_path.exists():
        warnings.append(f"stub file is missing: {stub_path}")
    if not edited_path.exists():
        errors.append(f"edited label batch is missing: {edited_path}")
        result = _result(stub_path, edited_path, False, 0, 0, 0, errors, warnings)
        _write_report(result, out)
        return result
    if edited_path.name.lower().endswith("stub.yaml") or "template" in edited_path.name.lower():
        errors.append("edited path looks like a stub/template, not a human-edited label batch")
    if stub_path.exists() and _sha256(stub_path) == _sha256(edited_path):
        errors.append("edited label batch is byte-identical to the stub")

    data = _load_yaml(edited_path)
    window_entries = data.get("windows", {}) if isinstance(data.get("windows", {}), dict) else {}
    pair_entries = data.get("pair_windows", {}) if isinstance(data.get("pair_windows", {}), dict) else {}
    usable = 0
    empty = 0
    weak_label_entries = 0

    for section, entries, known_ids, kind in [
        ("windows", window_entries, window_ids, "window_id"),
        ("pair_windows", pair_entries, pair_window_ids, "pair_window_id"),
    ]:
        for item_id, entry in entries.items():
            item_id = str(item_id)
            if item_id not in known_ids:
                errors.append(f"{section} entry references unknown or stale {kind}: {item_id}")
            if _is_empty_manual_entry(entry):
                empty += 1
                continue
            weak = _weak_labels(entry)
            if weak:
                weak_label_entries += 1
                errors.append(f"{section} entry `{item_id}` contains weak labels pasted as manual labels: {weak}")
                continue
            usable += 1

    if usable == 0:
        errors.append("edited label batch contains no usable non-empty manual entries")
    safe_to_merge = not errors and usable > 0
    result = _result(stub_path, edited_path, safe_to_merge, usable, empty, weak_label_entries, errors, warnings)
    result["total_window_entries"] = len(window_entries)
    result["total_pair_window_entries"] = len(pair_entries)
    _write_report(result, out)
    return result


def _result(
    stub: Path,
    edited: Path,
    safe_to_merge: bool,
    usable: int,
    empty: int,
    weak_label_entries: int,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "stub": str(stub),
        "edited": str(edited),
        "safe_to_merge": safe_to_merge,
        "usable_edited_entries": usable,
        "empty_default_entries_ignored": empty,
        "entries_with_weak_labels": weak_label_entries,
        "errors": errors,
        "warnings": warnings,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_empty_manual_entry(value: Any) -> bool:
    if not isinstance(value, dict):
        return not bool(value)
    labels: list[Any] = []
    for key in LABEL_FIELDS:
        labels.extend(value.get(key, []) or [])
    confidence = value.get("confidence")
    notes = str(value.get("notes", "") or "").strip()
    role = value.get("semantic_role", "unknown")
    focus_actor = value.get("focus_actor", "unknown")
    include = value.get("include_for_ml", False)
    quality = value.get("movement_quality", "questionable")
    pair_refs = [value.get("rider_window_id"), value.get("receiver_window_id"), value.get("rider_atom_id"), value.get("receiver_atom_id")]
    return (
        not labels
        and not notes
        and all(not str(ref or "").strip() for ref in pair_refs)
        and confidence in {0, 0.0, "0", "0.0", None, ""}
        and role in {"unknown", None, ""}
        and focus_actor in {"unknown", None, ""}
        and quality in {"questionable", None, ""}
        and include is False
    )


def _weak_labels(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    found: list[str] = []
    for key in LABEL_FIELDS:
        for label in value.get(key, []) or []:
            if str(label).startswith("weak_"):
                found.append(str(label))
    return sorted(set(found))


def _write_report(result: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Edited Label Batch Inspection",
        "",
        f"- Stub: `{result['stub']}`",
        f"- Edited: `{result['edited']}`",
        f"- Safe to merge: {result['safe_to_merge']}",
        f"- Usable edited entries: {result['usable_edited_entries']}",
        f"- Empty/default entries ignored: {result['empty_default_entries_ignored']}",
        f"- Entries containing weak labels: {result['entries_with_weak_labels']}",
        f"- Total window entries: {result.get('total_window_entries', 0)}",
        f"- Total pair-window entries: {result.get('total_pair_window_entries', 0)}",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {e}" for e in result["errors"]] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {w}" for w in result["warnings"]] or ["- None"])
    lines.extend([
        "",
        "## Notes",
        "",
        "Weak labels are review hints only and are not accepted as manual labels.",
        "Unknown IDs are treated as stale or non-clean-run references.",
    ])
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

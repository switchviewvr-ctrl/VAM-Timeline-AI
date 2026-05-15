"""Discover the latest local review batch for a clean run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl
from vam_timeline_ai.semantics.edited_label_batch import _is_empty_manual_entry, _weak_labels


BATCH_RE = re.compile(r"^batch_(\d+)$")


def find_latest_review_batch(run_dir: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    batches_dir = run_path / "labels" / "batches"
    batch_infos = [_inspect_batch(path) for path in sorted(batches_dir.glob("batch_*")) if path.is_dir()]
    valid = [b for b in batch_infos if b["has_review_batch"] and b["has_stub"] and b["batch_number"] is not None]
    selected = None
    if valid:
        selected = sorted(valid, key=lambda b: (b["batch_number"], b["has_previews_index"]))[-1]
    status = "no_valid_batch"
    if selected:
        status = "ready_for_ingestion" if selected["has_edited"] and selected["usable_edited_entries"] > 0 and not selected["errors"] else "waiting_for_human_labels"
    result = {
        "run_dir": str(run_path),
        "batches_dir": str(batches_dir),
        "status": status,
        "latest_batch": selected,
        "batches": batch_infos,
    }
    if out:
        out_path = Path(out)
        json_path = out_path.with_name("latest_review_batch.json")
        dump_json(json_path, result)
        _write_report(result, out_path)
    return result


def _inspect_batch(path: Path) -> dict[str, Any]:
    match = BATCH_RE.match(path.name)
    review = path / "review_batch.jsonl"
    stub = path / "manual_labels.stub.yaml"
    edited = path / "manual_labels.edited.yaml"
    preview_index = path / "previews" / "index.html"
    manifest = path / "previews" / "preview_manifest.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    rows = load_jsonl(review) if review.exists() else []
    manifest_rows = load_jsonl(manifest) if manifest.exists() else []
    usable = 0
    empty = 0
    weak_entries = 0
    if edited.exists():
        data = _load_yaml(edited)
        for section in ["windows", "pair_windows"]:
            entries = data.get(section, {}) if isinstance(data.get(section, {}), dict) else {}
            for item_id, entry in entries.items():
                if _is_empty_manual_entry(entry):
                    empty += 1
                    continue
                weak = _weak_labels(entry)
                if weak:
                    weak_entries += 1
                    errors.append(f"{section} `{item_id}` contains weak labels: {weak}")
                    continue
                usable += 1
    if not review.exists():
        warnings.append("missing review_batch.jsonl")
    if not stub.exists():
        warnings.append("missing manual_labels.stub.yaml")
    if review.exists() and not rows:
        warnings.append("review_batch.jsonl is empty")
    return {
        "batch_name": path.name,
        "batch_number": int(match.group(1)) if match else None,
        "path": str(path),
        "has_review_batch": review.exists(),
        "has_stub": stub.exists(),
        "has_edited": edited.exists(),
        "has_previews_index": preview_index.exists(),
        "review_row_count": len(rows),
        "preview_manifest_count": len(manifest_rows),
        "usable_edited_entries": usable,
        "empty_edited_entries": empty,
        "edited_entries_with_weak_labels": weak_entries,
        "appears_label_ready": review.exists() and stub.exists() and bool(rows) and preview_index.exists(),
        "appears_already_edited": edited.exists() and usable > 0,
        "errors": errors,
        "warnings": warnings,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_report(result: dict[str, Any], out: Path) -> None:
    lines = [
        "# Latest Review Batch Report",
        "",
        f"- Run dir: `{result['run_dir']}`",
        f"- Status: {result['status']}",
        "",
    ]
    latest = result.get("latest_batch")
    if latest:
        lines.extend([
            "## Selected Batch",
            "",
            f"- Batch: `{latest['batch_name']}`",
            f"- Path: `{latest['path']}`",
            f"- Review rows: {latest['review_row_count']}",
            f"- Has previews/index.html: {latest['has_previews_index']}",
            f"- Has edited labels: {latest['has_edited']}",
            f"- Usable edited entries: {latest['usable_edited_entries']}",
            "",
        ])
    else:
        lines.extend(["## Selected Batch", "", "- None", ""])
    lines.extend(["## All Batches", ""])
    for batch in result["batches"]:
        lines.extend([
            f"### `{batch['batch_name']}`",
            "",
            f"- Number: {batch['batch_number']}",
            f"- Has review_batch.jsonl: {batch['has_review_batch']}",
            f"- Has manual_labels.stub.yaml: {batch['has_stub']}",
            f"- Has manual_labels.edited.yaml: {batch['has_edited']}",
            f"- Has previews/index.html: {batch['has_previews_index']}",
            f"- Review rows: {batch['review_row_count']}",
            f"- Preview manifest rows: {batch['preview_manifest_count']}",
            f"- Usable edited entries: {batch['usable_edited_entries']}",
            f"- Label-ready: {batch['appears_label_ready']}",
            f"- Already edited: {batch['appears_already_edited']}",
            "",
        ])
        if batch["warnings"]:
            lines.append("Warnings:")
            lines.extend(f"- {w}" for w in batch["warnings"])
            lines.append("")
        if batch["errors"]:
            lines.append("Errors:")
            lines.extend(f"- {e}" for e in batch["errors"])
            lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

"""Markdown report writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_raw_scan_report(index: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    totals = index.get("totals", {})
    files = index.get("files", [])

    lines = [
        "# Raw VaM JSON Scan",
        "",
        "This is a lightweight technical scan. It does not bake motion, infer semantic actor roles, export Timeline clips, run VaM, or train ML.",
        "",
        "## Summary",
        "",
        f"- Raw folder: `{index.get('raw_dir')}`",
        f"- Total JSON files: {totals.get('total_json_files', 0)}",
        f"- Parsed JSON files: {totals.get('parsed_json_files', 0)}",
        f"- Parse failures: {totals.get('parse_failures', 0)}",
        f"- Likely VaM scenes: {totals.get('vam_scenes', 0)}",
        f"- External Timeline exports: {totals.get('external_timeline_exports', 0)}",
        f"- Scenes with native tracks: {totals.get('with_native_motion_tracks', 0)}",
        f"- Files with Timeline markers: {totals.get('with_timeline', 0)}",
        "",
        "## Files",
        "",
        "| File | Kind | Persons | Native Tracks | Timeline Controller Clips | Timeline FloatParam Clips | Trigger-only Clips | Filename Hints |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for item in files:
        hints = ", ".join(item.get("likely_filename_tags", []))
        lines.append(
            "| {file} | {kind} | {persons} | {native} | {controllers} | {floats} | {triggers} | {hints} |".format(
                file=item.get("file_name", ""),
                kind=item.get("json_kind", ""),
                persons=item.get("person_atoms_count", 0),
                native=item.get("native_track_count", 0),
                controllers=item.get("timeline_controller_clip_count", 0),
                floats=item.get("timeline_floatparam_clip_count", 0),
                triggers=item.get("timeline_trigger_only_clip_count", 0),
                hints=hints,
            )
        )

    warnings = [
        {"file": item.get("file_name"), "warnings": item.get("warnings", [])}
        for item in files
        if item.get("warnings")
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for item in warnings:
            for warning in item["warnings"]:
                lines.append(f"- `{item['file']}`: {warning}")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Use this scan to choose which scenes deserve technical extraction and manual semantic review. Do not treat filename hints or atom IDs as actor-role truth.",
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")

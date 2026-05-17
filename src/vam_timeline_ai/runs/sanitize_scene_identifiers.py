"""Neutralize local scene identifiers in generated run artifacts.

This is intended for ignored ``data/runs`` artifacts before sharing a review
package or database excerpt. It keeps stable aliases while moving the original
scene names into a local private alias map.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv
import json
import re
from json import JSONDecodeError

from vam_timeline_ai.io.json_utils import dump_json, load_json, safe_id_for_path


SCENE_KEY_HINTS = (
    "scene_file",
    "scene_path",
    "source_scene",
    "raw_scene",
    "scene_name",
    "scene_relative_path",
)


def sanitize_run_scene_identifiers(
    run_dir: str | Path,
    alias_map_out: str | Path,
    report: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(run)
    alias_path = Path(alias_map_out)
    aliases: dict[str, str] = {}
    existing = _load_existing_aliases(alias_path)
    aliases.update(existing)

    changed_files: list[str] = []
    skipped_files: list[str] = []
    replacements = 0
    for path in _iter_artifact_files(run):
        suffix = path.suffix.lower()
        before = len(aliases)
        try:
            if suffix == ".jsonl":
                changed, count = _sanitize_jsonl(path, aliases, dry_run)
            elif suffix == ".json":
                changed, count = _sanitize_json(path, aliases, dry_run)
            elif suffix == ".csv":
                changed, count = _sanitize_csv(path, aliases, dry_run)
            else:
                continue
        except (JSONDecodeError, UnicodeDecodeError, csv.Error):
            skipped_files.append(str(path))
            continue
        if changed:
            changed_files.append(str(path))
            replacements += count
        if len(aliases) != before and dry_run:
            # Dry runs should still report prospective aliases but not mutate.
            pass

    if not dry_run:
        dump_json(alias_path, {"schema": "scene_identifier_alias_map_v1", "aliases": aliases})
    _write_report(Path(report), run, alias_path, changed_files, skipped_files, aliases, replacements, dry_run)
    return {
        "status": "ok",
        "run_dir": str(run),
        "changed_files": len(changed_files),
        "skipped_files": len(skipped_files),
        "scene_aliases": len(aliases),
        "replacements": replacements,
        "alias_map": str(alias_path),
        "report": str(report),
        "dry_run": dry_run,
    }


def _iter_artifact_files(run: Path) -> list[Path]:
    ignored_names = {"local_scene_aliases.private.json"}
    return [
        p
        for p in run.rglob("*")
        if p.is_file()
        and p.name not in ignored_names
        and p.suffix.lower() in {".jsonl", ".json", ".csv"}
        and ".private." not in p.name
    ]


def _load_existing_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = load_json(path)
    aliases = data.get("aliases") if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in (aliases or {}).items()}


def _sanitize_jsonl(path: Path, aliases: dict[str, str], dry_run: bool) -> tuple[bool, int]:
    rows: list[dict[str, Any]] = []
    changed = False
    count = 0
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            new_row, row_changed, row_count = _sanitize_value(row, aliases)
            rows.append(new_row)
            changed = changed or row_changed
            count += row_count
    if changed and not dry_run:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
    return changed, count


def _sanitize_json(path: Path, aliases: dict[str, str], dry_run: bool) -> tuple[bool, int]:
    data = load_json(path)
    new_data, changed, count = _sanitize_value(data, aliases)
    if changed and not dry_run:
        dump_json(path, new_data)
    return changed, count


def _sanitize_csv(path: Path, aliases: dict[str, str], dry_run: bool) -> tuple[bool, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    changed = False
    count = 0
    for row in rows:
        for key in fields:
            if _is_scene_key(key) and row.get(key):
                original = str(row[key])
                row[key] = _alias_for(original, aliases)
                changed = changed or row[key] != original
                count += 1 if row[key] != original else 0
    if changed and not dry_run:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return changed, count


def _sanitize_value(value: Any, aliases: dict[str, str], key: str = "") -> tuple[Any, bool, int]:
    if isinstance(value, dict):
        changed = False
        count = 0
        out: dict[str, Any] = {}
        for k, v in value.items():
            new_v, child_changed, child_count = _sanitize_value(v, aliases, str(k))
            out[k] = new_v
            changed = changed or child_changed
            count += child_count
        return out, changed, count
    if isinstance(value, list):
        changed = False
        count = 0
        out = []
        for item in value:
            new_item, child_changed, child_count = _sanitize_value(item, aliases, key)
            out.append(new_item)
            changed = changed or child_changed
            count += child_count
        return out, changed, count
    if isinstance(value, str) and _is_scene_key(key) and value.strip():
        alias = _alias_for(value, aliases)
        return alias, alias != value, 1 if alias != value else 0
    return value, False, 0


def _is_scene_key(key: str) -> bool:
    lower = key.lower()
    return any(hint in lower for hint in SCENE_KEY_HINTS)


def _alias_for(original: str, aliases: dict[str, str]) -> str:
    text = str(original)
    if _looks_like_public_scene_alias(text):
        return text
    if text in aliases:
        return aliases[text]
    index = len(aliases) + 1
    suffix = Path(text).suffix or ".json"
    if len(suffix) > 12 or any(ch in suffix for ch in {"/", "\\"}):
        suffix = ".json"
    alias = f"scene_{index:06d}{suffix}"
    aliases[text] = alias
    return alias


def _looks_like_public_scene_alias(text: str) -> bool:
    return re.match(r"^scene_\d{6}\.[A-Za-z0-9]+$", text.strip()) is not None


def _write_report(path: Path, run: Path, alias_path: Path, changed_files: list[str], skipped_files: list[str], aliases: dict[str, str], replacements: int, dry_run: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_suffix = Counter(Path(alias).suffix for alias in aliases.values())
    lines = [
        "# Scene Identifier Sanitization",
        "",
        f"- Run: `{run}`",
        f"- Dry run: `{dry_run}`",
        f"- Changed files: `{len(changed_files)}`",
        f"- Skipped unreadable files: `{len(skipped_files)}`",
        f"- Replacement count: `{replacements}`",
        f"- Scene aliases: `{len(aliases)}`",
        f"- Alias suffixes: `{dict(by_suffix)}`",
        f"- Private alias map: `{alias_path}`",
        "",
        "Original local scene names are intentionally kept out of public databases.",
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in changed_files[:500])
    if len(changed_files) > 500:
        lines.append(f"- ... {len(changed_files) - 500} more")
    if skipped_files:
        lines.extend(["", "## Skipped Files", ""])
        lines.extend(f"- `{path}`" for path in skipped_files[:500])
        if len(skipped_files) > 500:
            lines.append(f"- ... {len(skipped_files) - 500} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Import handmade labeled Timeline reference animations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv
import json
import zipfile

from vam_timeline_ai.io.identity import stable_hash
from vam_timeline_ai.io.json_utils import safe_id_for_path, write_jsonl
from vam_timeline_ai.references.handmade_parser import bake_handmade_manifest


def import_handmade_reference_animations(zip_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    zip_file = Path(zip_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "handmade_animation_manifest.jsonl"
    csv_path = out / "handmade_animation_manifest.csv"
    report_path = out / "handmade_import_report.md"

    if not zip_file.exists():
        summary = {"status": "missing_zip", "zip_path": str(zip_file), "json_count": 0, "jpg_count": 0, "family_counts": {}}
        write_jsonl(manifest_path, [])
        _write_manifest_csv(csv_path, [])
        _write_report(report_path, summary, [])
        bake_handmade_manifest(manifest_path, out / "handmade_sample_index.jsonl", out / "baked")
        return summary

    rows: list[dict[str, Any]] = []
    jpg_by_stem: dict[str, str] = {}
    with zipfile.ZipFile(zip_file) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in names:
            suffix = Path(name).suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                jpg_by_stem[_norm_stem(name)] = name
        for name in names:
            if Path(name).suffix.lower() != ".json":
                continue
            safe_name = safe_id_for_path(Path(name).name)
            target = raw_dir / safe_name
            target.write_bytes(zf.read(name))
            jpg_name = jpg_by_stem.get(_norm_stem(name))
            preview_path = None
            if jpg_name:
                preview_target = raw_dir / safe_id_for_path(Path(jpg_name).name)
                preview_target.write_bytes(zf.read(jpg_name))
                preview_path = str(preview_target)
            meta = parse_reference_filename(Path(name).name)
            reference_id = "ref_" + safe_id_for_path(Path(name).stem)[:48] + "_" + stable_hash([Path(name).name], 8)
            rows.append(
                {
                    "reference_id": reference_id,
                    "source_zip": str(zip_file),
                    "archive_path": name,
                    "json_path": str(target),
                    "preview_jpg_path": preview_path,
                    **meta,
                }
            )
    rows.sort(key=lambda r: (r["label_family"], r["archive_path"]))
    write_jsonl(manifest_path, rows)
    _write_manifest_csv(csv_path, rows)
    sample_summary = bake_handmade_manifest(manifest_path, out / "handmade_sample_index.jsonl", out / "baked")
    summary = {
        "status": "ok",
        "zip_path": str(zip_file),
        "json_count": len(rows),
        "jpg_count": sum(1 for r in rows if r.get("preview_jpg_path")),
        "family_counts": dict(Counter(r["label_family"] for r in rows)),
        "sample_parse_ok": sample_summary.get("parse_ok", 0),
        "sample_bake_ok": sample_summary.get("bake_ok", 0),
    }
    _write_report(report_path, summary, rows)
    return summary


def parse_reference_filename(filename: str) -> dict[str, Any]:
    lower = Path(filename).stem.lower().replace("_", " ").replace("-", " ")
    tokens = lower.split()
    if "cowgirl" in tokens or "cowgirl" in lower:
        family = "cowgirl"
    elif "doggy" in tokens or "doggy" in lower:
        family = "doggy"
    elif "bj" in tokens or "deepthroat" in lower:
        family = "bj"
    elif "hand" in tokens:
        family = "hand"
    elif "head" in tokens:
        family = "head"
    elif "shoulder" in tokens:
        family = "shoulder"
    else:
        family = "unknown"

    intensity = next((t for t in ("soft", "medium", "hard", "basic", "advanced") if t in lower), "unknown")
    depth = "deep" if "deep" in lower or "deepthroat" in lower else ("shallow" if "shallow" in lower else "unknown")
    styles = {
        "vertical": "vertical",
        "horizontal": "horizontal",
        "circular": "circular",
        "twist left": "twist_left",
        "twist right": "twist_right",
        "bouncy": "bouncy",
        "grind": "grinding",
        "grinding": "grinding",
        "riding": "riding",
        "realign": "realign",
        "flinch": "flinch",
        "look": "look",
        "nod": "nod",
        "shake": "shake",
        "cover": "cover",
        "push": "push",
    }
    style = next((value for key, value in styles.items() if key in lower), "unknown")
    is_transition = "realign" in lower or "transition" in lower or "adjust" in lower
    subtype = " ".join(t for t in tokens if t not in {"female", family, intensity, depth, "animation", "seconds", "second"}).strip() or style
    return {
        "label_family": family,
        "label_subtype": subtype,
        "intensity": intensity,
        "depth": depth,
        "style": style,
        "direction_style": style,
        "is_transition_or_realign": bool(is_transition),
    }


def _norm_stem(path: str) -> str:
    return Path(path).stem.lower().replace("_", " ").replace("-", " ").strip()


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["reference_id", "archive_path", "label_family", "label_subtype", "intensity", "depth", "style", "is_transition_or_realign", "json_path", "preview_jpg_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Handmade Animation Import Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- ZIP: `{summary.get('zip_path')}`",
        f"- JSON animations: {summary.get('json_count', 0)}",
        f"- JPG previews paired: {summary.get('jpg_count', 0)}",
        "",
        "Filename labels are used only because this ZIP is a handmade labeled reference set. They are not used as truth for wild scene data.",
        "",
        "## Family Counts",
        "",
    ]
    for family, count in (summary.get("family_counts") or {}).items():
        lines.append(f"- `{family}`: {count}")
    if summary.get("status") == "missing_zip":
        lines.extend(["", "The ZIP was not found; downstream reference commands will write empty/unknown reports."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

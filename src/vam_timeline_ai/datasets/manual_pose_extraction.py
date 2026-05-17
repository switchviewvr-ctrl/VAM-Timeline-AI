"""Extract manual VaM pose capture ZIPs into ignored run folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import zipfile


CAPTURE_ID_RE = re.compile(r"(pose_capture_\d{8}_\d{6})", re.IGNORECASE)


def capture_id_from_name(name: str | Path) -> str | None:
    match = CAPTURE_ID_RE.search(str(name))
    return match.group(1) if match else None


def extract_manual_pose_captures_v1(zip_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    source = Path(zip_path)
    target = Path(out_dir)
    reports = target.parent / "reports"
    target.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    if not source.exists():
        raise FileNotFoundError(source)

    with zipfile.ZipFile(source, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name:
                continue
            # Keep extraction flat and inside the requested ignored run folder.
            # The source ZIP is manual local data, but we still avoid trusting
            # paths embedded in it.
            dest = target / name
            with zf.open(info, "r") as src, dest.open("wb") as dst:
                dst.write(src.read())
            extracted.append(name)

    json_files = sorted(p for p in target.glob("*.json") if capture_id_from_name(p.name))
    png_files = sorted(p for p in target.glob("*.png") if capture_id_from_name(p.name))
    json_ids = {capture_id_from_name(p.name) for p in json_files}
    png_ids = {capture_id_from_name(p.name) for p in png_files}
    capture_ids = sorted((json_ids | png_ids) - {None})
    unmatched_json = sorted(str(v) for v in (json_ids - png_ids) if v)
    unmatched_png = sorted(str(v) for v in (png_ids - json_ids) if v)
    report = reports / "extract_report.md"
    report.write_text(
        "# Manual Pose Capture Extract Report V1\n\n"
        f"- ZIP: `{source}`\n"
        f"- Output dir: `{target}`\n"
        f"- Files extracted: `{len(extracted)}`\n"
        f"- JSON count: `{len(json_files)}`\n"
        f"- PNG count: `{len(png_files)}`\n"
        f"- Capture IDs found: `{len(capture_ids)}`\n"
        f"- Unmatched JSON: `{unmatched_json}`\n"
        f"- Unmatched PNG: `{unmatched_png}`\n\n"
        "## Capture IDs\n\n"
        + "\n".join(f"- `{cid}`" for cid in capture_ids)
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "zip": str(source),
        "out_dir": str(target),
        "report": str(report),
        "files_extracted": len(extracted),
        "json_count": len(json_files),
        "png_count": len(png_files),
        "capture_ids": capture_ids,
        "unmatched_json": unmatched_json,
        "unmatched_png": unmatched_png,
    }

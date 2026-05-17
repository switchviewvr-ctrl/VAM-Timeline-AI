"""Build contact sheets and optional GIFs from real VaM capture frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl


def build_vam_capture_contact_sheets_v0(capture_results: str | Path, out_dir: str | Path) -> dict[str, Any]:
    rows = load_jsonl(capture_results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        rid = str(row.get("review_id") or f"capture_{len(manifest)+1:03d}")
        item_dir = out / rid
        item_dir.mkdir(parents=True, exist_ok=True)
        frames = _frame_paths(row)
        entry = {"review_id": rid, "status": "unavailable", "frame_paths": [str(p) for p in frames], "contact_sheet_path": None, "gif_path": None, "warnings": []}
        if not frames:
            entry["warnings"].append("no frame paths in capture result")
        else:
            sheet = item_dir / "contact_sheet.png"
            if _make_contact_sheet(sheet, frames, rid, entry["warnings"]):
                entry["status"] = "ok"
                entry["contact_sheet_path"] = str(sheet)
            gif = item_dir / "preview.gif"
            if _make_gif(gif, frames, entry["warnings"]):
                entry["gif_path"] = str(gif)
        dump_json(item_dir / "metadata.json", {"capture_result": row, "contact_sheet": entry})
        manifest.append(entry)
    write_jsonl(out / "vam_capture_contact_sheet_manifest.jsonl", manifest)
    (out / "vam_capture_contact_sheet_report.md").write_text(
        "# VaM Capture Contact Sheets V0\n\n"
        f"- Capture results: {len(rows)}\n"
        f"- Sheets created: {sum(1 for r in manifest if r.get('contact_sheet_path'))}\n"
        f"- GIFs created: {sum(1 for r in manifest if r.get('gif_path'))}\n",
        encoding="utf-8",
    )
    return {"status": "ok", "items": len(rows), "sheets": sum(1 for r in manifest if r.get("contact_sheet_path")), "gifs": sum(1 for r in manifest if r.get("gif_path")), "out_dir": str(out)}


def _frame_paths(row: dict[str, Any]) -> list[Path]:
    paths = row.get("frame_paths") or row.get("frames") or []
    return [Path(str(p)) for p in paths if Path(str(p)).exists()]


def _make_contact_sheet(path: Path, frames: list[Path], review_id: str, warnings: list[str]) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Pillow unavailable; contact sheet skipped: {exc}")
        return False
    try:
        imgs = [Image.open(p).convert("RGB") for p in frames[:16]]
        if not imgs:
            return False
        thumb_w = 400
        thumb_h = 225
        cols = 4
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 24)), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, img in enumerate(imgs):
            img.thumbnail((thumb_w, thumb_h))
            x = (idx % cols) * thumb_w
            y = (idx // cols) * (thumb_h + 24) + 24
            sheet.paste(img, (x + (thumb_w - img.width) // 2, y))
            draw.text((x + 8, y - 20), f"{review_id} frame {idx:02d}", fill=(0, 0, 0))
        sheet.save(path)
        for img in imgs:
            img.close()
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"contact sheet failed: {exc}")
        return False


def _make_gif(path: Path, frames: list[Path], warnings: list[str]) -> bool:
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Pillow unavailable; GIF skipped: {exc}")
        return False
    try:
        imgs = [Image.open(p).convert("P", palette=getattr(Image, "ADAPTIVE", 1)) for p in frames[:32]]
        if not imgs:
            return False
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
        for img in imgs:
            img.close()
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"GIF failed: {exc}")
        return False

"""Build manual VaM viewport capture requests for review items."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def build_vam_capture_requests_v0(review_dir: str | Path, out_jsonl: str | Path, output_root: str | Path, frame_count: int = 16, duration_seconds: float = 4.0) -> dict[str, Any]:
    review = Path(review_dir)
    rows = _merged_review_rows(review)
    output = Path(output_root)
    requests: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row.get("review_id") or row.get("window_id") or f"review_{len(requests)+1:03d}")
        scene_path = row.get("source_scene_path") or row.get("source_scene_file")
        requests.append(
            {
                "schema": "vam_timeline_ai_vam_capture_request_v0",
                "review_id": rid,
                "scene_path": scene_path,
                "actor": row.get("technical_atom_id") or row.get("technical_actor_id"),
                "clip_name": row.get("clip_name"),
                "clip_index": row.get("clip_index"),
                "timeline_export_path": row.get("timeline_export_path") or row.get("vam_animation_path"),
                "start_seconds": row.get("start_seconds"),
                "end_seconds": row.get("end_seconds"),
                "system_guess": {
                    "semantic_family": row.get("semantic_family"),
                    "pose": row.get("pose_subtype") or row.get("pose_family"),
                    "motion": row.get("motion_subtype"),
                    "phase": row.get("phase"),
                    "contact_support": row.get("contact_support"),
                },
                "output_dir": str(output / rid),
                "frame_count": int(frame_count),
                "duration_seconds": float(duration_seconds),
                "capture_interval_seconds": float(duration_seconds) / max(1, int(frame_count) - 1),
                "suggested_user_action": [
                    "load scene",
                    "select/import timeline segment if available",
                    "set camera angle",
                    "run capture",
                ],
                "captures_scene_automatically": False,
            }
        )
    write_jsonl(out_jsonl, requests)
    report = Path(out_jsonl).with_suffix(".md")
    report.write_text(
        "# VaM Capture Requests V0\n\n"
        f"- Requests: {len(requests)}\n"
        "- This file does not capture frames itself.\n"
        "- User loads scenes manually; BepInEx bridge only captures current viewport.\n",
        encoding="utf-8",
    )
    return {"status": "ok", "requests": len(requests), "out_jsonl": str(out_jsonl), "report": str(report)}


def _merged_review_rows(review: Path) -> list[dict[str, Any]]:
    rows = {r.get("review_id"): r for r in load_jsonl(review / "semantic_review_010.jsonl") if r.get("review_id")}
    manifest = {r.get("review_id"): r for r in load_jsonl(review / "vam_review_package" / "vam_review_manifest.jsonl") if r.get("review_id")}
    out = []
    for rid in sorted(set(rows) | set(manifest)):
        merged: dict[str, Any] = {}
        merged.update(rows.get(rid) or {})
        merged.update({k: v for k, v in (manifest.get(rid) or {}).items() if _has_value(v)})
        out.append(merged)
    return out


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return False
    return True

"""Build local visual-judge request manifests from review previews.

The requests contain file paths and audit context only. They do not call
external APIs and they do not turn visual model output into truth.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def build_visual_judge_requests_v0(review_dir: str | Path, preview_dir: str | Path, out_jsonl: str | Path, mode: str = "blind") -> dict[str, Any]:
    review = Path(review_dir)
    preview = Path(preview_dir)
    out = Path(out_jsonl)
    review_rows = {str(row.get("review_id")): row for row in load_jsonl(review / "semantic_review_010.jsonl") if row.get("review_id")}
    preview_rows = {str(row.get("review_id")): row for row in load_jsonl(preview / "digital_twin_preview_manifest_v1.jsonl") if row.get("review_id")}
    requests: list[dict[str, Any]] = []
    for rid, row in sorted(review_rows.items()):
        visual = preview_rows.get(rid) or {}
        request = _request_for(row, visual, review, preview, mode)
        requests.append(request)
    write_jsonl(out, requests)
    report = out.with_suffix(".md")
    _write_report(report, requests)
    counts = Counter(row.get("primary_visual_type") for row in requests)
    return {
        "status": "ok",
        "requests": len(requests),
        "out_jsonl": str(out),
        "report": str(report),
        "by_primary_visual_type": dict(counts),
        "external_api_calls": False,
        "visual_outputs_are_ground_truth": False,
    }


def _request_for(row: dict[str, Any], visual: dict[str, Any], review: Path, preview: Path, mode: str) -> dict[str, Any]:
    primary_path, primary_type, quality, fallbacks = _choose_visual(row, visual, review, preview)
    prompt_context = {} if mode == "blind" else {
        "system_semantic_family": row.get("semantic_family"),
        "system_pose": row.get("pose_subtype") or row.get("pose_family"),
        "system_motion": row.get("motion_subtype"),
        "system_contact": row.get("contact_support"),
    }
    return {
        "schema": "vam_timeline_ai_visual_judge_request_v0",
        "review_id": row.get("review_id"),
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "mode": mode,
        "primary_visual_path": primary_path,
        "primary_visual_type": primary_type,
        "fallback_paths": fallbacks,
        "visual_quality": quality,
        "visual_review_assist_only": True,
        "human_verification_required": True,
        "visual_output_is_ground_truth": False,
        "source_scene_file": row.get("source_scene_file"),
        "technical_actor_id": row.get("technical_actor_id") or row.get("technical_atom_id"),
        "time_range": [row.get("start_seconds"), row.get("end_seconds")],
        "prompt_context": prompt_context,
        "requested_optional_fields": [
            "visual_family_guess",
            "visual_pose_guess",
            "visual_motion_guess",
            "visual_contact_guess",
            "visual_pose_broken_score",
            "visual_confidence",
        ],
        "warnings": _visual_warnings(visual, quality),
    }


def _choose_visual(row: dict[str, Any], visual: dict[str, Any], review: Path, preview: Path) -> tuple[str | None, str, str, list[str]]:
    fallbacks: list[str] = []
    candidates = [
        ("mp4_path", "mp4", "high"),
        ("gif_path", "gif", "high"),
        ("contact_sheet_large_path", "contact_sheet", "medium"),
    ]
    for key, typ, quality in candidates:
        value = visual.get(key)
        if value:
            fallbacks.extend(_existing_fallbacks(visual, exclude=value))
            return str(value), typ, quality, fallbacks
    rid = str(row.get("review_id") or "")
    if rid:
        old_static = review / "digital_twin_previews" / rid / "contact_sheet.png"
        if old_static.exists():
            return str(old_static), "static_plot", "low", []
    return None, "none", "unavailable", []


def _existing_fallbacks(visual: dict[str, Any], exclude: Any) -> list[str]:
    out = []
    for key in ["gif_path", "contact_sheet_large_path", "mp4_path"]:
        value = visual.get(key)
        if value and value != exclude:
            out.append(str(value))
    return out


def _visual_warnings(visual: dict[str, Any], quality: str) -> list[str]:
    warnings = list(visual.get("warnings") or [])
    if quality == "low":
        warnings.append("Only static plot available; visual judge confidence must remain weak.")
    if quality == "unavailable":
        warnings.append("No visual preview available.")
    return warnings


def _write_report(path: Path, requests: list[dict[str, Any]]) -> None:
    counts = Counter(row.get("primary_visual_type") for row in requests)
    low = [row for row in requests if row.get("visual_quality") in {"low", "unavailable"}]
    lines = [
        "# Visual Judge Requests V0",
        "",
        "Local request manifest only. No external API was called. Visual outputs are review-assist only.",
        "",
        f"- Requests: {len(requests)}",
        f"- MP4 primary: {counts.get('mp4', 0)}",
        f"- GIF primary: {counts.get('gif', 0)}",
        f"- Contact sheet primary: {counts.get('contact_sheet', 0)}",
        f"- Static plot primary: {counts.get('static_plot', 0)}",
        f"- Missing visual: {counts.get('none', 0)}",
        "",
        "## Low Quality / Missing",
        "",
    ]
    lines.extend([f"- `{row.get('review_id')}`: {row.get('primary_visual_type')} {row.get('warnings')}" for row in low] or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

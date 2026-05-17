"""Build Visual Judge requests preferring real VaM captures over fallbacks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.vision.visual_judge_prompts import build_visual_judge_prompt
from vam_timeline_ai.vision.visual_judge_schema import visual_judge_result_schema_v0


def build_visual_judge_requests_v1(
    review_dir: str | Path,
    vam_capture_sheets: str | Path,
    digital_twin_previews: str | Path,
    out_jsonl: str | Path,
    mode: str = "blind",
) -> dict[str, Any]:
    review = Path(review_dir)
    capture = Path(vam_capture_sheets)
    digital = Path(digital_twin_previews)
    review_rows = {str(row.get("review_id")): row for row in load_jsonl(review / "semantic_review_010.jsonl") if row.get("review_id")}
    capture_rows = {str(row.get("review_id")): row for row in load_jsonl(capture / "vam_capture_contact_sheet_manifest.jsonl") if row.get("review_id")}
    digital_rows = {str(row.get("review_id")): row for row in load_jsonl(digital / "digital_twin_preview_manifest_v1.jsonl") if row.get("review_id")}
    requests = []
    for rid, row in sorted(review_rows.items()):
        cap = capture_rows.get(rid) or {}
        dig = digital_rows.get(rid) or {}
        primary, typ, quality, fallbacks, warnings = _choose_visual(cap, dig, review, rid)
        system_guess = {
            "semantic_family": row.get("semantic_family"),
            "pose_subtype": row.get("pose_subtype") or row.get("pose_family"),
            "motion_subtype": row.get("motion_subtype"),
            "phase": row.get("phase"),
            "contact_support": row.get("contact_support"),
        }
        requests.append(
            {
                "schema": "vam_timeline_ai_visual_judge_request_v1",
                "review_id": rid,
                "window_id": row.get("window_id"),
                "sample_id": row.get("sample_id"),
                "mode": mode,
                "primary_visual_path": primary,
                "primary_visual_type": typ,
                "visual_quality": quality,
                "fallback_paths": fallbacks,
                "prompt_text": build_visual_judge_prompt(mode=mode, system_guess=system_guess),
                "system_guess": system_guess if mode == "compare" else None,
                "expected_schema": visual_judge_result_schema_v0(),
                "warnings": warnings,
                "cloud_api_allowed": False,
                "auto_label_allowed": False,
            }
        )
    write_jsonl(out_jsonl, requests)
    report = Path(out_jsonl).with_suffix(".md")
    _write_report(report, requests)
    return {"status": "ok", "requests": len(requests), "out_jsonl": str(out_jsonl), "report": str(report), "by_primary_visual_type": dict(Counter(r.get("primary_visual_type") for r in requests))}


def _choose_visual(capture: dict[str, Any], digital: dict[str, Any], review: Path, rid: str) -> tuple[str | None, str, str, list[str], list[str]]:
    warnings: list[str] = []
    fallbacks: list[str] = []
    if capture.get("contact_sheet_path"):
        if capture.get("gif_path"):
            fallbacks.append(str(capture["gif_path"]))
        fallbacks.extend(_digital_fallbacks(digital))
        return str(capture["contact_sheet_path"]), "contact_sheet", "high_real_vam_capture", fallbacks, list(capture.get("warnings") or [])
    if capture.get("gif_path"):
        fallbacks.extend(_digital_fallbacks(digital))
        return str(capture["gif_path"]), "gif", "high_real_vam_capture", fallbacks, list(capture.get("warnings") or [])
    if digital.get("contact_sheet_large_path"):
        fallbacks.extend([str(p) for p in [digital.get("gif_path"), digital.get("mp4_path")] if p])
        return str(digital["contact_sheet_large_path"]), "contact_sheet", "medium_digital_twin", fallbacks, list(digital.get("warnings") or [])
    if digital.get("gif_path"):
        fallbacks.extend([str(p) for p in [digital.get("mp4_path")] if p])
        return str(digital["gif_path"]), "gif", "medium_digital_twin", fallbacks, list(digital.get("warnings") or [])
    if digital.get("mp4_path"):
        warnings.append("MP4 is a fallback visual input; prefer contact sheets for LM Studio VLM calls.")
        return str(digital["mp4_path"]), "mp4", "medium_digital_twin", [], list(digital.get("warnings") or []) + warnings
    old_static = review / "digital_twin_previews" / rid / "contact_sheet.png"
    if old_static.exists():
        warnings.append("Only static technical plot available; VLM result may be unreliable.")
        return str(old_static), "static_plot", "low_static_plot", [], warnings
    warnings.append("No visual input available.")
    return None, "static_plot", "unknown", [], warnings


def _digital_fallbacks(digital: dict[str, Any]) -> list[str]:
    return [str(p) for p in [digital.get("contact_sheet_large_path"), digital.get("gif_path"), digital.get("mp4_path")] if p]


def _write_report(path: Path, requests: list[dict[str, Any]]) -> None:
    counts = Counter(r.get("primary_visual_type") for r in requests)
    quality = Counter(r.get("visual_quality") for r in requests)
    lines = [
        "# Visual Judge Requests V1",
        "",
        "Local request manifest only. No cloud API call. No auto-labeling.",
        "",
        f"- Requests: {len(requests)}",
        f"- Primary visual types: {dict(counts)}",
        f"- Visual quality: {dict(quality)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Evaluate calibration outputs and assign a conservative visual judge trust gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.vision.lmstudio_vlm_judge import run_lmstudio_vlm_judge_v0
from vam_timeline_ai.vision.visual_judge_prompts import build_visual_judge_prompt


def evaluate_vlm_visual_judge_v1(calibration_set: str | Path, base_url: str, model: str, out_dir: str | Path, dry_run: bool = True) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(calibration_set)
    request_path = out / "calibration_visual_judge_requests.jsonl"
    requests = []
    for item in items:
        requests.append(
            {
                "review_id": item.get("calibration_id"),
                "primary_visual_path": item.get("image_or_contact_sheet_path"),
                "primary_visual_type": "contact_sheet",
                "visual_quality": "medium_contact_sheet" if item.get("image_or_contact_sheet_path") else "unknown",
                "prompt_text": build_visual_judge_prompt(),
            }
        )
    write_jsonl(request_path, requests)
    judge = run_lmstudio_vlm_judge_v0(request_path, base_url, model, out / "calibration_visual_judge_results.jsonl", out / "raw", dry_run=dry_run)
    results = load_jsonl(out / "calibration_visual_judge_results.jsonl")
    trust = _trust_level(results, dry_run)
    summary = {"status": "ok", "dry_run": dry_run, "model": model, "items": len(items), "trust_gate": trust, "judge": judge}
    dump_json(out / "trust_gate_summary.json", summary)
    (out / "trust_gate_report.md").write_text(
        "# VLM Visual Judge Trust Gate\n\n"
        f"- Model: `{model}`\n"
        f"- Dry run: {dry_run}\n"
        f"- Trust gate: `{trust}`\n"
        "- Default is disabled unless live calibration is strong.\n",
        encoding="utf-8",
    )
    return summary


def _trust_level(results: list[dict[str, Any]], dry_run: bool) -> str:
    if dry_run or not results:
        return "disabled"
    parsed = [r for r in results if r.get("parse_status") == "parsed"]
    if len(parsed) < max(3, len(results) // 2):
        return "disabled"
    hallucinated = [r for r in parsed if r.get("evidence_sufficient_for_family") is False and r.get("raw_suggested_family") not in {None, "unknown"}]
    if len(hallucinated) > len(parsed) * 0.25:
        return "coarse_pose_only"
    return "review_assist_low_trust"

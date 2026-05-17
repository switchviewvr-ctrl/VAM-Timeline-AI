"""Combine heuristic, ML, visual judge, and human-review availability for triage."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def build_multisignal_review_priorities_v0(run_dir: str | Path, review_dir: str | Path, model_scores: str | Path, visual_results: str | Path, out_jsonl: str | Path, report: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    review = Path(review_dir)
    rows = load_jsonl(review / "semantic_review_010.jsonl")
    scores = {str(r.get("window_id")): r for r in load_jsonl(model_scores)}
    visual = {str(r.get("review_id")): r for r in load_jsonl(visual_results)}
    human = {str(r.get("review_id")): r for r in load_jsonl(review / "human_review_ui_answers.jsonl")}
    trust_gate = _load_trust_gate(run)
    out_rows = []
    for row in rows:
        rid = str(row.get("review_id"))
        v = visual.get(rid) or {}
        m = scores.get(str(row.get("window_id"))) or {}
        priority, reason = _priority(row, m, v, rid in human, trust_gate)
        out_rows.append(
            {
                "review_id": rid,
                "window_id": row.get("window_id"),
                "heuristic_family": row.get("semantic_family"),
                "ml_cowgirl_probability": m.get("model_cowgirl_probability") or row.get("model_cowgirl_probability"),
                "visual_suggested_family": v.get("suggested_family"),
                "visual_family_confidence": v.get("family_confidence"),
                "visual_parse_status": v.get("parse_status"),
                "visual_evidence_sufficient": v.get("evidence_sufficient_for_family"),
                "visual_trust_gate": trust_gate,
                "human_answer_present": rid in human,
                "multisignal_priority": priority,
                "reason": reason,
            }
        )
    write_jsonl(out_jsonl, out_rows)
    counts = {}
    for row in out_rows:
        counts[row["multisignal_priority"]] = counts.get(row["multisignal_priority"], 0) + 1
    Path(report).write_text(
        "# Multisignal Triage Report\n\n"
        "Human answer overrides everything. Low-trust or insufficient visual evidence is not used as classification.\n\n"
        f"- Items: {len(out_rows)}\n"
        f"- Visual trust gate: `{trust_gate}`\n"
        f"- Priority counts: {counts}\n",
        encoding="utf-8",
    )
    return {"status": "ok", "items": len(out_rows), "visual_trust_gate": trust_gate, "priority_counts": counts, "out_jsonl": str(out_jsonl), "report": str(report)}


def _priority(row: dict[str, Any], model: dict[str, Any], visual: dict[str, Any], has_human: bool, trust_gate: str) -> tuple[str, str]:
    if has_human:
        return "human_reviewed", "human answer exists"
    if visual.get("parse_status") in {"dry_run", "blocked", "unavailable", "parse_failed"}:
        return "must_review", "visual judge unavailable or not parsed"
    if trust_gate not in {"review_assist_medium", "review_assist_high"}:
        return "review_queue", f"visual trust gate `{trust_gate}` is below review_assist_medium; VLM family is informational only"
    if visual.get("evidence_sufficient_for_family") is False:
        return "must_review", "visual evidence insufficient for family"
    hfam = row.get("semantic_family")
    vfam = visual.get("suggested_family")
    mlp = model.get("model_cowgirl_probability") or row.get("model_cowgirl_probability")
    try:
        mlp_f = float(mlp)
    except (TypeError, ValueError):
        mlp_f = 0.0
    if hfam == vfam == "cowgirl" and mlp_f >= 0.75:
        return "spot_check", "heuristic, ML, and visual judge agree"
    if vfam and vfam != "unknown" and vfam != hfam:
        return "must_review", "visual judge disagrees with heuristic"
    if vfam == "unknown":
        return "must_review", "visual judge unknown"
    return "review_queue", "no strong multisignal conclusion"


def _load_trust_gate(run: Path) -> str:
    summary = run / "vision" / "visual_judge_calibration_v1" / "nsfwvision_eval" / "trust_gate_summary.json"
    if not summary.exists():
        return "disabled"
    try:
        return str(json.loads(summary.read_text(encoding="utf-8")).get("trust_gate") or "disabled")
    except json.JSONDecodeError:
        return "disabled"

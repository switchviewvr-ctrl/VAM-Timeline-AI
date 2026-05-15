"""Clean Cowgirl candidate scoring for audit review selection.

Scores here are review triage, not labels and not training targets.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.domain_guards import evaluate_domain_guards
from vam_timeline_ai.semantics.motion_phase_classifier import classify_motion_phase


def score_cowgirl_candidates_v2(
    run_dir: str | Path,
    wild_reference_matches: str | Path,
    body_quality: str | Path,
    features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    matches = {r.get("window_id"): r for r in load_jsonl(wild_reference_matches) if r.get("window_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    rows = [
        score_window(feature_rows[wid], body.get(wid, {}), matches.get(wid, {}), windows.get(wid, {}))
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: float(r.get("final_clean_cowgirl_candidate_score") or 0.0), reverse=True)
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def score_window(feature_row: dict[str, Any], body: dict[str, Any], match: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    values = feature_row.get("feature_values", {}) or {}
    duration = _num(window.get("duration_seconds") or window.get("window_size_seconds") or 0.0)
    body_quality = body.get("body_motion_quality", "unknown")
    phase = classify_motion_phase(feature_row, body)
    guard = evaluate_domain_guards(feature_row, body)
    pelvis_range = max(_num(values.get("pelvis_total_position_range")), _num(values.get("pelvis_vertical_amplitude")), _num(values.get("pelvis_forward_back_amplitude")), _num(values.get("pelvis_lateral_amplitude")))
    pelvis_energy = _num(values.get("pelvis_movement_energy"))
    cowgirl_reference_score = _num(match.get("cowgirl_reference_score"))
    body_score = {"good_body_motion": 1.0, "partial_body_motion": 0.65}.get(str(body_quality), 0.0)
    pelvis_motion_score = max(min(pelvis_range / 0.12, 1.0), min(pelvis_energy / 0.04, 1.0)) if np.isfinite(pelvis_range) or np.isfinite(pelvis_energy) else 0.0
    duration_score = 1.0 if duration >= 8.0 else 0.85 if duration >= 4.0 else 0.45 if duration >= 3.0 else 0.18 if duration >= 2.0 else 0.0
    anti_static_score = 0.0 if body.get("body_motion_quality") in {"static_or_empty", "static_or_micro_motion"} or body.get("static_or_micro_motion") else 1.0
    anti_micro_motion_score = max(0.0, 1.0 - _num(body.get("micro_motion_score")))
    anti_head_only_score = 0.0 if body.get("minimal_head_motion_only") or "possible_non_cowgirl_head_dominant_motion" in guard.get("domain_guard_audit_labels", []) else 1.0
    anti_hand_jitter_score = 0.0 if body.get("minimal_hand_jitter_only") else 1.0
    transition_penalty = 0.45 if phase.get("motion_phase_candidate") == "transition_adjustment_candidate" or match.get("recommended_review_status") == "likely_transition_or_realign" else 0.0
    root_world_penalty = 1.0 if body_quality in {"controller_only_whole_person_motion", "root_only_motion"} else 0.0
    domain_guard_penalty = 0.55 if guard.get("cowgirl_confidence_multiplier", 1.0) < 0.8 else 0.0
    support_count = int(body.get("active_bodypart_count_above_threshold") or 0)
    support_penalty = 0.35 if support_count < 2 else 0.0
    raw_score = (
        0.30 * cowgirl_reference_score
        + 0.20 * body_score
        + 0.20 * pelvis_motion_score
        + 0.12 * duration_score
        + 0.08 * anti_static_score
        + 0.05 * anti_micro_motion_score
        + 0.03 * anti_head_only_score
        + 0.02 * anti_hand_jitter_score
    )
    penalty = max(0.0, 1.0 - transition_penalty - root_world_penalty - domain_guard_penalty - support_penalty)
    final_score = max(0.0, min(1.0, raw_score * penalty))
    reject_reasons = []
    if root_world_penalty:
        reject_reasons.append("root_or_controller_only")
    if anti_static_score == 0.0 or anti_micro_motion_score < 0.4:
        reject_reasons.append("static_or_micro_motion")
    if anti_head_only_score == 0.0:
        reject_reasons.append("head_only_or_head_dominant")
    if anti_hand_jitter_score == 0.0:
        reject_reasons.append("hand_jitter_only")
    if support_count < 2:
        reject_reasons.append("insufficient_supporting_bodyparts")
    if duration < 4.0:
        reject_reasons.append("too_short_for_semantic_judgment")
    if transition_penalty:
        reject_reasons.append("transition_or_realign_candidate")
    return {
        "window_id": feature_row.get("window_id"),
        "sample_id": feature_row.get("sample_id"),
        "source_id": feature_row.get("source_id"),
        "source_scene_file": feature_row.get("source_scene_file"),
        "technical_atom_id": feature_row.get("technical_atom_id"),
        "duration_seconds": duration,
        "cowgirl_reference_score": round(float(cowgirl_reference_score), 6),
        "body_motion_quality_score": round(float(body_score), 6),
        "pelvis_motion_score": round(float(pelvis_motion_score), 6),
        "duration_score": round(float(duration_score), 6),
        "anti_static_score": round(float(anti_static_score), 6),
        "anti_micro_motion_score": round(float(anti_micro_motion_score), 6),
        "anti_head_only_score": round(float(anti_head_only_score), 6),
        "anti_hand_jitter_score": round(float(anti_hand_jitter_score), 6),
        "transition_penalty": round(float(transition_penalty), 6),
        "root_world_penalty": round(float(root_world_penalty), 6),
        "domain_guard_penalty": round(float(domain_guard_penalty), 6),
        "support_penalty": round(float(support_penalty), 6),
        "final_clean_cowgirl_candidate_score": round(float(final_score), 6),
        "clean_cowgirl_candidate": bool(final_score >= 0.50 and not reject_reasons),
        "reject_reasons": reject_reasons,
        "body_motion_quality": body_quality,
        "static_or_micro_motion": bool(body.get("static_or_micro_motion")),
        "minimal_head_motion_only": bool(body.get("minimal_head_motion_only")),
        "minimal_hand_jitter_only": bool(body.get("minimal_hand_jitter_only")),
        "active_bodypart_count_above_threshold": support_count,
        "meaningful_motion_duration_ratio": body.get("meaningful_motion_duration_ratio"),
        "motion_phase_candidate": phase.get("motion_phase_candidate"),
        "domain_guard_warnings": guard.get("domain_guard_warnings", []),
        "reference_review_status": match.get("recommended_review_status"),
        "nearest_reference_families": match.get("nearest_reference_families", []),
        "is_human_ground_truth": False,
    }


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = [r for r in rows if r.get("clean_cowgirl_candidate")]
    rejection_counts = Counter(reason for r in rows for reason in r.get("reject_reasons", []))
    lines = [
        "# Cowgirl Candidate Score V2 Report",
        "",
        "Scores are review triage only. They are not labels and not ML targets.",
        "",
        f"- Windows scored: {len(rows)}",
        f"- Clean Cowgirl candidates: {len(clean)}",
        "",
        "## Rejection Reasons",
        "",
    ]
    for reason, count in rejection_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Top Candidates", ""])
    for row in rows[:20]:
        lines.append(f"- `{row.get('window_id')}` score={row.get('final_clean_cowgirl_candidate_score')} duration={row.get('duration_seconds')} scene=`{row.get('source_scene_file')}` reject={row.get('reject_reasons')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0

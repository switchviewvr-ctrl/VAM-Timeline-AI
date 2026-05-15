"""Rider vs receiver/body-response scoring.

This is an audit/review heuristic layer. It does not infer semantic roles from
atom names and does not create training labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.motion_phase_classifier import classify_motion_phase


def score_rider_receiver_v1(
    run_dir: str | Path,
    features: str | Path,
    pair_features: str | Path,
    pair_windows: str | Path,
    body_quality: str | Path,
    wild_reference_matches: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    feature_rows = {r.get("window_id"): r for r in load_jsonl(features) if r.get("window_id")}
    body_rows = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    match_rows = {r.get("window_id"): r for r in load_jsonl(wild_reference_matches) if r.get("window_id")}
    pair_window_rows = {r.get("pair_window_id"): r for r in load_jsonl(pair_windows) if r.get("pair_window_id")}
    pair_evidence = _collect_pair_evidence(load_jsonl(pair_features), pair_window_rows)
    rows = [
        score_window_role(feature_rows[wid], body_rows.get(wid, {}), match_rows.get(wid, {}), pair_evidence.get(wid, []))
        for wid in feature_rows
    ]
    rows.sort(key=lambda r: (r.get("rider_receiver_status") != "likely_active_rider", -float(r.get("active_rider_score") or 0.0), r.get("window_id") or ""))
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def score_window_role(feature: dict[str, Any], body: dict[str, Any], match: dict[str, Any], pair_items: list[dict[str, Any]]) -> dict[str, Any]:
    values = feature.get("feature_values", {}) or {}
    pelvis_energy = _num(values.get("pelvis_movement_energy"))
    pelvis_range = max(_num(values.get("pelvis_total_position_range")), _num(values.get("pelvis_vertical_amplitude")), _num(values.get("pelvis_forward_back_amplitude")), _num(values.get("pelvis_lateral_amplitude")))
    torso = _num(values.get("torso_motion_energy"))
    head = _num(values.get("head_motion_energy"))
    hands = _num(values.get("left_hand_motion_energy")) + _num(values.get("right_hand_motion_energy"))
    legs = _num(values.get("knee_motion_energy_left")) + _num(values.get("knee_motion_energy_right")) + _num(values.get("foot_motion_energy_left")) + _num(values.get("foot_motion_energy_right"))
    support_count = int(body.get("active_bodypart_count_above_threshold") or 0)
    body_quality = body.get("body_motion_quality", "unknown")
    phase = classify_motion_phase(feature, body).get("motion_phase_candidate")
    cowgirl_score = _num(match.get("cowgirl_reference_score"))
    doggy_score = _num(match.get("doggy_reference_score"))
    head_score = max(_num(match.get("head_reference_score")), _num(match.get("bj_reference_score")))

    pair_active = max([p["active_support"] for p in pair_items] or [0.0])
    pair_receiver = max([p["receiver_support"] for p in pair_items] or [0.0])
    pair_passive = max([p["passive_support"] for p in pair_items] or [0.0])
    pair_below = max([p["below_other_score"] for p in pair_items] or [0.0])
    pair_other_active = max([p["other_active_confidence"] for p in pair_items] or [0.0])
    pair_count = len(pair_items)

    body_initiative = 0.35 * min(pelvis_range / 0.12, 1.0) + 0.25 * min(pelvis_energy / 0.04, 1.0) + 0.25 * min(support_count / 3.0, 1.0) + 0.15 * min((torso + hands + legs + head) / 0.02, 1.0)
    hip_only_response = min(pelvis_range / 0.12, 1.0) * (1.0 - min(support_count / 2.0, 1.0))
    static_penalty = 0.6 if body_quality in {"static_or_empty", "static_or_micro_motion", "controller_only_whole_person_motion", "root_only_motion"} or body.get("static_or_micro_motion") else 0.0
    head_penalty = 0.25 if head_score > max(cowgirl_score, doggy_score, 0.0) and pelvis_energy < 0.02 else 0.0

    active_rider_score = max(0.0, min(1.0, 0.38 * body_initiative + 0.22 * cowgirl_score + 0.30 * pair_active + 0.10 * (1.0 - min(head_penalty + static_penalty, 1.0)) - 0.35 * pair_receiver - 0.20 * hip_only_response))
    receiver_body_response_score = max(0.0, min(1.0, 0.42 * pair_receiver + 0.18 * pair_below + 0.18 * pair_other_active + 0.18 * hip_only_response + 0.10 * max(0.0, 1.0 - min((torso + hands + legs + head) / 0.02, 1.0))))
    passive_context_score = max(0.0, min(1.0, 0.45 * pair_passive + 0.35 * static_penalty + 0.20 * (1.0 - min(body_initiative, 1.0))))
    if pair_count == 0:
        role_unclear_score = max(0.35, 1.0 - active_rider_score)
        status = "insufficient_pair_context"
    else:
        role_unclear_score = max(0.0, 1.0 - max(active_rider_score, receiver_body_response_score, passive_context_score))
        if receiver_body_response_score >= 0.50 and receiver_body_response_score > active_rider_score + 0.08:
            status = "likely_receiver_body_response"
        elif passive_context_score >= 0.65 and passive_context_score > active_rider_score:
            status = "likely_passive_context"
        elif active_rider_score >= 0.50 and active_rider_score > receiver_body_response_score + 0.10:
            status = "likely_active_rider"
        else:
            status = "role_unclear"
    if status == "likely_receiver_body_response" and body_quality in {"controller_only_whole_person_motion", "root_only_motion", "static_or_micro_motion", "static_or_empty"}:
        status = "likely_passive_context"

    evidence = {
        "pelvis_movement_energy": pelvis_energy,
        "pelvis_total_range": pelvis_range,
        "torso_energy": torso,
        "head_energy": head,
        "hand_energy": hands,
        "leg_energy": legs,
        "support_bodypart_count": support_count,
        "body_initiative_score": round(body_initiative, 6),
        "hip_only_response_score": round(hip_only_response, 6),
        "pair_active_support": round(pair_active, 6),
        "pair_receiver_support": round(pair_receiver, 6),
        "pair_passive_support": round(pair_passive, 6),
        "pair_below_other_score": round(pair_below, 6),
        "pair_other_active_confidence": round(pair_other_active, 6),
        "cowgirl_reference_score": cowgirl_score,
        "doggy_reference_score": doggy_score,
        "head_or_bj_reference_score": head_score,
        "body_motion_quality": body_quality,
        "motion_phase": phase,
        "pair_context_count": pair_count,
    }
    warnings = []
    if pair_count == 0:
        warnings.append("No pair context; role cannot be forced.")
    if hip_only_response > 0.3:
        warnings.append("Hip/pelvis motion without enough supporting bodypart initiative can be receiver/body-response.")
    if status == "likely_receiver_body_response":
        warnings.append("Do not treat as active rider Cowgirl without human confirmation.")
    if body_quality in {"controller_only_whole_person_motion", "root_only_motion"}:
        warnings.append("Root/controller-only motion is not body response and is unsafe as animation output.")
    return {
        "window_id": feature.get("window_id"),
        "sample_id": feature.get("sample_id"),
        "source_id": feature.get("source_id"),
        "source_scene_file": feature.get("source_scene_file"),
        "technical_atom_id": feature.get("technical_atom_id"),
        "active_rider_score": round(float(active_rider_score), 6),
        "receiver_body_response_score": round(float(receiver_body_response_score), 6),
        "passive_context_score": round(float(passive_context_score), 6),
        "role_unclear_score": round(float(role_unclear_score), 6),
        "rider_receiver_status": status,
        "evidence": evidence,
        "pair_evidence": pair_items[:5],
        "warnings": warnings,
        "is_human_ground_truth": False,
    }


def _collect_pair_evidence(pair_rows: list[dict[str, Any]], pair_windows: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        q = row.get("feature_quality", {}) or {}
        values = row.get("feature_values", {}) or {}
        active = q.get("active_actor_candidate", "unknown")
        active_conf = _num(q.get("active_actor_confidence"))
        pair_id = row.get("pair_window_id")
        pwin = pair_windows.get(pair_id, {})
        for side, other in (("a", "b"), ("b", "a")):
            wid = row.get(f"window_id_{side}") or pwin.get(f"window_id_{side}")
            if not wid:
                continue
            other_active = active == other and active_conf > 0
            this_active = active == side and active_conf > 0
            ratio_this_over_other = _num(values.get(f"activity_ratio_{side}_over_{other}"))
            ratio_other_over_this = _num(values.get(f"activity_ratio_{other}_over_{side}"))
            below_other = _num(values.get(f"{other}_pelvis_above_{side}_pelvis_score_proxy"))
            receiver_static = _num(values.get(f"receiver_static_context_proxy_{other}_active"))
            if side == "a":
                below_other = _num(values.get("b_pelvis_above_a_pelvis_score_proxy"))
            elif side == "b":
                below_other = _num(values.get("a_pelvis_above_b_pelvis_score_proxy"))
            active_support = active_conf * (0.7 + 0.3 * min(ratio_this_over_other / 3.0, 1.0)) if this_active else 0.0
            receiver_support = active_conf * (0.45 + 0.25 * min(ratio_other_over_this / 3.0, 1.0) + 0.20 * below_other + 0.10 * receiver_static) if other_active else 0.0
            passive_support = max(receiver_support * 0.8, receiver_static if other_active else 0.0)
            out[str(wid)].append(
                {
                    "pair_window_id": pair_id,
                    "side": side,
                    "active_actor_candidate_side": active,
                    "active_actor_confidence": round(active_conf, 6),
                    "active_support": round(max(0.0, min(1.0, active_support)), 6),
                    "receiver_support": round(max(0.0, min(1.0, receiver_support)), 6),
                    "passive_support": round(max(0.0, min(1.0, passive_support)), 6),
                    "below_other_score": round(max(0.0, min(1.0, below_other)), 6),
                    "other_active_confidence": round(active_conf if other_active else 0.0, 6),
                    "pair_confidence": pwin.get("pair_confidence"),
                }
            )
    return out


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(r.get("rider_receiver_status") for r in rows)
    receiver = sorted(rows, key=lambda r: float(r.get("receiver_body_response_score") or 0.0), reverse=True)[:20]
    active = sorted(rows, key=lambda r: float(r.get("active_rider_score") or 0.0), reverse=True)[:20]
    lines = [
        "# Rider / Receiver Score V1 Report",
        "",
        "Scores are audit/review heuristics only. They do not infer roles from atom names and are not training labels.",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in counts.most_common():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## High Active-Rider Candidates", ""])
    for row in active:
        lines.append(f"- `{row.get('window_id')}` active={row.get('active_rider_score')} receiver={row.get('receiver_body_response_score')} status=`{row.get('rider_receiver_status')}` scene=`{row.get('source_scene_file')}`")
    lines.extend(["", "## High Receiver Body-Response Candidates", ""])
    for row in receiver:
        lines.append(f"- `{row.get('window_id')}` receiver={row.get('receiver_body_response_score')} active={row.get('active_rider_score')} status=`{row.get('rider_receiver_status')}` scene=`{row.get('source_scene_file')}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0

"""Compare wild windows against handmade reference signatures."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl, write_jsonl


FEATURE_MAP = {
    "pelvis_movement_energy": "pelvis_movement_energy",
    "pelvis_vertical_amplitude": "pelvis_vertical_amplitude",
    "pelvis_forward_back_amplitude": "pelvis_forward_back_amplitude",
    "pelvis_lateral_amplitude": "pelvis_lateral_amplitude",
    "head_motion_energy": "head_motion_energy",
    "left_hand_motion_energy": "hand_motion_energy",
    "right_hand_motion_energy": "hand_motion_energy",
}


def compare_wild_to_handmade_references(
    wild_features: str | Path,
    wild_body_quality: str | Path,
    handmade_features: str | Path,
    signatures: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    wild_rows = load_jsonl(wild_features)
    body_by_window = {r.get("window_id"): r for r in load_jsonl(wild_body_quality)}
    hand_rows = load_jsonl(handmade_features)
    sig = load_json(signatures) if Path(signatures).exists() else {"families": {}}
    family_profiles = _family_profiles(hand_rows, sig)
    rows = [_match_row(row, body_by_window.get(row.get("window_id"), {}), family_profiles) for row in wild_rows]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, family_profiles, report)
    return rows


def _family_profiles(rows: list[dict[str, Any]], signatures: dict[str, Any]) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for family, sig in (signatures.get("families") or {}).items():
        med = sig.get("feature_medians", {}) or {}
        profiles[family] = {key: _num(med.get(key)) for key in {"pelvis_movement_energy", "pelvis_vertical_amplitude", "pelvis_forward_back_amplitude", "pelvis_lateral_amplitude", "head_motion_energy", "hand_motion_energy", "leg_motion_energy"}}
    if not profiles and rows:
        for row in rows:
            family = row.get("label_family") or "unknown"
            profiles.setdefault(family, {})
            for key, value in (row.get("feature_values") or {}).items():
                profiles[family].setdefault(key, _num(value))
    return profiles


def _match_row(row: dict[str, Any], body: dict[str, Any], profiles: dict[str, dict[str, float]]) -> dict[str, Any]:
    values = row.get("feature_values", {}) or {}
    wild_vec = _wild_vec(values)
    scores = {}
    for family in ["cowgirl", "doggy", "bj", "hand", "head", "shoulder", "transition"]:
        profile = profiles.get(family) or {}
        scores[f"{family}_reference_score"] = _score(wild_vec, profile)
    q = body.get("body_motion_quality")
    if q in {"controller_only_whole_person_motion", "root_only_motion"}:
        status = "root_or_controller_only_false_positive"
        scores["cowgirl_reference_score"] = min(scores.get("cowgirl_reference_score", 0.0), 0.1)
    elif scores.get("head_reference_score", 0.0) > max(scores.get("cowgirl_reference_score", 0.0), scores.get("doggy_reference_score", 0.0)) and _num(values.get("head_motion_energy")) > _num(values.get("pelvis_movement_energy")) * 2.0:
        status = "likely_not_cowgirl_head_or_bj"
    elif scores.get("cowgirl_reference_score", 0.0) >= 0.55 and q in {"good_body_motion", "partial_body_motion"}:
        status = "likely_cowgirl_candidate"
    elif scores.get("doggy_reference_score", 0.0) >= 0.55:
        status = "likely_doggy_or_other_hip_motion"
    elif scores.get("transition_reference_score", 0.0) >= 0.5:
        status = "likely_transition_or_realign"
    elif max(scores.get("hand_reference_score", 0.0), scores.get("head_reference_score", 0.0)) >= 0.55:
        status = "likely_isolated_gesture"
    else:
        status = "unknown_needs_review"
    best = sorted(((k.replace("_reference_score", ""), v) for k, v in scores.items()), key=lambda x: x[1], reverse=True)[:3]
    return {
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_atom_id": row.get("technical_atom_id"),
        **{k: round(float(v), 4) for k, v in scores.items()},
        "unknown_score": round(1.0 - max(scores.values() or [0.0]), 4),
        "nearest_reference_families": [{"family": k, "score": round(float(v), 4)} for k, v in best],
        "body_motion_quality": q,
        "recommended_review_status": status,
        "is_human_ground_truth": False,
        "warnings": ["Reference matching is review triage only, not wild-data label truth."],
    }


def _wild_vec(values: dict[str, Any]) -> dict[str, float]:
    out = {
        "pelvis_movement_energy": _num(values.get("pelvis_movement_energy")),
        "pelvis_vertical_amplitude": _num(values.get("pelvis_vertical_amplitude")),
        "pelvis_forward_back_amplitude": _num(values.get("pelvis_forward_back_amplitude")),
        "pelvis_lateral_amplitude": _num(values.get("pelvis_lateral_amplitude")),
        "head_motion_energy": _num(values.get("head_motion_energy")),
        "hand_motion_energy": _num(values.get("left_hand_motion_energy")) + _num(values.get("right_hand_motion_energy")),
        "leg_motion_energy": _num(values.get("knee_motion_energy_left")) + _num(values.get("knee_motion_energy_right")) + _num(values.get("foot_motion_energy_left")) + _num(values.get("foot_motion_energy_right")),
    }
    return out


def _score(wild: dict[str, float], profile: dict[str, float]) -> float:
    if not profile:
        return 0.0
    distances = []
    for key, wval in wild.items():
        pval = profile.get(key)
        if not np.isfinite(wval) or pval is None or not np.isfinite(pval):
            continue
        scale = max(abs(wval), abs(pval), 1e-3)
        distances.append(abs(wval - pval) / scale)
    if not distances:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - np.mean(distances))))


def _write_report(rows: list[dict[str, Any]], profiles: dict[str, dict[str, float]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(r.get("recommended_review_status") for r in rows)
    lines = [
        "# Wild-to-Handmade Reference Match Report",
        "",
        "This is not a classifier and does not label wild data as truth. It uses handmade reference signatures to suppress obvious false positives and choose better review examples.",
        "",
        f"- Wild windows compared: {len(rows)}",
        f"- Reference families available: {', '.join(sorted(profiles)) or 'none'}",
        "",
        "## Review Status Counts",
        "",
    ]
    for status, count in counts.most_common():
        lines.append(f"- `{status}`: {count}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0

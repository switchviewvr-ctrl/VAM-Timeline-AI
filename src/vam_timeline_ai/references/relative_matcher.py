"""Reference matching in relative feature and trajectory-shape space."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


FAMILIES = ["cowgirl", "doggy", "bj", "hand", "head", "transition"]


def compare_relative_wild_to_handmade(
    wild_relative_features: str | Path,
    wild_trajectory_features: str | Path,
    handmade_relative_features: str | Path,
    handmade_trajectory_features: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    wild_rel = {r.get("window_id"): r for r in load_jsonl(wild_relative_features) if r.get("window_id")}
    wild_traj = {r.get("window_id"): r for r in load_jsonl(wild_trajectory_features) if r.get("window_id")}
    handmade_rel = load_jsonl(handmade_relative_features)
    handmade_traj = {r.get("reference_id"): r for r in load_jsonl(handmade_trajectory_features) if r.get("reference_id")}
    signatures = _build_family_profiles(handmade_rel, handmade_traj)
    rows: list[dict[str, Any]] = []
    for wid, rel in wild_rel.items():
        traj = wild_traj.get(wid, {})
        rows.append(_match_row(rel, traj, signatures, handmade_rel, handmade_traj))
    write_jsonl(out_jsonl, rows)
    _write_report(rows, signatures, report)
    return rows


def _build_family_profiles(rel_rows: list[dict[str, Any]], traj_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rel_rows:
        family = str(row.get("label_family") or "unknown")
        if row.get("is_transition_or_realign"):
            family = "transition"
        if family not in FAMILIES:
            continue
        merged = _merged_features(row, traj_rows.get(row.get("reference_id"), {}))
        grouped[family].append(merged)
    profiles: dict[str, dict[str, Any]] = {}
    for family, rows in grouped.items():
        keys = sorted({k for r in rows for k in r})
        med = {k: float(np.nanmedian([r.get(k, np.nan) for r in rows])) for k in keys}
        spread = {k: float(np.nanstd([r.get(k, np.nan) for r in rows]) + 0.05) for k in keys}
        profiles[family] = {"count": len(rows), "median": med, "spread": spread}
    return profiles


def _match_row(rel: dict[str, Any], traj: dict[str, Any], profiles: dict[str, dict[str, Any]], refs: list[dict[str, Any]], traj_refs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = _merged_features(rel, traj)
    scores = {family: _score_to_profile(merged, profile) for family, profile in profiles.items()}
    for family in FAMILIES:
        scores.setdefault(family, 0.0)
    safe = bool(rel.get("feature_values", {}).get("safe_for_learning")) and bool(traj.get("safe_for_learning", True))
    if not safe:
        scores = {k: v * 0.35 for k, v in scores.items()}
    traj_vals = traj.get("feature_values", {}) or {}
    grind = _num(traj_vals.get("grind_pattern_score"))
    bounce = _num(traj_vals.get("bounce_pattern_score"))
    forward = _num(traj_vals.get("forward_back_rock_pattern_score"))
    transition = max(_num(traj_vals.get("transition_path_score")), scores["transition"])
    jitter = _num(traj_vals.get("jitter_score"))
    status = _review_status(scores, grind, bounce, forward, transition, jitter, safe)
    nearest = _nearest_refs(merged, refs, traj_refs)
    return {
        "window_id": rel.get("window_id"),
        "sample_id": rel.get("sample_id"),
        "source_id": rel.get("source_id"),
        "source_scene_file": rel.get("source_scene_file"),
        "technical_atom_id": rel.get("technical_atom_id"),
        "cowgirl_relative_score": round(scores["cowgirl"], 6),
        "doggy_relative_score": round(scores["doggy"], 6),
        "bj_relative_score": round(scores["bj"], 6),
        "hand_relative_score": round(scores["hand"], 6),
        "head_relative_score": round(scores["head"], 6),
        "transition_relative_score": round(scores["transition"], 6),
        "unknown_relative_score": round(1.0 - max(scores.values() or [0.0]), 6),
        "cowgirl_grind_trajectory_score": round(grind, 6),
        "cowgirl_bounce_trajectory_score": round(bounce, 6),
        "cowgirl_forward_back_rock_score": round(forward, 6),
        "transition_trajectory_score": round(transition, 6),
        "jitter_static_score": round(jitter, 6),
        "recommended_review_status": status,
        "safe_for_learning": safe,
        "trajectory_shape_classification": traj.get("trajectory_shape_classification"),
        "nearest_handmade_references": nearest,
        "warning": "Relative reference matching is review triage, not wild-data truth.",
        "is_human_ground_truth": False,
    }


def _merged_features(rel: dict[str, Any], traj: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix, row in [("rel", rel), ("traj", traj)]:
        for key, value in (row.get("feature_values", {}) or {}).items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                out[f"{prefix}_{key}"] = float(value)
    return out


def _score_to_profile(values: dict[str, float], profile: dict[str, Any]) -> float:
    median = profile.get("median", {})
    spread = profile.get("spread", {})
    common = [k for k in values if k in median]
    if not common:
        return 0.0
    distances = []
    for key in common:
        distances.append(abs(values[key] - median[key]) / max(spread.get(key, 0.1), 0.05))
    distance = float(np.nanmean(distances)) if distances else 999.0
    return float(np.clip(1.0 / (1.0 + distance), 0.0, 1.0))


def _nearest_refs(values: dict[str, float], refs: list[dict[str, Any]], traj_refs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for ref in refs:
        merged = _merged_features(ref, traj_refs.get(ref.get("reference_id"), {}))
        common = [k for k in values if k in merged]
        if not common:
            continue
        distance = float(np.nanmean([abs(values[k] - merged[k]) for k in common]))
        scored.append((distance, ref))
    out = []
    for distance, ref in sorted(scored, key=lambda x: x[0])[:5]:
        out.append({"reference_id": ref.get("reference_id"), "label_family": ref.get("label_family"), "style": ref.get("style"), "distance": round(distance, 6)})
    return out


def _review_status(scores: dict[str, float], grind: float, bounce: float, forward: float, transition: float, jitter: float, safe: bool) -> str:
    if not safe:
        return "unsafe_relative_motion"
    if jitter >= 0.70:
        return "likely_isolated_gesture"
    if max(scores["bj"], scores["head"]) > scores["cowgirl"] + 0.10:
        return "likely_not_cowgirl_head_or_bj"
    if transition >= 0.58:
        return "likely_transition_or_realign"
    if scores["doggy"] > scores["cowgirl"] + 0.08:
        return "likely_doggy_or_other_hip_motion"
    if scores["cowgirl"] >= 0.42 or max(grind, bounce, forward) >= 0.50:
        return "likely_cowgirl_candidate"
    return "unknown_needs_review"


def _write_report(rows: list[dict[str, Any]], profiles: dict[str, dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    statuses = Counter(r.get("recommended_review_status") for r in rows)
    lines = [
        "# Relative Reference Match Report",
        "",
        "Wild windows are compared to handmade references in relative + trajectory feature space. Scores are review triage, not labels.",
        "",
        f"- Wild windows: {len(rows)}",
        "",
        "## Reference Profiles",
        "",
    ]
    lines.extend(f"- `{family}`: {profile.get('count', 0)} references" for family, profile in sorted(profiles.items()))
    lines.extend(["", "## Review Status Counts", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in statuses.most_common())
    lines.extend(["", "## Top Relative Cowgirl Matches", ""])
    for row in sorted(rows, key=lambda r: float(r.get("cowgirl_relative_score") or 0.0), reverse=True)[:20]:
        lines.append(f"- `{row.get('window_id')}` cowgirl={row.get('cowgirl_relative_score')} grind={row.get('cowgirl_grind_trajectory_score')} status={row.get('recommended_review_status')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


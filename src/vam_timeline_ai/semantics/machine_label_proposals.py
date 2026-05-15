"""Generate numeric-rule machine label proposals.

These proposals are not human ground truth. They use feature proxies and pair
features only, with weak labels as supporting evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vam_timeline_ai.io.identity import stable_hash
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.machine_label_schema import MachineLabelProposal


def generate_machine_label_proposals_v1(
    run_dir: str | Path,
    features: str | Path,
    pair_features: str | Path,
    weak_labels: str | Path,
    windows: str | Path,
    pair_windows: str | Path,
    out_jsonl: str | Path,
    out_yaml: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    feature_rows = load_jsonl(features)
    pair_feature_rows = load_jsonl(pair_features) if Path(pair_features).exists() else []
    weak_by_window = {r.get("window_id"): r for r in load_jsonl(weak_labels)}
    windows_by_id = {r.get("window_id"): r for r in load_jsonl(windows)}
    pair_windows_by_id = {r.get("pair_window_id"): r for r in load_jsonl(pair_windows)} if Path(pair_windows).exists() else {}
    thresholds = _thresholds(feature_rows, pair_feature_rows)
    proposals: list[dict[str, Any]] = []
    for row in feature_rows:
        proposals.extend(_window_proposals(row, windows_by_id.get(row.get("window_id"), {}), weak_by_window.get(row.get("window_id"), {}), thresholds))
    proposals.extend(_pair_proposals(pair_feature_rows, pair_windows_by_id, thresholds))
    proposals.sort(key=lambda r: (r.get("window_id") or "", -(r.get("confidence") or 0), r.get("label") or ""))
    write_jsonl(out_jsonl, proposals)
    _write_yaml(proposals, thresholds, out_yaml)
    _write_report(proposals, thresholds, len(feature_rows), len(pair_feature_rows), report)
    return proposals


def _window_proposals(row: dict[str, Any], window: dict[str, Any], weak: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    v = row.get("feature_values", {})
    out: list[dict[str, Any]] = []
    weak_names = {item.get("label") for item in weak.get("weak_labels", [])}
    warnings = ["machine proposal from numeric proxies only; no visual inspection performed"]
    meta = {
        "window_id": str(row.get("window_id")),
        "pair_window_id": None,
        "sample_id": str(row.get("sample_id") or window.get("sample_id") or ""),
        "source_id": str(row.get("source_id") or window.get("source_id") or ""),
        "source_scene_file": str(row.get("source_scene_file") or window.get("source_scene_file") or ""),
        "technical_atom_id": str(row.get("technical_atom_id") or window.get("technical_atom_id") or ""),
    }
    vertical = _f(v, "pelvis_vertical_amplitude")
    fb = _f(v, "pelvis_forward_back_amplitude")
    lat = _f(v, "pelvis_lateral_amplitude")
    energy = _f(v, "pelvis_movement_energy")
    mean_speed = _f(v, "pelvis_mean_speed")
    fast = _f(v, "fast_motion_score_proxy")
    slow = _f(v, "slow_motion_score_proxy")
    pause = _f(v, "pause_hold_score_proxy")
    irregular = _f(v, "irregular_rhythm_score_proxy")
    accel = _f(v, "pelvis_acceleration_peak_count")
    circular = _f(v, "pelvis_circularity_score_proxy")
    grind = _f(v, "pelvis_grind_score_proxy")
    depth = _f(v, "depth_increase_proxy")
    rock = _f(v, "pelvis_rock_score_proxy")
    torso_forward = _f(v, "torso_lean_forward_proxy")
    torso_back = _f(v, "torso_lean_back_proxy")

    if vertical >= thresholds["p80"].get("pelvis_vertical_amplitude", np.inf) and vertical > max(fb, lat) * 1.15 and energy > thresholds["p50"].get("pelvis_movement_energy", 0):
        out.append(_proposal(meta, "cowgirl_vertical_bounce", "movement", "positive", _conf(vertical, thresholds, "pelvis_vertical_amplitude", weak_names, "weak_v2_high_vertical_motion"), "machine_rule_v1", "pelvis_vertical_dominant_v1", ["pelvis_vertical_amplitude", "pelvis_movement_energy"], v, warnings))
    if fb >= thresholds["p80"].get("pelvis_forward_back_amplitude", np.inf) and fb > lat * 1.2 and rock >= thresholds["p50"].get("pelvis_rock_score_proxy", 0):
        out.append(_proposal(meta, "cowgirl_forward_back_rock", "movement", "positive", _conf(fb, thresholds, "pelvis_forward_back_amplitude", weak_names, "weak_v2_forward_back_dominant"), "machine_rule_v1", "pelvis_forward_back_dominant_v1", ["pelvis_forward_back_amplitude", "pelvis_rock_score_proxy"], v, warnings))
    if lat >= thresholds["p80"].get("pelvis_lateral_amplitude", np.inf) and lat > fb * 1.15:
        out.append(_proposal(meta, "cowgirl_lateral_sway", "movement", "positive", _conf(lat, thresholds, "pelvis_lateral_amplitude", weak_names, "weak_v2_lateral_dominant"), "machine_rule_v1", "pelvis_lateral_dominant_v1", ["pelvis_lateral_amplitude"], v, warnings))
    if circular >= thresholds["p80"].get("pelvis_circularity_score_proxy", np.inf) and grind >= thresholds["p70"].get("pelvis_grind_score_proxy", 0) and min(fb, lat) > 0:
        out.append(_proposal(meta, "cowgirl_circular_grind", "movement", "positive", _mean_conf([_conf(circular, thresholds, "pelvis_circularity_score_proxy", weak_names, "weak_v2_circular_grind_candidate"), _conf(grind, thresholds, "pelvis_grind_score_proxy", weak_names, "weak_v2_circular_grind_candidate")]), "machine_rule_v1", "pelvis_circular_grind_proxy_v1", ["pelvis_circularity_score_proxy", "pelvis_grind_score_proxy"], v, warnings))
    if pause >= thresholds["p85"].get("pause_hold_score_proxy", np.inf) and mean_speed <= thresholds["p35"].get("pelvis_mean_speed", np.inf):
        out.append(_proposal(meta, "cowgirl_pause_hold", "movement", "positive", _conf(pause, thresholds, "pause_hold_score_proxy", weak_names, "weak_v2_pause_hold_candidate"), "machine_rule_v1", "pause_hold_low_speed_v1", ["pause_hold_score_proxy", "pelvis_mean_speed"], v, warnings))
    if irregular >= thresholds["p85"].get("irregular_rhythm_score_proxy", np.inf) or accel >= thresholds["p90"].get("pelvis_acceleration_peak_count", np.inf):
        out.append(_proposal(meta, "cowgirl_adjustment_transition", "movement", "positive", _mean_conf([_conf(irregular, thresholds, "irregular_rhythm_score_proxy", weak_names, "weak_v2_irregular_motion_candidate"), _conf(accel, thresholds, "pelvis_acceleration_peak_count", weak_names, None)]), "machine_rule_v1", "irregular_acceleration_transition_v1", ["irregular_rhythm_score_proxy", "pelvis_acceleration_peak_count"], v, warnings))
    if slow >= thresholds["p75"].get("slow_motion_score_proxy", np.inf) and fast < thresholds["p60"].get("fast_motion_score_proxy", np.inf) and max(vertical, fb, lat, depth) >= thresholds["p60"].get("pelvis_total_position_range", 0):
        out.append(_proposal(meta, "cowgirl_deep_slow", "movement", "positive", _conf(slow, thresholds, "slow_motion_score_proxy", weak_names, "weak_v2_slow_motion_candidate"), "machine_rule_v1", "slow_deep_proxy_v1", ["slow_motion_score_proxy", "pelvis_total_position_range"], v, warnings))
    if fast >= thresholds["p80"].get("fast_motion_score_proxy", np.inf) and mean_speed >= thresholds["p70"].get("pelvis_mean_speed", 0) and max(vertical, fb, lat) <= thresholds["p85"].get("pelvis_total_position_range", np.inf):
        out.append(_proposal(meta, "cowgirl_fast_shallow", "movement", "positive", _conf(fast, thresholds, "fast_motion_score_proxy", weak_names, "weak_v2_fast_motion_candidate"), "machine_rule_v1", "fast_shallow_proxy_v1", ["fast_motion_score_proxy", "pelvis_mean_speed"], v, warnings))
    if torso_forward >= thresholds["p90"].get("torso_lean_forward_proxy", np.inf):
        out.append(_proposal(meta, "cowgirl_lean_forward", "movement", "uncertain", min(0.72, _conf(torso_forward, thresholds, "torso_lean_forward_proxy", weak_names, None)), "machine_rule_v1", "torso_lean_forward_uncertain_axis_v1", ["torso_lean_forward_proxy"], v, warnings + ["axis interpretation uncertain; proposal kept uncertain"]))
    if torso_back >= thresholds["p90"].get("torso_lean_back_proxy", np.inf):
        out.append(_proposal(meta, "cowgirl_lean_back", "movement", "uncertain", min(0.72, _conf(torso_back, thresholds, "torso_lean_back_proxy", weak_names, None)), "machine_rule_v1", "torso_lean_back_uncertain_axis_v1", ["torso_lean_back_proxy"], v, warnings + ["axis interpretation uncertain; proposal kept uncertain"]))
    if irregular >= thresholds["p80"].get("irregular_rhythm_score_proxy", np.inf) and energy >= thresholds["p40"].get("pelvis_movement_energy", 0):
        out.append(_proposal(meta, "cowgirl_irregular_human_motion", "movement", "positive", _conf(irregular, thresholds, "irregular_rhythm_score_proxy", weak_names, "weak_v2_irregular_motion_candidate"), "machine_rule_v1", "irregular_human_motion_proxy_v1", ["irregular_rhythm_score_proxy", "pelvis_movement_energy"], v, warnings))
    if energy <= thresholds["p20"].get("pelvis_movement_energy", -np.inf) and mean_speed <= thresholds["p20"].get("pelvis_mean_speed", -np.inf):
        out.append(_proposal(meta, "machine_negative_low_rider_motion_candidate", "negative_candidate", "negative", 0.65, "machine_rule_v1", "low_rider_motion_negative_candidate_v1", ["pelvis_movement_energy", "pelvis_mean_speed"], v, warnings))
    return [p for p in out if p["confidence"] >= 0.4 and not p["label"].startswith("weak_")]


def _pair_proposals(pair_rows: list[dict[str, Any]], pair_windows_by_id: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in pair_rows:
        q = row.get("feature_quality", {})
        v = row.get("feature_values", {})
        active = q.get("active_actor_candidate", "unknown")
        active_conf = float(q.get("active_actor_confidence") or 0.0)
        pair_window = pair_windows_by_id.get(row.get("pair_window_id"), {})
        for side, other in [("a", "b"), ("b", "a")]:
            window_id = row.get(f"window_id_{side}")
            if not window_id:
                continue
            meta = {
                "window_id": str(window_id),
                "pair_window_id": str(row.get("pair_window_id")),
                "sample_id": str(row.get(f"sample_id_{side}") or ""),
                "source_id": "",
                "source_scene_file": str(row.get("source_scene_file") or pair_window.get("source_scene_file") or ""),
                "technical_atom_id": str(row.get(f"technical_atom_id_{side}") or ""),
            }
            warnings = ["pair proposal from numeric proxies only; no visual inspection performed", "active actor is motion-based candidate only"]
            prefix = f"{side}_"
            other_prefix = f"_{other}_"
            chest_vals = [_f(v, f"{prefix}left_hand_to_{other}_chest_distance_mean"), _f(v, f"{prefix}right_hand_to_{other}_chest_distance_mean")]
            pelvis_vals = [_f(v, f"{prefix}left_hand_to_{other}_pelvis_distance_mean"), _f(v, f"{prefix}right_hand_to_{other}_pelvis_distance_mean")]
            chest_near = _finite_min(chest_vals)
            pelvis_near = _finite_min(pelvis_vals)
            support = _f(v, f"{prefix}static_hand_support_on_{other}_candidate_proxy")
            low_chest = min(
                thresholds["p20"].get(f"{prefix}left_hand_to_{other}_chest_distance_mean", np.inf),
                thresholds["p20"].get(f"{prefix}right_hand_to_{other}_chest_distance_mean", np.inf),
            )
            low_pelvis = min(
                thresholds["p20"].get(f"{prefix}left_hand_to_{other}_pelvis_distance_mean", np.inf),
                thresholds["p20"].get(f"{prefix}right_hand_to_{other}_pelvis_distance_mean", np.inf),
            )
            if active == side and active_conf >= 0.65:
                out.append(_proposal(meta, "rider_active", "role_candidate", "role_candidate", min(0.9, active_conf), "machine_pair_rule_v1", "motion_contrast_active_actor_v1", ["active_actor_confidence", f"{side}_motion_energy", f"{other}_motion_energy"], v, warnings))
                other_meta = {
                    "window_id": str(row.get(f"window_id_{other}") or ""),
                    "pair_window_id": str(row.get("pair_window_id")),
                    "sample_id": str(row.get(f"sample_id_{other}") or ""),
                    "source_id": "",
                    "source_scene_file": str(row.get("source_scene_file") or pair_window.get("source_scene_file") or ""),
                    "technical_atom_id": str(row.get(f"technical_atom_id_{other}") or ""),
                }
                out.append(_proposal(other_meta, "partner_context_static", "role_candidate", "role_candidate", min(0.82, active_conf * 0.9), "machine_pair_rule_v1", "motion_contrast_context_actor_v1", ["active_actor_confidence"], v, warnings))
            if active == side and active_conf >= 0.55 and np.isfinite(chest_near) and chest_near <= low_chest and support >= thresholds["p60"].get(f"{prefix}static_hand_support_on_{other}_candidate_proxy", 0):
                out.append(_proposal(meta, "cowgirl_hand_supported_on_partner", "contact_candidate", "contact_candidate", 0.74 + min(0.16, active_conf * 0.12), "machine_pair_rule_v1", "hands_near_other_body_support_v1", [f"{prefix}left_hand_to_{other}_chest_distance_mean", f"{prefix}static_hand_support_on_{other}_candidate_proxy"], v, warnings))
                out.append(_proposal(meta, "cowgirl_hand_supported_on_partner_chest", "contact_candidate", "contact_candidate", 0.76 + min(0.14, active_conf * 0.1), "machine_pair_rule_v1", "hands_near_other_chest_support_v1", [f"{prefix}left_hand_to_{other}_chest_distance_mean", f"{prefix}right_hand_to_{other}_chest_distance_mean"], v, warnings))
            if active == side and active_conf >= 0.55 and np.isfinite(pelvis_near) and pelvis_near <= low_pelvis:
                out.append(_proposal(meta, "cowgirl_hand_supported_on_partner_hips", "contact_candidate", "contact_candidate", 0.72 + min(0.14, active_conf * 0.1), "machine_pair_rule_v1", "hands_near_other_pelvis_v1", [f"{prefix}left_hand_to_{other}_pelvis_distance_mean", f"{prefix}right_hand_to_{other}_pelvis_distance_mean"], v, warnings))
            if row.get("pair_window_id") and active == "unknown" and q.get("has_hand_to_partner_features"):
                out.append(_proposal(meta, "contact_unknown", "contact_candidate", "uncertain", 0.5, "machine_pair_rule_v1", "pair_contact_ambiguous_v1", ["has_hand_to_partner_features"], v, warnings))
    return [p for p in out if not p["label"].startswith("weak_")]


def _proposal(meta: dict[str, str | None], label: str, group: str, ptype: str, confidence: float, source: str, rule_id: str, evidence_features: list[str], values: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    evidence_values = {key: _json_float(values.get(key)) for key in evidence_features if key in values}
    proposal_id = "prop_" + stable_hash([meta.get("window_id", ""), meta.get("pair_window_id", ""), label, ptype, rule_id], 14)
    return MachineLabelProposal(
        proposal_id=proposal_id,
        window_id=str(meta.get("window_id") or ""),
        pair_window_id=meta.get("pair_window_id"),
        sample_id=str(meta.get("sample_id") or ""),
        source_id=str(meta.get("source_id") or ""),
        source_scene_file=str(meta.get("source_scene_file") or ""),
        technical_atom_id=str(meta.get("technical_atom_id") or ""),
        label=label,
        label_group=group,  # type: ignore[arg-type]
        proposal_type=ptype,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, float(confidence))),
        source=source,  # type: ignore[arg-type]
        rule_id=rule_id,
        evidence_features=evidence_features,
        evidence_values=evidence_values,
        warnings=warnings,
        is_silver_candidate=float(confidence) >= 0.75 and ptype != "uncertain",
        is_human_ground_truth=False,
    ).to_dict()


def _thresholds(feature_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    all_values: dict[str, list[float]] = defaultdict(list)
    for row in feature_rows + pair_rows:
        for key, value in row.get("feature_values", {}).items():
            val = _f(row.get("feature_values", {}), key)
            if np.isfinite(val):
                all_values[key].append(val)
    percentiles = {"p20": 20, "p35": 35, "p40": 40, "p50": 50, "p60": 60, "p70": 70, "p75": 75, "p80": 80, "p85": 85, "p90": 90}
    return {name: {key: float(np.percentile(vals, pct)) for key, vals in all_values.items() if vals} for name, pct in percentiles.items()}


def _conf(value: float, thresholds: dict[str, dict[str, float]], key: str, weak: set[str], weak_name: str | None) -> float:
    p75 = thresholds["p75"].get(key, 0.0)
    p90 = thresholds["p90"].get(key, p75 + 1e-6)
    if not np.isfinite(value):
        return 0.0
    base = 0.55 + 0.35 * max(0.0, min(1.0, (value - p75) / max(p90 - p75, 1e-6)))
    if weak_name and weak_name in weak:
        base += 0.05
    return max(0.0, min(0.95, base))


def _mean_conf(values: list[float]) -> float:
    vals = [v for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def _f(values: dict[str, Any], key: str) -> float:
    try:
        val = float(values.get(key))
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def _finite_min(values: list[float]) -> float:
    vals = [v for v in values if np.isfinite(v)]
    return float(min(vals)) if vals else np.nan


def _json_float(value: Any) -> Any:
    try:
        val = float(value)
        return val if np.isfinite(val) else None
    except Exception:
        return value


def _write_yaml(proposals: list[dict[str, Any]], thresholds: dict[str, Any], out: str | Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prop in proposals:
        grouped[prop["window_id"]].append({
            "label": prop["label"],
            "proposal_type": prop["proposal_type"],
            "confidence": prop["confidence"],
            "rule_id": prop["rule_id"],
            "is_silver_candidate": prop["is_silver_candidate"],
            "is_human_ground_truth": False,
        })
    data = {
        "metadata": {
            "label_source": "machine_label_proposals_v1",
            "is_human_ground_truth": False,
            "warning": "Machine proposals are generated from numeric rules/proxies and are not human semantic ground truth.",
        },
        "windows": dict(grouped),
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_report(proposals: list[dict[str, Any]], thresholds: dict[str, Any], window_count: int, pair_count: int, out: str | Path) -> None:
    label_counts = Counter(p["label"] for p in proposals)
    silver_counts = Counter(p["label"] for p in proposals if p.get("is_silver_candidate"))
    bins = Counter(_bin(p["confidence"]) for p in proposals)
    lines = [
        "# Machine Label Proposals v1",
        "",
        "These proposals are generated from numeric features and pair proxies. No visual inspection was performed.",
        "",
        f"- Windows considered: {window_count}",
        f"- Pair feature rows considered: {pair_count}",
        f"- Proposals: {len(proposals)}",
        f"- Silver candidates: {sum(1 for p in proposals if p.get('is_silver_candidate'))}",
        "",
        "## Proposal Counts By Label",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in label_counts.most_common())
    lines.extend(["", "## Silver Candidate Counts By Label", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in silver_counts.most_common())
    lines.extend(["", "## Confidence Distribution", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in sorted(bins.items()))
    lines.extend(["", "## Feature Thresholds Used", ""])
    for pct in ["p20", "p50", "p75", "p90"]:
        sample = {k: round(v, 5) for k, v in list(thresholds.get(pct, {}).items())[:30]}
        lines.append(f"- `{pct}`: {sample}")
    lines.extend(["", "## Warnings", "", "- Labels are proxy-only and not human ground truth.", "- Axis-dependent lean/contact labels require human review.", "- Weak_v2 labels were used only as supporting evidence."])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bin(confidence: float) -> str:
    if confidence >= 0.9:
        return "0.90-1.00"
    if confidence >= 0.75:
        return "0.75-0.90"
    if confidence >= 0.6:
        return "0.60-0.75"
    return "0.40-0.60"

"""Partner-relative feature proxies for clean_v3 interaction semantics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def extract_partner_relative_features_v0(
    pair_windows: str | Path,
    pair_features: str | Path,
    relative_index: str | Path,
    pose_semantics: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    feature_rows = {r.get("pair_window_id"): r for r in load_jsonl(pair_features) if r.get("pair_window_id")}
    pose = {r.get("window_id"): r for r in load_jsonl(pose_semantics) if r.get("window_id")}
    relative = {r.get("window_id"): r for r in load_jsonl(relative_index) if r.get("window_id")}
    rows: list[dict[str, Any]] = []
    for pair in load_jsonl(pair_windows):
        pid = pair.get("pair_window_id")
        if not pid:
            continue
        feature = feature_rows.get(pid, {})
        rows.append(_partner_feature_row(pair, feature, pose, relative))
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def _partner_feature_row(pair: dict[str, Any], feature: dict[str, Any], pose: dict[str, dict[str, Any]], relative: dict[str, dict[str, Any]]) -> dict[str, Any]:
    f = feature.get("feature_values") or {}
    q = feature.get("feature_quality") or {}
    active = str(q.get("active_actor_candidate") or "a")
    a_window = pair.get("window_id_a")
    b_window = pair.get("window_id_b")
    actor_window = a_window if active == "a" else b_window
    partner_window = b_window if active == "a" else a_window
    actor_pose = pose.get(actor_window, {})
    partner_pose = pose.get(partner_window, {})
    above_score = _num(f.get(f"{active}_pelvis_above_{'b' if active == 'a' else 'a'}_pelvis_score_proxy"))
    if active == "a":
        chest_score = _num(f.get("a_hands_near_b_chest_proxy"), 0.0)
        hip_score = _num(f.get("a_hands_near_b_pelvis_proxy"), 0.0)
        left_chest = _num(f.get("a_left_hand_to_b_chest_distance_mean"), None)
        right_chest = _num(f.get("a_right_hand_to_b_chest_distance_mean"), None)
        left_head = _num(f.get("a_left_hand_to_b_head_distance_mean"), None)
        right_head = _num(f.get("a_right_hand_to_b_head_distance_mean"), None)
        left_pelvis = _num(f.get("a_left_hand_to_b_pelvis_distance_mean"), None)
        right_pelvis = _num(f.get("a_right_hand_to_b_pelvis_distance_mean"), None)
    else:
        chest_score = _num(f.get("b_hands_near_a_chest_proxy"), 0.0)
        hip_score = _num(f.get("b_hands_near_a_pelvis_proxy"), 0.0)
        left_chest = _num(f.get("b_left_hand_to_a_chest_distance_mean"), None)
        right_chest = _num(f.get("b_right_hand_to_a_chest_distance_mean"), None)
        left_head = _num(f.get("b_left_hand_to_a_head_distance_mean"), None)
        right_head = _num(f.get("b_right_hand_to_a_head_distance_mean"), None)
        left_pelvis = _num(f.get("b_left_hand_to_a_pelvis_distance_mean"), None)
        right_pelvis = _num(f.get("b_right_hand_to_a_pelvis_distance_mean"), None)
    pelvis_distance = _num(f.get("pelvis_to_pelvis_distance_mean"), None)
    vertical = _num(f.get("pelvis_vertical_offset_a_minus_b_mean"), 0.0)
    if active == "b":
        vertical *= -1.0
    partner_lying = 1.0 if partner_pose.get("pose_family") == "lying_receiver" else 0.0
    if partner_lying == 0.0 and "empty" in str(pair.get(f"technical_atom_id_{'b' if active == 'a' else 'a'}") or "").lower():
        partner_lying = 0.35
    alignment = 0.0
    if pelvis_distance is not None:
        alignment = max(0.0, min(1.0, 1.0 - abs(float(pelvis_distance) - 1.0) / 2.0))
    confidence = max(_num(q.get("active_actor_confidence")), above_score * 0.5, chest_score)
    return {
        "window_id": actor_window,
        "pair_window_id": pair.get("pair_window_id"),
        "partner_window_id": partner_window,
        "source_scene_file": pair.get("source_scene_file"),
        "rider_actor_id": pair.get(f"technical_atom_id_{active}"),
        "partner_actor_id": pair.get(f"technical_atom_id_{'b' if active == 'a' else 'a'}"),
        "active_actor_slot": active,
        "rider_pelvis_to_partner_pelvis_offset": {
            "vertical": round(vertical, 6),
            "forward_back_uncertain": _num(f.get("pelvis_forward_offset_a_minus_b_mean_uncertain_axis"), 0.0),
            "lateral": 0.0,
        },
        "vertical_offset_proxy": round(vertical, 6),
        "forward_back_offset_proxy": _num(f.get("pelvis_forward_offset_a_minus_b_mean_uncertain_axis"), 0.0),
        "lateral_offset_proxy": 0.0,
        "pelvis_alignment_score": round(alignment, 6),
        "rider_above_partner_score": round(above_score, 6),
        "partner_lying_score": round(partner_lying, 6),
        "rider_facing_partner_proxy": 0.5 if alignment > 0.25 else 0.0,
        "lHand_to_partner_chest_distance": left_chest,
        "rHand_to_partner_chest_distance": right_chest,
        "lHand_to_partner_head_distance": left_head,
        "rHand_to_partner_head_distance": right_head,
        "lHand_to_partner_pelvis_distance": left_pelvis,
        "rHand_to_partner_pelvis_distance": right_pelvis,
        "hands_on_partner_chest_score": round(chest_score, 6),
        "hands_on_partner_hips_score": round(hip_score, 6),
        "hands_on_floor_or_bed_proxy": 0.0,
        "partner_context_confidence": round(min(1.0, confidence), 6),
        "rider_pose_family": actor_pose.get("pose_family") or "unknown",
        "partner_pose_family": partner_pose.get("pose_family") or "unknown",
        "warnings": list(pair.get("warnings") or []) + list(feature.get("warnings") or []),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    buckets = Counter()
    for row in rows:
        if _num(row.get("hands_on_partner_chest_score")) >= 0.55:
            buckets["hands_on_partner_chest_candidate"] += 1
        elif _num(row.get("rider_above_partner_score")) >= 0.55:
            buckets["rider_above_partner_candidate"] += 1
        else:
            buckets["unknown_or_weak_partner_context"] += 1
    lines = [
        "# Partner Relative Feature Report V0",
        "",
        "Partner features are relation/contact proxies. They do not use atom names as semantic truth.",
        "",
        f"- Rows: {len(rows)}",
        "",
        "## Buckets",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in buckets.most_common()) if buckets else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        if value != value:
            return default
        return float(value)
    except Exception:
        return default

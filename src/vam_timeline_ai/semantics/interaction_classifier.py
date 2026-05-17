"""Interaction classifier v0 for clean_v3."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.interaction_semantics import InteractionSemanticRecord


def classify_interactions_v0(
    partner_relative_features: str | Path,
    pose_semantics: str | Path,
    semantic_actions: str | Path | None,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    pose = {r.get("window_id"): r for r in load_jsonl(pose_semantics) if r.get("window_id")}
    rows = [_classify_interaction_row(row, pose) for row in load_jsonl(partner_relative_features)]
    data = [r.to_dict() for r in rows]
    write_jsonl(out_jsonl, data)
    _write_report(data, report)
    return data


def _classify_interaction_row(row: dict[str, Any], pose: dict[str, dict[str, Any]]) -> InteractionSemanticRecord:
    wid = str(row.get("window_id"))
    actor_pose = pose.get(wid, {})
    partner_pose = pose.get(row.get("partner_window_id"), {})
    chest = _num(row.get("hands_on_partner_chest_score"))
    hips = _num(row.get("hands_on_partner_hips_score"))
    legs = _num(row.get("hands_on_partner_legs_score"))
    thighs = _num(row.get("hands_on_partner_thighs_score"))
    behind = _num(row.get("hands_behind_partner_support_score"))
    above = _num(row.get("rider_above_partner_score"))
    align = _num(row.get("pelvis_alignment_score"))
    partner_lying = _num(row.get("partner_lying_score"))
    relations: list[str] = []
    if above >= 0.55:
        relations.append("rider_above_partner")
    if align >= 0.35:
        relations.append("pelvis_aligned")
    if partner_lying >= 0.45:
        relations.append("receiver_lying_on_back")
    if align >= 0.35:
        relations.append("rider_facing_partner")
    support = "unknown"
    contact_targets: dict[str, Any] = {}
    lower_body_score = max(legs, thighs)
    contact_scores = {
        "partner_chest": chest,
        "partner_hips": hips,
        "partner_legs_or_thighs": lower_body_score,
    }
    ranked = sorted(contact_scores.items(), key=lambda item: item[1], reverse=True)
    best_target, best_score = ranked[0]
    second_target, second_score = ranked[1]
    margin = max(0.0, best_score - second_score)
    ambiguous = best_score >= 0.35 and margin < 0.12
    if behind >= 0.45 and lower_body_score >= 0.45 and lower_body_score >= max(chest, hips) + 0.08:
        support = "hands_on_partner_legs_or_thighs"
        contact_targets = {"lHand": "partner.leg_or_thigh", "rHand": "partner.leg_or_thigh"}
    elif behind >= 0.45 and max(legs, thighs, hips) >= 0.35:
        support = "ambiguous_behind_support"
        contact_targets = {"lHand": "partner.lower_body_uncertain", "rHand": "partner.lower_body_uncertain"}
    elif behind >= 0.45:
        support = "hands_behind_support"
        contact_targets = {"lHand": "behind_support_unknown", "rHand": "behind_support_unknown"}
    elif chest >= 0.55 and chest >= max(hips, legs, thighs) + 0.12:
        support = "hands_on_partner_chest"
        contact_targets = {"lHand": "partner.chest", "rHand": "partner.chest"}
    elif hips >= 0.45 and hips >= max(chest, legs, thighs) + 0.08:
        support = "hands_on_partner_hips"
        contact_targets = {"lHand": "partner.pelvis", "rHand": "partner.pelvis"}
    elif ambiguous:
        support = "ambiguous_partner_contact"
        contact_targets = {"lHand": best_target, "rHand": best_target}
    elif "hands_forward_support" in (actor_pose.get("support_context") or []):
        support = "hands_on_floor_or_bed"
        contact_targets = {"lHand": "floor_or_bed", "rHand": "floor_or_bed"}
    elif row.get("pair_window_id"):
        support = "hands_free"
    family = "unknown"
    if actor_pose.get("pose_family") == "cowgirl" and above >= 0.35:
        family = "cowgirl"
    elif actor_pose.get("pose_family") == "bj_oral":
        family = "bj_oral"
    elif support.startswith("hands_on") or support in {"hands_behind_support", "ambiguous_behind_support"}:
        family = "hand_support"
    confidence = max(chest, hips, legs, thighs, behind, above * 0.7, align * 0.4)
    warnings = list(row.get("warnings") or [])
    if not row.get("pair_window_id"):
        warnings.append("Pair context is missing; interaction remains unknown.")
    if support == "hands_on_partner_chest" and not row.get("partner_actor_id"):
        warnings.append("Partner chest support requires partner target evidence.")
    if support in {"hands_on_partner_legs_or_thighs", "ambiguous_behind_support"} and row.get("partner_leg_thigh_approximation_used"):
        warnings.append("Partner leg/thigh support used approximate lower-body targets; review before generation.")
    return InteractionSemanticRecord(
        window_id=wid,
        pair_window_id=row.get("pair_window_id"),
        rider_actor_id=row.get("rider_actor_id"),
        partner_actor_id=row.get("partner_actor_id"),
        actor_role="rider" if family == "cowgirl" and above >= 0.35 else "unknown",
        partner_role="receiver" if family == "cowgirl" and row.get("partner_actor_id") else "unknown",
        interaction_family=family,
        rider_pose_family=actor_pose.get("pose_family") or row.get("rider_pose_family") or "unknown",
        partner_pose_family=partner_pose.get("pose_family") or row.get("partner_pose_family") or "unknown",
        partner_relation=relations or ["unknown"],
        contact_targets=contact_targets,
        support_context=support,
        interaction_confidence=round(min(1.0, confidence), 6),
        contact_support_confidence=round(min(1.0, best_score), 6),
        contact_support_margin=round(margin, 6),
        contact_support_ambiguous=bool(ambiguous or support in {"ambiguous_partner_contact", "ambiguous_behind_support"}),
        best_contact_target=best_target,
        second_best_contact_target=second_target,
        partner_context_confidence=round(_num(row.get("partner_context_confidence")), 6),
        hands_on_partner_legs_score=round(legs, 6),
        hands_on_partner_thighs_score=round(thighs, 6),
        hands_behind_partner_support_score=round(behind, 6),
        partner_leg_thigh_approximation_used=bool(row.get("partner_leg_thigh_approximation_used")),
        warnings=_dedupe(warnings),
    )


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    families = Counter(r.get("interaction_family") for r in rows)
    supports = Counter(r.get("support_context") for r in rows)
    lines = [
        "# Interaction Semantics Report V0",
        "",
        "Interactions combine pair evidence, pose context, partner relation, and contact/support targets.",
        "",
        f"- Rows: {len(rows)}",
        "",
        "## Interaction Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common()) if families else lines.append("- None")
    lines.extend(["", "## Support Context", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in supports.most_common()) if supports else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        if value != value:
            return 0.0
        return float(value or 0.0)
    except Exception:
        return 0.0


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(str(item))
    return out

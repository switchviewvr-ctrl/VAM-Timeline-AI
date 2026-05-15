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
    if chest >= 0.55:
        support = "hands_on_partner_chest"
        contact_targets = {"lHand": "partner.chest", "rHand": "partner.chest"}
    elif hips >= 0.45:
        support = "hands_on_partner_hips"
        contact_targets = {"lHand": "partner.pelvis", "rHand": "partner.pelvis"}
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
    elif support.startswith("hands_on"):
        family = "hand_support"
    confidence = max(chest, hips, above * 0.7, align * 0.4)
    warnings = list(row.get("warnings") or [])
    if not row.get("pair_window_id"):
        warnings.append("Pair context is missing; interaction remains unknown.")
    if support == "hands_on_partner_chest" and not row.get("partner_actor_id"):
        warnings.append("Partner chest support requires partner target evidence.")
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

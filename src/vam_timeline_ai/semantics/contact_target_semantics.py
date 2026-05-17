"""Infer target/contact regions from candidate and interaction evidence."""

from __future__ import annotations

from typing import Any


def infer_target_region(candidate: dict[str, Any], interaction: dict[str, Any] | None = None) -> tuple[str, str]:
    inter = interaction or {}
    contact = str(candidate.get("contact_support") or inter.get("support_context") or "unknown")
    best = str(candidate.get("best_contact_target") or "")
    targets = inter.get("contact_targets") or {}
    joined_targets = " ".join(str(v) for v in targets.values())
    hay = " ".join([contact, best, joined_targets]).lower()

    if "chest" in hay:
        return "partner_chest", "contact_support"
    if "leg" in hay or "thigh" in hay or "knee" in hay:
        return "partner_legs_or_thighs", "contact_support"
    if "hip" in hay:
        return "partner_hips", "contact_support"
    if "pelvis" in hay or "genital" in hay:
        return "partner_pelvis_or_genital_area", "contact_support"
    if "floor" in hay or "bed" in hay:
        return "floor_or_bed", "contact_support"
    if "free" in hay:
        return "none", "hands_free"
    if "possible" in hay or "ambiguous" in hay:
        return "partner_unknown", "ambiguous"
    return "unknown", "unknown"


def normalize_contact_support(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "partner_chest": "hands_on_partner_chest",
        "partner_hips": "hands_on_partner_hips",
        "partner_legs": "hands_on_partner_legs_or_thighs",
        "partner_thighs": "hands_on_partner_legs_or_thighs",
        "floor_or_bed": "hands_on_floor_or_bed",
    }
    return aliases.get(text, text or "unknown")

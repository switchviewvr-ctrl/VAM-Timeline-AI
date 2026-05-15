"""Provisional Cowgirl/Riding semantic label taxonomy.

The labels are intentionally multi-label compatible. They are not final ground
truth and should be corrected with manual labels as real VaM visual review
accumulates.
"""

from __future__ import annotations

from vam_timeline_ai.semantics.schema import SemanticLabel


COWGIRL_VERTICAL_BOUNCE = "cowgirl_vertical_bounce"
COWGIRL_FORWARD_BACK_ROCK = "cowgirl_forward_back_rock"
COWGIRL_LATERAL_SWAY = "cowgirl_lateral_sway"
COWGIRL_CIRCULAR_GRIND = "cowgirl_circular_grind"
COWGIRL_DEEP_SLOW = "cowgirl_deep_slow"
COWGIRL_FAST_SHALLOW = "cowgirl_fast_shallow"
COWGIRL_UPRIGHT = "cowgirl_upright"
COWGIRL_LEAN_FORWARD = "cowgirl_lean_forward"
COWGIRL_LEAN_BACK = "cowgirl_lean_back"
COWGIRL_HAND_SUPPORTED_ON_PARTNER = "cowgirl_hand_supported_on_partner"
COWGIRL_HAND_SUPPORTED_ON_PARTNER_CHEST = "cowgirl_hand_supported_on_partner_chest"
COWGIRL_HAND_SUPPORTED_ON_PARTNER_SHOULDERS = "cowgirl_hand_supported_on_partner_shoulders"
COWGIRL_HAND_SUPPORTED_ON_FLOOR_OR_BED = "cowgirl_hand_supported_on_floor_or_bed"
COWGIRL_HANDS_ON_OWN_THIGHS = "cowgirl_hands_on_own_thighs"
COWGIRL_HANDS_ON_OWN_BODY = "cowgirl_hands_on_own_body"
COWGIRL_PAUSE_HOLD = "cowgirl_pause_hold"
COWGIRL_ADJUSTMENT_TRANSITION = "cowgirl_adjustment_transition"
COWGIRL_TEMPO_INCREASE = "cowgirl_tempo_increase"
COWGIRL_TEMPO_DECREASE = "cowgirl_tempo_decrease"
COWGIRL_DEPTH_INCREASE = "cowgirl_depth_increase"
COWGIRL_DEPTH_DECREASE = "cowgirl_depth_decrease"
COWGIRL_IRREGULAR_HUMAN_MOTION = "cowgirl_irregular_human_motion"
UNKNOWN_NEEDS_MANUAL_REVIEW = "unknown_needs_manual_review"


COWGIRL_LABEL_DESCRIPTIONS: dict[str, str] = {
    COWGIRL_VERTICAL_BOUNCE: "Dominant vertical pelvis/hip movement by the rider.",
    COWGIRL_FORWARD_BACK_ROCK: "Dominant forward/back rocking movement.",
    COWGIRL_LATERAL_SWAY: "Noticeable left/right rider weight shift or sway.",
    COWGIRL_CIRCULAR_GRIND: "Circular or grinding pelvis path rather than straight strokes.",
    COWGIRL_DEEP_SLOW: "Lower tempo movement with larger depth/stroke proxy.",
    COWGIRL_FAST_SHALLOW: "Higher tempo movement with smaller depth/stroke proxy.",
    COWGIRL_UPRIGHT: "Rider torso remains mostly upright.",
    COWGIRL_LEAN_FORWARD: "Rider torso leans forward toward partner or support.",
    COWGIRL_LEAN_BACK: "Rider torso leans backward away from partner.",
    COWGIRL_HAND_SUPPORTED_ON_PARTNER: "Hands appear to use partner as support.",
    COWGIRL_HAND_SUPPORTED_ON_PARTNER_CHEST: "Hands appear supported on partner chest area.",
    COWGIRL_HAND_SUPPORTED_ON_PARTNER_SHOULDERS: "Hands appear supported on partner shoulder area.",
    COWGIRL_HAND_SUPPORTED_ON_FLOOR_OR_BED: "Hands appear supported on bed, floor, or environment.",
    COWGIRL_HANDS_ON_OWN_THIGHS: "Rider hands appear placed on own thighs.",
    COWGIRL_HANDS_ON_OWN_BODY: "Rider hands appear placed elsewhere on own body.",
    COWGIRL_PAUSE_HOLD: "Brief low-motion hold or pause with semantic value.",
    COWGIRL_ADJUSTMENT_TRANSITION: "Posture, support, tempo, or alignment adjustment.",
    COWGIRL_TEMPO_INCREASE: "Tempo increases across the movement window.",
    COWGIRL_TEMPO_DECREASE: "Tempo decreases across the movement window.",
    COWGIRL_DEPTH_INCREASE: "Depth/stroke proxy increases across the movement window.",
    COWGIRL_DEPTH_DECREASE: "Depth/stroke proxy decreases across the movement window.",
    COWGIRL_IRREGULAR_HUMAN_MOTION: "Natural irregularity, non-looping nuance, or imperfect rhythm.",
    UNKNOWN_NEEDS_MANUAL_REVIEW: "Window needs manual review before semantic use.",
}


COWGIRL_LABELS: tuple[str, ...] = tuple(COWGIRL_LABEL_DESCRIPTIONS.keys())

MULTI_LABEL_EXAMPLE: tuple[str, ...] = (
    COWGIRL_LEAN_FORWARD,
    COWGIRL_HAND_SUPPORTED_ON_PARTNER_CHEST,
    COWGIRL_DEEP_SLOW,
)


def semantic_labels() -> dict[str, SemanticLabel]:
    return {
        label_id: SemanticLabel(label_id=label_id, description=description)
        for label_id, description in COWGIRL_LABEL_DESCRIPTIONS.items()
    }


def unknown_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    known = set(COWGIRL_LABELS)
    return [label for label in labels if label not in known]


def validate_label_set(labels: list[str] | tuple[str, ...]) -> None:
    unknown = unknown_labels(labels)
    if unknown:
        raise ValueError(f"Unknown Cowgirl/Riding labels: {unknown}")

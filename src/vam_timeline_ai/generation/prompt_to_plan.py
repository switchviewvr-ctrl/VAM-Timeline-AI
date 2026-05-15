"""Rule-based prompt-to-plan prototype.

This is only an internal planning prototype, not final text-to-animation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.semantic_motion_plan import PrimitiveQuery, SemanticMotionPlan, SemanticMotionPlanPhase, new_plan_id
from vam_timeline_ai.io.json_utils import dump_json


def draft_motion_plan_v0(prompt: str, out: str | Path) -> dict[str, Any]:
    plan = plan_from_prompt(prompt)
    data = plan.to_dict()
    dump_json(out, data)
    return data


def draft_motion_plan_v1(prompt: str, out: str | Path) -> dict[str, Any]:
    plan = plan_from_prompt_v1(prompt)
    data = plan.to_dict()
    dump_json(out, data)
    return data


def plan_from_prompt(prompt: str) -> SemanticMotionPlan:
    text = prompt.lower()
    family = "cowgirl" if "cowgirl" in text or "riding" in text or "ride" in text else "unknown"
    subtype = "riding"
    trajectory = "forward_back_rock"
    if "grind" in text or "grinding" in text:
        subtype = "oval_grind"
        trajectory = "oval_grind"
    elif "bounce" in text:
        subtype = "vertical_bounce"
        trajectory = "vertical_bounce"
    elif "rock" in text or "riding" in text or "ride" in text:
        subtype = "forward_back_rock"
        trajectory = "forward_back_rock"
    tempo = "slow" if "slow" in text else "fast" if "fast" in text or "hard" in text else "medium"
    depth = "deep" if "deep" in text else "shallow" if "shallow" in text else "medium"
    intensity = "high" if "hard" in text or "intense" in text else "low" if "gentle" in text or "soft" in text else "medium"
    amplitude = "large" if depth == "deep" else "small" if depth == "shallow" else "medium"
    body_parameters = {}
    if "leaning forward" in text or "lean forward" in text:
        body_parameters["torso_lean"] = "forward"
    if "leaning back" in text or "lean back" in text:
        body_parameters["torso_lean"] = "back"
    contact_parameters = {}
    if "hand" in text or "hands" in text:
        contact_parameters["hand_support"] = "requested"
    query = PrimitiveQuery(
        family=family,
        subtype=subtype,
        trajectory_shape=trajectory,
        tempo=tempo,
        intensity=intensity,
        depth=depth,
        amplitude=amplitude,
        duration_range={"min_seconds": 4.0, "max_seconds": 8.0},
        generation_safe_required=True,
        anchor_complete_required=True,
    )
    phase = SemanticMotionPlanPhase(
        phase_id="phase_001",
        phase_type="clean_motion",
        primitive_query=query,
        duration=6.0 if tempo == "slow" else 4.0,
        style_parameters={"tempo": tempo, "intensity": intensity, "depth": depth, "amplitude": amplitude},
        body_parameters=body_parameters,
        contact_parameters=contact_parameters,
        safety_requirements={
            "coordinate_space": "relative_body_motion",
            "no_world_coordinates": True,
            "no_person_root_tracks": True,
            "no_timeline_clip_stitching": True,
        },
    )
    return SemanticMotionPlan(
        plan_id=new_plan_id(),
        source_prompt=prompt,
        family=family,
        requested_subtypes=[subtype],
        sequence=[phase],
        warnings=["Rule-based draft only. This is not final text-to-animation and exports no Timeline."],
        is_final_text_to_animation=False,
    )


def plan_from_prompt_v1(prompt: str) -> SemanticMotionPlan:
    text = prompt.lower()
    base = plan_from_prompt(prompt)
    phase = base.sequence[0]
    query = phase.primitive_query
    contact_targets: dict[str, str] = {}
    support_mode = "hands_free"
    if _contains_any(text, ["hands on partner chest", "hands on man's chest", "hands on his chest", "hände auf brust", "haende auf brust", "stützt sich an der brust ab", "stuetzt sich an der brust ab"]):
        support_mode = "hands_on_partner_chest"
        contact_targets = {"lHand": "partner.chest", "rHand": "partner.chest"}
    elif _contains_any(text, ["hands on shoulders", "hands on his shoulders", "hände auf schultern", "haende auf schultern"]):
        support_mode = "hands_on_partner_shoulders"
        contact_targets = {"lHand": "partner.shoulder", "rHand": "partner.shoulder"}
    elif _contains_any(text, ["hands on hips", "hands on his hips", "hände auf hüfte", "haende auf huefte"]):
        support_mode = "hands_on_partner_hips"
        contact_targets = {"lHand": "partner.hips", "rHand": "partner.hips"}
    elif _contains_any(text, ["hands free", "no hands", "hände frei", "haende frei"]):
        support_mode = "hands_free"
    pose_subtype = "cowgirl_lean_forward_supported" if support_mode == "hands_on_partner_chest" or phase.body_parameters.get("torso_lean") == "forward" else "cowgirl_kneeling"
    query.requested_pose_family = "cowgirl" if base.family == "cowgirl" else "unknown"
    query.requested_pose_subtype = pose_subtype
    query.partner_relation = "rider_over_receiver" if base.family == "cowgirl" else "unknown"
    query.coordinate_frame = "partner_pelvis_local" if base.family == "cowgirl" else "body_relative"
    query.contact_targets = contact_targets
    phase.contact_parameters.update({
        "support_mode": support_mode,
        "contact_targets": contact_targets,
    })
    phase.interaction = {
        "partner_relation": "rider_over_receiver" if base.family == "cowgirl" else "unknown",
        "coordinate_frame": query.coordinate_frame,
        "contact_targets": contact_targets,
        "support_mode": support_mode,
    }
    phase.anchors = {
        "required": ["lKneeControl", "rKneeControl", "lFootControl", "rFootControl"],
        "hands_support_required": support_mode != "hands_free",
    }
    phase.constraints = [
        "keep_pelvis_aligned_to_partner",
        "no_world_coords",
        "no_person_root_tracks",
    ]
    if support_mode == "hands_on_partner_chest":
        phase.constraints.append("keep_hands_near_partner_chest")
    base.actor_role = "rider" if base.family == "cowgirl" else "unknown"
    base.partner_role = "receiver" if base.family == "cowgirl" else "unknown"
    base.requested_pose_family = query.requested_pose_family
    base.requested_pose_subtype = pose_subtype
    base.warnings = [
        "Rule-based interaction plan only. This is not final text-to-animation.",
        "Plan includes pose, partner relation, and contact/support constraints for generation.",
    ]
    return base


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)

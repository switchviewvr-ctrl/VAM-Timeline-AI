"""Translate prompts into top-down MotionIntentPlan records."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

from vam_timeline_ai.generation.motion_intent_plan import MotionDriverSpec, MotionIntentPlan, new_motion_intent_plan_id
from vam_timeline_ai.io.json_utils import dump_json
from vam_timeline_ai.semantics.ontology_loader import load_motion_families, load_yaml


def translate_motion_intent_v1(prompt: str, ontology: str | Path, phrases: str | Path, out: str | Path) -> dict[str, Any]:
    families = load_motion_families(ontology)
    phrase_map = (load_yaml(phrases).get("phrases") or {})
    matches = _match_phrases(prompt, phrase_map)
    inferred = _merge_matches(matches)
    family = inferred.get("family") or _infer_family(prompt) or "unknown"
    fam_def = families.get(family, {})
    motion_subtype = inferred.get("motion_subtype") or fam_def.get("default_motion_subtype") or _default_motion_subtype(family)
    if family == "reverse_cowgirl" and str(motion_subtype).startswith("cowgirl_"):
        motion_subtype = str(motion_subtype).replace("cowgirl_", "reverse_cowgirl_", 1)
    pose_subtype = inferred.get("pose_subtype") or _pose_from_modifier(family, inferred.get("pose_modifier")) or fam_def.get("default_pose_subtype") or _default_pose_subtype(family)
    contact_support = inferred.get("contact_support") or inferred.get("support_context") or "unknown"
    torso_lean = inferred.get("torso_lean") or _torso_from_pose(pose_subtype)
    facing_context = inferred.get("facing_context") or ("back_to_partner" if family == "reverse_cowgirl" else "front_cowgirl" if family == "cowgirl" else "unknown")
    invalid_prevented = _invalid_mappings_prevented(prompt, inferred, family, pose_subtype, facing_context)
    contact_targets = _contact_targets(contact_support)
    unresolved = []
    if family in {"cowgirl", "reverse_cowgirl", "doggy", "bj_oral", "missionary"} and not fam_def.get("required_partner_relations"):
        unresolved.append("partner_relation")
    if contact_support == "unknown" and "hands" in prompt.lower():
        unresolved.append("contact_target")
    driver = MotionDriverSpec(
        primary_bodyparts=list(fam_def.get("primary_motion_centers") or []),
        secondary_bodyparts=list(fam_def.get("secondary_motion_centers") or []),
        shape=_shape_for(motion_subtype),
        rhythm="cyclic" if family in {"cowgirl", "reverse_cowgirl", "doggy", "bj_oral"} else "unknown",
        tempo=inferred.get("tempo") or "medium",
        amplitude=_amplitude_for(motion_subtype),
        axis_priority=_axis_priority_for(fam_def, motion_subtype),
        phase_relationships=_phase_relationships_for(motion_subtype),
    )
    plan = MotionIntentPlan(
        plan_id=new_motion_intent_plan_id(),
        source="prompt",
        source_prompt=prompt,
        family=family,
        motion_subtype=motion_subtype,
        pose_subtype=pose_subtype,
        actor_role=str(fam_def.get("actor_role") or "unknown"),
        partner_role=str(fam_def.get("partner_role") or "unknown"),
        facing_context=facing_context,
        torso_lean=torso_lean,
        coordinate_frame=_first(fam_def.get("coordinate_frame"), "body_relative"),
        partner_relation_requirements=list(fam_def.get("required_partner_relations") or []),
        contact_support=contact_support,
        contact_targets=contact_targets,
        motion_driver=driver,
        followers=dict(fam_def.get("followers") or {}),
        anchors=list(fam_def.get("anchors") or []),
        constraints=_constraints_for(family, pose_subtype, contact_support),
        micro_states=list(fam_def.get("micro_states") or []),
        limiter_rules=_limiter_rules_for(fam_def, pose_subtype, motion_subtype),
        anomaly_guards=list(fam_def.get("anomaly_guards") or []),
        sourcebook_trace=_sourcebook_trace(families, family),
        safety_rules=["no_person_root_tracks", "no_world_transform_targets", "controllers_are_output_layer"],
        confidence=0.72 if family != "unknown" else 0.2,
        completeness="ready_for_scene_context" if family != "unknown" else "incomplete",
        unresolved_requirements=unresolved,
        invalid_mappings_prevented=invalid_prevented,
    )
    dump_json(out, plan.to_dict())
    return {"status": "ok", "out": str(out), "family": plan.family, "motion_subtype": plan.motion_subtype, "pose_subtype": plan.pose_subtype, "contact_support": plan.contact_support, "invalid_mappings_prevented": invalid_prevented}


def _match_phrases(prompt: str, phrase_map: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    normalized = _norm(prompt)
    matches = []
    for phrase, mapping in phrase_map.items():
        if _norm(phrase) in normalized:
            matches.append((phrase, dict(mapping or {})))
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    return matches


def _merge_matches(matches: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _phrase, mapping in reversed(matches):
        out.update(mapping)
    return out


def _infer_family(prompt: str) -> str | None:
    p = _norm(prompt)
    if "reverse cowgirl" in p or "rueckwaerts" in p or "ruecken zu" in p:
        return "reverse_cowgirl"
    if "cowgirl" in p:
        return "cowgirl"
    if "doggy" in p:
        return "doggy"
    if re.search(r"\b(bj|blowjob|oral)\b", p):
        return "bj_oral"
    if "handjob" in p or re.search(r"\bhj\b", p):
        return "handjob"
    if "missionary" in p or "missionar" in p:
        return "missionary"
    return None


def _pose_from_modifier(family: str, pose_modifier: Any) -> str | None:
    modifier = str(pose_modifier or "")
    if family == "cowgirl" and modifier == "lean_forward":
        return "cowgirl_lean_forward_supported"
    if family == "cowgirl" and modifier == "lean_back":
        return "cowgirl_lean_back_supported"
    return None


def _invalid_mappings_prevented(prompt: str, inferred: dict[str, Any], family: str, pose_subtype: str, facing_context: str) -> list[str]:
    p = _norm(prompt)
    prevented = []
    if ("zurueckgelehnt" in p or "leaning back" in p or "lean back" in p) and family == "cowgirl" and facing_context != "back_to_partner":
        prevented.append("cowgirl_lean_back_not_mapped_to_reverse")
    if "kneeling" in p and family != "doggy":
        prevented.append("kneeling_alone_not_mapped_to_doggy")
    if family == "cowgirl" and pose_subtype == "cowgirl_lean_back_supported":
        prevented.append("lean_back_cowgirl_kept_front_facing")
    return prevented


def _default_motion_subtype(family: str) -> str:
    return {
        "cowgirl": "cowgirl_grinding",
        "reverse_cowgirl": "reverse_cowgirl_grinding",
        "doggy": "doggy_forward_back",
        "bj_oral": "bj_head_bob",
        "handjob": "hand_repetitive_up_down",
        "missionary": "missionary_counter_thrust",
    }.get(family, "unknown")


def _default_pose_subtype(family: str) -> str:
    return {
        "cowgirl": "cowgirl_kneeling",
        "reverse_cowgirl": "reverse_cowgirl_kneeling",
        "doggy": "doggy_all_fours",
        "bj_oral": "bj_kneeling",
        "handjob": "any_stable_pose",
        "missionary": "missionary_supine",
    }.get(family, "unknown")


def _torso_from_pose(pose_subtype: str) -> str:
    if "lean_back" in pose_subtype:
        return "backward"
    if "lean_forward" in pose_subtype or "doggy" in pose_subtype or "bj" in pose_subtype:
        return "forward"
    if "upright" in pose_subtype:
        return "upright"
    return "unknown"


def _contact_targets(contact_support: str) -> dict[str, str]:
    target = {
        "hands_on_partner_chest": "partner.chest",
        "hands_on_partner_hips": "partner.hips",
        "hands_on_partner_pelvis_or_thighs": "partner.pelvis_or_thighs",
        "hands_on_partner_legs_or_thighs": "partner.legs_or_thighs",
        "hands_on_floor_or_bed": "support.floor_or_bed",
        "forearms_on_floor_or_bed": "support.floor_or_bed_forearms",
        "hands_behind_support": "support.behind_body",
        "rear_floor_drop": "support.floor_or_bed_behind_body",
        "hands_on_own_thighs": "self.thighs",
        "synchronized_hand_tracking": "partner.pelvis_target_vector",
        "static_pinning": "partner.hips_or_thighs",
    }.get(contact_support)
    return {"lHand": target, "rHand": target} if target else {}


def _shape_for(motion_subtype: str) -> str:
    if "grind" in motion_subtype:
        return "oval_circular_figure8"
    if "bounce" in motion_subtype:
        return "vertical_loop"
    if "rock" in motion_subtype:
        return "forward_back_rock"
    if "corkscrew" in motion_subtype or "helical" in motion_subtype:
        return "helical_translation_rotation_coupling"
    if "spine_wave" in motion_subtype:
        return "sequential_spine_wave"
    if "forward_back" in motion_subtype:
        return "z_axis_thrust"
    if "counter" in motion_subtype:
        return "reactive_counter_motion"
    if "head" in motion_subtype:
        return "head_bob"
    return "unknown"


def _amplitude_for(motion_subtype: str) -> str:
    if "bounce" in motion_subtype:
        return "medium_high"
    if "grind" in motion_subtype:
        return "low_medium"
    return "medium"


def _constraints_for(family: str, pose_subtype: str, contact_support: str) -> list[str]:
    constraints = ["no_person_root_tracks", "no_world_coords"]
    if family == "cowgirl":
        constraints.extend(["keep_pelvis_aligned_to_partner", "pelvis_hip_primary_driver"])
    if pose_subtype == "cowgirl_lean_back_supported":
        constraints.extend(["keep_torso_lean_back", "front_cowgirl_not_reverse"])
    if contact_support in {"hands_on_partner_chest", "hands_on_partner_legs_or_thighs", "hands_behind_support"}:
        constraints.append("keep_hands_near_contact_target")
    return constraints


def _axis_priority_for(family_def: dict[str, Any], motion_subtype: str) -> list[str]:
    axis = family_def.get("axis_priority") or {}
    if not isinstance(axis, dict):
        return []
    key_hints: list[str] = []
    if "bounce" in motion_subtype:
        key_hints.extend(["clean_bounce", "z_thrust", "counter_thrust"])
    if "grind" in motion_subtype:
        key_hints.extend(["grinding", "planar_grind"])
    if "corkscrew" in motion_subtype:
        key_hints.append("grinding")
    if "spine_wave" in motion_subtype:
        key_hints.append("clean_bounce")
    if "head" in motion_subtype or "bj" in motion_subtype:
        key_hints.extend(["bobbing", "helical", "deep_alignment"])
    for key in key_hints:
        value = axis.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
    for value in axis.values():
        if isinstance(value, list):
            return [str(v) for v in value]
    return []


def _phase_relationships_for(motion_subtype: str) -> list[str]:
    if "spine_wave" in motion_subtype:
        return ["pelvis_initiates", "lower_spine_lag_150ms", "chest_lag_300ms", "head_lag_450ms"]
    if "bounce" in motion_subtype:
        return ["chest_head_phase_shift_after_pelvis", "bottom_impact_fast_deceleration", "top_apex_soft_ease"]
    if "helical" in motion_subtype:
        return ["rotation_max_at_translation_apex"]
    if "hand_assisted" in motion_subtype:
        return ["hands_phase_shift_against_head"]
    return []


def _limiter_rules_for(family_def: dict[str, Any], pose_subtype: str, motion_subtype: str) -> list[str]:
    rules: list[str] = []
    if pose_subtype == "bj_deep_alignment" or motion_subtype == "bj_deep_alignment":
        rules.extend(["neck_stress_limiter", "head_dof_tolerance_under_2deg"])
    if "corkscrew" in motion_subtype:
        rules.extend(["foot_yaw_slip_tolerance", "thigh_torsion_guard"])
    if "doggy" in motion_subtype:
        rules.append("knee_friction_support_required")
    for guard in family_def.get("anomaly_guards") or []:
        if "limiter" in str(guard) or "guard" in str(guard):
            rules.append(str(guard))
    return sorted(set(rules))


def _sourcebook_trace(families: dict[str, Any], family: str) -> dict[str, Any]:
    fam = families.get(family, {})
    return {
        "source": "Semantik_Master_Konsolidiert.docx",
        "meaning_source": "semantik_master_konsolidiert_docx",
        "family_id": fam.get("family_id", family),
        "root_mapping": "sourcebook root/root-node means pelvisControl/hipControl/abdomen region, never VaM Person/root/world",
    }


def _first(value: Any, default: str) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if value:
        return str(value)
    return default


def _norm(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace("ü", "ue")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ß", "ss")
    )

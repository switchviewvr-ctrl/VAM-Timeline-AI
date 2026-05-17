"""Build canonical semantic stickman base poses from the motion ontology."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.semantic_stickman import SemanticStickmanPose
from vam_timeline_ai.io.json_utils import dump_json
from vam_timeline_ai.semantics.ontology_loader import load_motion_families


def build_semantic_stickman_pose_library_v1(ontology: str | Path, out_json: str | Path, report: str | Path) -> dict[str, Any]:
    families = load_motion_families(ontology)
    poses = _canonical_poses(families)
    data = {
        "schema_version": "semantic_stickman_pose_library_v1",
        "source_ontology": str(ontology),
        "production_vam_targets": False,
        "uses_person_root_or_world": False,
        "poses": [pose.to_dict() for pose in poses],
    }
    dump_json(out_json, data)
    counts: dict[str, int] = {}
    for pose in poses:
        counts[pose.family] = counts.get(pose.family, 0) + 1
    lines = [
        "# Semantic Stickman Pose Library V1",
        "",
        "Schematic relative poses for ontology sanity checking. These are not VaM production controller targets.",
        "",
        f"- Source ontology: `{ontology}`",
        f"- Pose concepts: {len(poses)}",
        f"- Counts by family: {counts}",
        "- Person/root/world transforms used: false",
        "- Timeline animation generated: false",
        "- ML training performed: false",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "pose_count": len(poses), "family_counts": counts, "out_json": str(out_json), "report": str(report)}


def _canonical_poses(families: dict[str, Any]) -> list[SemanticStickmanPose]:
    partner = _partner_supine()
    standing_partner = _partner_standing()
    return [
        _pose("cowgirl_upright", "cowgirl", "cowgirl_upright", _cowgirl_points("upright"), partner, "front_cowgirl", "upright", ["feet", "knees"], ["pelvis_hip"], ["chest", "head"]),
        _pose("cowgirl_kneeling", "cowgirl", "cowgirl_kneeling", _cowgirl_points("kneeling"), partner, "front_cowgirl", "upright", ["feet", "knees"], ["pelvis_hip"], ["chest", "head"]),
        _pose("cowgirl_squat", "cowgirl", "cowgirl_squat", _cowgirl_points("squat"), partner, "front_cowgirl", "upright", ["feet", "knees"], ["pelvis_hip"], ["chest", "head"]),
        _pose("cowgirl_lean_forward_supported", "cowgirl", "cowgirl_lean_forward_supported", _cowgirl_points("lean_forward"), partner, "front_cowgirl", "forward", ["feet", "knees", "hands"], ["pelvis_hip"], ["chest", "head"], {"lHand": "partner.chest", "rHand": "partner.chest"}, "hands_on_partner_chest"),
        _pose("cowgirl_lean_back_supported", "cowgirl", "cowgirl_lean_back_supported", _cowgirl_points("lean_back"), partner, "front_cowgirl", "backward", ["feet", "knees", "hands"], ["pelvis_hip"], ["chest", "head"], {"lHand": "partner.legs_or_thighs", "rHand": "partner.legs_or_thighs"}, "hands_behind_support", ["reverse_cowgirl"]),
        _pose("reverse_cowgirl_standing_squat", "reverse_cowgirl", "reverse_cowgirl_standing_squat", _reverse_cowgirl_points("squat"), partner, "back_to_partner", "upright", ["feet", "knees"], ["pelvis_hip"], ["chest", "head"], not_labels=["cowgirl_lean_back_supported"]),
        _pose("reverse_cowgirl_kneeling", "reverse_cowgirl", "reverse_cowgirl_kneeling", _reverse_cowgirl_points("kneeling"), partner, "back_to_partner", "upright", ["feet", "knees"], ["pelvis_hip"], ["chest", "head"], not_labels=["cowgirl_lean_back_supported"]),
        _pose("doggy_all_fours", "doggy", "doggy_all_fours", _doggy_points("all_fours"), _partner_behind(), "partner_behind", "forward", ["hands", "knees", "feet"], ["pelvis_hip"], ["chest", "head"], {"lHand": "floor", "rHand": "floor"}, "hands_on_floor_or_bed"),
        _pose("doggy_bent_forward", "doggy", "doggy_bent_forward", _doggy_points("bent_forward"), _partner_behind(), "partner_behind", "forward", ["hands", "knees", "feet"], ["pelvis_hip"], ["chest", "head"], {"lHand": "floor", "rHand": "floor"}, "hands_on_floor_or_bed"),
        _pose("doggy_elevated_support", "doggy", "doggy_elevated_support", _doggy_points("elevated"), _partner_behind(), "partner_behind", "forward", ["hands", "knees", "feet"], ["pelvis_hip"], ["chest", "head"], {"lHand": "elevated_support", "rHand": "elevated_support"}, "chest_elevated_support"),
        _pose("bj_kneeling_forward", "bj_oral", "bj_kneeling", _bj_points("kneeling"), standing_partner, "partner_in_front", "forward", ["knees", "feet"], ["head_neck", "chest_abdomen"], ["pelvis"], {"head": "partner.pelvis"}, "hands_on_partner_hips"),
        _pose("bj_low_forward", "bj_oral", "bj_low_or_lying", _bj_points("low"), standing_partner, "partner_in_front", "forward", ["knees", "hands"], ["head_neck", "chest_abdomen"], ["pelvis"], {"head": "partner.pelvis"}, "hands_on_floor_or_bed"),
        _pose("bj_hand_assisted", "bj_oral", "bj_kneeling", _bj_points("hand_assisted"), standing_partner, "partner_in_front", "forward", ["knees"], ["head_neck", "chest_abdomen"], ["pelvis", "hands"], {"head": "partner.pelvis", "lHand": "partner.pelvis", "rHand": "partner.pelvis"}, "synchronized_hand_tracking"),
        _pose("missionary_supine", "missionary", "missionary_supine", _missionary_points("supine"), _partner_above(), "active_partner_above", "supine", ["chest", "head", "back"], ["pelvis_counter_driver"], ["legs", "head"], support_context="back_support"),
        _pose("missionary_legs_up", "missionary", "missionary_legs_up", _missionary_points("legs_up"), _partner_above(), "active_partner_above", "supine", ["chest", "head", "shoulders"], ["pelvis_counter_driver"], ["legs"], {"lFoot": "partner.back", "rFoot": "partner.back"}, "partner_hooking"),
        _pose("missionary_flat_passive", "missionary", "missionary_flat_passive", _missionary_points("flat"), _partner_above(), "active_partner_above", "supine", ["chest", "head", "back"], ["static_pose"], ["pelvis"], support_context="body_grounded", not_labels=["cowgirl"]),
        _pose("standing_hand_head_gesture", "standing_hand_head", "standing_upright", _standing_points(), {}, "unknown", "upright", ["feet"], ["hands", "head_neck"], ["pelvis"], support_context="hands_free", not_labels=["cowgirl"]),
        _pose("broken_pose_example", "unknown", "unknown", _broken_points(), {}, "unknown", "unknown", [], ["unknown"], [], warnings=["intentionally broken schematic pose"]),
    ]


def _pose(
    concept_id: str,
    family: str,
    pose_subtype: str,
    points: dict[str, tuple[float, float, float]],
    partner: dict[str, tuple[float, float, float]],
    facing: str,
    torso: str,
    anchors: list[str],
    drivers: list[str],
    followers: list[str],
    contact_targets: dict[str, str] | None = None,
    support_context: str = "unknown",
    not_labels: list[str] | None = None,
    warnings: list[str] | None = None,
) -> SemanticStickmanPose:
    return SemanticStickmanPose(
        concept_id=concept_id,
        family=family,
        pose_subtype=pose_subtype,
        actor_role=_actor_role(family),
        partner_role=_partner_role(family),
        coordinate_frame="partner_pelvis_local" if partner else "body_relative",
        facing_context=facing,
        torso_lean=torso,
        controller_points=points,
        partner_reference_points=partner,
        anchors=anchors,
        drivers=drivers,
        followers=followers,
        contact_targets=contact_targets or {},
        support_context=support_context,
        not_labels=not_labels or [],
        warnings=warnings or [],
    )


def _actor_role(family: str) -> str:
    return {"cowgirl": "rider", "reverse_cowgirl": "rider", "doggy": "receiver_or_front_actor", "bj_oral": "giver", "missionary": "receiver_or_bottom_actor"}.get(family, "actor")


def _partner_role(family: str) -> str:
    return {"cowgirl": "receiver", "reverse_cowgirl": "receiver", "doggy": "driver_or_behind_actor", "bj_oral": "receiver", "missionary": "active_partner_above"}.get(family, "unknown")


def _cowgirl_points(mode: str) -> dict[str, tuple[float, float, float]]:
    pelvis_y = {"upright": 1.20, "kneeling": 1.05, "squat": 0.88, "lean_forward": 1.08, "lean_back": 1.08}[mode]
    z = 0.0
    chest = (0, pelvis_y + 0.62, 0.05)
    head = (0, pelvis_y + 0.95, 0.07)
    hands = ((-0.38, pelvis_y + 0.38, 0.16), (0.38, pelvis_y + 0.38, 0.16))
    if mode == "lean_forward":
        chest, head = (0, pelvis_y + 0.48, 0.42), (0, pelvis_y + 0.77, 0.55)
        hands = ((-0.34, pelvis_y + 0.26, 0.82), (0.34, pelvis_y + 0.26, 0.82))
    if mode == "lean_back":
        chest, head = (0, pelvis_y + 0.50, -0.42), (0, pelvis_y + 0.78, -0.58)
        hands = ((-0.42, pelvis_y + 0.18, -0.72), (0.42, pelvis_y + 0.18, -0.72))
    return {
        "pelvis": (0, pelvis_y, z),
        "abdomen": (0, pelvis_y + 0.28, (chest[2]) * 0.45),
        "chest": chest,
        "head": head,
        "lElbow": (-0.28, pelvis_y + 0.44, (hands[0][2] + chest[2]) / 2),
        "rElbow": (0.28, pelvis_y + 0.44, (hands[1][2] + chest[2]) / 2),
        "lHand": hands[0],
        "rHand": hands[1],
        "lThigh": (-0.28, pelvis_y - 0.12, 0.08),
        "rThigh": (0.28, pelvis_y - 0.12, 0.08),
        "lKnee": (-0.55, 0.55, 0.22),
        "rKnee": (0.55, 0.55, 0.22),
        "lFoot": (-0.72, 0.18, 0.45),
        "rFoot": (0.72, 0.18, 0.45),
    }


def _reverse_cowgirl_points(mode: str) -> dict[str, tuple[float, float, float]]:
    pts = _cowgirl_points("squat" if mode == "squat" else "kneeling")
    return {k: (v[0], v[1], -v[2]) for k, v in pts.items()}


def _doggy_points(mode: str) -> dict[str, tuple[float, float, float]]:
    pelvis_y = 1.10 if mode != "elevated" else 1.22
    hand_y = 0.18 if mode != "elevated" else 0.62
    chest_y = 0.55 if mode != "bent_forward" else 0.42
    return {
        "pelvis": (0, pelvis_y, -0.10),
        "abdomen": (0, 0.88, 0.20),
        "chest": (0, chest_y, 0.72),
        "head": (0, chest_y + 0.18, 1.05),
        "lElbow": (-0.32, hand_y + 0.18, 0.88),
        "rElbow": (0.32, hand_y + 0.18, 0.88),
        "lHand": (-0.42, hand_y, 1.08),
        "rHand": (0.42, hand_y, 1.08),
        "lThigh": (-0.24, 0.92, -0.35),
        "rThigh": (0.24, 0.92, -0.35),
        "lKnee": (-0.42, 0.22, -0.55),
        "rKnee": (0.42, 0.22, -0.55),
        "lFoot": (-0.48, 0.12, -1.02),
        "rFoot": (0.48, 0.12, -1.02),
    }


def _bj_points(mode: str) -> dict[str, tuple[float, float, float]]:
    low = mode == "low"
    pelvis_y = 0.45 if low else 0.62
    chest_y = 0.72 if low else 0.98
    head_z = 1.18
    return {
        "pelvis": (0, pelvis_y, -0.45),
        "abdomen": (0, (pelvis_y + chest_y) / 2, -0.05),
        "chest": (0, chest_y, 0.48),
        "head": (0, chest_y + 0.08, head_z),
        "lElbow": (-0.26, chest_y - 0.08, 0.70),
        "rElbow": (0.26, chest_y - 0.08, 0.70),
        "lHand": (-0.36, 0.62, 1.02),
        "rHand": (0.36, 0.62, 1.02),
        "lThigh": (-0.28, pelvis_y - 0.12, -0.55),
        "rThigh": (0.28, pelvis_y - 0.12, -0.55),
        "lKnee": (-0.38, 0.16, -0.85),
        "rKnee": (0.38, 0.16, -0.85),
        "lFoot": (-0.45, 0.10, -1.15),
        "rFoot": (0.45, 0.10, -1.15),
    }


def _missionary_points(mode: str) -> dict[str, tuple[float, float, float]]:
    leg_up = mode == "legs_up"
    flat = mode == "flat"
    pelvis_y = 0.30 if not flat else 0.18
    knee_y = 1.08 if leg_up else 0.50
    foot_y = 1.42 if leg_up else 0.35
    return {
        "pelvis": (0, pelvis_y, 0.0),
        "abdomen": (0, 0.22, 0.30),
        "chest": (0, 0.18, 0.68),
        "head": (0, 0.20, 1.02),
        "lElbow": (-0.36, 0.20, 0.70),
        "rElbow": (0.36, 0.20, 0.70),
        "lHand": (-0.55, 0.20, 0.55),
        "rHand": (0.55, 0.20, 0.55),
        "lThigh": (-0.28, 0.42 if leg_up else 0.30, -0.12),
        "rThigh": (0.28, 0.42 if leg_up else 0.30, -0.12),
        "lKnee": (-0.55, knee_y, -0.08 if leg_up else -0.45),
        "rKnee": (0.55, knee_y, -0.08 if leg_up else -0.45),
        "lFoot": (-0.65, foot_y, 0.12 if leg_up else -0.78),
        "rFoot": (0.65, foot_y, 0.12 if leg_up else -0.78),
    }


def _standing_points() -> dict[str, tuple[float, float, float]]:
    return {
        "pelvis": (0, 1.10, 0),
        "abdomen": (0, 1.38, 0),
        "chest": (0, 1.68, 0),
        "head": (0, 2.02, 0.02),
        "lElbow": (-0.38, 1.62, 0.05),
        "rElbow": (0.38, 1.86, 0.06),
        "lHand": (-0.55, 1.38, 0.08),
        "rHand": (0.52, 2.05, 0.08),
        "lThigh": (-0.18, 0.85, 0),
        "rThigh": (0.18, 0.85, 0),
        "lKnee": (-0.20, 0.48, 0),
        "rKnee": (0.20, 0.48, 0),
        "lFoot": (-0.25, 0.08, 0.05),
        "rFoot": (0.25, 0.08, 0.05),
    }


def _broken_points() -> dict[str, tuple[float, float, float]]:
    pts = _standing_points()
    pts["head"] = (0.9, 0.45, 0.4)
    pts["lFoot"] = (0.2, 1.7, -0.8)
    return pts


def _partner_supine() -> dict[str, tuple[float, float, float]]:
    return {
        "partner_pelvis": (0, 0.38, 0.0),
        "partner_chest": (0, 0.34, 0.70),
        "partner_head": (0, 0.36, 1.08),
        "partner_lThigh": (-0.35, 0.30, -0.22),
        "partner_rThigh": (0.35, 0.30, -0.22),
        "partner_lLeg": (-0.45, 0.20, -0.75),
        "partner_rLeg": (0.45, 0.20, -0.75),
    }


def _partner_behind() -> dict[str, tuple[float, float, float]]:
    return {
        "partner_pelvis": (0, 1.00, -1.10),
        "partner_chest": (0, 1.45, -1.45),
        "partner_head": (0, 1.78, -1.70),
    }


def _partner_standing() -> dict[str, tuple[float, float, float]]:
    return {
        "partner_pelvis": (0, 1.02, 1.45),
        "partner_chest": (0, 1.55, 1.48),
        "partner_head": (0, 1.88, 1.50),
        "partner_lThigh": (-0.22, 0.72, 1.45),
        "partner_rThigh": (0.22, 0.72, 1.45),
    }


def _partner_above() -> dict[str, tuple[float, float, float]]:
    return {
        "partner_pelvis": (0, 0.92, -0.15),
        "partner_chest": (0, 1.22, 0.52),
        "partner_head": (0, 1.42, 0.85),
    }

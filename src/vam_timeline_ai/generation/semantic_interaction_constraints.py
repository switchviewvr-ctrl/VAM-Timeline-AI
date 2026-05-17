"""Contact-aware interaction constraints for semantic stickman previews.

This layer is still only an ontology visualization sanity check. It constrains
schematic stickman poses around partner-relative interaction targets; it does
not create VaM Timeline controller targets and never uses Person/root/world
transforms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import copy
import math

from vam_timeline_ai.generation.semantic_stickman import as_point3


Point3 = tuple[float, float, float]


@dataclass
class InteractionConstraint:
    constraint_id: str
    family: str
    required_targets: list[str]
    actor_anchor_bodypart: str
    partner_target_bodypart: str
    target_distance_max: float
    target_distance_ideal: float
    vertical_offset_range: tuple[float, float]
    forward_back_offset_range: tuple[float, float]
    lateral_offset_range: tuple[float, float]
    orientation_requirement: str
    support_requirement: str
    contact_requirement: str
    violation_severity: str = "high"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vertical_offset_range"] = list(self.vertical_offset_range)
        data["forward_back_offset_range"] = list(self.forward_back_offset_range)
        data["lateral_offset_range"] = list(self.lateral_offset_range)
        return data


def constraint_for_example(example: dict[str, Any]) -> InteractionConstraint | None:
    family = str(example.get("family") or "")
    cid = str(example.get("concept_id") or "")
    if family == "cowgirl":
        return InteractionConstraint(
            constraint_id="cowgirl_pelvis_alignment",
            family=family,
            required_targets=["partner_pelvis_target", "contact_zone"],
            actor_anchor_bodypart="pelvis",
            partner_target_bodypart="partner_pelvis",
            target_distance_max=0.34 if "bounce" not in cid else 0.38,
            target_distance_ideal=0.16,
            vertical_offset_range=(0.06, 0.28),
            forward_back_offset_range=(-0.26, 0.26),
            lateral_offset_range=(-0.20, 0.20),
            orientation_requirement="front_to_partner_or_neutral",
            support_requirement="knees_or_feet_or_hands_support",
            contact_requirement="rider_pelvis_close_to_partner_pelvis",
        )
    if family == "reverse_cowgirl":
        return InteractionConstraint(
            constraint_id="reverse_cowgirl_pelvis_alignment",
            family=family,
            required_targets=["partner_pelvis_target", "contact_zone"],
            actor_anchor_bodypart="pelvis",
            partner_target_bodypart="partner_pelvis",
            target_distance_max=0.38,
            target_distance_ideal=0.16,
            vertical_offset_range=(0.06, 0.30),
            forward_back_offset_range=(-0.28, 0.28),
            lateral_offset_range=(-0.20, 0.20),
            orientation_requirement="back_to_partner_or_facing_away",
            support_requirement="knees_or_feet_support",
            contact_requirement="rider_pelvis_close_to_partner_pelvis",
        )
    if family == "doggy":
        return InteractionConstraint(
            constraint_id="doggy_partner_behind_alignment",
            family=family,
            required_targets=["partner_behind_reference", "receiver_pelvis_target"],
            actor_anchor_bodypart="pelvis",
            partner_target_bodypart="partner_pelvis",
            target_distance_max=0.72,
            target_distance_ideal=0.46,
            vertical_offset_range=(-0.18, 0.18),
            forward_back_offset_range=(0.24, 0.70),
            lateral_offset_range=(-0.18, 0.18),
            orientation_requirement="partner_behind",
            support_requirement="hands_knees_or_elevated_front_support",
            contact_requirement="partner_behind_close_to_receiver_pelvis",
        )
    if family == "bj_oral":
        return InteractionConstraint(
            constraint_id="bj_head_target_alignment",
            family=family,
            required_targets=["partner_pelvis_target", "head_chest_target_path"],
            actor_anchor_bodypart="head",
            partner_target_bodypart="partner_pelvis",
            target_distance_max=0.60,
            target_distance_ideal=0.24,
            vertical_offset_range=(-0.32, 0.32),
            forward_back_offset_range=(-0.48, -0.02),
            lateral_offset_range=(-0.16, 0.16),
            orientation_requirement="partner_in_front",
            support_requirement="pelvis_static_base",
            contact_requirement="head_chest_path_to_partner_pelvis_target",
        )
    if family == "missionary":
        return InteractionConstraint(
            constraint_id="missionary_supine_alignment",
            family=family,
            required_targets=["partner_pelvis_target", "close_body_relation"],
            actor_anchor_bodypart="pelvis",
            partner_target_bodypart="partner_pelvis",
            target_distance_max=0.60,
            target_distance_ideal=0.30,
            vertical_offset_range=(-0.44, -0.16),
            forward_back_offset_range=(-0.20, 0.20),
            lateral_offset_range=(-0.16, 0.16),
            orientation_requirement="partner_above_or_front",
            support_requirement="receiver_supine_grounded",
            contact_requirement="close_supine_body_relation",
        )
    if family == "handjob":
        return InteractionConstraint(
            constraint_id="hand_interaction_partner_pelvis_alignment",
            family=family,
            required_targets=["partner_pelvis_target", "hand_target_path"],
            actor_anchor_bodypart="rHand",
            partner_target_bodypart="partner_pelvis",
            target_distance_max=0.60,
            target_distance_ideal=0.24,
            vertical_offset_range=(-0.24, 0.24),
            forward_back_offset_range=(-0.45, 0.25),
            lateral_offset_range=(-0.42, 0.42),
            orientation_requirement="partner_target_visible",
            support_requirement="pelvis_static_base",
            contact_requirement="hand_path_to_partner_pelvis_target",
        )
    return None


def make_contact_aware_example(example: dict[str, Any]) -> dict[str, Any]:
    ex = copy.deepcopy(example)
    constraint = constraint_for_example(ex)
    if not constraint:
        ex["interaction_constraints"] = []
        ex["target_points"] = {}
        ex["support_targets"] = dict(ex.get("contact_targets") or {})
        ex["alignment_valid_expected"] = False
        ex["alignment_validation"] = {"valid": True, "reason": "no interaction constraint required"}
        return ex
    frames = list(ex.get("frames") or [])
    if frames:
        if ex.get("family") in {"cowgirl", "reverse_cowgirl"}:
            _align_rider_to_partner_pelvis(frames, ex)
        elif ex.get("family") == "doggy":
            _align_partner_behind_receiver(frames, ex)
        elif ex.get("family") == "bj_oral":
            _align_bj_head_to_partner_pelvis(frames, ex)
        elif ex.get("family") == "missionary":
            _align_missionary_partner_above(frames, ex)
        elif ex.get("family") == "handjob":
            _align_hand_target_to_partner_pelvis(frames, ex)
    ex["frames"] = frames
    ex["motion_trails"] = _rebuild_trails(frames)
    first = frames[0] if frames else {}
    partner = {k: as_point3(v) for k, v in (first.get("partner_reference_points") or {}).items()}
    target_points = {}
    if "partner_pelvis" in partner:
        target_points["partner_pelvis_target"] = partner["partner_pelvis"]
    if "partner_chest" in partner:
        target_points["partner_chest_target"] = partner["partner_chest"]
    if "partner_head" in partner:
        target_points["partner_head_target"] = partner["partner_head"]
    ex["interaction_constraints"] = [constraint.to_dict()]
    ex["target_points"] = _jsonable_points(target_points)
    ex["contact_zone"] = {
        "center": _jsonable_points({"center": target_points.get("partner_pelvis_target", (0.0, 0.0, 0.0))})["center"],
        "radius": constraint.target_distance_max,
        "label": "interaction target tolerance zone",
    }
    ex["allowed_offset_ranges"] = {
        "vertical": list(constraint.vertical_offset_range),
        "forward_back": list(constraint.forward_back_offset_range),
        "lateral": list(constraint.lateral_offset_range),
    }
    ex["support_targets"] = dict(ex.get("contact_targets") or {})
    ex["alignment_valid_expected"] = True
    ex["alignment_validation"] = evaluate_interaction_constraints(ex)
    warnings = list(ex.get("warnings") or [])
    if not ex["alignment_validation"].get("valid"):
        warnings.append("interaction alignment invalid in contact-aware preview")
    ex["warnings"] = sorted(set(warnings))
    return ex


def evaluate_interaction_constraints(example: dict[str, Any]) -> dict[str, Any]:
    constraint = constraint_for_example(example)
    if not constraint:
        return {"valid": True, "max_distance": 0.0, "mean_distance": 0.0, "failed_constraints": []}
    distances: list[float] = []
    failed: list[str] = []
    for frame in example.get("frames") or []:
        pts = {k: as_point3(v) for k, v in (frame.get("controller_points") or {}).items()}
        partner = {k: as_point3(v) for k, v in (frame.get("partner_reference_points") or {}).items()}
        actor_key = _actor_anchor_key(constraint.actor_anchor_bodypart, pts)
        target_key = constraint.partner_target_bodypart
        if actor_key not in pts:
            failed.append(f"missing_actor_anchor:{actor_key}")
            continue
        if target_key not in partner:
            failed.append(f"missing_partner_target:{target_key}")
            continue
        actor = pts[actor_key]
        target = partner[target_key]
        distance = _distance(actor, target)
        distances.append(distance)
        if distance > constraint.target_distance_max:
            failed.append(f"{constraint.constraint_id}:distance {distance:.3f} > {constraint.target_distance_max:.3f}")
        dy = actor[1] - target[1]
        dz = actor[2] - target[2]
        dx = actor[0] - target[0]
        if not _in_range(dy, constraint.vertical_offset_range):
            failed.append(f"{constraint.constraint_id}:vertical_offset {dy:.3f} outside {constraint.vertical_offset_range}")
        if not _in_range(dz, constraint.forward_back_offset_range):
            failed.append(f"{constraint.constraint_id}:forward_back_offset {dz:.3f} outside {constraint.forward_back_offset_range}")
        if not _in_range(dx, constraint.lateral_offset_range):
            failed.append(f"{constraint.constraint_id}:lateral_offset {dx:.3f} outside {constraint.lateral_offset_range}")
    unique_failed = sorted(set(failed))
    max_distance = max(distances) if distances else 0.0
    mean_distance = sum(distances) / len(distances) if distances else 0.0
    return {
        "valid": not unique_failed,
        "constraint_id": constraint.constraint_id,
        "max_distance": round(max_distance, 4),
        "mean_distance": round(mean_distance, 4),
        "target_distance_max": constraint.target_distance_max,
        "target_distance_ideal": constraint.target_distance_ideal,
        "failed_constraints": unique_failed,
    }


def _align_rider_to_partner_pelvis(frames: list[dict[str, Any]], example: dict[str, Any]) -> None:
    if not frames:
        return
    first_pts = {k: as_point3(v) for k, v in (frames[0].get("controller_points") or {}).items()}
    first_pelvis = first_pts.get("pelvis", (0.0, 0.0, 0.0))
    is_bounce = "bounce" in str(example.get("concept_id"))
    for frame in frames:
        pts = {k: as_point3(v) for k, v in (frame.get("controller_points") or {}).items()}
        partner = {k: as_point3(v) for k, v in (frame.get("partner_reference_points") or {}).items()}
        if "pelvis" not in pts or "partner_pelvis" not in partner:
            continue
        current = pts["pelvis"]
        delta = (current[0] - first_pelvis[0], current[1] - first_pelvis[1], current[2] - first_pelvis[2])
        delta_scale = (0.65, 0.35 if is_bounce else 0.45, 0.65)
        target = partner["partner_pelvis"]
        desired = (
            target[0] + delta[0] * delta_scale[0],
            target[1] + 0.14 + delta[1] * delta_scale[1],
            target[2] + delta[2] * delta_scale[2],
        )
        shift = (desired[0] - current[0], desired[1] - current[1], desired[2] - current[2])
        _move_all_points(pts, shift)
        for key in ("lFoot", "rFoot"):
            if key in pts and pts[key][1] < 0.12:
                pts[key] = (pts[key][0], 0.12, pts[key][2])
        for key in ("lKnee", "rKnee"):
            if key in pts and pts[key][1] < 0.22:
                pts[key] = (pts[key][0], 0.22, pts[key][2])
        frame["controller_points"] = _jsonable_points(pts)


def _align_partner_behind_receiver(frames: list[dict[str, Any]], example: dict[str, Any]) -> None:
    for frame in frames:
        pts = {k: as_point3(v) for k, v in (frame.get("controller_points") or {}).items()}
        pelvis = pts.get("pelvis")
        if not pelvis:
            continue
        frame["partner_reference_points"] = _jsonable_points({
            "partner_pelvis": (pelvis[0], pelvis[1] - 0.03, pelvis[2] - 0.46),
            "partner_chest": (pelvis[0], pelvis[1] + 0.42, pelvis[2] - 0.72),
            "partner_head": (pelvis[0], pelvis[1] + 0.74, pelvis[2] - 0.94),
        })


def _align_bj_head_to_partner_pelvis(frames: list[dict[str, Any]], example: dict[str, Any]) -> None:
    first_pts = {k: as_point3(v) for k, v in ((frames[0] if frames else {}).get("controller_points") or {}).items()}
    head = first_pts.get("head")
    if not head:
        return
    partner = {
        "partner_pelvis": (head[0], head[1] - 0.04, head[2] + 0.24),
        "partner_chest": (head[0], head[1] + 0.48, head[2] + 0.28),
        "partner_head": (head[0], head[1] + 0.80, head[2] + 0.30),
        "partner_lThigh": (-0.22, head[1] - 0.30, head[2] + 0.20),
        "partner_rThigh": (0.22, head[1] - 0.30, head[2] + 0.20),
    }
    for frame in frames:
        frame["partner_reference_points"] = _jsonable_points(partner)


def _align_missionary_partner_above(frames: list[dict[str, Any]], example: dict[str, Any]) -> None:
    for frame in frames:
        pts = {k: as_point3(v) for k, v in (frame.get("controller_points") or {}).items()}
        pelvis = pts.get("pelvis")
        chest = pts.get("chest", pelvis)
        if not pelvis:
            continue
        frame["partner_reference_points"] = _jsonable_points({
            "partner_pelvis": (pelvis[0], pelvis[1] + 0.30, pelvis[2] - 0.04),
            "partner_chest": (chest[0], chest[1] + 0.55, chest[2] - 0.06),
            "partner_head": (chest[0], chest[1] + 0.82, chest[2] + 0.22),
        })


def _align_hand_target_to_partner_pelvis(frames: list[dict[str, Any]], example: dict[str, Any]) -> None:
    first_pts = {k: as_point3(v) for k, v in ((frames[0] if frames else {}).get("controller_points") or {}).items()}
    hands = [first_pts[k] for k in ("lHand", "rHand") if k in first_pts]
    if not hands:
        return
    center = (
        sum(p[0] for p in hands) / len(hands),
        sum(p[1] for p in hands) / len(hands),
        sum(p[2] for p in hands) / len(hands),
    )
    partner = {
        "partner_pelvis": (center[0], center[1], center[2] + 0.22),
        "partner_chest": (center[0], center[1] + 0.50, center[2] + 0.28),
        "partner_head": (center[0], center[1] + 0.82, center[2] + 0.32),
    }
    for frame in frames:
        frame["partner_reference_points"] = _jsonable_points(partner)


def _rebuild_trails(frames: list[dict[str, Any]]) -> dict[str, list[list[float]]]:
    trails: dict[str, list[list[float]]] = {"pelvis": [], "head": [], "lHand": [], "rHand": []}
    for frame in frames:
        pts = {k: as_point3(v) for k, v in (frame.get("controller_points") or {}).items()}
        for key in trails:
            if key in pts:
                trails[key].append(list(pts[key]))
    return trails


def _move_all_points(points: dict[str, Point3], shift: Point3) -> None:
    for key, p in list(points.items()):
        points[key] = (p[0] + shift[0], p[1] + shift[1], p[2] + shift[2])


def _actor_anchor_key(anchor: str, points: dict[str, Point3]) -> str:
    if anchor == "head/mouth_proxy":
        return "head"
    if anchor in points:
        return anchor
    if anchor == "receiver_pelvis":
        return "pelvis"
    return anchor


def _distance(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _in_range(value: float, span: tuple[float, float]) -> bool:
    return span[0] <= value <= span[1]


def _jsonable_points(points: dict[str, Point3]) -> dict[str, list[float]]:
    return {k: [round(float(v[0]), 5), round(float(v[1]), 5), round(float(v[2]), 5)] for k, v in points.items()}

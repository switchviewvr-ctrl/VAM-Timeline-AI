"""Derived measurements for real manual VaM pose captures."""

from __future__ import annotations

from typing import Any
import math

from vam_timeline_ai.io.json_utils import as_float


CORE_CONTROLLER_GROUPS = {
    "rider_pelvis": ["pelvisControl", "hipControl"],
    "rider_chest": ["chestControl", "abdomen2Control", "abdomenControl"],
    "rider_head": ["headControl"],
    "rider_hands": ["lHandControl", "rHandControl"],
    "rider_knees": ["lKneeControl", "rKneeControl"],
    "rider_feet": ["lFootControl", "rFootControl"],
    "partner_pelvis": ["pelvisControl", "hipControl"],
    "partner_chest": ["chestControl", "abdomen2Control", "abdomenControl"],
    "partner_head": ["headControl"],
}


def compute_manual_pose_measurements(capture: dict[str, Any], human_label: dict[str, Any] | None = None) -> dict[str, Any]:
    human_label = human_label or {}
    atoms = capture.get("atoms") or {}
    derived = capture.get("derived") or {}
    completeness = _controller_completeness(atoms)

    measurements: dict[str, Any] = {
        "controller_completeness": completeness,
        "partner_relative": {
            "rider_pelvis_to_partner_pelvis_world_delta": _relation_vec(derived, "rider_pelvis_to_partner_pelvis", "world_delta")
            or _delta(atoms, "rider", "pelvisControl", "partner", "pelvisControl"),
            "rider_pelvis_to_partner_pelvis_distance": _relation_distance(derived, "rider_pelvis_to_partner_pelvis")
            or _distance_between(atoms, "rider", "pelvisControl", "partner", "pelvisControl"),
            "rider_pelvis_to_partner_pelvis_partner_local_delta": _relation_vec(derived, "rider_pelvis_to_partner_pelvis", "partner_local_delta"),
            "rider_chest_to_partner_chest_distance": _distance_between(atoms, "rider", "chestControl", "partner", "chestControl"),
            "rider_head_to_partner_pelvis_distance": _relation_distance(derived, "rider_head_to_partner_pelvis")
            or _distance_between(atoms, "rider", "headControl", "partner", "pelvisControl"),
            "rider_lhand_to_partner_chest_distance": _relation_distance(derived, "rider_lhand_to_partner_chest")
            or _distance_between(atoms, "rider", "lHandControl", "partner", "chestControl"),
            "rider_rhand_to_partner_chest_distance": _relation_distance(derived, "rider_rhand_to_partner_chest")
            or _distance_between(atoms, "rider", "rHandControl", "partner", "chestControl"),
            "rider_lhand_to_partner_pelvis_distance": _relation_distance(derived, "rider_lhand_to_partner_pelvis")
            or _distance_between(atoms, "rider", "lHandControl", "partner", "pelvisControl"),
            "rider_rhand_to_partner_pelvis_distance": _relation_distance(derived, "rider_rhand_to_partner_pelvis")
            or _distance_between(atoms, "rider", "rHandControl", "partner", "pelvisControl"),
            "rider_lhand_to_partner_thigh_or_leg_distance": _nearest_distance(atoms, "rider", "lHandControl", "partner", ["lThighControl", "rThighControl", "lKneeControl", "rKneeControl"]),
            "rider_rhand_to_partner_thigh_or_leg_distance": _nearest_distance(atoms, "rider", "rHandControl", "partner", ["lThighControl", "rThighControl", "lKneeControl", "rKneeControl"]),
        },
        "pose_geometry": {
            "torso_vector_chest_minus_pelvis": _delta_same_role(atoms, "rider", "chestControl", "pelvisControl"),
            "head_vector_head_minus_chest": _delta_same_role(atoms, "rider", "headControl", "chestControl"),
            "leg_spread_foot_distance": _distance_same_role(atoms, "rider", "lFootControl", "rFootControl"),
            "knee_spread_distance": _distance_same_role(atoms, "rider", "lKneeControl", "rKneeControl"),
            "feet_height_relative_to_pelvis": {
                "left": _relative_axis(atoms, "rider", "lFootControl", "pelvisControl", axis=1),
                "right": _relative_axis(atoms, "rider", "rFootControl", "pelvisControl", axis=1),
            },
            "knee_height_relative_to_pelvis": {
                "left": _relative_axis(atoms, "rider", "lKneeControl", "pelvisControl", axis=1),
                "right": _relative_axis(atoms, "rider", "rKneeControl", "pelvisControl", axis=1),
            },
            "rider_feet_relative_to_pelvis": derived.get("rider_feet_relative_to_pelvis") or _paired_relative(atoms, "rider", ["lFootControl", "rFootControl"], "pelvisControl"),
            "rider_knees_relative_to_pelvis": derived.get("rider_knees_relative_to_pelvis") or _paired_relative(atoms, "rider", ["lKneeControl", "rKneeControl"], "pelvisControl"),
        },
        "facing_hints": {
            "rider_facing_relative_to_partner": ((derived.get("orientation_hints") or {}).get("rider_facing_relative_to_partner")) or "unknown",
            "pose_hint": ((derived.get("orientation_hints") or {}).get("pose_hint")) or "unknown",
        },
        "hand_target_candidates": _hand_target_candidates(atoms),
        "motion_semantics_from_human_label": {
            "expected_primary_driver": human_label.get("primary_driver", "unknown"),
            "expected_secondary_drivers": human_label.get("secondary_drivers", []),
            "valid_motion_types": human_label.get("generation_valid_motions", []),
        },
        "anchor_expectations": {
            "feet_should_be_static": human_label.get("foot_behavior") == "mostly_static",
            "knees_may_phase": human_label.get("knee_behavior") in {"may_phase_out_in", "reactive_or_mixed"},
            "hands_support_or_driver": human_label.get("hand_support_options", []),
            "pelvis_should_drive": human_label.get("primary_driver") == "pelvis_hip",
            "head_should_drive": human_label.get("primary_driver") == "head_neck",
            "hand_should_drive": human_label.get("primary_driver") == "hand",
        },
    }
    return measurements


def _controller_completeness(atoms: dict[str, Any]) -> dict[str, bool]:
    return {
        key: any(_controller_pos(atoms, role, name) is not None for name in names)
        for key, names in CORE_CONTROLLER_GROUPS.items()
        for role in [key.split("_", 1)[0]]
    }


def _relation_distance(derived: dict[str, Any], key: str) -> float | None:
    value = ((derived.get(key) or {}).get("distance"))
    return as_float(value)


def _relation_vec(derived: dict[str, Any], key: str, field: str) -> list[float] | None:
    value = ((derived.get(key) or {}).get(field))
    return _vec(value)


def _vec(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    parsed = [as_float(value[i]) for i in range(3)]
    if any(v is None for v in parsed):
        return None
    return [float(v) for v in parsed if v is not None]


def _controller_pos(atoms: dict[str, Any], role: str, controller: str) -> list[float] | None:
    ctrl = (((atoms.get(role) or {}).get("controllers") or {}).get(controller) or {})
    return _vec(ctrl.get("world_position"))


def _delta(atoms: dict[str, Any], role_a: str, ctrl_a: str, role_b: str, ctrl_b: str) -> list[float] | None:
    a = _controller_pos(atoms, role_a, ctrl_a)
    b = _controller_pos(atoms, role_b, ctrl_b)
    if a is None or b is None:
        return None
    return [round(a[i] - b[i], 6) for i in range(3)]


def _distance_between(atoms: dict[str, Any], role_a: str, ctrl_a: str, role_b: str, ctrl_b: str) -> float | None:
    delta = _delta(atoms, role_a, ctrl_a, role_b, ctrl_b)
    if delta is None:
        return None
    return round(math.sqrt(sum(v * v for v in delta)), 6)


def _delta_same_role(atoms: dict[str, Any], role: str, ctrl_a: str, ctrl_b: str) -> list[float] | None:
    return _delta(atoms, role, ctrl_a, role, ctrl_b)


def _distance_same_role(atoms: dict[str, Any], role: str, ctrl_a: str, ctrl_b: str) -> float | None:
    return _distance_between(atoms, role, ctrl_a, role, ctrl_b)


def _relative_axis(atoms: dict[str, Any], role: str, ctrl_a: str, ctrl_b: str, *, axis: int) -> float | None:
    delta = _delta_same_role(atoms, role, ctrl_a, ctrl_b)
    if delta is None:
        return None
    return delta[axis]


def _paired_relative(atoms: dict[str, Any], role: str, controllers: list[str], base: str) -> dict[str, list[float] | None]:
    return {name: _delta_same_role(atoms, role, name, base) for name in controllers}


def _nearest_distance(atoms: dict[str, Any], role_a: str, ctrl_a: str, role_b: str, targets: list[str]) -> float | None:
    distances = [_distance_between(atoms, role_a, ctrl_a, role_b, target) for target in targets]
    valid = [v for v in distances if v is not None]
    return min(valid) if valid else None


def _hand_target_candidates(atoms: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = {
        "partner_chest": "chestControl",
        "partner_pelvis": "pelvisControl",
        "partner_left_thigh": "lThighControl",
        "partner_right_thigh": "rThighControl",
        "self_left_thigh": "lThighControl",
        "self_right_thigh": "rThighControl",
    }
    result: dict[str, dict[str, Any]] = {}
    for hand in ("lHandControl", "rHandControl"):
        distances: dict[str, float] = {}
        for label, target in targets.items():
            role_b = "rider" if label.startswith("self_") else "partner"
            distance = _distance_between(atoms, "rider", hand, role_b, target)
            if distance is not None:
                distances[label] = distance
        nearest = min(distances.items(), key=lambda item: item[1]) if distances else None
        result[hand] = {
            "distances": distances,
            "nearest_target": nearest[0] if nearest else None,
            "nearest_distance": nearest[1] if nearest else None,
        }
    return result

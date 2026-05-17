"""Biomechanical gates for cycle-aware semantic motion classification."""

from __future__ import annotations

from typing import Any


GATE_PASS = "pass"
GATE_SOFT_PASS = "soft_pass"
MIN_CLEAN_CYCLE_COUNT = 3.0
MIN_CLEAN_FREQUENCY_HZ = 1.5
MIN_SOFT_CYCLE_COUNT = 2.0
MIN_SOFT_FREQUENCY_HZ = 1.0
ACTIVE_CONTACT_TARGET_MAX_DISTANCE_M = 0.20
COWGIRL_CONTACT_TARGET_MAX_DISTANCE_M = 0.35
MALE_THRUST_TARGET_MAX_DISTANCE_M = 0.30
MIN_THRUST_TRANSFORM_DISTANCE_M = 0.07


def evaluate_biomechanical_gates(pose_row: dict[str, Any], cycle_row: dict[str, Any], relational_row: dict[str, Any] | None = None) -> dict[str, Any]:
    family = str(pose_row.get("resolved_semantic_family") or "unknown")
    if family in {"cowgirl", "reverse_cowgirl"}:
        return _cowgirl_gates(pose_row, cycle_row, relational_row or {})
    if family == "doggy":
        return _doggy_gates(pose_row, cycle_row, relational_row or {})
    if family == "bj_oral":
        return _bj_gates(pose_row, cycle_row, relational_row or {})
    if family in {"handjob", "hand_touching"}:
        return _hj_gates(pose_row, cycle_row, relational_row or {})
    if family == "missionary":
        return _missionary_gates(pose_row, cycle_row, relational_row or {})
    return _gate_result("fail_wrong_driver", "fail_wrong_driver", ["no supported clean-motion family"])


def evaluate_male_active_thrust_gates(pose_row: dict[str, Any], cycle_row: dict[str, Any], relational_row: dict[str, Any] | None = None) -> dict[str, Any]:
    relational = relational_row or {}
    m = _metrics(cycle_row)
    hip_control = _controller(m, "hipControl")
    pelvis_control = _controller(m, "pelvisControl")
    driver = hip_control or pelvis_control
    reasons: list[str] = []
    male_hint = _looks_male_actor(pose_row, cycle_row, relational)
    if not male_hint:
        result = _full_result("not_applicable", "not_applicable", "not_applicable", "not_applicable", "not_applicable", "not_applicable", "not_applicable", "not_applicable", "not_applicable", ["actor is not a male-active candidate"])
        result["male_active_candidate"] = False
        return result
    controller_gate = "pass" if driver else "fail_wrong_driver"
    if not driver:
        reasons.append("male active thrust requires hipControl or pelvisControl")
    driver_gate = "pass" if _has_real_motion_threshold(driver, min_range=MIN_THRUST_TRANSFORM_DISTANCE_M, min_path=MIN_THRUST_TRANSFORM_DISTANCE_M * 2.0) else "fail_wrong_driver"
    if driver_gate != "pass":
        reasons.append("male hip/pelvis thrust transform distance is below 7cm or empty")
    cycle_gate = _strict_cycle_gate(driver, min_cycles=3.0, min_frequency=0.5, min_cyclicity=0.35)
    if cycle_gate not in {GATE_PASS, GATE_SOFT_PASS}:
        reasons.append("male hip/pelvis driver must complete at least 3 plausible thrust cycles")
    transform_gate = _male_thrust_transform_gate(driver, relational)
    if transform_gate != "pass":
        reasons.append("male thrust contact-axis movement is below 7cm")
    target_distance = _rel_float(pose_row, relational, "actor_pelvis_partner_alignment_distance_mean", "relational_actor_pelvis_partner_alignment_distance_mean")
    partner_gate = "pass" if target_distance is not None and target_distance <= MALE_THRUST_TARGET_MAX_DISTANCE_M else "fail_partner_alignment_lost"
    if target_distance is None:
        reasons.append("male active thrust requires partner pelvis/hip target distance")
    elif target_distance > MALE_THRUST_TARGET_MAX_DISTANCE_M:
        reasons.append("male hip/pelvis is farther than 30cm from partner pelvis/hip contact proxy")
    break_gate = "pass"
    if _transition_score(driver) >= 0.72:
        break_gate = "fail_transition"
        reasons.append("male hip/pelvis movement is monotonic, not repeated thrusting")
    final = _final_gate(controller_gate, driver_gate, cycle_gate, transform_gate, "pass", "pass", partner_gate, break_gate)
    result = _full_result(controller_gate, driver_gate, cycle_gate, transform_gate, "pass", "pass", partner_gate, break_gate, final, reasons)
    result["male_active_candidate"] = True
    result["target_proximity_gate_result"] = partner_gate
    result["target_proximity_distance_m"] = target_distance
    result["target_proximity_max_distance_m"] = MALE_THRUST_TARGET_MAX_DISTANCE_M
    result["thrust_transform_min_distance_m"] = MIN_THRUST_TRANSFORM_DISTANCE_M
    result["thrust_contact_axis_gate_result"] = transform_gate
    result["dyad_role_context"] = "male_active_female_receiver"
    return result


def _cowgirl_gates(pose: dict[str, Any], cycle: dict[str, Any], relational: dict[str, Any]) -> dict[str, Any]:
    m = _metrics(cycle)
    completeness = _completeness(cycle)
    hip_control = _controller(m, "hipControl")
    pelvis_control = _controller(m, "pelvisControl")
    hip = hip_control or pelvis_control
    head = _controller(m, "headControl")
    lh = _controller(m, "lHandControl")
    rh = _controller(m, "rHandControl")
    reasons: list[str] = []
    controller_gate = _cowgirl_controller_gate(completeness, hip_control, pelvis_control)
    if controller_gate != GATE_PASS:
        reasons.extend(_as_list(completeness.get("reasons")))
    driver_gate = "pass" if hip_control and _has_real_motion(hip_control) else "fail_wrong_driver"
    if not hip_control and not pelvis_control:
        reasons.append("hipControl and pelvisControl missing; cannot be Cowgirl motion")
    elif not hip_control:
        reasons.append("hipControl missing; pelvisControl alone is not accepted as clean Cowgirl driver")
    elif not _has_real_motion(hip_control):
        reasons.append("hipControl has no real transform-distance motion")
    elif pelvis_control and not _has_real_motion(pelvis_control):
        reasons.append("pelvisControl is static/follower; hipControl remains the only possible driver")
    cycle_gate = _cycle_gate(hip)
    if cycle_gate not in {GATE_PASS, GATE_SOFT_PASS}:
        reasons.append("hip/pelvis driver lacks reliable repeated cycle")
    transform_gate = "pass" if hip_control and _has_real_motion(hip_control) else "fail_pose_hold"
    if transform_gate != "pass":
        reasons.append("hipControl transform distance is too small or empty for Cowgirl animation")
    pattern_gate = _cowgirl_pattern_gate(hip_control)
    if pattern_gate not in {GATE_PASS, GATE_SOFT_PASS}:
        reasons.append("hipControl movement does not match Cowgirl pattern: up/down, forward/back, or oval/circular loop")
    anchor_gate = _anchor_gate(cycle, feet=True, knees=True)
    if anchor_gate != GATE_PASS:
        reasons.append("required lower-body anchors moved too much")
    relation = set(_as_list(pose.get("partner_relation")))
    alignment_score = _rel_float(pose, relational, "actor_pelvis_partner_alignment_score", "relational_actor_pelvis_partner_alignment_score")
    alignment_distance = _rel_float(pose, relational, "actor_pelvis_partner_alignment_distance_mean", "relational_actor_pelvis_partner_alignment_distance_mean")
    actor_above_score = _rel_float(pose, relational, "actor_above_partner_score", "relational_actor_above_partner_score")
    partner_gate = "pass" if relation & {"rider_above_partner", "pelvis_aligned", "pelvis_aligned_or_near"} else "fail_partner_alignment_lost"
    if alignment_score is not None and alignment_score >= 0.35:
        partner_gate = "pass"
    if alignment_distance is not None and alignment_distance > COWGIRL_CONTACT_TARGET_MAX_DISTANCE_M:
        partner_gate = "fail_partner_alignment_lost"
        reasons.append("actor hip/pelvis is farther than 35cm from partner pelvis/hip contact proxy")
    if alignment_distance is not None and alignment_distance > 0.55:
        partner_gate = "fail_partner_alignment_lost"
        reasons.append("actor hip/pelvis is too far from partner pelvis interaction zone")
    if actor_above_score is not None and actor_above_score < 0.75:
        partner_gate = "fail_partner_alignment_lost"
        reasons.append("actor hip/pelvis is not above partner pelvis in world space; lying/flat pose cannot be clean Cowgirl")
    if partner_gate != GATE_PASS:
        reasons.append("rider-over-receiver / pelvis alignment not established")
    pose_gate = "pass" if "cowgirl" in str(pose.get("pose_subtype") or pose.get("pose_family") or "") else "fail_pose_broken"
    break_gate = "pass"
    if (cycle.get("driver_summary") or {}).get("global_motion_suspect"):
        break_gate = "fail_transition"
        reasons.append("many controllers move together in the same direction; likely global/empty Timeline motion, not articulated animation")
    if _transitionish(pose) or _transition_score(hip) >= 0.72:
        break_gate = "fail_transition"
        reasons.append("driver motion looks monotonic or source clip name indicates transition/setup")
    if max(_cyclicity(head), _cyclicity(lh), _cyclicity(rh)) > _cyclicity(hip) + 0.2:
        break_gate = "fail_wrong_driver"
        reasons.append("head or hands appear more cyclic than hip/pelvis")
    if break_gate == "pass" and _hand_driver_competes_with_hip(hip_control, lh, rh):
        break_gate = "fail_wrong_driver"
        reasons.append("hand controller motion competes with hipControl; likely mixed Cowgirl-to-HJ/reach segment")
    if (cycle.get("anchor_summary") or {}).get("possible_locomotion") and not _feet_are_stable(cycle):
        break_gate = "fail_locomotion"
        reasons.append("hands/feet motion suggests locomotion or pose break, not seated Cowgirl loop")
    final = _final_gate(controller_gate, driver_gate, cycle_gate, transform_gate, pattern_gate, anchor_gate, pose_gate, partner_gate, break_gate)
    result = _full_result(controller_gate, driver_gate, cycle_gate, transform_gate, anchor_gate, pose_gate, partner_gate, break_gate, final, reasons)
    result["motion_pattern_gate_result"] = pattern_gate
    result["motion_pattern"] = ((hip_control or {}).get("cowgirl_motion_pattern") or {}).get("pattern", "unknown")
    result["relational_alignment_score"] = alignment_score
    result["relational_alignment_distance_mean"] = alignment_distance
    result["relational_actor_above_partner_score"] = actor_above_score
    result["target_proximity_gate_result"] = "pass" if partner_gate == "pass" else "fail_partner_alignment_lost"
    result["target_proximity_distance_m"] = alignment_distance
    result["target_proximity_max_distance_m"] = COWGIRL_CONTACT_TARGET_MAX_DISTANCE_M
    return result


def _doggy_gates(pose: dict[str, Any], cycle: dict[str, Any], relational: dict[str, Any]) -> dict[str, Any]:
    m = _metrics(cycle)
    completeness = _completeness(cycle)
    hip = _controller(m, "hipControl") or _controller(m, "pelvisControl")
    reasons: list[str] = []
    controller_gate = "pass" if completeness.get("has_hip_or_pelvis") else "fail_wrong_driver"
    if controller_gate != "pass":
        reasons.extend(_as_list(completeness.get("reasons")))
    relation = set(_as_list(pose.get("partner_relation")))
    pose_ok = "doggy" in str(pose.get("pose_subtype") or "") or bool(relation & {"partner_behind", "actor_in_front_of_partner_local", "actor_behind_partner_local"})
    pose_gate = "pass" if pose_ok else "fail_pose_broken"
    if not pose_ok:
        reasons.append("doggy requires all-fours/bent/elevated support or partner-behind relation")
    cycle_gate = _cycle_gate(hip, axis="z")
    anchor_gate = _anchor_gate(cycle, hands=True, knees=True, feet=True)
    break_gate = "fail_locomotion" if (cycle.get("anchor_summary") or {}).get("possible_locomotion") else "pass"
    if break_gate != "pass":
        reasons.append("hands/feet move like locomotion/crawling")
    transform_gate = "pass" if _has_real_motion(hip) else "fail_pose_hold"
    final = _final_gate(controller_gate, "pass" if hip else "fail_wrong_driver", cycle_gate, transform_gate, anchor_gate, pose_gate, "pass", break_gate)
    return _full_result(controller_gate, "pass" if hip else "fail_wrong_driver", cycle_gate, transform_gate, anchor_gate, pose_gate, "pass", break_gate, final, reasons)


def _bj_gates(pose: dict[str, Any], cycle: dict[str, Any], relational: dict[str, Any]) -> dict[str, Any]:
    m = _metrics(cycle)
    head = _controller(m, "headControl")
    chest = _controller(m, "chestControl")
    hip = _controller(m, "hipControl") or _controller(m, "pelvisControl")
    reasons: list[str] = []
    controller_gate = "pass" if head else "fail_wrong_driver"
    if not head:
        reasons.append("headControl missing; chestControl alone cannot define clean BJ/oral motion")
    driver_gate = "pass" if head and _has_real_motion_threshold(head, min_range=0.035, min_path=0.12) else "fail_wrong_driver"
    if head and driver_gate != "pass":
        reasons.append("headControl has no real transform-distance motion; empty/static keyframes ignored")
    cycle_gate = _strict_cycle_gate(head, min_cycles=3.0, min_frequency=1.5, min_cyclicity=0.45)
    if cycle_gate not in {GATE_PASS, GATE_SOFT_PASS}:
        reasons.append("headControl must complete at least 3 plausible approach-withdraw cycles")
    transform_gate = "pass" if _has_real_motion_threshold(head, min_range=0.035, min_path=0.12) else "fail_pose_hold"
    anchor_gate = _anchor_gate(cycle, feet=True)
    break_gate = "pass"
    target_score = _rel_float(pose, relational, "head_to_partner_pelvis_target_score", "relational_head_to_partner_pelvis_target_score")
    head_target_distance = _rel_float(pose, relational, "head_to_partner_pelvis_distance_mean", "relational_head_to_partner_pelvis_distance_mean")
    if head_target_distance is not None and head_target_distance > ACTIVE_CONTACT_TARGET_MAX_DISTANCE_M:
        break_gate = "fail_partner_alignment_lost"
        reasons.append("headControl cycle is farther than 20cm from partner pelvis/hip contact proxy")
    elif target_score is not None and target_score < 0.15:
        break_gate = "fail_partner_alignment_lost"
        reasons.append("head/chest path is not close to partner pelvis target")
    if _has_real_motion_threshold(hip, min_range=0.06, min_path=0.16) and (
        _cyclicity(hip) >= max(_cyclicity(head), _cyclicity(chest), 0.35)
        or float((hip or {}).get("estimated_cycle_count") or 0.0) >= 1.0
    ):
        break_gate = "fail_wrong_driver"
        reasons.append("hip/pelvis macro-cycle competes with head/chest driver")
    if _transition_score(head) >= 0.72:
        break_gate = "fail_transition"
        reasons.append("head/chest motion is one-way lean/reach, not approach-withdraw loop")
    partner_gate = "fail_partner_alignment_lost" if break_gate == "fail_partner_alignment_lost" else "pass"
    final = _final_gate(controller_gate, driver_gate, cycle_gate, transform_gate, anchor_gate, "pass", partner_gate, break_gate)
    result = _full_result(controller_gate, driver_gate, cycle_gate, transform_gate, anchor_gate, "pass", partner_gate, break_gate, final, reasons)
    result["target_proximity_gate_result"] = partner_gate
    result["target_proximity_distance_m"] = head_target_distance
    result["target_proximity_max_distance_m"] = ACTIVE_CONTACT_TARGET_MAX_DISTANCE_M
    return result


def _hj_gates(pose: dict[str, Any], cycle: dict[str, Any], relational: dict[str, Any]) -> dict[str, Any]:
    m = _metrics(cycle)
    lh = _controller(m, "lHandControl")
    rh = _controller(m, "rHandControl")
    hand = _best_hand(lh, rh)
    hand_name = "lHandControl" if hand is lh else "rHandControl" if hand is rh else ""
    hip = _controller(m, "hipControl") or _controller(m, "pelvisControl")
    chest = _controller(m, "chestControl")
    reasons: list[str] = []
    controller_gate = "pass" if hand else "fail_wrong_driver"
    if not hand:
        reasons.append("lHandControl/rHandControl missing; cannot classify clean HJ/manual cycle")
    driver_gate = "pass" if hand and _has_real_motion_threshold(hand, min_range=0.03, min_path=0.10) else "fail_wrong_driver"
    if hand and driver_gate != "pass":
        reasons.append("active hand controller has no real transform-distance motion; empty/static keyframes ignored")
    cycle_gate = _strict_cycle_gate(hand, min_cycles=3.0, min_frequency=1.5, min_cyclicity=0.35)
    if cycle_gate not in {GATE_PASS, GATE_SOFT_PASS}:
        reasons.append("active hand must complete at least 3 plausible back-and-forth cycles")
    transform_gate = "pass" if _has_real_motion_threshold(hand, min_range=0.03, min_path=0.10) else "fail_pose_hold"
    break_gate = "pass"
    targets = {
        str(_rel_value(pose, relational, "best_lHand_partner_target", "relational_best_lHand_partner_target") or ""),
        str(_rel_value(pose, relational, "best_rHand_partner_target", "relational_best_rHand_partner_target") or ""),
    }
    hand_target_distance = _hand_partner_contact_distance(pose, relational, hand_name)
    partner_gate = "pass"
    if hand_target_distance is None and targets and not (targets & {"partner_pelvis", "partner_hip", "partner_lThigh", "partner_rThigh"}):
        reasons.append("hand cycle is not clearly near partner pelvis/hip/thigh target")
    if hand_target_distance is not None and hand_target_distance > ACTIVE_CONTACT_TARGET_MAX_DISTANCE_M:
        partner_gate = "fail_partner_alignment_lost"
        break_gate = "fail_partner_alignment_lost"
        reasons.append("active hand cycle is farther than 20cm from partner pelvis/hip contact proxy")
    if _has_real_motion_threshold(hip, min_range=0.06, min_path=0.16) and _cyclicity(hip) >= max(_cyclicity(hand), 0.35):
        break_gate = "fail_wrong_driver"
        reasons.append("hip/pelvis driver competes with hand motion")
    if _transition_score(hand) >= 0.72:
        break_gate = "fail_transition"
        reasons.append("hand motion is one-way reach/acquisition, not repeated HJ cycle")
    if _has_real_motion_threshold(chest, min_range=0.08, min_path=0.20) and float((chest or {}).get("total_path_length") or 0.0) >= float((hand or {}).get("total_path_length") or 0.0) * 0.65:
        break_gate = "fail_wrong_driver"
        reasons.append("upper body moves with the hand; likely reach/body movement rather than isolated HJ cycle")
    final = _final_gate(controller_gate, driver_gate, cycle_gate, transform_gate, "pass", "pass", partner_gate, break_gate)
    result = _full_result(controller_gate, driver_gate, cycle_gate, transform_gate, "pass", "pass", partner_gate, break_gate, final, reasons)
    result["target_proximity_gate_result"] = partner_gate
    result["target_proximity_distance_m"] = hand_target_distance
    result["target_proximity_max_distance_m"] = ACTIVE_CONTACT_TARGET_MAX_DISTANCE_M
    return result


def _missionary_gates(pose: dict[str, Any], cycle: dict[str, Any], relational: dict[str, Any]) -> dict[str, Any]:
    m = _metrics(cycle)
    completeness = _completeness(cycle)
    hip = _controller(m, "hipControl") or _controller(m, "pelvisControl")
    chest = _controller(m, "chestControl")
    reasons: list[str] = []
    controller_gate = "pass" if completeness.get("has_hip_or_pelvis") else "fail_wrong_driver"
    if controller_gate != "pass":
        reasons.extend(_as_list(completeness.get("reasons")))
    pose_gate = "pass" if "missionary" in str(pose.get("pose_subtype") or pose.get("pose_family") or "") or "supine" in str(pose.get("pose_subtype") or "") else "fail_pose_broken"
    cycle_gate = _cycle_gate(hip)
    break_gate = "pass"
    if _transition_score(chest) > 0.65 and _range(chest) > 0.12:
        break_gate = "fail_transition"
        reasons.append("chest/head rising suggests getting up or sitting transition")
    transform_gate = "pass" if _has_real_motion(hip) else "fail_pose_hold"
    final = _final_gate(controller_gate, "pass" if hip else "fail_wrong_driver", cycle_gate, transform_gate, "pass", pose_gate, "pass", break_gate)
    return _full_result(controller_gate, "pass" if hip else "fail_wrong_driver", cycle_gate, transform_gate, "pass", pose_gate, "pass", break_gate, final, reasons)


def _cycle_gate(controller: dict[str, Any] | None, axis: str | None = None) -> str:
    if not controller:
        return "fail_wrong_driver"
    if not _has_real_motion(controller):
        return "fail_pose_hold"
    if _range(controller) < 0.025 or float(controller.get("pose_hold_score") or 0.0) >= 0.8:
        return "fail_pose_hold"
    axis_metrics = controller.get("axis_metrics") or {}
    metric = axis_metrics.get(axis) if axis else None
    cyc = float((metric or controller).get("estimated_cycle_count") or controller.get("estimated_cycle_count") or 0.0)
    freq = float((metric or controller).get("estimated_frequency_hz") or controller.get("estimated_frequency_hz") or 0.0)
    score = float((metric or controller).get("cyclicity_score") or controller.get("cyclicity_score") or 0.0)
    trans = float((metric or controller).get("transition_score") or controller.get("transition_score") or 0.0)
    if trans >= 0.75 and cyc < 1.5:
        return "fail_monotonic_transition"
    if cyc >= MIN_CLEAN_CYCLE_COUNT and freq >= MIN_CLEAN_FREQUENCY_HZ and score >= 0.35:
        return "pass"
    if cyc >= MIN_SOFT_CYCLE_COUNT and freq >= MIN_SOFT_FREQUENCY_HZ and score >= 0.3:
        return "soft_pass"
    return "fail_insufficient_cycles"


def _strict_cycle_gate(controller: dict[str, Any] | None, min_cycles: float, min_frequency: float, min_cyclicity: float) -> str:
    if not controller:
        return "fail_wrong_driver"
    if not _has_real_motion(controller):
        return "fail_pose_hold"
    if _range(controller) < 0.025 or float(controller.get("pose_hold_score") or 0.0) >= 0.8:
        return "fail_pose_hold"
    cycles = float(controller.get("estimated_cycle_count") or 0.0)
    freq = float(controller.get("estimated_frequency_hz") or 0.0)
    score = float(controller.get("cyclicity_score") or 0.0)
    trans = float(controller.get("transition_score") or 0.0)
    if trans >= 0.72 and cycles < min_cycles:
        return "fail_monotonic_transition"
    if cycles >= min_cycles and freq >= min_frequency and score >= min_cyclicity:
        return "pass"
    return "fail_insufficient_cycles"


def _cowgirl_pattern_gate(controller: dict[str, Any] | None) -> str:
    if not controller or not _has_real_motion(controller):
        return "fail_pose_hold"
    pattern = controller.get("cowgirl_motion_pattern") or {}
    if pattern.get("valid_clean_cowgirl_pattern"):
        return "pass"
    if pattern.get("valid_soft_cowgirl_pattern"):
        return "soft_pass"
    name = str(pattern.get("pattern") or "")
    if name in {"no_real_motion"}:
        return "fail_pose_hold"
    if name in {"monotonic_transition"}:
        return "fail_transition"
    return "fail_insufficient_cycles"


def _anchor_gate(cycle: dict[str, Any], feet: bool = False, knees: bool = False, hands: bool = False) -> str:
    anchor = cycle.get("anchor_summary") or {}
    if feet and anchor.get("feet_stable") is False:
        return "fail_anchor_lost"
    if knees and anchor.get("knees_stable") is False:
        return "fail_anchor_lost"
    if hands and anchor.get("hands_stable") is False:
        return "fail_anchor_lost"
    return "pass"


def _final_gate(*gates: str) -> str:
    if any(g.startswith("fail_") for g in gates):
        for gate in gates:
            if gate.startswith("fail_"):
                return gate
    if any(g == "soft_pass" for g in gates):
        return "soft_pass"
    return "pass"


def _full_result(controller: str, driver: str, cycle: str, transform: str, anchor: str, pose: str, partner: str, break_gate: str, final: str, reasons: list[str]) -> dict[str, Any]:
    if not reasons and final not in {"pass", "soft_pass"}:
        reasons = [final]
    return {
        "controller_gate_result": controller,
        "driver_gate_result": driver,
        "cycle_gate_result": cycle,
        "transform_distance_gate_result": transform,
        "anchor_gate_result": anchor,
        "pose_preservation_gate_result": pose,
        "partner_alignment_gate_result": partner,
        "break_state_gate_result": break_gate,
        "final_clean_motion_gate": final,
        "gate_failure_reasons": sorted(set(reasons)),
    }


def _gate_result(final: str, break_gate: str, reasons: list[str]) -> dict[str, Any]:
    return _full_result(final, final, final, final, "pass", "pass", "pass", break_gate, final, reasons)


def _metrics(cycle: dict[str, Any]) -> dict[str, Any]:
    return cycle.get("controller_metrics") if isinstance(cycle.get("controller_metrics"), dict) else {}


def _completeness(cycle: dict[str, Any]) -> dict[str, Any]:
    return cycle.get("controller_completeness_summary") if isinstance(cycle.get("controller_completeness_summary"), dict) else {}


def _cowgirl_controller_gate(completeness: dict[str, Any], hip: dict[str, Any] | None, pelvis: dict[str, Any] | None) -> str:
    status = str(completeness.get("status") or "")
    if status in {"empty", "only_feet", "only_upper_body_hands_head", "missing_hip_and_pelvis"}:
        return "fail_wrong_driver"
    if not hip:
        return "fail_wrong_driver"
    if not _has_real_motion(hip):
        return "fail_pose_hold"
    return "pass"


def _controller(metrics: dict[str, Any], name: str) -> dict[str, Any] | None:
    return metrics.get(name)


def _cyclicity(metric: dict[str, Any] | None) -> float:
    return float((metric or {}).get("cyclicity_score") or 0.0)


def _transition_score(metric: dict[str, Any] | None) -> float:
    return float((metric or {}).get("transition_score") or 0.0)


def _range(metric: dict[str, Any] | None) -> float:
    return float((metric or {}).get("max_displacement_range") or (metric or {}).get("dominant_axis_range") or 0.0)


def _has_real_motion(metric: dict[str, Any] | None) -> bool:
    if not metric:
        return False
    if metric.get("has_real_motion") is not None:
        return bool(metric.get("has_real_motion"))
    return _range(metric) >= 0.025 and float(metric.get("total_path_length") or 0.0) >= 0.05


def _has_real_motion_threshold(metric: dict[str, Any] | None, min_range: float, min_path: float) -> bool:
    if not metric:
        return False
    return _has_real_motion(metric) and _range(metric) >= min_range and float(metric.get("total_path_length") or 0.0) >= min_path


def _best_hand(lh: dict[str, Any] | None, rh: dict[str, Any] | None) -> dict[str, Any] | None:
    if not lh:
        return rh
    if not rh:
        return lh
    lscore = float(lh.get("estimated_cycle_count") or 0.0) * 2.0 + _cyclicity(lh) + float(lh.get("total_path_length") or 0.0)
    rscore = float(rh.get("estimated_cycle_count") or 0.0) * 2.0 + _cyclicity(rh) + float(rh.get("total_path_length") or 0.0)
    return lh if lscore >= rscore else rh


def _hand_partner_contact_distance(pose: dict[str, Any], relational: dict[str, Any], hand_name: str) -> float | None:
    side = "lHand" if hand_name == "lHandControl" else "rHand" if hand_name == "rHandControl" else ""
    if not side:
        return None
    targets = relational.get("hand_partner_targets") if isinstance(relational.get("hand_partner_targets"), dict) else None
    if not targets:
        targets = pose.get("relational_hand_partner_targets") if isinstance(pose.get("relational_hand_partner_targets"), dict) else None
    if not targets:
        return None
    hand_targets = targets.get(side) if isinstance(targets.get(side), dict) else {}
    distances = hand_targets.get("distances") if isinstance(hand_targets.get("distances"), dict) else {}
    candidates: list[float] = []
    for target in ("pelvis", "hip"):
        value = distances.get(target)
        if isinstance(value, dict):
            try:
                candidates.append(float(value.get("mean")))
            except (TypeError, ValueError):
                pass
    if candidates:
        return min(candidates)
    try:
        return float(hand_targets.get("best_distance_mean"))
    except (TypeError, ValueError):
        return None


def _male_thrust_transform_gate(driver: dict[str, Any] | None, relational: dict[str, Any]) -> str:
    if not driver:
        return "fail_wrong_driver"
    contact_axis = relational.get("hip_motion_contact_axis") if isinstance(relational.get("hip_motion_contact_axis"), dict) else None
    if not contact_axis:
        contact_axis = relational.get("pelvis_motion_contact_axis") if isinstance(relational.get("pelvis_motion_contact_axis"), dict) else None
    if contact_axis and contact_axis.get("available"):
        axis_range = float(contact_axis.get("range") or 0.0)
        axis_path = float(contact_axis.get("path") or 0.0)
        return "pass" if axis_range >= MIN_THRUST_TRANSFORM_DISTANCE_M and axis_path >= MIN_THRUST_TRANSFORM_DISTANCE_M * 2.0 else "fail_pose_hold"
    return "pass" if _has_real_motion_threshold(driver, min_range=MIN_THRUST_TRANSFORM_DISTANCE_M, min_path=MIN_THRUST_TRANSFORM_DISTANCE_M * 2.0) else "fail_pose_hold"


def _looks_male_actor(pose: dict[str, Any], cycle: dict[str, Any], relational: dict[str, Any]) -> bool:
    hay = " ".join(
        str(v or "")
        for v in [
            pose.get("technical_atom_id"),
            pose.get("technical_actor_id"),
            cycle.get("technical_atom_id"),
            relational.get("actor_atom_id"),
        ]
    ).lower()
    tokens = ["man", "male", "guy", "_guy", "p_him", "__h", "__m", " him", "boy", "dude"]
    female_tokens = ["woman", "female", "girl", "her", "p_her", "fem", "tifa", "yumi", "eve", "leia", "leah", "hanna", "felicity"]
    return any(token in hay for token in tokens) and not any(token in hay for token in female_tokens)


def _hand_driver_competes_with_hip(hip: dict[str, Any] | None, lh: dict[str, Any] | None, rh: dict[str, Any] | None) -> bool:
    if not hip or not _has_real_motion(hip):
        return False
    hand = lh if float((lh or {}).get("total_path_length") or 0.0) >= float((rh or {}).get("total_path_length") or 0.0) else rh
    if not hand or not _has_real_motion(hand):
        return False
    hip_path = float(hip.get("total_path_length") or 0.0)
    hand_path = float(hand.get("total_path_length") or 0.0)
    hip_cycles = float(hip.get("estimated_cycle_count") or 0.0)
    hand_cycles = float(hand.get("estimated_cycle_count") or 0.0)
    hip_cyc = _cyclicity(hip)
    hand_cyc = _cyclicity(hand)
    if hand_path >= hip_path * 0.6 and hand_cycles >= hip_cycles + 1.0 and hand_cyc >= max(0.35, hip_cyc * 0.75):
        return True
    return hand_path >= hip_path * 0.85 and hand_cyc >= hip_cyc * 0.9 and hand_cycles >= hip_cycles


def _feet_are_stable(cycle: dict[str, Any]) -> bool:
    anchor = cycle.get("anchor_summary") or {}
    return anchor.get("feet_stable") is not False


def _transitionish(pose: dict[str, Any]) -> bool:
    hay = " ".join(str(pose.get(k) or "") for k in ["sample_id", "source_id", "old_motion_subtype", "resolved_motion_subtype"]).lower()
    return any(token in hay for token in ["maintrans", "mount", "switch", "transition", "getting_up"])


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def _rel_value(pose: dict[str, Any], relational: dict[str, Any], raw_key: str, merged_key: str) -> Any:
    if relational and relational.get(raw_key) is not None:
        return relational.get(raw_key)
    return pose.get(merged_key)


def _rel_float(pose: dict[str, Any], relational: dict[str, Any], raw_key: str, merged_key: str) -> float | None:
    value = _rel_value(pose, relational, raw_key, merged_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

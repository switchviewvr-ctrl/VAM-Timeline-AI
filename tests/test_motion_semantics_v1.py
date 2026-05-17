import math

from vam_timeline_ai.features.motion_cycle_features import compute_signal_cycle_metrics
from vam_timeline_ai.semantics.biomechanical_motion_gates import evaluate_biomechanical_gates
from vam_timeline_ai.semantics.motion_semantic_resolver import resolve_motion_candidate_v1


def _pose(family="cowgirl"):
    return {
        "window_id": "w",
        "resolved_semantic_family": family,
        "pose_family": family,
        "pose_subtype": "cowgirl_kneeling" if family == "cowgirl" else family,
        "partner_relation": ["rider_above_partner", "pelvis_aligned"],
        "sample_id": "sample_clean",
        "source_id": "src_clean",
        "confidence": 0.7,
    }


def _male_pose():
    pose = _pose("unknown")
    pose["technical_atom_id"] = "Man"
    pose["technical_actor_id"] = "Man"
    pose["pose_subtype"] = "standing_partner"
    return pose


def _metric(cycles=3.0, cyc=0.8, trans=0.1, rng=0.2, axis="y", freq=1.5):
    valid_clean = cycles >= 3.0 and freq >= 1.5 and cyc >= 0.35 and rng >= 0.025
    valid_soft = cycles >= 2.0 and freq >= 1.0 and cyc >= 0.3 and rng >= 0.025
    pattern = "vertical_bounce" if axis == "y" else "forward_back_or_lateral_cycle"
    if rng < 0.025:
        pattern = "no_real_motion"
    elif trans >= 0.75:
        pattern = "monotonic_transition"
    return {
        "dominant_axis": axis,
        "max_displacement_range": rng,
        "dominant_axis_range": rng,
        "total_path_length": max(rng * 4.0, 0.0),
        "net_displacement_distance": rng * (0.2 if cycles >= 1.0 else 1.0),
        "net_to_path_ratio": 0.2 if cycles >= 1.0 else 1.0,
        "moving_step_count": 20 if rng >= 0.025 else 0,
        "has_real_motion": rng >= 0.025,
        "estimated_cycle_count": cycles,
        "estimated_frequency_hz": freq,
        "cyclicity_score": cyc,
        "transition_score": trans,
        "pose_hold_score": 0.0,
        "axis_metrics": {axis: {"estimated_cycle_count": cycles, "cyclicity_score": cyc, "transition_score": trans}},
        "cowgirl_motion_pattern": {
            "pattern": pattern,
            "valid_clean_cowgirl_pattern": valid_clean,
            "valid_soft_cowgirl_pattern": (not valid_clean) and valid_soft,
            "clean_cycle_axes": [axis] if valid_clean else [],
            "soft_cycle_axes": [axis] if valid_soft else [],
        },
    }


def _cycle(**controllers):
    return {
        "controller_metrics": controllers,
        "anchor_summary": {"feet_stable": True, "knees_stable": True, "hands_stable": True, "possible_locomotion": False},
        "controller_completeness_summary": {
            "status": "usable",
            "has_hip_control": "hipControl" in controllers,
            "has_pelvis_control": "pelvisControl" in controllers,
            "has_hip_or_pelvis": bool({"hipControl", "pelvisControl"} & set(controllers)),
            "has_real_hip_motion": controllers.get("hipControl", {}).get("has_real_motion", False) if "hipControl" in controllers else False,
            "has_real_pelvis_motion": controllers.get("pelvisControl", {}).get("has_real_motion", False) if "pelvisControl" in controllers else False,
            "reasons": [],
        },
        "has_hip_control": "hipControl" in controllers,
    }


def test_cycle_extractor_does_not_count_jitter():
    signal = [0.001 * math.sin(i) for i in range(80)]
    m = compute_signal_cycle_metrics(signal)
    assert m["pose_hold_score"] == 1.0
    assert m["cyclicity_score"] < 0.2


def test_monotonic_hip_upward_is_transition_not_cowgirl():
    pose = _pose("cowgirl")
    cycle = _cycle(hipControl=_metric(cycles=0.0, cyc=0.0, trans=0.9, rng=0.3))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["category"] if "category" in row else True
    assert row["motion_state"] == "intro_transition"
    assert row["resolved_motion_family"] != "cowgirl"


def test_cyclic_hip_y_cowgirl_passes():
    row = resolve_motion_candidate_v1(_pose("cowgirl"), _cycle(hipControl=_metric(cycles=3.0, cyc=0.75, trans=0.1, axis="y", freq=1.5)))
    assert row["resolved_motion_family"] == "cowgirl"
    assert row["motion_state"] == "clean_motion"
    assert row["resolved_motion_subtype"] == "cowgirl_vertical_bounce"


def test_hip_cycle_with_moving_feet_fails_cowgirl_anchor():
    cycle = _cycle(hipControl=_metric())
    cycle["anchor_summary"]["feet_stable"] = False
    row = resolve_motion_candidate_v1(_pose("cowgirl"), cycle)
    assert row["final_clean_motion_gate"] == "fail_anchor_lost"
    assert row["motion_state"] == "intro_transition"


def test_cowgirl_requires_real_hip_controller_motion():
    cycle = _cycle(hipControl=_metric(cycles=3.0, cyc=0.9, rng=0.001, freq=1.5), pelvisControl=_metric(cycles=3.0, cyc=0.9, rng=0.12, freq=1.5))
    row = resolve_motion_candidate_v1(_pose("cowgirl"), cycle)
    assert row["final_clean_motion_gate"] == "fail_pose_hold"
    assert "hipControl has no real transform-distance motion" in row["gate_failure_reasons"]
    assert row["transform_distance_gate_result"] == "fail_pose_hold"


def test_cowgirl_missing_hip_and_pelvis_fails_driver():
    row = resolve_motion_candidate_v1(_pose("cowgirl"), _cycle(headControl=_metric(cycles=2.0, cyc=0.9, rng=0.2)))
    assert row["final_clean_motion_gate"] == "fail_wrong_driver"
    assert "hipControl and pelvisControl missing" in " ".join(row["gate_failure_reasons"])


def test_cowgirl_short_hip_cycle_no_longer_passes_clean_gate():
    row = resolve_motion_candidate_v1(_pose("cowgirl"), _cycle(hipControl=_metric(cycles=1.0, cyc=0.7, trans=0.1, rng=0.2, freq=0.5)))
    assert row["final_clean_motion_gate"] == "fail_insufficient_cycles"
    assert row["motion_state"] != "clean_motion"


def test_cowgirl_needs_three_cycles_per_two_seconds_for_clean_gate():
    row = resolve_motion_candidate_v1(_pose("cowgirl"), _cycle(hipControl=_metric(cycles=2.0, cyc=0.8, trans=0.1, rng=0.2, freq=1.0)))
    assert row["cycle_gate_result"] == "soft_pass"
    assert row["motion_state"] == "short_cycle_candidate"

    clean = resolve_motion_candidate_v1(_pose("cowgirl"), _cycle(hipControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, rng=0.2, freq=1.5)))
    assert clean["cycle_gate_result"] == "pass"
    assert clean["motion_pattern_gate_result"] == "pass"
    assert clean["motion_state"] == "clean_motion"


def test_cowgirl_pattern_gate_rejects_non_cowgirl_hip_motion():
    bad = _metric(cycles=3.0, cyc=0.8, rng=0.2, freq=1.5)
    bad["cowgirl_motion_pattern"] = {
        "pattern": "monotonic_transition",
        "valid_clean_cowgirl_pattern": False,
        "valid_soft_cowgirl_pattern": False,
    }
    row = resolve_motion_candidate_v1(_pose("cowgirl"), _cycle(hipControl=bad))
    assert row["motion_pattern_gate_result"] == "fail_transition"
    assert row["motion_state"] != "clean_motion"


def test_global_same_direction_controller_motion_rejects_cowgirl():
    cycle = _cycle(
        hipControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, axis="x", freq=1.5),
        pelvisControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, axis="x", freq=1.5),
        chestControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, axis="x", freq=1.5),
        headControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, axis="x", freq=1.5),
        lHandControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, axis="x", freq=1.5),
        rHandControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, axis="x", freq=1.5),
    )
    cycle["driver_summary"] = {"global_motion_suspect": True}
    row = resolve_motion_candidate_v1(_pose("cowgirl"), cycle)
    assert row["final_clean_motion_gate"] == "fail_transition"
    assert "many controllers move together" in " ".join(row["gate_failure_reasons"])


def test_cowgirl_missing_controller_completeness_gate_is_red():
    cycle = _cycle(headControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, freq=1.5), lHandControl=_metric(cycles=3.0, cyc=0.9, rng=0.2, freq=1.5))
    cycle["controller_completeness_summary"] = {
        "status": "only_upper_body_hands_head",
        "has_hip_control": False,
        "has_pelvis_control": False,
        "has_hip_or_pelvis": False,
        "reasons": ["only chest/head/hand controllers are present"],
    }
    row = resolve_motion_candidate_v1(_pose("cowgirl"), cycle)
    assert row["controller_gate_result"] == "fail_wrong_driver"
    assert row["final_clean_motion_gate"] == "fail_wrong_driver"
    assert "only chest/head/hand controllers are present" in row["gate_failure_reasons"]


def test_cowgirl_mixed_hand_cycle_competes_with_hip_fails_clean():
    cycle = _cycle(
        hipControl=_metric(cycles=3.0, cyc=0.75, rng=0.2, freq=1.5),
        pelvisControl=_metric(cycles=2.0, cyc=0.5, rng=0.08, freq=1.0),
        rHandControl=_metric(cycles=5.0, cyc=0.8, rng=0.19, freq=2.5),
    )
    cycle["controller_metrics"]["rHandControl"]["total_path_length"] = 0.8
    cycle["controller_metrics"]["hipControl"]["total_path_length"] = 1.0
    row = resolve_motion_candidate_v1(_pose("cowgirl"), cycle)
    assert row["final_clean_motion_gate"] == "fail_wrong_driver"
    assert "hand controller motion competes" in " ".join(row["gate_failure_reasons"])


def test_cowgirl_motion_in_lying_pose_fails_partner_pose_gate():
    relational = {
        "actor_pelvis_partner_alignment_distance_mean": 0.82,
        "actor_pelvis_partner_alignment_score": 0.65,
        "actor_above_partner_score": 0.0,
    }
    row = resolve_motion_candidate_v1(
        _pose("cowgirl"),
        _cycle(hipControl=_metric(cycles=11.0, cyc=1.0, trans=0.0, axis="z", freq=2.7, rng=0.06)),
        relational,
    )
    assert row["final_clean_motion_gate"] == "fail_partner_alignment_lost"
    assert row["motion_state"] == "intro_transition"
    assert "lying/flat pose cannot be clean Cowgirl" in " ".join(row["gate_failure_reasons"])


def test_cowgirl_fails_when_hip_too_far_from_partner_contact_proxy():
    relational = {
        "actor_pelvis_partner_alignment_distance_mean": 0.42,
        "actor_pelvis_partner_alignment_score": 0.8,
        "actor_above_partner_score": 1.0,
    }
    row = resolve_motion_candidate_v1(
        _pose("cowgirl"),
        _cycle(hipControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, axis="y", freq=1.5)),
        relational,
    )
    assert row["target_proximity_gate_result"] == "fail_partner_alignment_lost"
    assert row["final_clean_motion_gate"] == "fail_partner_alignment_lost"
    assert "farther than 35cm" in " ".join(row["gate_failure_reasons"])


def test_head_one_way_lowering_is_not_bj_clean():
    pose = _pose("bj_oral")
    cycle = _cycle(headControl=_metric(cycles=0.0, cyc=0.0, trans=0.9, rng=0.2), hipControl=_metric(cycles=0.0, cyc=0.0, trans=0.0, rng=0.01))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "bj_oral_reaching_or_alignment"


def test_cyclic_head_static_pelvis_is_bj():
    pose = _pose("bj_oral")
    cycle = _cycle(headControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, freq=1.5), hipControl=_metric(cycles=0.0, cyc=0.0, trans=0.0, rng=0.01, freq=0.0))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "bj_oral"
    assert row["motion_state"] == "clean_motion"


def test_bj_head_cycle_must_be_near_partner_contact_proxy():
    pose = _pose("bj_oral")
    cycle = _cycle(headControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, freq=1.5), hipControl=_metric(cycles=0.0, cyc=0.0, trans=0.0, rng=0.01, freq=0.0))
    row = resolve_motion_candidate_v1(pose, cycle, {"head_to_partner_pelvis_distance_mean": 0.31, "head_to_partner_pelvis_target_score": 0.9})
    assert row["target_proximity_gate_result"] == "fail_partner_alignment_lost"
    assert row["resolved_motion_family"] == "bj_oral_reaching_or_alignment"
    assert "farther than 20cm" in " ".join(row["gate_failure_reasons"])


def test_bj_requires_head_controller_not_chest_only():
    pose = _pose("bj_oral")
    cycle = _cycle(chestControl=_metric(cycles=4.0, cyc=0.9, trans=0.1, freq=2.0), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01, freq=0.0))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "bj_oral_reaching_or_alignment"
    assert row["controller_gate_result"] == "fail_wrong_driver"
    assert "headControl missing" in " ".join(row["gate_failure_reasons"])


def test_bj_head_two_cycles_fails_strict_cycle_gate():
    pose = _pose("bj_oral")
    cycle = _cycle(headControl=_metric(cycles=2.0, cyc=0.9, trans=0.1, freq=2.0), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01, freq=0.0))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "bj_oral_reaching_or_alignment"
    assert row["cycle_gate_result"] == "fail_insufficient_cycles"


def test_bj_empty_head_keyframes_fail_transform_distance_gate():
    pose = _pose("bj_oral")
    cycle = _cycle(headControl=_metric(cycles=4.0, cyc=0.9, trans=0.1, freq=2.0, rng=0.0), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01, freq=0.0))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "bj_oral_reaching_or_alignment"
    assert row["transform_distance_gate_result"] == "fail_pose_hold"


def test_hand_single_reach_fails_hj():
    pose = _pose("handjob")
    cycle = _cycle(lHandControl=_metric(cycles=0.0, cyc=0.0, trans=0.9, rng=0.2), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "hand_reaching_or_touching"


def test_cyclic_hand_static_pelvis_is_hj():
    pose = _pose("handjob")
    cycle = _cycle(lHandControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, freq=1.5), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01, freq=0.0))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "handjob"
    assert row["motion_state"] == "clean_motion"


def test_hj_missing_hand_controller_fails():
    pose = _pose("handjob")
    cycle = _cycle(hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "hand_reaching_or_touching"
    assert row["controller_gate_result"] == "fail_wrong_driver"


def test_hj_two_hand_cycles_fails_strict_cycle_gate():
    pose = _pose("handjob")
    cycle = _cycle(rHandControl=_metric(cycles=2.0, cyc=0.9, trans=0.1, freq=2.0), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "hand_reaching_or_touching"
    assert row["cycle_gate_result"] == "fail_insufficient_cycles"


def test_hj_empty_hand_keyframes_fail_transform_distance_gate():
    pose = _pose("handjob")
    cycle = _cycle(rHandControl=_metric(cycles=4.0, cyc=0.9, trans=0.1, freq=2.0, rng=0.0), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "hand_reaching_or_touching"
    assert row["transform_distance_gate_result"] == "fail_pose_hold"


def test_hj_right_hand_cycle_can_pass():
    pose = _pose("handjob")
    cycle = _cycle(rHandControl=_metric(cycles=4.0, cyc=0.9, trans=0.1, freq=2.0), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "handjob"
    assert row["primary_driver_controller"] == "rHandControl"


def test_hj_hand_cycle_must_be_near_partner_contact_proxy():
    pose = _pose("handjob")
    cycle = _cycle(rHandControl=_metric(cycles=4.0, cyc=0.9, trans=0.1, freq=2.0), hipControl=_metric(cycles=0.0, cyc=0.0, rng=0.01))
    relational = {
        "hand_partner_targets": {
            "rHand": {
                "distances": {
                    "pelvis": {"mean": 0.33},
                    "hip": {"mean": 0.31},
                }
            }
        }
    }
    row = resolve_motion_candidate_v1(pose, cycle, relational)
    assert row["target_proximity_gate_result"] == "fail_partner_alignment_lost"
    assert row["resolved_motion_family"] == "hand_reaching_or_touching"
    assert "farther than 20cm" in " ".join(row["gate_failure_reasons"])


def test_male_active_thrust_passes_with_three_cycles_distance_and_proximity():
    cycle = _cycle(hipControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, rng=0.09, axis="z", freq=1.0))
    relational = {
        "actor_pelvis_partner_alignment_distance_mean": 0.22,
        "hip_motion_contact_axis": {"available": True, "range": 0.09, "path": 0.24, "net": 0.01, "net_to_path": 0.04},
    }
    row = resolve_motion_candidate_v1(_male_pose(), cycle, relational)
    assert row["resolved_motion_family"] == "male_active_thrust"
    assert row["motion_state"] == "clean_motion"
    assert row["resolved_motion_subtype"] == "male_active_penetration_thrust"


def test_male_active_thrust_below_seven_cm_fails():
    cycle = _cycle(hipControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, rng=0.06, axis="z", freq=1.0))
    relational = {
        "actor_pelvis_partner_alignment_distance_mean": 0.22,
        "hip_motion_contact_axis": {"available": True, "range": 0.06, "path": 0.18},
    }
    row = resolve_motion_candidate_v1(_male_pose(), cycle, relational)
    assert row["resolved_motion_family"] == "male_active_thrust_transition_or_invalid"
    assert row["transform_distance_gate_result"] == "fail_pose_hold"
    assert "below 7cm" in " ".join(row["gate_failure_reasons"])


def test_male_active_thrust_far_from_partner_fails():
    cycle = _cycle(hipControl=_metric(cycles=3.0, cyc=0.8, trans=0.1, rng=0.09, axis="z", freq=1.0))
    relational = {
        "actor_pelvis_partner_alignment_distance_mean": 0.42,
        "hip_motion_contact_axis": {"available": True, "range": 0.09, "path": 0.24},
    }
    row = resolve_motion_candidate_v1(_male_pose(), cycle, relational)
    assert row["resolved_motion_family"] == "male_active_thrust_transition_or_invalid"
    assert row["partner_alignment_gate_result"] == "fail_partner_alignment_lost"
    assert "farther than 30cm" in " ".join(row["gate_failure_reasons"])


def test_doggy_locomotion_rejects_clean_loop():
    pose = _pose("doggy")
    pose["pose_subtype"] = "doggy_all_fours"
    cycle = _cycle(hipControl=_metric(axis="z"))
    cycle["anchor_summary"]["possible_locomotion"] = True
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["motion_state"] == "crawling_locomotion"


def test_missionary_chest_rising_rejects_clean():
    pose = _pose("missionary")
    pose["pose_subtype"] = "missionary_supine"
    cycle = _cycle(hipControl=_metric(), chestControl=_metric(cycles=0.0, cyc=0.0, trans=0.9, rng=0.2))
    row = resolve_motion_candidate_v1(pose, cycle)
    assert row["resolved_motion_family"] == "missionary_getting_up_or_transition"


def test_no_training_or_manual_label_markers():
    row = resolve_motion_candidate_v1(_pose("cowgirl"), _cycle(hipControl=_metric()))
    assert row["ml_training_performed"] is False
    assert row["manual_labels_modified"] is False

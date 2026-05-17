"""Plan review-only Timeline examples from real manual VaM pose captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.vam_controller_mapping import (
    BJ_STATIC_HIPS,
    COWGIRL_HIP_FOLLOWERS,
    COWGIRL_HIP_PRIMARY_DRIVER,
    COWGIRL_STATIC_ANCHORS,
    HJ_STATIC_HIPS,
)
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl


CORE_CONTROLLERS = [
    "hipControl",
    "pelvisControl",
    "abdomenControl",
    "chestControl",
    "headControl",
    "lHandControl",
    "rHandControl",
    "lElbowControl",
    "rElbowControl",
    "lKneeControl",
    "rKneeControl",
    "lFootControl",
    "rFootControl",
    "lThighControl",
    "rThighControl",
]


TARGET_EXAMPLES = [
    {
        "capture_id": "pose_capture_20260516_204110",
        "clip_id": "manualgt_cowgirl_classic_grinding",
        "motion_example_name": "cowgirl_grinding",
        "motion_curve_type": "oval_grind",
        "driver_controllers": ["pelvisControl"],
        "follower_controllers": ["abdomenControl", "chestControl", "headControl"],
        "static_anchor_controllers": ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 1.0,
        "allowed_motion_axes": ["x", "z", "y_small"],
        "forbidden_motion_axes": ["feet_all_axes", "hands_primary", "head_primary"],
        "expected_motion": "Pelvis oval/low vertical grind; feet and hands stay captured.",
    },
    {
        "capture_id": "pose_capture_20260516_204110",
        "clip_id": "manualgt_cowgirl_classic_bounce",
        "motion_example_name": "cowgirl_vertical_bounce",
        "motion_curve_type": "vertical_bounce",
        "driver_controllers": ["pelvisControl"],
        "follower_controllers": ["abdomenControl", "chestControl", "headControl"],
        "static_anchor_controllers": ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 1.0,
        "allowed_motion_axes": ["y", "z_small"],
        "forbidden_motion_axes": ["feet_all_axes", "hands_primary", "head_primary"],
        "expected_motion": "Pelvis vertical bounce/riding from captured Cowgirl baseline.",
    },
    {
        "capture_id": "pose_capture_20260516_210540",
        "clip_id": "manualgt_cowgirl_lean_back_grind",
        "motion_example_name": "cowgirl_lean_back_grind",
        "motion_curve_type": "slow_grind",
        "driver_controllers": ["pelvisControl"],
        "follower_controllers": ["abdomenControl", "chestControl", "headControl"],
        "static_anchor_controllers": ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 0.75,
        "allowed_motion_axes": ["x", "z", "y_small"],
        "forbidden_motion_axes": ["hands_support_motion", "feet_all_axes"],
        "expected_motion": "Gentle grind while preserving lean-back and hand support.",
    },
    {
        "capture_id": "pose_capture_20260516_210528",
        "clip_id": "manualgt_sitting_cowgirl_small_grind",
        "motion_example_name": "sitting_cowgirl_small_grind",
        "motion_curve_type": "small_grind",
        "driver_controllers": ["pelvisControl"],
        "follower_controllers": ["abdomenControl", "chestControl", "headControl"],
        "static_anchor_controllers": ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 0.55,
        "allowed_motion_axes": ["x", "z", "y_tiny"],
        "forbidden_motion_axes": ["feet_all_axes", "hands_primary"],
        "expected_motion": "Small close-partner sitting Cowgirl grind.",
    },
    {
        "capture_id": "pose_capture_20260516_203203",
        "clip_id": "manualgt_bj_kneeling_head_bob",
        "motion_example_name": "bj_kneeling_head_bob",
        "motion_curve_type": "head_chest_forward_back",
        "driver_controllers": ["headControl"],
        "follower_controllers": ["chestControl"],
        "static_anchor_controllers": ["pelvisControl", "lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 1.0,
        "allowed_motion_axes": ["z", "y_small"],
        "forbidden_motion_axes": ["pelvis_riding", "feet_all_axes", "hands_primary"],
        "expected_motion": "Head/chest bob; pelvis and feet stay static.",
    },
    {
        "capture_id": "pose_capture_20260516_205956",
        "clip_id": "manualgt_hj_kneeling_hand_motion",
        "motion_example_name": "hj_kneeling_hand_motion",
        "motion_curve_type": "single_hand_forward_back",
        "driver_controllers": ["rHandControl"],
        "follower_controllers": [],
        "static_anchor_controllers": ["pelvisControl", "lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "headControl", "chestControl"],
        "amplitude_scale": 1.0,
        "allowed_motion_axes": ["z", "y_tiny"],
        "forbidden_motion_axes": ["pelvis_riding", "feet_all_axes", "other_hand_primary"],
        "expected_motion": "One hand forward/back; pelvis, feet, and other hand static.",
    },
    {
        "capture_id": "pose_capture_20260516_204615",
        "clip_id": "manualgt_doggy_classic_receiver_response",
        "motion_example_name": "doggy_classic_receiver_response",
        "motion_curve_type": "receiver_response_z",
        "driver_controllers": ["pelvisControl"],
        "follower_controllers": ["chestControl"],
        "static_anchor_controllers": ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 0.65,
        "allowed_motion_axes": ["z", "y_tiny"],
        "forbidden_motion_axes": ["cowgirl_vertical_bounce", "hands_primary", "feet_all_axes"],
        "expected_motion": "Subtle receiver body response, not Cowgirl bounce.",
    },
    {
        "capture_id": "pose_capture_20260516_210510",
        "clip_id": "manualgt_standing_doggy_table_response",
        "motion_example_name": "standing_doggy_table_response",
        "motion_curve_type": "standing_receiver_response_z",
        "driver_controllers": ["pelvisControl"],
        "follower_controllers": ["chestControl"],
        "static_anchor_controllers": ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 0.45,
        "allowed_motion_axes": ["z", "y_tiny"],
        "forbidden_motion_axes": ["cowgirl_vertical_bounce", "hands_primary", "feet_all_axes"],
        "expected_motion": "Small standing/table-supported receiver response.",
    },
    {
        "capture_id": "pose_capture_20260516_210433",
        "clip_id": "manualgt_missionary_legs_up_counter_motion",
        "motion_example_name": "missionary_legs_up_counter_motion",
        "motion_curve_type": "pelvis_counter_lift",
        "driver_controllers": ["pelvisControl"],
        "follower_controllers": ["lKneeControl", "rKneeControl", "lFootControl", "rFootControl"],
        "static_anchor_controllers": ["headControl", "chestControl", "lHandControl", "rHandControl"],
        "amplitude_scale": 0.6,
        "allowed_motion_axes": ["y", "z_tiny", "legs_reactive"],
        "forbidden_motion_axes": ["cowgirl_riding_loop", "head_bj_motion"],
        "expected_motion": "Subtle pelvis counter-lift with light leg reaction.",
    },
]


def build_manual_gt_timeline_plans_v1(
    ground_truth: str | Path,
    out_json: str | Path,
    *,
    duration: float = 4.0,
    fps: int | float = 60,
    keyframe_rate: float | None = None,
    mapping_version: str = "v1",
    require_hip_control: bool = False,
) -> dict[str, Any]:
    rows = load_jsonl(ground_truth)
    by_id = {str(row.get("capture_id")): row for row in rows}
    plans: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    specs = _target_examples_for_mapping(mapping_version)
    for spec in specs:
        capture_id = spec["capture_id"]
        row = by_id.get(capture_id)
        if row is None:
            skipped.append({"capture_id": capture_id, "clip_id": spec["clip_id"], "reason": "capture_not_found"})
            continue
        labels = row.get("human_labels") or {}
        existing = set(((row.get("atoms") or {}).get("rider") or {}).get("controllers") or {})
        if require_hip_control and "hipControl" not in existing:
            skipped.append({"capture_id": capture_id, "clip_id": spec["clip_id"], "reason": "missing_required_hipControl"})
            continue
        driver = [name for name in spec["driver_controllers"] if name in existing]
        followers = [name for name in spec["follower_controllers"] if name in existing]
        anchors = [name for name in spec["static_anchor_controllers"] if name in existing and name not in driver]
        explicitly_static = [name for name in CORE_CONTROLLERS if name in existing and name not in set(driver + followers)]
        warnings: list[str] = []
        if len(driver) != len(spec["driver_controllers"]):
            warnings.append("missing_some_driver_controllers")
        if require_hip_control and "hipControl" not in driver + followers + anchors + explicitly_static:
            warnings.append("hipControl_not_exported")
        plans.append(
            {
                "schema_version": f"manual_gt_timeline_plan_{mapping_version}",
                "clip_id": spec["clip_id"],
                "capture_id": capture_id,
                "family": labels.get("family"),
                "subtype": labels.get("pose_subtype"),
                "baseline_source_capture": row.get("raw_capture_path"),
                "screenshot_path": row.get("screenshot_path"),
                "motion_example_name": spec["motion_example_name"],
                "duration_seconds": float(duration),
                "fps": float(fps),
                "keyframe_rate": float(keyframe_rate if keyframe_rate is not None else fps),
                "driver_controllers": driver,
                "follower_controllers": followers,
                "static_anchor_controllers": sorted(set(anchors)),
                "explicitly_static_controllers": sorted(set(explicitly_static)),
                "allowed_motion_axes": spec["allowed_motion_axes"],
                "forbidden_motion_axes": spec["forbidden_motion_axes"],
                "motion_curve_type": spec["motion_curve_type"],
                "amplitude_scale": float(spec["amplitude_scale"]),
                "controller_mapping_version": mapping_version,
                "require_hip_control": bool(require_hip_control),
                "expected_motion": spec["expected_motion"],
                "human_notes": labels.get("raw_notes"),
                "warnings": warnings,
                "review_only": True,
                "ml_training_run": False,
                "manual_labels_yaml_modified": False,
            }
        )
    payload = {
        "schema_version": f"manual_gt_timeline_plans_{mapping_version}",
        "ground_truth": str(ground_truth),
        "duration_seconds": float(duration),
        "fps": float(fps),
        "keyframe_rate": float(keyframe_rate if keyframe_rate is not None else fps),
        "controller_mapping_version": mapping_version,
        "require_hip_control": bool(require_hip_control),
        "plans": plans,
        "skipped": skipped,
        "review_only": True,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    dump_json(out_json, payload)
    return {
        "status": "ok",
        "out_json": str(out_json),
        "plans": len(plans),
        "skipped": skipped,
    }


def _target_examples_for_mapping(mapping_version: str) -> list[dict[str, Any]]:
    if mapping_version != "v3":
        return TARGET_EXAMPLES
    specs: list[dict[str, Any]] = []
    for item in TARGET_EXAMPLES:
        spec = dict(item)
        family_hint = str(spec.get("clip_id") or "")
        if family_hint.startswith("manualgt_cowgirl") or "cowgirl" in family_hint:
            spec["driver_controllers"] = list(COWGIRL_HIP_PRIMARY_DRIVER)
            spec["follower_controllers"] = list(COWGIRL_HIP_FOLLOWERS)
            spec["static_anchor_controllers"] = list(COWGIRL_STATIC_ANCHORS)
            spec["expected_motion"] = str(spec.get("expected_motion") or "").replace("Pelvis", "Hip/hipControl").replace("pelvis", "hipControl")
            spec["forbidden_motion_axes"] = sorted(set(spec.get("forbidden_motion_axes") or []) | {"pelvisControl_primary_driver"})
        elif family_hint == "manualgt_bj_kneeling_head_bob":
            spec["static_anchor_controllers"] = sorted(set(spec["static_anchor_controllers"]) | set(BJ_STATIC_HIPS))
        elif family_hint == "manualgt_hj_kneeling_hand_motion":
            spec["static_anchor_controllers"] = sorted(set(spec["static_anchor_controllers"]) | set(HJ_STATIC_HIPS))
        elif "doggy" in family_hint:
            spec["driver_controllers"] = ["hipControl"]
            spec["follower_controllers"] = ["pelvisControl", "chestControl"]
            spec["static_anchor_controllers"] = ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"]
            spec["expected_motion"] = "Subtle hipControl-led receiver response, not Cowgirl bounce."
        elif "missionary" in family_hint:
            spec["driver_controllers"] = ["hipControl"]
            spec["follower_controllers"] = ["pelvisControl", "lKneeControl", "rKneeControl", "lFootControl", "rFootControl"]
            spec["static_anchor_controllers"] = ["headControl", "chestControl", "lHandControl", "rHandControl"]
            spec["expected_motion"] = "Subtle hipControl counter-lift with light leg reaction."
        specs.append(spec)
    return specs

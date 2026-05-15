from pathlib import Path

from vam_timeline_ai.datasets.cowgirl_candidate_database import build_cowgirl_candidate_db_v5
from vam_timeline_ai.datasets.semantic_candidate_database import build_semantic_candidate_db_from_actions_v0
from vam_timeline_ai.generation.baseline_pose import select_interaction_baseline_for_plan_v0
from vam_timeline_ai.generation.interaction_validation import validate_partner_relative_flow_v0
from vam_timeline_ai.generation.partner_relative_flow import synthesize_partner_relative_flow_v0
from vam_timeline_ai.generation.primitive_extractor import extract_cowgirl_motion_primitives_v1
from vam_timeline_ai.generation.primitive_groups import group_cowgirl_motion_primitives_v1
from vam_timeline_ai.generation.prompt_to_plan import draft_motion_plan_v1
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl, write_jsonl


def test_prompt_parser_recognizes_hands_on_partner_chest(tmp_path: Path):
    out = tmp_path / "plan.json"
    plan = draft_motion_plan_v1("slow cowgirl grinding, leaning forward, hands on partner chest", out)
    phase = plan["sequence"][0]
    assert plan["family"] == "cowgirl"
    assert plan["requested_pose_subtype"] == "cowgirl_lean_forward_supported"
    assert phase["interaction"]["contact_targets"]["lHand"] == "partner.chest"
    assert "keep_hands_near_partner_chest" in phase["constraints"]


def test_interaction_baseline_contains_partner_chest_reference(tmp_path: Path):
    plan = tmp_path / "plan.json"
    draft_motion_plan_v1("slow cowgirl grinding, hands on his chest", plan)
    baseline = select_interaction_baseline_for_plan_v0(plan, tmp_path / "baseline.json")
    assert baseline["person_root_included"] is False
    assert baseline["world_coords_allowed"] is False
    assert "partner_chest_reference" in baseline["partner_references"]
    hand_positions = {p["controller_name"]: p["baseline_position"] for p in baseline["controller_poses"]}
    assert hand_positions["lHandControl"][2] < 0.0


def test_partner_relative_flow_keeps_hand_targets_near_partner_chest(tmp_path: Path):
    plan = tmp_path / "plan.json"
    draft_motion_plan_v1("slow cowgirl grinding, hands on partner chest", plan)
    baseline = tmp_path / "baseline.json"
    select_interaction_baseline_for_plan_v0(plan, baseline)
    groups = tmp_path / "groups.json"
    dump_json(groups, {"groups": [{"primitive_set_id": "cowgirl_oval_grind", "primitives": ["p1"]}]})
    flow = synthesize_partner_relative_flow_v0(plan, groups, baseline, tmp_path / "flow.json", tmp_path / "flow.md")
    hand_tracks = [t for t in flow["controller_tracks"] if t["controller_name"] in {"lHandControl", "rHandControl"}]
    assert len(hand_tracks) == 2
    assert all(t["target"] == "partner.chest" for t in hand_tracks)
    result = validate_partner_relative_flow_v0(tmp_path / "flow.json", tmp_path / "validation.md")
    assert result["passed"] is True


def test_interaction_validation_fails_without_partner_target(tmp_path: Path):
    flow = {
        "source_world_coords_used": False,
        "person_root_tracks_included": False,
        "clip_stitching_used": False,
        "support_mode": "hands_on_partner_chest",
        "partner_references": {},
        "controller_tracks": [
            {"controller_name": "pelvisControl", "role": "driver", "position_deltas": [[0, 0, 0]]},
            {"controller_name": "lHandControl", "role": "support", "target": "partner.chest", "position_deltas": [[0, 0, 0]]},
        ],
    }
    dump_json(tmp_path / "flow.json", flow)
    result = validate_partner_relative_flow_v0(tmp_path / "flow.json", tmp_path / "validation.md")
    assert result["passed"] is False


def test_semantic_db_and_cowgirl_v5_derive_from_actions(tmp_path: Path):
    actions = tmp_path / "actions.jsonl"
    write_jsonl(actions, [
        {
            "window_id": "cowgirl",
            "semantic_family": "cowgirl",
            "pose_family": "cowgirl",
            "pose_subtype": "cowgirl_lean_forward_supported",
            "motion_subtype": "oval_grind",
            "partner_relation": ["rider_above_partner"],
            "contact_support": "hands_on_partner_chest",
            "phase": "clean_motion",
            "generation_safe": True,
            "semantic_score": 0.9,
            "pose_score": 0.8,
            "motion_score": 0.9,
            "interaction_score": 0.8,
            "consistency_score": 0.9,
        },
        {
            "window_id": "bj",
            "semantic_family": "bj_oral",
            "pose_family": "bj_oral",
            "motion_subtype": "bj_head_dominant_motion",
            "contact_support": "unknown",
            "phase": "clean_motion",
            "generation_safe": False,
            "semantic_score": 0.7,
        },
    ])
    semantic = build_semantic_candidate_db_from_actions_v0(actions, tmp_path / "semantic.jsonl", tmp_path / "semantic.csv", tmp_path / "semantic.md")
    assert {r["semantic_family"] for r in semantic} == {"cowgirl", "bj_oral"}
    cowgirl = build_cowgirl_candidate_db_v5(tmp_path / "semantic.jsonl", tmp_path / "cowgirl.jsonl", tmp_path / "cowgirl.csv", tmp_path / "cowgirl.md")
    categories = {r["category"] for r in cowgirl}
    assert "cowgirl_hands_on_partner_chest" in categories
    assert "not_cowgirl_bj_oral" in categories


def test_primitives_v1_include_pose_partner_contact_requirements(tmp_path: Path):
    candidate_db = tmp_path / "cowgirl.jsonl"
    relative = tmp_path / "relative.jsonl"
    trajectory = tmp_path / "trajectory.jsonl"
    pose = tmp_path / "pose.jsonl"
    interaction = tmp_path / "interaction.jsonl"
    write_jsonl(candidate_db, [{
        "candidate_id": "c1",
        "window_id": "w1",
        "category": "cowgirl_hands_on_partner_chest",
        "semantic_family": "cowgirl",
        "generation_safe": True,
        "duration_seconds": 4.0,
        "motion_subtype": "oval_grind",
        "pose_subtype": "cowgirl_lean_forward_supported",
        "contact_support": "hands_on_partner_chest",
        "partner_relation": ["rider_above_partner"],
        "semantic_score": 0.9,
    }])
    write_jsonl(relative, [{"window_id": "w1", "controllers_used": ["pelvisControl", "chestControl"], "feature_values": {"relative_pelvis_vertical_amplitude": 0.03, "relative_pelvis_forward_back_amplitude": 0.08, "relative_pelvis_lateral_amplitude": 0.04, "safe_for_learning": True}}])
    write_jsonl(trajectory, [{"window_id": "w1", "trajectory_shape_classification": "oval_grind", "feature_values": {"oval_path_score": 0.8}}])
    write_jsonl(pose, [{"window_id": "w1", "pose_family": "cowgirl", "pose_subtype": "cowgirl_lean_forward_supported"}])
    write_jsonl(interaction, [{"window_id": "w1", "support_context": "hands_on_partner_chest", "partner_relation": ["rider_above_partner"]}])
    rows = extract_cowgirl_motion_primitives_v1(candidate_db, relative, trajectory, pose, interaction, tmp_path / "primitives.jsonl", tmp_path / "primitives.md")
    assert len(rows) == 1
    primitive = rows[0]
    assert primitive["required_pose_family"] == "cowgirl"
    assert primitive["interaction_frame"] == "partner_chest_target"
    assert primitive["contact_support_requirements"]["hands_on_partner_chest_requires_partner_chest_target"] is True
    groups = group_cowgirl_motion_primitives_v1(tmp_path / "primitives.jsonl", tmp_path / "groups.json", tmp_path / "groups.md")
    assert groups["schema"] == "cowgirl_motion_primitive_groups_v1"


def test_no_manual_labels_or_ml_training_markers_in_generated_plan(tmp_path: Path):
    plan = draft_motion_plan_v1("slow cowgirl grinding, hands on partner chest", tmp_path / "plan.json")
    text = str(plan).lower()
    assert "manual_labels.yaml" not in text
    assert "ml_training" not in text

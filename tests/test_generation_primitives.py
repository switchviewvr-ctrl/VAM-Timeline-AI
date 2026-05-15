from pathlib import Path

from vam_timeline_ai.generation.motion_flow_generator import generate_motion_flow_skeleton_v0
from vam_timeline_ai.generation.primitive_extractor import extract_cowgirl_motion_primitives_v0
from vam_timeline_ai.generation.primitive_groups import group_cowgirl_motion_primitives_v0
from vam_timeline_ai.generation.primitive_retrieval import retrieve_primitives_for_plan_v0
from vam_timeline_ai.generation.prompt_to_plan import draft_motion_plan_v0
from vam_timeline_ai.io.json_utils import load_json, load_jsonl, write_jsonl


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    candidate_db = tmp_path / "candidate_db.jsonl"
    relative = tmp_path / "relative.jsonl"
    trajectory = tmp_path / "trajectory.jsonl"
    relative_index = tmp_path / "relative_index.jsonl"
    write_jsonl(candidate_db, [
        {
            "candidate_id": "cand_safe",
            "window_id": "safe",
            "category": "semantic_cowgirl_generation_safe",
            "semantic_family": "cowgirl",
            "generation_safe": True,
            "duration_seconds": 8.0,
            "cowgirl_subtype": "oval_grind",
            "generation_candidate_score": 0.9,
            "semantic_cowgirl_score": 0.9,
            "clean_motion_score": 0.8,
            "pose_anchor_status": "complete",
            "core_gate_status": "pass",
        },
        {
            "candidate_id": "cand_soft",
            "window_id": "soft",
            "category": "semantic_cowgirl_core_soft_fail_generation_safe",
            "semantic_family": "cowgirl",
            "generation_safe": True,
            "duration_seconds": 6.0,
            "cowgirl_subtype": "forward_back_rock",
            "generation_candidate_score": 0.7,
            "semantic_cowgirl_score": 0.8,
            "clean_motion_score": 0.7,
            "pose_anchor_status": "partial",
            "core_gate_status": "soft_fail",
            "core_gate_can_be_overridden": True,
        },
        {
            "candidate_id": "cand_bj",
            "window_id": "bj",
            "category": "not_cowgirl_bj_oral",
            "semantic_family": "bj_oral",
            "generation_safe": False,
            "excluded_from_cowgirl": True,
        },
    ])
    write_jsonl(relative, [
        {"window_id": "safe", "feature_values": {"safe_for_learning": True, "relative_pelvis_vertical_amplitude": 0.03, "relative_pelvis_forward_back_amplitude": 0.10, "relative_pelvis_lateral_amplitude": 0.08, "local_path_length": 0.6, "local_motion_energy": 0.5, "local_velocity_mean": 0.2, "local_velocity_max": 0.4, "local_rhythm_regularity": 0.8, "root_world_motion_removed": True}},
        {"window_id": "soft", "feature_values": {"safe_for_learning": True, "relative_pelvis_vertical_amplitude": 0.02, "relative_pelvis_forward_back_amplitude": 0.12, "relative_pelvis_lateral_amplitude": 0.01, "local_path_length": 0.5, "local_motion_energy": 0.4, "local_velocity_mean": 0.18, "local_velocity_max": 0.35, "local_rhythm_regularity": 0.7}},
    ])
    write_jsonl(trajectory, [
        {"window_id": "safe", "trajectory_shape_classification": "oval_grind", "dominant_motion_plane": "local_xz", "feature_values": {"oval_path_score": 0.9, "ellipse_fit_score": 0.8, "closed_loop_ratio": 0.7, "cycle_count_estimate": 2.0, "rhythm_repeat_score": 0.8}},
        {"window_id": "soft", "trajectory_shape_classification": "forward_back_rock", "dominant_motion_plane": "local_xz", "feature_values": {"linearity_score": 0.75, "closed_loop_ratio": 0.2, "cycle_count_estimate": 1.5, "rhythm_repeat_score": 0.6}},
    ])
    write_jsonl(relative_index, [
        {"window_id": "safe", "controllers": ["hipControl", "chestControl", "lFootControl", "rFootControl", "lKneeControl", "rKneeControl"]},
        {"window_id": "soft", "controllers": ["hipControl", "chestControl"]},
    ])
    return {"candidate_db": candidate_db, "relative": relative, "trajectory": trajectory, "relative_index": relative_index}


def test_primitive_extraction_uses_generation_safe_cowgirl_only(tmp_path):
    paths = _write_fixture(tmp_path)
    out = tmp_path / "primitives.jsonl"
    rows = extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], out, tmp_path / "report.md")
    assert {row["source_window_ids"][0] for row in rows} == {"safe", "soft"}
    assert all(row["semantic_family"] == "cowgirl" for row in rows)
    assert all(row["contains_absolute_world_coordinates"] is False for row in rows)
    assert all(row["is_timeline_clip"] is False for row in rows)
    assert rows[0]["trajectory_shape"]["classification"]
    assert "forward_back" in rows[0]["amplitude_profile"]


def test_primitive_grouping_creates_expected_subtype_groups(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "report.md")
    data = group_cowgirl_motion_primitives_v0(primitive_path, tmp_path / "groups.json", tmp_path / "groups.md")
    groups = {g["primitive_set_id"]: g for g in data["groups"]}
    assert groups["cowgirl_oval_grind"]["cluster_summary"]["count"] == 1
    assert groups["cowgirl_forward_back_rock"]["cluster_summary"]["count"] == 1


def test_prompt_to_plan_maps_cowgirl_grinding_slow(tmp_path):
    plan = draft_motion_plan_v0("slow cowgirl grinding, leaning forward", tmp_path / "plan.json")
    phase = plan["sequence"][0]
    assert plan["family"] == "cowgirl"
    assert phase["primitive_query"]["subtype"] == "oval_grind"
    assert phase["primitive_query"]["tempo"] == "slow"
    assert phase["body_parameters"]["torso_lean"] == "forward"
    assert plan["is_final_text_to_animation"] is False


def test_retrieval_and_flow_skeleton_do_not_export_timeline(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    retrieved_path = tmp_path / "retrieved.json"
    flow_path = tmp_path / "flow.json"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    retrieved = retrieve_primitives_for_plan_v0(plan_path, group_path, primitive_path, retrieved_path, tmp_path / "r.md")
    assert retrieved["timeline_export_performed"] is False
    assert retrieved["clip_stitching_performed"] is False
    assert retrieved["matches"][0]["candidate_primitive_ids"]
    flow = generate_motion_flow_skeleton_v0(plan_path, retrieved_path, flow_path, tmp_path / "f.md")
    assert flow["coordinate_space"] == "relative_body_motion"
    assert flow["export_ready"] is False
    assert flow["timeline_export_performed"] is False
    assert flow["clip_stitching_performed"] is False


def test_generation_pipeline_does_not_modify_manual_labels(tmp_path):
    paths = _write_fixture(tmp_path)
    run = tmp_path / "data" / "runs" / "clean_v2"
    out = tmp_path / "primitives.jsonl"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], out, tmp_path / "report.md")
    assert not (run / "labels" / "manual_labels.yaml").exists()
    assert "train" not in "\n".join(p.name for p in tmp_path.rglob("*")).lower()

from pathlib import Path

from vam_timeline_ai.generation.motion_flow_generator import generate_motion_flow_skeleton_v0
from vam_timeline_ai.generation.motion_flow_synthesis import synthesize_motion_flow_v0, synthesize_motion_flow_v1
from vam_timeline_ai.generation.generated_motion_validation import validate_generated_motion_flow_v0
from vam_timeline_ai.generation.baseline_pose import create_synthetic_baseline_pose_v0, create_cowgirl_review_baseline_pose_v1
from vam_timeline_ai.generation.relative_flow_retargeter import retarget_motion_flow_v0, retarget_motion_flow_v1
from vam_timeline_ai.generation.retarget_validation import validate_retargeted_motion_flow_v0, validate_retargeted_motion_flow_v1
from vam_timeline_ai.generation.timeline_from_retargeted_flow import export_retargeted_flow_timeline_v0
from vam_timeline_ai.generation.review_player_export import export_generated_flow_for_vam_review, export_generated_flow_for_vam_review_v1, prepare_vam_review_player_v0, write_review_export_status
from vam_timeline_ai.generation.native_timeline_exporter import export_generated_flow_native_timeline_v0, run_native_timeline_export_review_v0
from vam_timeline_ai.generation.native_timeline_exporter import export_generated_flow_native_timeline_v1, run_native_timeline_export_review_v1
from vam_timeline_ai.generation.native_timeline_validation import validate_native_timeline_export_v0, validate_native_timeline_export_v1
from vam_timeline_ai.timeline.codec import decode_keyframe_sequence
from vam_timeline_ai.generation.primitive_extractor import extract_cowgirl_motion_primitives_v0
from vam_timeline_ai.generation.primitive_groups import group_cowgirl_motion_primitives_v0
from vam_timeline_ai.generation.primitive_retrieval import retrieve_primitives_for_plan_v0
from vam_timeline_ai.generation.prompt_to_plan import draft_motion_plan_v0
from vam_timeline_ai.generation.trajectory_synthesizer import synthesize_oval_grind
from vam_timeline_ai.visualization.generated_motion_preview import render_generated_motion_preview_v0
from vam_timeline_ai.visualization.retargeted_motion_preview import render_retargeted_motion_preview_v0
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl, write_jsonl


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


def test_oval_grind_trajectory_is_centered_relative():
    _times, path = synthesize_oval_grind(4.0, 60.0, 2.0, 0.12, 0.08, 0.03, seed=7)
    assert path.shape[1] == 3
    assert max(abs(float(v)) for v in path.mean(axis=0)) < 1e-5
    assert float(abs(path[:, 0]).max()) < 0.2
    assert float(abs(path[:, 2]).max()) < 0.2


def test_synthesized_flow_is_relative_and_contains_driver_and_anchors(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    flow_path = tmp_path / "generated_flow.json"
    npz_path = tmp_path / "generated_flow.npz"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    flow = synthesize_motion_flow_v0(plan_path, group_path, primitive_path, flow_path, npz_path, tmp_path / "flow.md", duration=4.0, fps=30.0, seed=42)
    assert flow["selected_primitive_group"] == "cowgirl_oval_grind"
    assert flow["trajectory_shape"] == "oval_grind"
    assert flow["coordinate_space"] == "relative_body_motion"
    assert flow["no_world_coordinates"] is True
    assert flow["no_person_root_tracks"] is True
    assert flow["clip_stitching_used"] is False
    assert flow["timeline_export_performed"] is False
    assert flow["export_ready"] is False
    tracks = {track["controller_name"]: track for track in flow["controller_tracks"]}
    assert "hipControl" in tracks or "pelvisControl" in tracks
    assert tracks["lFootControl"]["role"] == "anchor"
    assert all(point == [0.0, 0.0, 0.0] for point in tracks["lFootControl"]["position_deltas"])
    assert not any("root" in name.lower() or "person" in name.lower() for name in tracks)


def test_generated_flow_validator_catches_world_flag_and_violent_jump(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    flow_path = tmp_path / "generated_flow.json"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    flow = synthesize_motion_flow_v0(plan_path, group_path, primitive_path, flow_path, tmp_path / "flow.npz", tmp_path / "flow.md", duration=4.0, fps=30.0, seed=42)
    assert validate_generated_motion_flow_v0(flow)["passed"] is True
    bad_flag = dict(flow)
    bad_flag["no_world_coordinates"] = False
    assert validate_generated_motion_flow_v0(bad_flag)["passed"] is False
    bad_jump = load_json(flow_path)
    bad_jump["controller_tracks"][0]["position_deltas"][5] = [3.0, 0.0, 0.0]
    result = validate_generated_motion_flow_v0(bad_jump)
    assert result["passed"] is False
    assert any(check["name"].endswith(":no_violent_jumps") and not check["passed"] for check in result["checks"])


def test_preview_command_handles_synthetic_flow(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    flow_path = tmp_path / "generated_flow.json"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    synthesize_motion_flow_v0(plan_path, group_path, primitive_path, flow_path, tmp_path / "flow.npz", tmp_path / "flow.md", duration=2.0, fps=20.0, seed=42)
    manifest = render_generated_motion_preview_v0(flow_path, tmp_path / "preview")
    assert manifest["timeline_export_performed"] is False
    assert (tmp_path / "preview" / "index.html").exists()
    assert (tmp_path / "preview" / "controller_track_summary.md").exists()


def _write_generated_flow_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    flow_path = tmp_path / "generated_flow.json"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    flow = synthesize_motion_flow_v0(plan_path, group_path, primitive_path, flow_path, tmp_path / "flow.npz", tmp_path / "flow.md", duration=2.0, fps=20.0, seed=42)
    baseline_path = tmp_path / "baseline.json"
    create_synthetic_baseline_pose_v0(baseline_path)
    return flow, flow_path, baseline_path


def test_synthetic_baseline_contains_required_controllers(tmp_path):
    out = tmp_path / "baseline.json"
    baseline = create_synthetic_baseline_pose_v0(out)
    names = {pose["controller_name"] for pose in baseline["controller_poses"]}
    assert {"pelvisControl", "chestControl", "lFootControl", "rFootControl", "lKneeControl", "rKneeControl"}.issubset(names)
    assert baseline["person_root_included"] is False
    assert baseline["world_coords_allowed"] is False


def test_cowgirl_baseline_pose_contains_kneeling_anchors(tmp_path):
    baseline = create_cowgirl_review_baseline_pose_v1(tmp_path / "cowgirl_baseline.json")
    poses = {p["controller_name"]: p for p in baseline["controller_poses"]}
    assert baseline["style"] == "kneeling_forward"
    assert baseline["intended_family"] == "cowgirl"
    assert poses["lKneeControl"]["is_anchor"] is True
    assert poses["rFootControl"]["baseline_position"][2] > poses["pelvisControl"]["baseline_position"][2]
    assert baseline["person_root_included"] is False


def test_retargeting_adds_relative_deltas_and_keeps_anchors_stable(tmp_path):
    flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retargeted = retarget_motion_flow_v0(flow_path, baseline_path, tmp_path / "retarget.json", tmp_path / "retarget.npz", tmp_path / "retarget.md")
    tracks = {track["controller_name"]: track for track in retargeted["controller_tracks"]}
    driver_name = "pelvisControl" if "pelvisControl" in tracks else "hipControl"
    generated_driver = next(track for track in flow["controller_tracks"] if track["controller_name"] == driver_name)
    base = tracks[driver_name]["baseline_position"]
    expected = [round(base[i] + generated_driver["position_deltas"][10][i], 6) for i in range(3)]
    assert tracks[driver_name]["retargeted_positions"][10] == expected
    assert len({tuple(point) for point in tracks["lFootControl"]["retargeted_positions"]}) == 1
    assert retargeted["person_root_included"] is False
    assert retargeted["source_world_coords_used"] is False
    assert retargeted["clip_stitching_used"] is False


def test_retarget_validation_catches_missing_anchor_and_huge_distance(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retargeted = retarget_motion_flow_v0(flow_path, baseline_path, tmp_path / "retarget.json", tmp_path / "retarget.npz", tmp_path / "retarget.md")
    assert validate_retargeted_motion_flow_v0(retargeted)["passed"] is True
    missing = dict(retargeted)
    missing["controller_tracks"] = [track for track in retargeted["controller_tracks"] if track["controller_name"] != "lFootControl"]
    assert validate_retargeted_motion_flow_v0(missing)["passed"] is False
    far = load_json(tmp_path / "retarget.json")
    for track in far["controller_tracks"]:
        if track["controller_name"] == "lFootControl":
            track["retargeted_positions"] = [[9.0, 9.0, 9.0] for _ in track["retargeted_positions"]]
    result = validate_retargeted_motion_flow_v0(far)
    assert result["passed"] is False
    assert any("not_too_far" in check["name"] and not check["passed"] for check in result["checks"])


def test_retarget_preview_and_review_export_gating(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retarget_path = tmp_path / "retarget.json"
    validation_path = tmp_path / "validation.md"
    retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    validate_retargeted_motion_flow_v0(retarget_path, validation_path)
    manifest = render_retargeted_motion_preview_v0(retarget_path, tmp_path / "retarget_preview")
    assert manifest["timeline_export_performed"] is False
    assert (tmp_path / "retarget_preview" / "index.html").exists()
    export = export_retargeted_flow_timeline_v0(retarget_path, validation_path, tmp_path / "timeline")
    assert export["status"] == "review_flow_json_written"
    bad_validation = tmp_path / "bad_validation.md"
    bad_validation.write_text("# Bad\n\n- Passed: `False`\n", encoding="utf-8")
    refused = export_retargeted_flow_timeline_v0(retarget_path, bad_validation, tmp_path / "timeline_bad")
    assert refused["status"] == "export_unavailable"


def test_motion_flow_v1_generates_coordinated_followers_and_reduces_lateral(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    v0 = synthesize_motion_flow_v0(plan_path, group_path, primitive_path, tmp_path / "v0.json", tmp_path / "v0.npz", tmp_path / "v0.md", duration=4.0, fps=30, seed=42)
    v1 = synthesize_motion_flow_v1(plan_path, group_path, primitive_path, "cowgirl_oval_grind_v1", tmp_path / "v1.json", tmp_path / "v1.npz", tmp_path / "v1.md", duration=4.0, fps=30, seed=42)
    tracks = {t["controller_name"]: t for t in v1["controller_tracks"]}
    assert {"pelvisControl", "abdomenControl", "chestControl", "headControl"}.issubset(tracks)
    assert v1["amplitude_profile"]["lateral"] < v0["amplitude_profile"]["lateral"]
    assert v1["amplitude_profile"]["vertical"] >= v0["amplitude_profile"]["vertical"]
    pelvis_span = max(abs(x) for row in tracks["pelvisControl"]["position_deltas"] for x in row)
    chest_span = max(abs(x) for row in tracks["chestControl"]["position_deltas"] for x in row)
    assert chest_span < pelvis_span
    assert not any("root" in name.lower() or "person" in name.lower() for name in tracks)


def test_v1_retarget_validation_and_review_player_schema(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    synthesize_motion_flow_v1(plan_path, group_path, primitive_path, "cowgirl_oval_grind_v1", tmp_path / "v1.json", tmp_path / "v1.npz", tmp_path / "v1.md", duration=2.0, fps=20, seed=42)
    create_cowgirl_review_baseline_pose_v1(tmp_path / "baseline.json")
    retargeted = retarget_motion_flow_v1(tmp_path / "v1.json", tmp_path / "baseline.json", tmp_path / "retarget.json", tmp_path / "retarget.npz", tmp_path / "retarget.md")
    assert retargeted["baseline_style"] == "kneeling_forward"
    result = validate_retargeted_motion_flow_v1(retargeted)
    assert result["passed"] is True
    data = export_generated_flow_for_vam_review_v1(tmp_path / "retarget.json", tmp_path / "player_v1.json", tmp_path / "player_v1.md")
    assert data["schema"] == "vam_generated_motion_review_player_v1"
    assert data["axis_scales"]
    assert data["baseline_style"] == "kneeling_forward"


def test_vam_review_player_export_is_relative_and_filters_disallowed_tracks(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retarget_path = tmp_path / "retarget.json"
    retargeted = retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    retargeted["controller_tracks"].append({
        "controller_name": "PersonRoot",
        "role": "root",
        "times": [0.0],
        "position_deltas_applied": [[1.0, 1.0, 1.0]],
    })
    bad_path = tmp_path / "retarget_with_root.json"
    from vam_timeline_ai.io.json_utils import dump_json
    dump_json(bad_path, retargeted)
    out = tmp_path / "review_player.json"
    data = export_generated_flow_for_vam_review(bad_path, out, tmp_path / "review_player.md")
    names = {row["name"] for row in data["controllers"]}
    assert data["schema"] == "vam_generated_motion_review_player_v0"
    assert data["coordinate_mode"] == "relative_to_playback_baseline"
    assert data["native_timeline_importable"] is False
    assert "PersonRoot" not in names
    assert data["person_root_tracks_included"] is False
    assert all(row["times"] and row["position_deltas"] for row in data["controllers"])


def test_prepare_review_player_writes_script_instructions_and_status(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retarget_path = tmp_path / "retarget.json"
    retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    summary = prepare_vam_review_player_v0(retarget_path, tmp_path / "review_player")
    script = Path(summary["script_source"])
    instructions = Path(summary["instructions"])
    assert Path(summary["review_player_json"]).exists()
    assert script.exists()
    assert instructions.exists()
    assert summary["vam_secure_json_path"] == "Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v0.json"
    text = script.read_text(encoding="utf-8")
    assert "allowedControllers" in text
    assert "lower.Contains(\"person\")" in text
    assert "baselinePosition + delta" in text
    assert "using System.IO" not in text
    assert "FileManagerSecure.ReadAllText" in text
    assert "ResolveSecurePath" in text
    status = tmp_path / "review_export_status.md"
    write_review_export_status(status)
    status_text = status.read_text(encoding="utf-8")
    assert "native_timeline_importable: false" in status_text
    assert "Generated Motion Review Player" in status_text


def test_native_timeline_export_writes_timeline_like_json_not_review_schema(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retarget_path = tmp_path / "retarget.json"
    retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    out = tmp_path / "generated.timeline.json"
    payload = export_generated_flow_native_timeline_v0(retarget_path, out, tmp_path / "export.md")
    assert out.exists()
    assert "schema" not in payload
    assert payload["SerializeVersion"] == "283"
    assert payload["AtomType"] == "Person"
    clip = payload["Clips"][0]
    assert clip["AnimationName"] == "Generated_Cowgirl_Grinding_V0"
    assert len(clip["Controllers"]) > 0
    names = {c["Controller"] for c in clip["Controllers"]}
    assert "PersonRoot" not in names
    assert payload["VAMTimelineAIGeneratedMetadata"]["generated_from_relative_flow"] is True
    assert payload["VAMTimelineAIGeneratedMetadata"]["review_player_not_required"] is True


def test_native_timeline_validation_catches_required_fields_and_disallowed_controllers(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retarget_path = tmp_path / "retarget.json"
    retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    out = tmp_path / "generated.timeline.json"
    payload = export_generated_flow_native_timeline_v0(retarget_path, out, tmp_path / "export.md")
    result = validate_native_timeline_export_v0(out, tmp_path / "validation.md")
    assert result["passed"] is True
    assert result["expected_importable"] == "unknown"
    for controller in payload["Clips"][0]["Controllers"]:
        keys = controller["X"]
        decoded = __import__("vam_timeline_ai.timeline.codec", fromlist=["decode_keyframe_sequence"]).decode_keyframe_sequence(keys, 283)
        times = [key.time for key in decoded]
        assert times == sorted(times)
    bad = dict(payload)
    bad.pop("SerializeVersion")
    assert validate_native_timeline_export_v0(bad)["passed"] is False
    bad2 = load_json(out)
    bad2["Clips"][0]["Controllers"].append({"Controller": "PersonRoot", "TargetsPosition": 1, "ControlPosition": 1, "X": [], "Y": [], "Z": []})
    assert validate_native_timeline_export_v0(bad2)["passed"] is False


def test_native_timeline_review_pipeline_writes_instructions(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    retarget_path = tmp_path / "retarget.json"
    retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    summary = run_native_timeline_export_review_v0(retarget_path, tmp_path / "native_review")
    assert Path(summary["timeline"]).exists()
    assert Path(summary["instructions"]).exists()
    assert summary["review_player_required"] is False
    instructions = Path(summary["instructions"]).read_text(encoding="utf-8")
    assert "Import" in instructions
    assert "Timeline" in instructions


def test_native_timeline_v1_includes_baseline_keyframe_and_rotations(tmp_path):
    paths = _write_fixture(tmp_path)
    primitive_path = tmp_path / "primitives.jsonl"
    group_path = tmp_path / "groups.json"
    plan_path = tmp_path / "plan.json"
    flow_path = tmp_path / "flow_v1.json"
    baseline_path = tmp_path / "cowgirl_baseline.json"
    retarget_path = tmp_path / "retarget_v1.json"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], primitive_path, tmp_path / "p.md")
    group_cowgirl_motion_primitives_v0(primitive_path, group_path, tmp_path / "g.md")
    draft_motion_plan_v0("slow cowgirl grinding, leaning forward", plan_path)
    synthesize_motion_flow_v1(plan_path, group_path, primitive_path, "cowgirl_oval_grind_v1", flow_path, tmp_path / "flow.npz", tmp_path / "flow.md", duration=2.0, fps=20, seed=42)
    baseline = create_cowgirl_review_baseline_pose_v1(baseline_path)
    retarget_motion_flow_v1(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    payload = export_generated_flow_native_timeline_v1(retarget_path, baseline_path, tmp_path / "generated_v1.timeline.json", tmp_path / "export.md")
    clip = payload["Clips"][0]
    assert clip["AnimationName"] == "Generated_Cowgirl_Grinding_V1"
    assert payload["VAMTimelineAIGeneratedMetadata"]["includes_baseline_keyframe"] is True
    assert payload["VAMTimelineAIGeneratedMetadata"]["includes_rotation_tracks"] is True
    controllers = {row["Controller"]: row for row in clip["Controllers"]}
    poses = {row["controller_name"]: row for row in baseline["controller_poses"]}
    pelvis = controllers["pelvisControl"]
    expected = poses["pelvisControl"]["baseline_position"]
    actual = [decode_keyframe_sequence(pelvis[axis], 283)[0].value for axis in ("X", "Y", "Z")]
    assert [round(v, 6) for v in actual] == [round(v, 6) for v in expected]
    assert all(len(decode_keyframe_sequence(pelvis[axis], 283)) > 0 for axis in ("RotX", "RotY", "RotZ", "RotW"))
    assert not any("root" in name.lower() or "person" in name.lower() for name in controllers)
    result = validate_native_timeline_export_v1(payload, baseline_path, tmp_path / "validation.md")
    assert result["passed"] is True
    assert result["expected_pose_context"] == "cowgirl_kneeling_forward"


def test_native_timeline_v1_validation_catches_missing_baseline_and_anchor(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    cowgirl_baseline = tmp_path / "cowgirl_baseline.json"
    create_cowgirl_review_baseline_pose_v1(cowgirl_baseline)
    retarget_path = tmp_path / "retarget.json"
    retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    payload = export_generated_flow_native_timeline_v1(retarget_path, cowgirl_baseline, tmp_path / "generated_v1.timeline.json", tmp_path / "export.md")
    bad = load_json(tmp_path / "generated_v1.timeline.json")
    bad["Clips"][0]["Controllers"][0]["X"][0] = bad["Clips"][0]["Controllers"][0]["X"][1]
    assert validate_native_timeline_export_v1(bad, cowgirl_baseline)["passed"] is False
    bad2 = load_json(tmp_path / "generated_v1.timeline.json")
    bad2["Clips"][0]["Controllers"] = [c for c in bad2["Clips"][0]["Controllers"] if c["Controller"] != "lFootControl"]
    assert validate_native_timeline_export_v1(bad2, cowgirl_baseline)["passed"] is False
    bad3 = load_json(tmp_path / "generated_v1.timeline.json")
    bad3["Clips"][0]["Controllers"].append({"Controller": "PersonRoot", "TargetsPosition": 1, "ControlPosition": 1, "X": [], "Y": [], "Z": []})
    assert validate_native_timeline_export_v1(bad3, cowgirl_baseline)["passed"] is False


def test_native_timeline_v1_review_pipeline_writes_import_instructions(tmp_path):
    _flow, flow_path, baseline_path = _write_generated_flow_fixture(tmp_path)
    cowgirl_baseline = tmp_path / "cowgirl_baseline.json"
    create_cowgirl_review_baseline_pose_v1(cowgirl_baseline)
    retarget_path = tmp_path / "retarget.json"
    retarget_motion_flow_v0(flow_path, baseline_path, retarget_path, tmp_path / "retarget.npz", tmp_path / "retarget.md")
    summary = run_native_timeline_export_review_v1(retarget_path, cowgirl_baseline, tmp_path / "native_v1")
    assert Path(summary["timeline"]).exists()
    assert Path(summary["instructions"]).exists()
    assert summary["includes_baseline_keyframe"] is True
    assert summary["includes_rotation_tracks"] is True
    text = Path(summary["instructions"]).read_text(encoding="utf-8")
    assert "Generated_Cowgirl_Grinding_V1" in text


def test_generation_pipeline_does_not_modify_manual_labels(tmp_path):
    paths = _write_fixture(tmp_path)
    run = tmp_path / "data" / "runs" / "clean_v2"
    out = tmp_path / "primitives.jsonl"
    extract_cowgirl_motion_primitives_v0(paths["candidate_db"], paths["relative"], paths["trajectory"], paths["relative_index"], out, tmp_path / "report.md")
    assert not (run / "labels" / "manual_labels.yaml").exists()
    assert "train" not in "\n".join(p.name for p in tmp_path.rglob("*")).lower()

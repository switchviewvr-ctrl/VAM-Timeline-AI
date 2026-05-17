from pathlib import Path

from vam_timeline_ai.generation.manual_gt_baseline_builder import build_manual_gt_baseline_for_plan
from vam_timeline_ai.generation.manual_gt_motion_synthesizer import synthesize_manual_gt_motion
from vam_timeline_ai.generation.manual_gt_timeline_exporter import export_manual_gt_timeline_examples_v1, export_manual_gt_timeline_examples_v2, export_manual_gt_timeline_examples_v3, export_manual_gt_timeline_examples_v4
from vam_timeline_ai.generation.manual_gt_timeline_planner import build_manual_gt_timeline_plans_v1
from vam_timeline_ai.generation.manual_gt_timeline_validation import validate_manual_gt_timeline_examples_v1, validate_manual_gt_timeline_examples_v2, validate_manual_gt_timeline_examples_v3, validate_manual_gt_timeline_examples_v4
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl


def _ctrl(x, y, z):
    return {
        "local_position_to_atom": [x, y, z],
        "world_position": [x, y, z],
        "local_rotation_to_atom_quat": [0, 0, 0, 1],
        "world_rotation_quat": [0, 0, 0, 1],
        "active": True,
    }


def _capture(path: Path, capture_id: str, family: str, subtype: str):
    controllers = {
        "hipControl": _ctrl(0, 0.95, 0),
        "pelvisControl": _ctrl(0, 1.0, 0),
        "abdomenControl": _ctrl(0, 1.22, 0.02),
        "chestControl": _ctrl(0, 1.45, 0.05),
        "headControl": _ctrl(0, 1.75, 0.08),
        "lHandControl": _ctrl(-0.25, 1.1, 0.08),
        "rHandControl": _ctrl(0.25, 1.1, 0.08),
        "lElbowControl": _ctrl(-0.22, 1.25, 0.06),
        "rElbowControl": _ctrl(0.22, 1.25, 0.06),
        "lKneeControl": _ctrl(-0.35, 0.55, 0.0),
        "rKneeControl": _ctrl(0.35, 0.55, 0.0),
        "lFootControl": _ctrl(-0.45, 0.05, 0.1),
        "rFootControl": _ctrl(0.45, 0.05, 0.1),
        "lThighControl": _ctrl(-0.25, 0.8, 0.0),
        "rThighControl": _ctrl(0.25, 0.8, 0.0),
    }
    raw_path = path / f"{capture_id}.json"
    dump_json(
        raw_path,
        {
            "schema_version": "pose_capture_v1",
            "atoms": {
                "rider": {"atom_uid": "Quiet", "controllers": controllers, "missing_controllers": []},
                "partner": {"atom_uid": "Person", "controllers": controllers, "missing_controllers": []},
            },
            "derived": {},
        },
    )
    return {
        "schema_version": "manual_pose_ground_truth_v1",
        "capture_id": capture_id,
        "raw_capture_path": str(raw_path),
        "screenshot_path": "",
        "human_labels": {
            "family": family,
            "pose_subtype": subtype,
            "raw_notes": "unit test",
        },
        "atoms": {"rider": {"controllers": controllers}},
        "measurements": {},
        "warnings": [],
    }


def _ground_truth(tmp_path: Path) -> Path:
    rows = [
        _capture(tmp_path, "pose_capture_20260516_204110", "cowgirl", "cowgirl_classic_lean_forward_light"),
        _capture(tmp_path, "pose_capture_20260516_210540", "cowgirl", "cowgirl_lean_back_object_supported"),
        _capture(tmp_path, "pose_capture_20260516_210528", "cowgirl", "sitting_cowgirl_intimate"),
        _capture(tmp_path, "pose_capture_20260516_203203", "bj_oral", "bj_kneeling_cowgirl_like"),
        _capture(tmp_path, "pose_capture_20260516_205956", "handjob", "hj_kneeling_to_standing_partner"),
        _capture(tmp_path, "pose_capture_20260516_204615", "doggy", "doggy_classic"),
        _capture(tmp_path, "pose_capture_20260516_210510", "doggy", "standing_doggy_table"),
        _capture(tmp_path, "pose_capture_20260516_210433", "missionary", "missionary_table"),
    ]
    out = tmp_path / "manual_pose_ground_truth_v1.jsonl"
    out.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")
    return out


def _range(track):
    cols = list(zip(*track["positions"]))
    return max(max(col) - min(col) for col in cols)


def test_planner_selects_known_capture_ids_and_roles(tmp_path):
    gt = _ground_truth(tmp_path)
    summary = build_manual_gt_timeline_plans_v1(gt, tmp_path / "plans.json", duration=2.0, fps=12)
    plans = load_json(tmp_path / "plans.json")["plans"]
    by_id = {plan["clip_id"]: plan for plan in plans}

    assert summary["plans"] == 9
    assert by_id["manualgt_cowgirl_classic_grinding"]["capture_id"] == "pose_capture_20260516_204110"
    assert by_id["manualgt_cowgirl_classic_grinding"]["driver_controllers"] == ["pelvisControl"]
    assert "lFootControl" in by_id["manualgt_cowgirl_classic_grinding"]["static_anchor_controllers"]
    assert "rFootControl" in by_id["manualgt_cowgirl_classic_grinding"]["static_anchor_controllers"]
    assert by_id["manualgt_bj_kneeling_head_bob"]["driver_controllers"] == ["headControl"]
    assert "pelvisControl" in by_id["manualgt_bj_kneeling_head_bob"]["static_anchor_controllers"]
    assert by_id["manualgt_hj_kneeling_hand_motion"]["driver_controllers"] == ["rHandControl"]


def test_baseline_builder_preserves_captured_controller_positions(tmp_path):
    gt = _ground_truth(tmp_path)
    build_manual_gt_timeline_plans_v1(gt, tmp_path / "plans.json", duration=2.0, fps=12)
    plan = load_json(tmp_path / "plans.json")["plans"][0]
    baseline = build_manual_gt_baseline_for_plan(plan)

    assert baseline["controller_baseline"]["pelvisControl"]["position"] == [0.0, 1.0, 0.0]
    assert baseline["controller_baseline"]["pelvisControl"]["rotation_quat"] == [0.0, 0.0, 0.0, 1.0]
    assert baseline["controller_baseline"]["pelvisControl"]["rotation_source"] == "local_rotation_to_atom_quat"
    assert baseline["source_world_tracks_included"] is False
    assert baseline["person_root_world_tracks_included"] is False


def test_motion_synthesizer_respects_family_static_rules(tmp_path):
    gt = _ground_truth(tmp_path)
    build_manual_gt_timeline_plans_v1(gt, tmp_path / "plans.json", duration=2.0, fps=12)
    plans = load_json(tmp_path / "plans.json")["plans"]
    cowgirl = next(plan for plan in plans if plan["clip_id"] == "manualgt_cowgirl_classic_grinding")
    bj = next(plan for plan in plans if plan["clip_id"] == "manualgt_bj_kneeling_head_bob")
    hj = next(plan for plan in plans if plan["clip_id"] == "manualgt_hj_kneeling_hand_motion")

    cow_clip = synthesize_manual_gt_motion(cowgirl, build_manual_gt_baseline_for_plan(cowgirl))
    cow_tracks = {track["controller_name"]: track for track in cow_clip["controller_tracks"]}
    assert _range(cow_tracks["pelvisControl"]) > 0.005
    assert _range(cow_tracks["lFootControl"]) == 0
    assert _range(cow_tracks["rFootControl"]) == 0

    bj_clip = synthesize_manual_gt_motion(bj, build_manual_gt_baseline_for_plan(bj))
    bj_tracks = {track["controller_name"]: track for track in bj_clip["controller_tracks"]}
    assert _range(bj_tracks["headControl"]) > 0.005
    assert _range(bj_tracks["pelvisControl"]) == 0

    hj_clip = synthesize_manual_gt_motion(hj, build_manual_gt_baseline_for_plan(hj))
    hj_tracks = {track["controller_name"]: track for track in hj_clip["controller_tracks"]}
    assert _range(hj_tracks["rHandControl"]) > 0.005
    assert _range(hj_tracks["pelvisControl"]) == 0


def test_exporter_writes_timeline_json_and_validation_passes(tmp_path):
    gt = _ground_truth(tmp_path)
    summary = export_manual_gt_timeline_examples_v1(gt, tmp_path / "pkg", duration=2.0, fps=12, copy_to_vam=False)
    validation = validate_manual_gt_timeline_examples_v1(tmp_path / "pkg", tmp_path / "pkg" / "reports" / "validation.md")
    payload = load_json(tmp_path / "pkg" / "clips" / "manualgt_cowgirl_classic_grinding.timeline.json")
    controllers = [row["Controller"] for row in payload["Clips"][0]["Controllers"]]

    assert summary["clips_exported"] == 9
    assert validation["status"] == "ok"
    assert payload["VAMTimelineAIManualGTMetadata"]["review_only"] is True
    assert "pelvisControl" in controllers
    assert all("root" not in name.lower() and "world" not in name.lower() and "person" not in name.lower() for name in controllers)


def test_validation_catches_foot_motion_in_cowgirl_and_pelvis_motion_in_bj(tmp_path):
    gt = _ground_truth(tmp_path)
    export_manual_gt_timeline_examples_v1(gt, tmp_path / "pkg", duration=2.0, fps=12, copy_to_vam=False)
    data_path = tmp_path / "pkg" / "preview_data" / "manual_gt_timeline_clips_v1.json"
    data = load_json(data_path)
    for clip in data["clips"]:
        if clip["clip_id"] == "manualgt_cowgirl_classic_grinding":
            for track in clip["controller_tracks"]:
                if track["controller_name"] == "lFootControl":
                    track["positions"] = [[p[0], p[1] + (0.03 if i % 2 else 0.0), p[2]] for i, p in enumerate(track["positions"])]
        if clip["clip_id"] == "manualgt_bj_kneeling_head_bob":
            for track in clip["controller_tracks"]:
                if track["controller_name"] == "pelvisControl":
                    track["positions"] = [[p[0], p[1], p[2] + (0.03 if i % 2 else 0.0)] for i, p in enumerate(track["positions"])]
    dump_json(data_path, data)
    validation = validate_manual_gt_timeline_examples_v1(tmp_path / "pkg", tmp_path / "pkg" / "reports" / "bad.md")

    assert validation["status"] == "failed"
    assert any("Cowgirl" in msg and "lFootControl" in msg for msg in validation["error_messages"])
    assert any("BJ pelvis" in msg for msg in validation["error_messages"])


def test_v2_export_writes_rotation_tracks_and_sparse_keys(tmp_path):
    gt = _ground_truth(tmp_path)
    export_manual_gt_timeline_examples_v1(gt, tmp_path / "manual_gt_timeline_examples_v1", duration=2.0, fps=12, copy_to_vam=False)
    summary = export_manual_gt_timeline_examples_v2(gt, tmp_path / "manual_gt_timeline_examples_v2", duration=4.0, keyframe_rate=2.0, copy_to_vam=False)
    validation = validate_manual_gt_timeline_examples_v2(tmp_path / "manual_gt_timeline_examples_v2", tmp_path / "manual_gt_timeline_examples_v2" / "reports" / "validation_v2.md")
    payload = load_json(tmp_path / "manual_gt_timeline_examples_v2" / "clips" / "manualgt_cowgirl_classic_grinding.timeline.json")
    controller = payload["Clips"][0]["Controllers"][0]

    assert summary["clips_exported"] == 9
    assert validation["status"] == "ok"
    assert controller["RotX"]
    assert controller["RotY"]
    assert controller["RotZ"]
    assert controller["RotW"]
    assert controller["TargetsRotation"] == 1
    assert len(controller["X"]) <= 10
    assert (tmp_path / "manual_gt_timeline_examples_v1" / "DEPRECATED_POSITION_ONLY.md").exists()


def test_v2_validation_fails_if_rotation_tracks_missing(tmp_path):
    gt = _ground_truth(tmp_path)
    export_manual_gt_timeline_examples_v2(gt, tmp_path / "pkg", duration=4.0, keyframe_rate=2.0, copy_to_vam=False)
    timeline = tmp_path / "pkg" / "clips" / "manualgt_cowgirl_classic_grinding.timeline.json"
    payload = load_json(timeline)
    payload["Clips"][0]["Controllers"][0].pop("RotX", None)
    dump_json(timeline, payload)
    validation = validate_manual_gt_timeline_examples_v2(tmp_path / "pkg", tmp_path / "pkg" / "reports" / "bad.md")

    assert validation["status"] == "failed"
    assert any("missing rotation axis RotX" in msg for msg in validation["error_messages"])


def test_v2_validation_fails_if_static_rotation_changes_or_bj_pelvis_rotates(tmp_path):
    gt = _ground_truth(tmp_path)
    export_manual_gt_timeline_examples_v2(gt, tmp_path / "pkg", duration=4.0, keyframe_rate=2.0, copy_to_vam=False)
    data_path = tmp_path / "pkg" / "preview_data" / "manual_gt_timeline_clips_v2.json"
    data = load_json(data_path)
    for clip in data["clips"]:
        if clip["clip_id"] == "manualgt_cowgirl_classic_grinding":
            for track in clip["controller_tracks"]:
                if track["controller_name"] == "lFootControl":
                    track["rotations"] = [[p[0] + (0.02 if i % 2 else 0.0), p[1], p[2], p[3]] for i, p in enumerate(track["rotations"])]
        if clip["clip_id"] == "manualgt_bj_kneeling_head_bob":
            for track in clip["controller_tracks"]:
                if track["controller_name"] == "pelvisControl":
                    track["rotations"] = [[p[0], p[1] + (0.02 if i % 2 else 0.0), p[2], p[3]] for i, p in enumerate(track["rotations"])]
    dump_json(data_path, data)
    validation = validate_manual_gt_timeline_examples_v2(tmp_path / "pkg", tmp_path / "pkg" / "reports" / "bad_rot.md")

    assert validation["status"] == "failed"
    assert any("Cowgirl foot rotation" in msg for msg in validation["error_messages"])
    assert any("BJ/HJ pelvis rotation" in msg for msg in validation["error_messages"])


def test_v3_planner_uses_hipcontrol_for_cowgirl_and_static_hips_for_bj_hj(tmp_path):
    gt = _ground_truth(tmp_path)
    summary = build_manual_gt_timeline_plans_v1(
        gt,
        tmp_path / "plans_v3.json",
        duration=4.0,
        fps=1.0,
        keyframe_rate=1.0,
        mapping_version="v3",
        require_hip_control=True,
    )
    plans = load_json(tmp_path / "plans_v3.json")["plans"]
    by_id = {plan["clip_id"]: plan for plan in plans}

    assert summary["plans"] == 9
    cow = by_id["manualgt_cowgirl_classic_grinding"]
    assert cow["driver_controllers"] == ["hipControl"]
    assert "pelvisControl" in cow["follower_controllers"]
    assert "lFootControl" in cow["static_anchor_controllers"]
    assert "rFootControl" in cow["static_anchor_controllers"]
    assert by_id["manualgt_bj_kneeling_head_bob"]["driver_controllers"] == ["headControl"]
    assert "hipControl" in by_id["manualgt_bj_kneeling_head_bob"]["static_anchor_controllers"]
    assert "hipControl" in by_id["manualgt_hj_kneeling_hand_motion"]["static_anchor_controllers"]


def test_v3_export_blocks_missing_hip_when_required(tmp_path):
    gt = _ground_truth(tmp_path)
    rows = load_jsonl(gt)
    raw = load_json(rows[0]["raw_capture_path"])
    raw["atoms"]["rider"]["controllers"].pop("hipControl", None)
    dump_json(rows[0]["raw_capture_path"], raw)
    rows[0]["atoms"]["rider"]["controllers"].pop("hipControl", None)
    gt.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = export_manual_gt_timeline_examples_v3(gt, tmp_path / "pkg", duration=4.0, keyframe_rate=1.0, copy_to_vam=False, require_hip_control=True)

    assert summary["clips_exported"] == 7
    skipped = summary["skipped"]
    assert any(row["reason"] == "missing_required_hipControl" for row in skipped)


def test_v3_export_writes_hip_rotation_tracks_and_sparse_one_fps_keys(tmp_path):
    gt = _ground_truth(tmp_path)
    export_manual_gt_timeline_examples_v1(gt, tmp_path / "manual_gt_timeline_examples_v1", duration=2.0, fps=12, copy_to_vam=False)
    export_manual_gt_timeline_examples_v2(gt, tmp_path / "manual_gt_timeline_examples_v2", duration=4.0, keyframe_rate=2.0, copy_to_vam=False)
    summary = export_manual_gt_timeline_examples_v3(gt, tmp_path / "manual_gt_timeline_examples_v3", duration=4.0, keyframe_rate=1.0, copy_to_vam=False)
    validation = validate_manual_gt_timeline_examples_v3(tmp_path / "manual_gt_timeline_examples_v3", tmp_path / "manual_gt_timeline_examples_v3" / "reports" / "validation_v3.md")
    payload = load_json(tmp_path / "manual_gt_timeline_examples_v3" / "clips" / "manualgt_cowgirl_classic_grinding.timeline.json")
    controllers = {row["Controller"]: row for row in payload["Clips"][0]["Controllers"]}
    preview = load_json(tmp_path / "manual_gt_timeline_examples_v3" / "preview_data" / "manual_gt_timeline_clips_v3.json")
    cow = next(clip for clip in preview["clips"] if clip["clip_id"] == "manualgt_cowgirl_classic_grinding")
    tracks = {track["controller_name"]: track for track in cow["controller_tracks"]}

    assert summary["clips_exported"] == 9
    assert validation["status"] == "ok"
    assert "hipControl" in controllers
    assert controllers["hipControl"]["RotX"]
    assert controllers["hipControl"]["TargetsRotation"] == 1
    assert len(controllers["hipControl"]["X"]) == 5
    assert _range(tracks["hipControl"]) > _range(tracks["pelvisControl"])
    assert _range(tracks["lFootControl"]) == 0
    assert (tmp_path / "manual_gt_timeline_examples_v1" / "DEPRECATED.md").exists()
    assert (tmp_path / "manual_gt_timeline_examples_v2" / "DEPRECATED.md").exists()


def test_v3_validation_fails_if_hip_missing_or_dense_export(tmp_path):
    gt = _ground_truth(tmp_path)
    export_manual_gt_timeline_examples_v3(gt, tmp_path / "pkg", duration=4.0, keyframe_rate=1.0, copy_to_vam=False)
    data_path = tmp_path / "pkg" / "preview_data" / "manual_gt_timeline_clips_v3.json"
    data = load_json(data_path)
    data["clips"][0]["keyframe_rate"] = 60.0
    data["clips"][0]["controller_tracks"] = [track for track in data["clips"][0]["controller_tracks"] if track["controller_name"] != "hipControl"]
    dump_json(data_path, data)
    timeline = tmp_path / "pkg" / "clips" / "manualgt_cowgirl_classic_grinding.timeline.json"
    payload = load_json(timeline)
    payload["VAMTimelineAIManualGTMetadata"]["keyframe_rate"] = 60.0
    payload["Clips"][0]["Controllers"] = [row for row in payload["Clips"][0]["Controllers"] if row["Controller"] != "hipControl"]
    dump_json(timeline, payload)
    validation = validate_manual_gt_timeline_examples_v3(tmp_path / "pkg", tmp_path / "pkg" / "reports" / "bad_v3.md")

    assert validation["status"] == "failed"
    assert any("hipControl missing" in msg for msg in validation["error_messages"])
    assert any("dense 60fps" in msg for msg in validation["error_messages"])


def test_v4_applies_amplitude_profiles_and_preserves_static_anchors(tmp_path):
    gt = _ground_truth(tmp_path)
    profile = tmp_path / "profiles.yaml"
    profile.write_text(
        "profiles:\n"
        "  cowgirl_grinding:\n"
        "    hip_lateral_scale: 1.6\n"
        "    hip_forward_back_scale: 1.4\n"
        "    hip_vertical_scale: 0.45\n"
        "    pelvis_follow_scale: 0.20\n"
        "    thigh_follow_scale: 0.35\n"
        "  bj_head_bob:\n"
        "    head_forward_back_scale: 1.35\n"
        "    head_vertical_scale: 0.35\n"
        "    chest_follow_scale: 0.65\n"
        "  hj_hand_motion:\n"
        "    active_hand_forward_back_scale: 1.6\n"
        "    active_hand_vertical_scale: 0.15\n"
        "  cowgirl_bounce: {hip_vertical_scale: 1.7, hip_forward_back_scale: 0.35, hip_lateral_scale: 0.15}\n"
        "  cowgirl_lean_back_grind: {hip_lateral_scale: 1.35, hip_forward_back_scale: 1.20, hip_vertical_scale: 0.30, pelvis_follow_scale: 0.18}\n"
        "  sitting_cowgirl_small_grind: {hip_lateral_scale: 1.15, hip_forward_back_scale: 1.10, hip_vertical_scale: 0.25, pelvis_follow_scale: 0.15}\n"
        "  doggy_receiver_response: {hip_response_scale: 0.65, pelvis_response_scale: 0.45, chest_response_scale: 0.25}\n"
        "  missionary_counter_motion: {pelvis_counter_scale: 0.65, leg_reactive_scale: 0.35}\n",
        encoding="utf-8",
    )
    export_manual_gt_timeline_examples_v3(gt, tmp_path / "manual_gt_timeline_examples_v3", duration=4.0, keyframe_rate=1.0, copy_to_vam=False)
    summary = export_manual_gt_timeline_examples_v4(
        gt,
        tmp_path / "manual_gt_timeline_examples_v4",
        duration=4.0,
        keyframe_rate=1.0,
        copy_to_vam=False,
        amplitude_profile=profile,
    )
    validation = validate_manual_gt_timeline_examples_v4(tmp_path / "manual_gt_timeline_examples_v4", tmp_path / "manual_gt_timeline_examples_v4" / "reports" / "validation_v4.md")
    preview = load_json(tmp_path / "manual_gt_timeline_examples_v4" / "preview_data" / "manual_gt_timeline_clips_v4.json")
    cow = next(clip for clip in preview["clips"] if clip["clip_id"] == "manualgt_cowgirl_classic_grinding")
    bj = next(clip for clip in preview["clips"] if clip["clip_id"] == "manualgt_bj_kneeling_head_bob")
    cow_tracks = {track["controller_name"]: track for track in cow["controller_tracks"]}
    bj_tracks = {track["controller_name"]: track for track in bj["controller_tracks"]}

    assert summary["clips_exported"] == 9
    assert validation["status"] == "ok"
    assert cow["amplitude_profile_key"] == "cowgirl_grinding"
    assert cow["amplitude_profile"]["hip_lateral_scale"] == 1.6
    assert _range(cow_tracks["hipControl"]) > 0.10
    assert _range(cow_tracks["pelvisControl"]) < _range(cow_tracks["hipControl"])
    assert _range(cow_tracks["lFootControl"]) == 0
    assert _range(cow_tracks["rFootControl"]) == 0
    assert _range(bj_tracks["hipControl"]) == 0
    assert _range(bj_tracks["pelvisControl"]) == 0
    assert (tmp_path / "manual_gt_timeline_examples_v4" / "reports" / "motion_amplitude_profile_report.md").exists()
    assert (tmp_path / "manual_gt_timeline_examples_v3" / "SUPERSEDED_BY_V4_LOW_AMPLITUDE.md").exists()


def test_v4_validation_fails_without_amplitude_profile_report(tmp_path):
    gt = _ground_truth(tmp_path)
    profile = tmp_path / "profiles.yaml"
    profile.write_text("profiles:\n  cowgirl_grinding: {hip_lateral_scale: 1.6}\n", encoding="utf-8")
    export_manual_gt_timeline_examples_v4(gt, tmp_path / "pkg", duration=4.0, keyframe_rate=1.0, copy_to_vam=False, amplitude_profile=profile)
    (tmp_path / "pkg" / "reports" / "motion_amplitude_profile_report.md").unlink()
    validation = validate_manual_gt_timeline_examples_v4(tmp_path / "pkg", tmp_path / "pkg" / "reports" / "bad_v4.md")

    assert validation["status"] == "failed"
    assert any("missing amplitude profile report" in msg for msg in validation["error_messages"])

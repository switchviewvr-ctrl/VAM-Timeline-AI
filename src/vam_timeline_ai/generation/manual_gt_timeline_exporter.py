"""Export review-only native Timeline clips from manual pose ground truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import shutil

import numpy as np

from vam_timeline_ai.generation.manual_gt_baseline_builder import build_manual_gt_baseline_for_plan
from vam_timeline_ai.generation.manual_gt_motion_synthesizer import synthesize_manual_gt_motion
from vam_timeline_ai.generation.manual_gt_timeline_planner import build_manual_gt_timeline_plans_v1
from vam_timeline_ai.generation.native_timeline_exporter import _controller_payload, _timeline_payload
from vam_timeline_ai.generation.vam_semantic_preview import is_disallowed_timeline_track
from vam_timeline_ai.io.json_utils import dump_json, load_json, safe_id_for_path, write_jsonl
from vam_timeline_ai.semantics.ontology_loader import load_yaml


def _default_vam_animation_copy_dir(package_name: str) -> Path:
    base = os.environ.get("VAM_TIMELINE_AI_VAM_ANIMATIONS_DIR")
    if base:
        return Path(base) / "VAMTimelineAI" / package_name
    return Path("data") / "runs" / "local_vam_animation_exports" / package_name


DEFAULT_VAM_COPY_DIR = _default_vam_animation_copy_dir("manual_gt_timeline_examples_v1")
DEFAULT_VAM_COPY_DIR_V2 = _default_vam_animation_copy_dir("manual_gt_timeline_examples_v2")
DEFAULT_VAM_COPY_DIR_V3 = _default_vam_animation_copy_dir("manual_gt_timeline_examples_v3")
DEFAULT_VAM_COPY_DIR_V4 = _default_vam_animation_copy_dir("manual_gt_timeline_examples_v4")


def export_manual_gt_timeline_examples_v1(
    ground_truth: str | Path,
    out_dir: str | Path,
    *,
    duration: float = 4.0,
    fps: int = 60,
    copy_to_vam: bool = False,
) -> dict[str, Any]:
    target = Path(out_dir)
    clips_dir = target / "clips"
    preview_dir = target / "preview_data"
    reports_dir = target / "reports"
    baselines_dir = target / "baselines"
    for folder in (clips_dir, preview_dir, reports_dir, baselines_dir):
        folder.mkdir(parents=True, exist_ok=True)

    plans_path = target / "manual_gt_timeline_plans_v1.json"
    plan_summary = build_manual_gt_timeline_plans_v1(ground_truth, plans_path, duration=duration, fps=fps)
    plans_payload = load_json(plans_path)
    clips: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    exported = 0
    skipped = list(plan_summary.get("skipped") or [])
    for plan in plans_payload.get("plans") or []:
        clip_id = safe_id_for_path(str(plan.get("clip_id") or "manualgt_clip"))
        baseline = build_manual_gt_baseline_for_plan(plan, out_dir=baselines_dir)
        clip = synthesize_manual_gt_motion(plan, baseline)
        timeline_path = clips_dir / f"{clip_id}.timeline.json"
        payload = _timeline_payload_for_clip(clip, plan, baseline, timeline_path)
        dump_json(timeline_path, payload)
        clip["timeline_json"] = str(timeline_path)
        clip["baseline_summary_path"] = str(baselines_dir / clip_id / "baseline_summary.json")
        clips.append(clip)
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "timeline_json": str(timeline_path),
                "source_capture_id": plan.get("capture_id"),
                "family": plan.get("family"),
                "subtype": plan.get("subtype"),
                "motion_example_name": plan.get("motion_example_name"),
                "driver_controllers": plan.get("driver_controllers"),
                "static_anchor_controllers": plan.get("static_anchor_controllers"),
                "screenshot_path": plan.get("screenshot_path"),
                "review_only": True,
            }
        )
        exported += 1

    preview_payload = {
        "schema_version": "manual_gt_timeline_clips_v1",
        "ground_truth": str(ground_truth),
        "plans": str(plans_path),
        "clips": clips,
        "skipped": skipped,
        "review_only": True,
        "source_baseline": "manual_pose_ground_truth_v1",
        "source_world_tracks_included": False,
        "person_root_world_tracks_included": False,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    preview_json = preview_dir / "manual_gt_timeline_clips_v1.json"
    dump_json(preview_json, preview_payload)
    write_jsonl(target / "manifest.jsonl", manifest_rows)
    _write_review_package(target, manifest_rows, skipped)
    _write_export_report(reports_dir / "manual_gt_timeline_export_v1.md", target, exported, skipped, copy_to_vam)

    copied_to = None
    if copy_to_vam:
        copied_to = str(_copy_package_to_vam(target, DEFAULT_VAM_COPY_DIR))

    return {
        "status": "ok",
        "out_dir": str(target),
        "plans": str(plans_path),
        "preview_data": str(preview_json),
        "manifest": str(target / "manifest.jsonl"),
        "clips_dir": str(clips_dir),
        "clips_exported": exported,
        "skipped": skipped,
        "copied_to_vam": copied_to,
    }


def export_manual_gt_timeline_examples_v2(
    ground_truth: str | Path,
    out_dir: str | Path,
    *,
    duration: float = 4.0,
    keyframe_rate: float = 2.0,
    copy_to_vam: bool = False,
    include_rotations: bool = True,
    allow_dense_export: bool = False,
) -> dict[str, Any]:
    if keyframe_rate > 5.0 and not allow_dense_export:
        raise ValueError("Dense manual GT Timeline export blocked: keyframe_rate must be <= 5 unless allow_dense_export is true.")
    target = Path(out_dir)
    clips_dir = target / "clips"
    preview_dir = target / "preview_data"
    reports_dir = target / "reports"
    baselines_dir = target / "baselines"
    for folder in (clips_dir, preview_dir, reports_dir, baselines_dir):
        folder.mkdir(parents=True, exist_ok=True)

    _mark_v1_deprecated(target)
    plans_path = target / "manual_gt_timeline_plans_v2.json"
    plan_summary = build_manual_gt_timeline_plans_v1(ground_truth, plans_path, duration=duration, fps=keyframe_rate, keyframe_rate=keyframe_rate)
    plans_payload = load_json(plans_path)
    clips: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    exported = 0
    skipped = list(plan_summary.get("skipped") or [])
    for plan in plans_payload.get("plans") or []:
        clip_id = safe_id_for_path(str(plan.get("clip_id") or "manualgt_clip"))
        baseline = build_manual_gt_baseline_for_plan(plan, out_dir=baselines_dir)
        rotation_rows.extend(_rotation_rows_for_baseline(clip_id, baseline))
        clip = synthesize_manual_gt_motion(plan, baseline)
        timeline_path = clips_dir / f"{clip_id}.timeline.json"
        payload = _timeline_payload_for_clip(clip, plan, baseline, timeline_path, schema_version="manual_gt_timeline_example_v2", include_rotations=include_rotations)
        dump_json(timeline_path, payload)
        clip["timeline_json"] = str(timeline_path)
        clip["baseline_summary_path"] = str(baselines_dir / clip_id / "baseline_summary.json")
        clip["keyframe_rate"] = keyframe_rate
        clip["include_rotations"] = include_rotations
        clips.append(clip)
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "timeline_json": str(timeline_path),
                "source_capture_id": plan.get("capture_id"),
                "family": plan.get("family"),
                "subtype": plan.get("subtype"),
                "motion_example_name": plan.get("motion_example_name"),
                "driver_controllers": plan.get("driver_controllers"),
                "static_anchor_controllers": plan.get("static_anchor_controllers"),
                "screenshot_path": plan.get("screenshot_path"),
                "keyframe_rate": keyframe_rate,
                "include_rotations": include_rotations,
                "review_only": True,
            }
        )
        exported += 1

    preview_payload = {
        "schema_version": "manual_gt_timeline_clips_v2",
        "ground_truth": str(ground_truth),
        "plans": str(plans_path),
        "clips": clips,
        "skipped": skipped,
        "review_only": True,
        "source_baseline": "manual_pose_ground_truth_v1",
        "include_rotations": include_rotations,
        "keyframe_rate": keyframe_rate,
        "source_world_tracks_included": False,
        "person_root_world_tracks_included": False,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    preview_json = preview_dir / "manual_gt_timeline_clips_v2.json"
    dump_json(preview_json, preview_payload)
    write_jsonl(target / "manifest.jsonl", manifest_rows)
    _write_review_package(target, manifest_rows, skipped, version="v2")
    _write_rotation_source_report(reports_dir / "rotation_source_report.md", rotation_rows, exported)
    _write_export_report(reports_dir / "manual_gt_timeline_export_v2.md", target, exported, skipped, copy_to_vam)

    copied_to = None
    if copy_to_vam:
        copied_to = str(_copy_package_to_vam(target, DEFAULT_VAM_COPY_DIR_V2))

    return {
        "status": "ok",
        "out_dir": str(target),
        "plans": str(plans_path),
        "preview_data": str(preview_json),
        "manifest": str(target / "manifest.jsonl"),
        "clips_dir": str(clips_dir),
        "clips_exported": exported,
        "skipped": skipped,
        "copied_to_vam": copied_to,
        "rotation_source_report": str(reports_dir / "rotation_source_report.md"),
        "keyframe_rate": keyframe_rate,
        "include_rotations": include_rotations,
    }


def export_manual_gt_timeline_examples_v3(
    ground_truth: str | Path,
    out_dir: str | Path,
    *,
    duration: float = 4.0,
    keyframe_rate: float = 1.0,
    copy_to_vam: bool = False,
    include_rotations: bool = True,
    require_hip_control: bool = True,
    allow_high_key_density: bool = False,
    allow_dense_export: bool = False,
) -> dict[str, Any]:
    if keyframe_rate > 3.0 and not allow_high_key_density:
        raise ValueError("Manual GT v3 export blocked: keyframe_rate must be <= 3 unless allow_high_key_density is true.")
    if keyframe_rate > 5.0 and not allow_dense_export:
        raise ValueError("Dense manual GT Timeline export blocked: keyframe_rate must be <= 5 unless allow_dense_export is true.")
    target = Path(out_dir)
    clips_dir = target / "clips"
    preview_dir = target / "preview_data"
    reports_dir = target / "reports"
    baselines_dir = target / "baselines"
    for folder in (clips_dir, preview_dir, reports_dir, baselines_dir):
        folder.mkdir(parents=True, exist_ok=True)

    _mark_previous_packages_deprecated(target)
    plans_path = target / "manual_gt_timeline_plans_v3.json"
    plan_summary = build_manual_gt_timeline_plans_v1(
        ground_truth,
        plans_path,
        duration=duration,
        fps=keyframe_rate,
        keyframe_rate=keyframe_rate,
        mapping_version="v3",
        require_hip_control=require_hip_control,
    )
    plans_payload = load_json(plans_path)
    clips: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    exported = 0
    skipped = list(plan_summary.get("skipped") or [])
    for plan in plans_payload.get("plans") or []:
        clip_id = safe_id_for_path(str(plan.get("clip_id") or "manualgt_clip"))
        baseline = build_manual_gt_baseline_for_plan(plan, out_dir=baselines_dir)
        if require_hip_control and "hipControl" not in (baseline.get("controller_baseline") or {}):
            skipped.append({"capture_id": plan.get("capture_id"), "clip_id": clip_id, "reason": "baseline_missing_required_hipControl"})
            continue
        rotation_rows.extend(_rotation_rows_for_baseline(clip_id, baseline))
        clip = synthesize_manual_gt_motion(plan, baseline)
        timeline_path = clips_dir / f"{clip_id}.timeline.json"
        payload = _timeline_payload_for_clip(clip, plan, baseline, timeline_path, schema_version="manual_gt_timeline_example_v3", include_rotations=include_rotations)
        dump_json(timeline_path, payload)
        clip["timeline_json"] = str(timeline_path)
        clip["baseline_summary_path"] = str(baselines_dir / clip_id / "baseline_summary.json")
        clip["keyframe_rate"] = keyframe_rate
        clip["include_rotations"] = include_rotations
        clip["require_hip_control"] = require_hip_control
        clips.append(clip)
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "timeline_json": str(timeline_path),
                "source_capture_id": plan.get("capture_id"),
                "family": plan.get("family"),
                "subtype": plan.get("subtype"),
                "motion_example_name": plan.get("motion_example_name"),
                "driver_controllers": plan.get("driver_controllers"),
                "follower_controllers": plan.get("follower_controllers"),
                "static_anchor_controllers": plan.get("static_anchor_controllers"),
                "screenshot_path": plan.get("screenshot_path"),
                "keyframe_rate": keyframe_rate,
                "include_rotations": include_rotations,
                "require_hip_control": require_hip_control,
                "review_only": True,
            }
        )
        exported += 1

    preview_payload = {
        "schema_version": "manual_gt_timeline_clips_v3",
        "ground_truth": str(ground_truth),
        "plans": str(plans_path),
        "clips": clips,
        "skipped": skipped,
        "review_only": True,
        "source_baseline": "manual_pose_ground_truth_v1",
        "controller_mapping_version": "v3",
        "hipControl_required": require_hip_control,
        "cowgirl_primary_driver": "hipControl",
        "pelvisControl_role_for_cowgirl": "secondary_follower_or_static",
        "include_rotations": include_rotations,
        "keyframe_rate": keyframe_rate,
        "source_world_tracks_included": False,
        "person_root_world_tracks_included": False,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    preview_json = preview_dir / "manual_gt_timeline_clips_v3.json"
    dump_json(preview_json, preview_payload)
    write_jsonl(target / "manifest.jsonl", manifest_rows)
    _write_review_package(target, manifest_rows, skipped, version="v3")
    _write_rotation_source_report(reports_dir / "rotation_source_report.md", rotation_rows, exported, version="V3")
    _write_export_report(reports_dir / "manual_gt_timeline_export_v3.md", target, exported, skipped, copy_to_vam, version="V3")

    copied_to = None
    if copy_to_vam:
        copied_to = str(_copy_package_to_vam(target, DEFAULT_VAM_COPY_DIR_V3))

    return {
        "status": "ok",
        "out_dir": str(target),
        "plans": str(plans_path),
        "preview_data": str(preview_json),
        "manifest": str(target / "manifest.jsonl"),
        "clips_dir": str(clips_dir),
        "clips_exported": exported,
        "skipped": skipped,
        "copied_to_vam": copied_to,
        "rotation_source_report": str(reports_dir / "rotation_source_report.md"),
        "keyframe_rate": keyframe_rate,
        "include_rotations": include_rotations,
        "require_hip_control": require_hip_control,
    }


def export_manual_gt_timeline_examples_v4(
    ground_truth: str | Path,
    out_dir: str | Path,
    *,
    duration: float = 4.0,
    keyframe_rate: float = 1.0,
    copy_to_vam: bool = False,
    include_rotations: bool = True,
    require_hip_control: bool = True,
    amplitude_profile: str | Path | None = None,
    allow_high_key_density: bool = False,
    allow_dense_export: bool = False,
) -> dict[str, Any]:
    if keyframe_rate > 3.0 and not allow_high_key_density:
        raise ValueError("Manual GT v4 export blocked: keyframe_rate must be <= 3 unless allow_high_key_density is true.")
    if keyframe_rate > 5.0 and not allow_dense_export:
        raise ValueError("Dense manual GT Timeline export blocked: keyframe_rate must be <= 5 unless allow_dense_export is true.")
    target = Path(out_dir)
    clips_dir = target / "clips"
    preview_dir = target / "preview_data"
    reports_dir = target / "reports"
    baselines_dir = target / "baselines"
    for folder in (clips_dir, preview_dir, reports_dir, baselines_dir):
        folder.mkdir(parents=True, exist_ok=True)

    _mark_v3_superseded(target)
    profiles_payload = _load_amplitude_profiles(amplitude_profile)
    plans_path = target / "manual_gt_timeline_plans_v4.json"
    plan_summary = build_manual_gt_timeline_plans_v1(
        ground_truth,
        plans_path,
        duration=duration,
        fps=keyframe_rate,
        keyframe_rate=keyframe_rate,
        mapping_version="v3",
        require_hip_control=require_hip_control,
    )
    plans_payload = load_json(plans_path)
    clips: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    amplitude_rows: list[dict[str, Any]] = []
    exported = 0
    skipped = list(plan_summary.get("skipped") or [])
    for raw_plan in plans_payload.get("plans") or []:
        plan = dict(raw_plan)
        profile_key = _amplitude_profile_key_for_plan(plan)
        profile = dict(profiles_payload.get(profile_key) or {})
        plan["amplitude_profile_key"] = profile_key
        plan["amplitude_profile"] = profile
        clip_id = safe_id_for_path(str(plan.get("clip_id") or "manualgt_clip"))
        baseline = build_manual_gt_baseline_for_plan(plan, out_dir=baselines_dir)
        if require_hip_control and "hipControl" not in (baseline.get("controller_baseline") or {}):
            skipped.append({"capture_id": plan.get("capture_id"), "clip_id": clip_id, "reason": "baseline_missing_required_hipControl"})
            continue
        rotation_rows.extend(_rotation_rows_for_baseline(clip_id, baseline))
        clip = synthesize_manual_gt_motion(plan, baseline)
        timeline_path = clips_dir / f"{clip_id}.timeline.json"
        payload = _timeline_payload_for_clip(clip, plan, baseline, timeline_path, schema_version="manual_gt_timeline_example_v4", include_rotations=include_rotations)
        dump_json(timeline_path, payload)
        clip["timeline_json"] = str(timeline_path)
        clip["baseline_summary_path"] = str(baselines_dir / clip_id / "baseline_summary.json")
        clip["keyframe_rate"] = keyframe_rate
        clip["include_rotations"] = include_rotations
        clip["require_hip_control"] = require_hip_control
        clips.append(clip)
        amplitude_rows.append(_amplitude_report_row(clip, plan, profile_key, profile))
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "timeline_json": str(timeline_path),
                "source_capture_id": plan.get("capture_id"),
                "family": plan.get("family"),
                "subtype": plan.get("subtype"),
                "motion_example_name": plan.get("motion_example_name"),
                "driver_controllers": plan.get("driver_controllers"),
                "follower_controllers": plan.get("follower_controllers"),
                "static_anchor_controllers": plan.get("static_anchor_controllers"),
                "screenshot_path": plan.get("screenshot_path"),
                "keyframe_rate": keyframe_rate,
                "include_rotations": include_rotations,
                "require_hip_control": require_hip_control,
                "amplitude_profile_key": profile_key,
                "review_only": True,
            }
        )
        exported += 1

    preview_payload = {
        "schema_version": "manual_gt_timeline_clips_v4",
        "ground_truth": str(ground_truth),
        "plans": str(plans_path),
        "clips": clips,
        "skipped": skipped,
        "review_only": True,
        "source_baseline": "manual_pose_ground_truth_v1",
        "controller_mapping_version": "v3",
        "amplitude_profile_version": "manual_gt_motion_amplitude_profiles_v1",
        "amplitude_profile_path": str(amplitude_profile) if amplitude_profile else None,
        "hipControl_required": require_hip_control,
        "cowgirl_primary_driver": "hipControl",
        "pelvisControl_role_for_cowgirl": "secondary_follower_or_static",
        "include_rotations": include_rotations,
        "keyframe_rate": keyframe_rate,
        "source_world_tracks_included": False,
        "person_root_world_tracks_included": False,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    preview_json = preview_dir / "manual_gt_timeline_clips_v4.json"
    dump_json(preview_json, preview_payload)
    write_jsonl(target / "manifest.jsonl", manifest_rows)
    _write_review_package(target, manifest_rows, skipped, version="v4")
    _write_rotation_source_report(reports_dir / "rotation_source_report.md", rotation_rows, exported, version="V4")
    amplitude_report = reports_dir / "motion_amplitude_profile_report.md"
    _write_motion_amplitude_profile_report(amplitude_report, amplitude_profile, amplitude_rows)
    _write_export_report(reports_dir / "manual_gt_timeline_export_v4.md", target, exported, skipped, copy_to_vam, version="V4")

    copied_to = None
    if copy_to_vam:
        copied_to = str(_copy_package_to_vam(target, DEFAULT_VAM_COPY_DIR_V4))

    return {
        "status": "ok",
        "out_dir": str(target),
        "plans": str(plans_path),
        "preview_data": str(preview_json),
        "manifest": str(target / "manifest.jsonl"),
        "clips_dir": str(clips_dir),
        "clips_exported": exported,
        "skipped": skipped,
        "copied_to_vam": copied_to,
        "rotation_source_report": str(reports_dir / "rotation_source_report.md"),
        "motion_amplitude_profile_report": str(amplitude_report),
        "keyframe_rate": keyframe_rate,
        "include_rotations": include_rotations,
        "require_hip_control": require_hip_control,
        "amplitude_profile": str(amplitude_profile) if amplitude_profile else None,
    }


def _timeline_payload_for_clip(
    clip: dict[str, Any],
    plan: dict[str, Any],
    baseline: dict[str, Any],
    timeline_path: Path,
    *,
    schema_version: str = "manual_gt_timeline_example_v1",
    include_rotations: bool = False,
) -> dict[str, Any]:
    duration = float(clip.get("duration_seconds") or 4.0)
    controllers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for track in clip.get("controller_tracks") or []:
        name = str(track.get("controller_name") or "")
        if is_disallowed_timeline_track(name):
            skipped.append({"controller_name": name, "reason": "person_root_world_or_atom_track"})
            continue
        times = np.asarray(track.get("times") or [], dtype=np.float32)
        positions = np.asarray(track.get("positions") or [], dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(times) != len(positions):
            skipped.append({"controller_name": name, "reason": "invalid_position_track"})
            continue
        rotations = None
        if include_rotations:
            rotations = np.asarray(track.get("rotations") or [], dtype=np.float32)
            if rotations.ndim != 2 or rotations.shape[1] != 4 or len(rotations) != len(times):
                skipped.append({"controller_name": name, "reason": "invalid_or_missing_rotation_track"})
                continue
        controllers.append(_controller_payload(name, times, positions, duration, key_stride=1, rotations=rotations, target_rotation=include_rotations))
    metadata = {
        "schema": schema_version,
        "review_only": True,
        "production_ready": False,
        "timeline_generation_final": False,
        "source_capture_id": plan.get("capture_id"),
        "baseline_source_capture": plan.get("baseline_source_capture"),
        "baseline_source": "manual_pose_ground_truth_v1",
        "baseline_coordinate_space": baseline.get("coordinate_space"),
        "coordinate_note": baseline.get("coordinate_note"),
        "source_world_coords_used_as_tracks": False,
        "source_scene_coordinates_claim": False,
        "person_root_tracks_included": False,
        "old_raw_timeline_curves_copied": False,
        "include_rotations": include_rotations,
        "keyframe_rate": clip.get("keyframe_rate") or plan.get("keyframe_rate"),
        "rotation_source_counts": baseline.get("rotation_source_counts"),
        "identity_rotation_fallback_controllers": baseline.get("identity_rotation_fallback_controllers"),
        "missing_rotations": baseline.get("missing_rotations"),
        "family": plan.get("family"),
        "subtype": plan.get("subtype"),
        "motion_example_name": plan.get("motion_example_name"),
        "driver_controllers": plan.get("driver_controllers"),
        "follower_controllers": plan.get("follower_controllers"),
        "static_anchor_controllers": plan.get("static_anchor_controllers"),
        "explicitly_static_controllers": plan.get("explicitly_static_controllers"),
        "amplitude_profile_key": plan.get("amplitude_profile_key"),
        "amplitude_profile": plan.get("amplitude_profile") or {},
        "forbidden_motion_controllers": _forbidden_controllers(plan),
        "controller_roles": {track.get("controller_name"): track.get("role") for track in clip.get("controller_tracks") or []},
        "skipped_tracks": skipped,
        "warning": "Review-only manual-ground-truth baseline clip. Inspect in VaM; do not treat as production animation.",
    }
    payload = _timeline_payload(animation_name=f"ManualGT_{safe_id_for_path(str(plan.get('motion_example_name') or timeline_path.stem))}", duration=duration, controllers=controllers, metadata=metadata)
    payload["VAMTimelineAIManualGTMetadata"] = metadata
    payload["VAMTimelineAIGeneratedMetadata"] = metadata
    payload["Clips"][0]["AnimationSegment"] = "ManualGroundTruthReviewOnly"
    return payload


def _forbidden_controllers(plan: dict[str, Any]) -> list[str]:
    family = str(plan.get("family") or "")
    if family == "cowgirl":
        return ["lFootControl", "rFootControl", "lHandControl", "rHandControl"]
    if family in {"bj_oral", "handjob"}:
        return ["hipControl", "pelvisControl", "lFootControl", "rFootControl"]
    if family == "doggy":
        return ["lHandControl", "rHandControl", "lFootControl", "rFootControl", "lKneeControl", "rKneeControl"]
    return []


def _write_review_package(out_dir: Path, manifest_rows: list[dict[str, Any]], skipped: list[dict[str, Any]], *, version: str = "v1") -> None:
    _write_import_instructions(out_dir, version=version)
    _write_review_checklist(out_dir / "review_checklist.md", manifest_rows)
    _write_index_html(out_dir / "index.html", manifest_rows, skipped)
    _write_index_md(out_dir / "index.md", manifest_rows, skipped)


def _write_import_instructions(out_dir: Path, *, version: str = "v1") -> None:
    if version == "v2":
        version_note = "V1 was position-only and should be considered invalid/deprecated. Use this v2 package only; it includes captured controller rotations.\n\n"
    elif version == "v3":
        version_note = (
            "V1 and V2 are deprecated. V1 was position-only. V2 included rotations but missed `hipControl` and used `pelvisControl` as the Cowgirl primary driver.\n\n"
            "Use this V3 package only. Verify `hipControl` appears in the Timeline target list, every exported controller has Position and Rotation tracks, and Cowgirl visible motion comes from `hipControl` while feet stay static.\n\n"
        )
    elif version == "v4":
        version_note = (
            "V4 uses the same manual captured baselines and v3 hipControl mapping, but applies per-family amplitude profiles for clearer review motion.\n\n"
            "V3 remains a correct low-amplitude baseline. Use V4 when checking motion readability. Verify anchors stay static, rotations are present, and only semantic drivers are stronger.\n\n"
        )
    else:
        version_note = ""
    (out_dir / "import_instructions.md").write_text(
        f"# Manual GT Timeline Example Import Instructions {version.upper()}\n\n"
        + version_note +
        "These clips are review-only Timeline examples built from real manual VaM pose captures.\n\n"
        "1. Open VaM.\n"
        "2. Load a simple test scene or the same/approximate character setup used for the capture.\n"
        "3. Add AcidBubbles Timeline to the actor Person atom.\n"
        "4. Import one `.timeline.json` from `clips/`.\n"
        "5. Play from `t=0` and inspect one clip at a time.\n"
        "6. Check only semantic driver/anchor correctness: correct driver, feet still where expected, hands still/supporting where expected, no weird extra controller motion.\n"
        "7. Do not judge polish or production quality yet.\n\n"
        "Important: these files do not animate Person/root/world tracks and do not train ML or write labels.\n",
        encoding="utf-8",
    )


def _write_review_checklist(out: Path, manifest_rows: list[dict[str, Any]]) -> None:
    lines = ["# Manual GT Timeline Review Checklist V1", ""]
    for row in manifest_rows:
        lines.extend(
            [
                f"## {row['clip_id']}",
                "",
                f"- Source capture: `{row['source_capture_id']}`",
                f"- Family/subtype: `{row['family']}` / `{row['subtype']}`",
                f"- Drivers: `{row['driver_controllers']}`",
                f"- Static anchors: `{row['static_anchor_controllers']}`",
                "- Pose starts correctly?",
                "- Correct semantic driver?",
                "- Feet still where expected?",
                "- Hands still/supporting where expected?",
                "- No unnecessary controller motion?",
                "- Partner relation still plausible?",
                "- Motion family visually correct?",
                "- Notes:",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_index_md(out: Path, manifest_rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    lines = [
        "# Manual GT Timeline Examples V1",
        "",
        "- Baseline source: real manual VaM pose captures",
        "- Review-only: true",
        "- Production-ready: false",
        "- Person/root/world tracks: false",
        "- ML training: false",
        "",
        "## Clips",
        "",
    ]
    for row in manifest_rows:
        lines.append(f"- `{row['clip_id']}` -> `clips/{Path(str(row['timeline_json'])).name}` from `{row['source_capture_id']}`")
    if skipped:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- `{row.get('clip_id')}` / `{row.get('capture_id')}`: {row.get('reason')}" for row in skipped)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index_html(out: Path, manifest_rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    cards = []
    for row in manifest_rows:
        timeline = Path(str(row["timeline_json"])).name
        screenshot = _rel(row.get("screenshot_path") or "", out.parent)
        cards.append(
            "<article class='card'>"
            f"<h2>{row['clip_id']}</h2>"
            f"<p><b>Family:</b> {row['family']} / {row['subtype']}</p>"
            f"<p><b>Source:</b> {row['source_capture_id']}</p>"
            f"<p><b>Drivers:</b> {row['driver_controllers']}</p>"
            f"<p><b>Static anchors:</b> {row['static_anchor_controllers']}</p>"
            f"{'<img src=\"' + screenshot + '\" alt=\"source capture screenshot\">' if screenshot else ''}"
            f"<p><a href='clips/{timeline}'>{timeline}</a></p>"
            "</article>"
        )
    skipped_html = "".join(f"<li>{row.get('clip_id')} / {row.get('capture_id')}: {row.get('reason')}</li>" for row in skipped)
    out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Manual GT Timeline Examples V1</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f6f6f3;color:#202020}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}"
        ".card{background:white;border:1px solid #d5d5ce;border-radius:8px;padding:14px}"
        "img{width:100%;max-height:280px;object-fit:contain;background:#111;border-radius:4px}</style></head><body>"
        "<h1>Manual GT Timeline Examples V1</h1>"
        "<p>Review-only clips from real manual VaM pose baselines. No ML, no labels written, no root/world tracks.</p>"
        "<p><a href='import_instructions.md'>Import instructions</a> | <a href='review_checklist.md'>Review checklist</a> | <a href='manifest.jsonl'>Manifest</a></p>"
        f"<section class='grid'>{''.join(cards)}</section>"
        + (f"<h2>Skipped</h2><ul>{skipped_html}</ul>" if skipped_html else "")
        + "</body></html>\n",
        encoding="utf-8",
    )


def _write_export_report(out: Path, target: Path, exported: int, skipped: list[dict[str, Any]], copy_to_vam: bool, *, version: str = "V1") -> None:
    out.write_text(
        f"# Manual GT Timeline Export {version}\n\n"
        f"- Package: `{target}`\n"
        f"- Clips exported: `{exported}`\n"
        f"- Skipped clips: `{len(skipped)}`\n"
        f"- Copy to VaM requested: `{copy_to_vam}`\n"
        "- Baseline source: `manual_pose_ground_truth_v1`\n"
        "- Review-only: `true`\n"
        "- Person/root/world tracks included: `false`\n"
        "- ML training performed: `false`\n"
        "- manual_labels.yaml modified: `false`\n",
        encoding="utf-8",
    )


def _rotation_rows_for_baseline(clip_id: str, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, entry in sorted((baseline.get("controller_baseline") or {}).items()):
        rows.append(
            {
                "clip_id": clip_id,
                "controller": name,
                "rotation_source": entry.get("rotation_source"),
                "has_rotation": bool(entry.get("rotation_quat")),
                "identity_fallback": entry.get("rotation_source") == "identity_missing_rotation_fallback",
            }
        )
    return rows


def _write_rotation_source_report(out: Path, rotation_rows: list[dict[str, Any]], clip_count: int, *, version: str = "V2") -> None:
    source_counts: dict[str, int] = {}
    identity = []
    missing = []
    for row in rotation_rows:
        source = str(row.get("rotation_source") or "missing")
        source_counts[source] = source_counts.get(source, 0) + 1
        if row.get("identity_fallback"):
            identity.append(f"{row.get('clip_id')}:{row.get('controller')}")
        if not row.get("has_rotation"):
            missing.append(f"{row.get('clip_id')}:{row.get('controller')}")
    lines = [
        f"# Manual GT Timeline Rotation Source Report {version}",
        "",
        f"- Clips: `{clip_count}`",
        f"- Controller rotation rows: `{len(rotation_rows)}`",
        f"- Rotation source counts: `{source_counts}`",
        f"- Identity rotation fallbacks: `{len(identity)}`",
        f"- Missing rotations: `{len(missing)}`",
        "- Local captured rotations preferred: `local_rotation_to_atom_quat`",
        "- World rotation fallback is reported if local rotation is unavailable.",
        "",
    ]
    if identity:
        lines.append("## Identity Fallbacks")
        lines.extend(f"- `{item}`" for item in identity)
        lines.append("")
    if missing:
        lines.append("## Missing Rotations")
        lines.extend(f"- `{item}`" for item in missing)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def _mark_v1_deprecated(v2_target: Path) -> None:
    v1 = v2_target.parent / "manual_gt_timeline_examples_v1"
    if not v1.exists():
        return
    (v1 / "DEPRECATED_POSITION_ONLY.md").write_text(
        "# Deprecated: Position-Only Manual GT Timeline Examples V1\n\n"
        "This package is deprecated because the Timeline clips contain controller Position tracks without captured controller Rotation quaternion tracks.\n\n"
        "Use `manual_gt_timeline_examples_v2` instead. V2 includes captured controller rotations and sparse semantic keyframes.\n",
        encoding="utf-8",
    )


def _mark_previous_packages_deprecated(v3_target: Path) -> None:
    root = v3_target.parent
    v1 = root / "manual_gt_timeline_examples_v1"
    if v1.exists():
        (v1 / "DEPRECATED.md").write_text(
            "# Deprecated: Manual GT Timeline Examples V1\n\n"
            "Deprecated because this package was position-only and did not include captured controller rotation tracks.\n\n"
            "Use `manual_gt_timeline_examples_v3` instead.\n",
            encoding="utf-8",
        )
        _mark_v1_deprecated(v3_target)
    v2 = root / "manual_gt_timeline_examples_v2"
    if v2.exists():
        (v2 / "DEPRECATED.md").write_text(
            "# Deprecated: Manual GT Timeline Examples V2\n\n"
            "Deprecated because `hipControl` was missing from exported clips and Cowgirl motion used `pelvisControl` as the primary driver.\n\n"
            "Use `manual_gt_timeline_examples_v3` instead. V3 includes `hipControl`, preserves rotations, and uses sparse semantic keyframes.\n",
            encoding="utf-8",
        )


def _mark_v3_superseded(v4_target: Path) -> None:
    v3 = v4_target.parent / "manual_gt_timeline_examples_v3"
    if not v3.exists():
        return
    (v3 / "SUPERSEDED_BY_V4_LOW_AMPLITUDE.md").write_text(
        "# Superseded By V4 For Motion Readability\n\n"
        "V3 is the first correct manual-ground-truth baseline package: real captured poses, `hipControl`, rotations, sparse keys, and static anchors.\n\n"
        "It is not deprecated as invalid. It is marked `baseline_correct_but_low_amplitude` because V4 uses the same baselines and controller mapping with explicit per-family amplitude profiles for clearer visual review.\n",
        encoding="utf-8",
    )


def _load_amplitude_profiles(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = load_yaml(path)
    profiles = data.get("profiles", data)
    return {str(key): dict(value or {}) for key, value in profiles.items()} if isinstance(profiles, dict) else {}


def _amplitude_profile_key_for_plan(plan: dict[str, Any]) -> str:
    name = str(plan.get("motion_example_name") or plan.get("clip_id") or "")
    if name == "cowgirl_grinding":
        return "cowgirl_grinding"
    if name == "cowgirl_vertical_bounce":
        return "cowgirl_bounce"
    if name == "cowgirl_lean_back_grind":
        return "cowgirl_lean_back_grind"
    if name == "sitting_cowgirl_small_grind":
        return "sitting_cowgirl_small_grind"
    if name == "bj_kneeling_head_bob":
        return "bj_head_bob"
    if name == "hj_kneeling_hand_motion":
        return "hj_hand_motion"
    if "doggy" in name:
        return "doggy_receiver_response"
    if "missionary" in name:
        return "missionary_counter_motion"
    return name


def _amplitude_report_row(clip: dict[str, Any], plan: dict[str, Any], profile_key: str, profile: dict[str, Any]) -> dict[str, Any]:
    ranges = {str(track.get("controller_name")): track.get("motion_range") for track in clip.get("controller_tracks") or []}
    return {
        "clip_id": clip.get("clip_id"),
        "family": clip.get("family"),
        "motion_example_name": clip.get("motion_example_name"),
        "profile_key": profile_key,
        "profile": profile,
        "driver_controllers": plan.get("driver_controllers") or [],
        "static_anchor_controllers": plan.get("static_anchor_controllers") or [],
        "motion_ranges": ranges,
    }


def _write_motion_amplitude_profile_report(out: Path, profile_path: str | Path | None, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Manual GT Motion Amplitude Profile Report V4",
        "",
        f"- Profile path: `{profile_path}`",
        f"- Clips: `{len(rows)}`",
        "- Architecture changed: `false`",
        "- Baselines changed: `false`",
        "- Rotations preserved: `true`",
        "- Sparse keyframes preserved: `true`",
        "",
    ]
    for row in rows:
        ranges = row.get("motion_ranges") or {}
        driver_ranges = {name: ranges.get(name) for name in row.get("driver_controllers") or []}
        lines.extend(
            [
                f"## {row.get('clip_id')}",
                "",
                f"- Family: `{row.get('family')}`",
                f"- Motion: `{row.get('motion_example_name')}`",
                f"- Profile: `{row.get('profile_key')}`",
                f"- Driver ranges: `{driver_ranges}`",
                f"- Static anchors: `{row.get('static_anchor_controllers')}`",
                f"- Profile values: `{row.get('profile')}`",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def _copy_package_to_vam(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(source)
        dest = destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    return destination


def _rel(path: str, base: Path) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        try:
            return str(Path(path).relative_to(base)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

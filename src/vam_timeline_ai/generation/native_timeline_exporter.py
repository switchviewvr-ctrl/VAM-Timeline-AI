"""Export generated retargeted flows as native AcidBubbles Timeline JSON.

This exporter is deliberately narrow: it maps generated/retargeted controller
positions into a single Timeline v283 clip. It does not use source scene
coordinates, Person/root tracks, or clip stitching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.generation.generated_motion import is_allowed_generated_controller
from vam_timeline_ai.generation.native_timeline_baker import bake_relative_flow_to_timeline_targets
from vam_timeline_ai.io.json_utils import dump_json, load_json, safe_id_for_path
from vam_timeline_ai.timeline.bezier import CurveTypeValues
from vam_timeline_ai.timeline.codec import TimelineKeyframe, encode_keyframe_sequence
from vam_timeline_ai.timeline.parser import POSITION_AXES, ROTATION_AXES


DEFAULT_ANIMATION_NAME = "Generated_Cowgirl_Grinding_V0"
DEFAULT_ANIMATION_NAME_V1 = "Generated_Cowgirl_Grinding_V1"


def export_generated_flow_native_timeline_v0(
    retargeted_flow: str | Path,
    out: str | Path,
    report: str | Path,
    *,
    animation_name: str = DEFAULT_ANIMATION_NAME,
    key_stride: int = 1,
) -> dict[str, Any]:
    flow = load_json(retargeted_flow)
    tracks = flow.get("controller_tracks", []) or []
    controllers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    duration = _duration_from_tracks(tracks, flow)
    fps = _fps_from_tracks(tracks, flow)
    key_stride = max(1, int(key_stride))

    for track in tracks:
        name = str(track.get("controller_name") or "")
        if not is_allowed_generated_controller(name):
            skipped.append({"controller_name": name, "reason": "disallowed_controller"})
            continue
        if _is_root_like(name):
            skipped.append({"controller_name": name, "reason": "root_or_world_controller"})
            continue
        times = _times(track, fps, duration)
        positions = _positions(track)
        if positions is None or len(positions) == 0:
            skipped.append({"controller_name": name, "reason": "missing_retargeted_positions"})
            continue
        if len(times) != len(positions):
            n = min(len(times), len(positions))
            times = times[:n]
            positions = positions[:n]
        controllers.append(_controller_payload(name, times, positions, duration, key_stride))

    payload = {
        "SerializeVersion": "283",
        "SerializeMode": "2",
        "AtomType": "Person",
        "Clips": [
            {
                "AnimationName": safe_id_for_path(animation_name),
                "AnimationLength": round(float(duration), 6),
                "BlendDuration": 0,
                "Loop": 1,
                "PreserveLastFrame": 1,
                "LoopSelfBlendDuration": 0,
                "NextAnimationRandomizeWeight": 1,
                "AutoTransitionPrevious": 0,
                "AutoTransitionNext": 0,
                "SyncTransitionTime": 1,
                "SyncTransitionTimeNL": 0,
                "EnsureQuaternionContinuity": 1,
                "AnimationLayer": "Main",
                "Speed": 1,
                "Weight": 1,
                "Uninterruptible": 0,
                "AnimationSegment": "GeneratedMotion",
                "NextAnimationName": "",
                "NextAnimationTime": round(float(duration), 6),
                "Controllers": controllers,
            }
        ],
        "VAMTimelineAIGeneratedMetadata": {
            "generated_from_relative_flow": True,
            "source_world_coords_used": False,
            "clip_stitching_used": False,
            "person_root_tracks_included": False,
            "review_player_not_required": True,
            "generation_template_candidate": False,
            "experimental_native_timeline_export": True,
            "source_retargeted_flow_id": flow.get("flow_id"),
            "source_schema": flow.get("schema"),
            "coordinate_space": flow.get("coordinate_space"),
            "baseline_pose_id": flow.get("baseline_pose_id"),
            "controller_count": len(controllers),
            "skipped_tracks": skipped,
            "warning": "Manual VaM Timeline import test is still required before claiming production compatibility.",
        },
    }
    dump_json(out, payload)
    _write_report(payload, report, out)
    return payload


def export_generated_flow_native_timeline_v1(
    retargeted_flow: str | Path,
    baseline_pose: str | Path,
    out: str | Path,
    report: str | Path,
    *,
    animation_name: str = DEFAULT_ANIMATION_NAME_V1,
    key_stride: int = 1,
    include_baseline_keyframe: bool = True,
    include_rotation_tracks: bool = True,
) -> dict[str, Any]:
    bake = bake_relative_flow_to_timeline_targets(
        retargeted_flow,
        baseline_pose,
        include_baseline_keyframe=include_baseline_keyframe,
        include_rotation_tracks=include_rotation_tracks,
    )
    controllers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    duration = float(bake.get("duration_seconds") or 0.001)
    key_stride = max(1, int(key_stride))

    for track in bake.get("controller_tracks", []) or []:
        name = str(track.get("controller_name") or "")
        if not is_allowed_generated_controller(name):
            skipped.append({"controller_name": name, "reason": "disallowed_controller"})
            continue
        if _is_root_like(name):
            skipped.append({"controller_name": name, "reason": "root_or_world_controller"})
            continue
        times = np.asarray(track.get("times") or [], dtype=np.float32)
        positions = np.asarray(track.get("positions") or [], dtype=np.float32)
        rotations = np.asarray(track.get("rotations") or [], dtype=np.float32) if track.get("rotations") is not None else None
        if positions.ndim != 2 or positions.shape[1] != 3 or len(times) != len(positions):
            skipped.append({"controller_name": name, "reason": "invalid_baked_positions"})
            continue
        if rotations is not None and (rotations.ndim != 2 or rotations.shape[1] != 4 or len(rotations) != len(times)):
            rotations = None
        controllers.append(_controller_payload(name, times, positions, duration, key_stride, rotations=rotations, target_rotation=include_rotation_tracks and rotations is not None))

    payload = _timeline_payload(
        animation_name=safe_id_for_path(animation_name),
        duration=duration,
        controllers=controllers,
        metadata={
            "generated_from_relative_flow": True,
            "baseline_pose_id": bake.get("baseline_pose_id"),
            "baseline_style": bake.get("baseline_style"),
            "source_world_coords_used": False,
            "clip_stitching_used": False,
            "person_root_tracks_included": False,
            "includes_baseline_keyframe": bool(include_baseline_keyframe),
            "includes_rotation_tracks": bool(include_rotation_tracks),
            "generated_baseline_pose": bool(bake.get("generated_baseline_pose")),
            "review_player_required": False,
            "review_player_not_required": True,
            "experimental_native_timeline_export": True,
            "generation_template_candidate": False,
            "source_retargeted_flow_id": bake.get("source_flow_id"),
            "source_schema": bake.get("source_flow_schema"),
            "controller_count": len(controllers),
            "skipped_tracks": skipped,
            "missing_baseline_controllers": bake.get("missing_baseline_controllers", []),
            "timeline_bake_schema": bake.get("schema"),
        },
    )
    dump_json(out, payload)
    _write_report(payload, report, out)
    return payload


def write_native_timeline_import_instructions(out_dir: str | Path, timeline_path: str | Path) -> Path:
    target = Path(out_dir) / "TIMELINE_IMPORT_TEST_INSTRUCTIONS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Native Timeline Import Test Instructions\n\n"
        "This export is intended to be native AcidBubbles Timeline JSON, not review-player JSON.\n\n"
        "1. Open VaM.\n"
        "2. Load a Person atom in a Cowgirl/kneeling test pose if needed.\n"
        "3. Add the Timeline plugin if the scene does not already have it.\n"
        f"4. Import `{Path(timeline_path).name}` through Timeline's import/load UI.\n"
        "5. Play `Generated_Cowgirl_Grinding_V0`.\n"
        "6. Check whether Timeline accepts the file.\n"
        "7. Check whether controller tracks are visible.\n"
        "8. Check whether pelvis motion reads as oval Cowgirl grind.\n"
        "9. Check whether knee/foot anchors stay stable.\n"
        "10. Confirm the Person/root atom stays still.\n"
        "11. Report whether the result looks Cowgirl-like or still hula-hoop-like.\n\n"
        "If Timeline refuses the file, report the VaM error text and whether any controller tracks were created.\n",
        encoding="utf-8",
    )
    return target


def write_native_timeline_import_instructions_v1(out_dir: str | Path, timeline_path: str | Path) -> Path:
    target = Path(out_dir) / "TIMELINE_IMPORT_TEST_INSTRUCTIONS_V1.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Native Timeline Import Test Instructions V1\n\n"
        "V1 bakes generated relative motion onto a generated Cowgirl/kneeling baseline pose before writing Timeline JSON.\n\n"
        "1. Open VaM.\n"
        "2. Add the Timeline plugin to the Person atom.\n"
        f"3. Import `{Path(timeline_path).name}` through Timeline's import/load UI.\n"
        "4. Select `Generated_Cowgirl_Grinding_V1`.\n"
        "5. Play from `t=0`.\n"
        "6. Check whether the first frame places the body in a Cowgirl/kneeling baseline.\n"
        "7. Check whether pelvis motion reads as oval Cowgirl grind.\n"
        "8. Check whether feet/knees remain stable.\n"
        "9. Check whether rotations are better than v0.\n"
        "10. Confirm the Person/root atom stays still and no teleport occurs.\n\n"
        "Please report: import accepted yes/no, t=0 pose correct yes/no, motion correct yes/no, rotations correct yes/no, feet/knees stable yes/no.\n",
        encoding="utf-8",
    )
    return target


def run_native_timeline_export_review_v0(retargeted_flow: str | Path, out_dir: str | Path) -> dict[str, Any]:
    from vam_timeline_ai.generation.native_timeline_validation import validate_native_timeline_export_v0

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    timeline = target / "generated_cowgirl_motion_v0.timeline.json"
    export_report = target / "generated_cowgirl_motion_v0_export_report.md"
    validation_report = target / "generated_cowgirl_motion_v0_validation.md"
    payload = export_generated_flow_native_timeline_v0(retargeted_flow, timeline, export_report)
    validation = validate_native_timeline_export_v0(timeline, validation_report)
    instructions = write_native_timeline_import_instructions(target, timeline)
    summary = {
        "schema": "native_timeline_export_review_v0",
        "timeline": str(timeline),
        "export_report": str(export_report),
        "validation_report": str(validation_report),
        "instructions": str(instructions),
        "validation_passed": validation.get("passed"),
        "expected_importable": validation.get("expected_importable"),
        "controller_count": len(((payload.get("Clips") or [{}])[0]).get("Controllers") or []),
        "review_player_required": False,
        "source_world_coords_used": False,
        "clip_stitching_used": False,
        "ml_training_run": False,
    }
    dump_json(target / "native_timeline_export_review_v0_summary.json", summary)
    _write_summary_report(summary, target / "native_timeline_export_review_v0_report.md")
    return summary


def run_native_timeline_export_review_v1(retargeted_flow: str | Path, baseline_pose: str | Path, out_dir: str | Path) -> dict[str, Any]:
    from vam_timeline_ai.generation.native_timeline_validation import validate_native_timeline_export_v1

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    timeline = target / "generated_cowgirl_motion_v1.timeline.json"
    export_report = target / "generated_cowgirl_motion_v1_export_report.md"
    validation_report = target / "generated_cowgirl_motion_v1_validation.md"
    payload = export_generated_flow_native_timeline_v1(
        retargeted_flow,
        baseline_pose,
        timeline,
        export_report,
        include_baseline_keyframe=True,
        include_rotation_tracks=True,
    )
    validation = validate_native_timeline_export_v1(timeline, baseline_pose, validation_report)
    instructions = write_native_timeline_import_instructions_v1(target, timeline)
    summary = {
        "schema": "native_timeline_export_review_v1",
        "timeline": str(timeline),
        "export_report": str(export_report),
        "validation_report": str(validation_report),
        "instructions": str(instructions),
        "validation_passed": validation.get("passed"),
        "expected_importable": validation.get("expected_importable"),
        "expected_pose_context": validation.get("expected_pose_context"),
        "controller_count": len(((payload.get("Clips") or [{}])[0]).get("Controllers") or []),
        "includes_baseline_keyframe": (payload.get("VAMTimelineAIGeneratedMetadata") or {}).get("includes_baseline_keyframe"),
        "includes_rotation_tracks": (payload.get("VAMTimelineAIGeneratedMetadata") or {}).get("includes_rotation_tracks"),
        "review_player_required": False,
        "source_world_coords_used": False,
        "clip_stitching_used": False,
        "ml_training_run": False,
    }
    dump_json(target / "native_timeline_export_review_v1_summary.json", summary)
    _write_summary_report(summary, target / "native_timeline_export_review_v1_report.md")
    return summary


def _timeline_payload(animation_name: str, duration: float, controllers: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "SerializeVersion": "283",
        "SerializeMode": "2",
        "AtomType": "Person",
        "Clips": [
            {
                "AnimationName": animation_name,
                "AnimationLength": round(float(duration), 6),
                "BlendDuration": 0,
                "Loop": 1,
                "PreserveLastFrame": 1,
                "LoopSelfBlendDuration": 0,
                "NextAnimationRandomizeWeight": 1,
                "AutoTransitionPrevious": 0,
                "AutoTransitionNext": 0,
                "SyncTransitionTime": 1,
                "SyncTransitionTimeNL": 0,
                "EnsureQuaternionContinuity": 1,
                "AnimationLayer": "Main",
                "Speed": 1,
                "Weight": 1,
                "Uninterruptible": 0,
                "AnimationSegment": "GeneratedMotion",
                "NextAnimationName": "",
                "NextAnimationTime": round(float(duration), 6),
                "Controllers": controllers,
            }
        ],
        "VAMTimelineAIGeneratedMetadata": metadata,
    }


def _controller_payload(
    name: str,
    times: np.ndarray,
    positions: np.ndarray,
    duration: float,
    key_stride: int,
    *,
    rotations: np.ndarray | None = None,
    target_rotation: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Controller": name,
        "TargetsPosition": 1,
        "TargetsRotation": 1 if target_rotation else 0,
        "ControlPosition": 1,
        "ControlRotation": 1 if target_rotation else 0,
    }
    for idx, axis in enumerate(POSITION_AXES):
        payload[axis] = _encode_curve(times, positions[:, idx], duration, key_stride, endpoint_value=float(positions[-1, idx]))
    if rotations is None:
        identity = {"RotX": 0.0, "RotY": 0.0, "RotZ": 0.0, "RotW": 1.0}
        for axis in ROTATION_AXES:
            payload[axis] = _encode_curve(times, np.full((len(times),), identity[axis], dtype=np.float32), duration, key_stride, endpoint_value=identity[axis])
    else:
        for idx, axis in enumerate(ROTATION_AXES):
            payload[axis] = _encode_curve(times, rotations[:, idx], duration, key_stride, endpoint_value=float(rotations[-1, idx]))
    return payload


def _encode_curve(times: np.ndarray, values: np.ndarray, duration: float, key_stride: int, endpoint_value: float) -> list[str]:
    keys: list[TimelineKeyframe] = []
    count = int(min(len(times), len(values)))
    if count == 0:
        keys = [TimelineKeyframe(0.0, float(endpoint_value), CurveTypeValues.Linear), TimelineKeyframe(float(duration), float(endpoint_value), CurveTypeValues.Linear)]
        return encode_keyframe_sequence(keys)
    indices = list(range(0, count, key_stride))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    for idx in indices:
        t = min(max(float(times[idx]), 0.0), float(duration))
        keys.append(TimelineKeyframe(time=t, value=float(values[idx]), curve_type=CurveTypeValues.Linear))
    if abs(keys[-1].time - float(duration)) > 1e-6:
        keys.append(TimelineKeyframe(time=float(duration), value=float(endpoint_value), curve_type=CurveTypeValues.Linear))
    return encode_keyframe_sequence(keys)


def _duration_from_tracks(tracks: list[dict[str, Any]], flow: dict[str, Any]) -> float:
    duration = float(flow.get("duration_seconds") or 0.0)
    for track in tracks:
        values = track.get("times") or []
        if values:
            duration = max(duration, float(max(values)))
    return round(max(duration, 0.001), 6)


def _fps_from_tracks(tracks: list[dict[str, Any]], flow: dict[str, Any]) -> float:
    if flow.get("fps"):
        return float(flow.get("fps"))
    for track in tracks:
        values = [float(v) for v in (track.get("times") or [])]
        if len(values) > 1 and values[1] > values[0]:
            return round(1.0 / (values[1] - values[0]), 6)
    return 60.0


def _times(track: dict[str, Any], fps: float, duration: float) -> np.ndarray:
    values = track.get("times") or []
    if values:
        return np.asarray(values, dtype=np.float32)
    positions = _positions(track)
    if positions is None:
        return np.zeros((0,), dtype=np.float32)
    return np.arange(len(positions), dtype=np.float32) / float(fps or 60.0)


def _positions(track: dict[str, Any]) -> np.ndarray | None:
    values = track.get("retargeted_positions")
    if values is None and track.get("baseline_position") is not None and track.get("position_deltas_applied") is not None:
        base = np.asarray(track.get("baseline_position"), dtype=np.float32).reshape(1, 3)
        values = base + np.asarray(track.get("position_deltas_applied"), dtype=np.float32)
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def _is_root_like(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in ("person", "root", "world", "atom"))


def _write_report(payload: dict[str, Any], report: str | Path, out: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    clip = (payload.get("Clips") or [{}])[0]
    metadata = payload.get("VAMTimelineAIGeneratedMetadata") or {}
    lines = [
        "# Generated Native Timeline Export V0",
        "",
        "This file uses the native AcidBubbles Timeline JSON structure observed in imported handmade references and the old segment exporter.",
        "",
        f"- Timeline JSON: `{Path(out)}`",
        f"- SerializeVersion: `{payload.get('SerializeVersion')}`",
        f"- AtomType: `{payload.get('AtomType')}`",
        f"- Animation: `{clip.get('AnimationName')}`",
        f"- Duration: `{clip.get('AnimationLength')}`",
        f"- Controllers: `{[c.get('Controller') for c in clip.get('Controllers', [])]}`",
        f"- Generated from relative flow: `{metadata.get('generated_from_relative_flow')}`",
        f"- Source world coords used: `{metadata.get('source_world_coords_used')}`",
        f"- Person/root tracks included: `{metadata.get('person_root_tracks_included')}`",
        f"- Clip stitching used: `{metadata.get('clip_stitching_used')}`",
        f"- Review player required: `{not metadata.get('review_player_not_required', False)}`",
        "",
        "Manual VaM Timeline import is still required before calling this production-compatible.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_report(summary: dict[str, Any], report: str | Path) -> None:
    Path(report).write_text(
        "# Native Timeline Export Review V0\n\n"
        f"- Timeline: `{summary.get('timeline')}`\n"
        f"- Validation passed: `{summary.get('validation_passed')}`\n"
        f"- Expected importable: `{summary.get('expected_importable')}`\n"
        f"- Controller count: `{summary.get('controller_count')}`\n"
        f"- Review player required: `{summary.get('review_player_required')}`\n"
        f"- Source world coords used: `{summary.get('source_world_coords_used')}`\n"
        f"- Clip stitching used: `{summary.get('clip_stitching_used')}`\n",
        encoding="utf-8",
    )

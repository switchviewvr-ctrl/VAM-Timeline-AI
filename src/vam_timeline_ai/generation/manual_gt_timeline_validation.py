"""Validate manual-ground-truth based review Timeline examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.vam_semantic_preview import is_disallowed_timeline_track
from vam_timeline_ai.io.json_utils import load_json


STATIC_THRESHOLD = 0.002
DRIVER_THRESHOLD = 0.005
AMPLITUDE_MAX = 0.16


def validate_manual_gt_timeline_examples_v1(preview_dir: str | Path, out: str | Path) -> dict[str, Any]:
    root = Path(preview_dir)
    data_path = root / "preview_data" / "manual_gt_timeline_clips_v1.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not data_path.exists():
        errors.append(f"missing preview data: {data_path}")
        data = {"clips": []}
    else:
        data = load_json(data_path)
    clips = list(data.get("clips") or [])
    for clip in clips:
        _validate_clip(root, clip, errors, warnings)
    status = "ok" if not errors else "failed"
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual GT Timeline Validation V1",
        "",
        f"- Status: `{status}`",
        f"- Clips: `{len(clips)}`",
        f"- Errors: `{len(errors)}`",
        f"- Warnings: `{len(warnings)}`",
        "- Review-only metadata required: `true`",
        "- Person/root/world tracks allowed: `false`",
        "- ML training performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {msg}" for msg in errors)
        lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {msg}" for msg in warnings)
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": status,
        "out": str(target),
        "clips": len(clips),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors,
        "warning_messages": warnings,
    }


def validate_manual_gt_timeline_examples_v2(
    preview_dir: str | Path,
    out: str | Path,
    *,
    allow_dense_export: bool = False,
) -> dict[str, Any]:
    root = Path(preview_dir)
    data_path = root / "preview_data" / "manual_gt_timeline_clips_v2.json"
    errors: list[str] = []
    warnings: list[str] = []
    key_counts: dict[str, dict[str, int]] = {}
    if not data_path.exists():
        errors.append(f"missing preview data: {data_path}")
        data = {"clips": []}
    else:
        data = load_json(data_path)
    clips = list(data.get("clips") or [])
    for clip in clips:
        _validate_clip(root, clip, errors, warnings)
        _validate_v2_rotation_rules(root, clip, errors, warnings, key_counts, allow_dense_export=allow_dense_export)
    status = "ok" if not errors else "failed"
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual GT Timeline Validation V2",
        "",
        f"- Status: `{status}`",
        f"- Clips: `{len(clips)}`",
        f"- Errors: `{len(errors)}`",
        f"- Warnings: `{len(warnings)}`",
        "- Position tracks required: `true`",
        "- Rotation quaternion tracks required: `true`",
        "- Dense 60fps export allowed: `false`",
        "- Review-only metadata required: `true`",
        "- Person/root/world tracks allowed: `false`",
        "- ML training performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Keyframe Counts",
        "",
    ]
    for clip_id, counts in sorted(key_counts.items()):
        lines.append(f"- `{clip_id}`: min `{counts.get('min')}`, max `{counts.get('max')}`, controllers `{counts.get('controllers')}`")
    lines.append("")
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {msg}" for msg in errors)
        lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {msg}" for msg in warnings)
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": status,
        "out": str(target),
        "clips": len(clips),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors,
        "warning_messages": warnings,
        "keyframe_counts": key_counts,
    }


def validate_manual_gt_timeline_examples_v3(
    preview_dir: str | Path,
    out: str | Path,
    *,
    allow_high_key_density: bool = False,
    allow_dense_export: bool = False,
) -> dict[str, Any]:
    root = Path(preview_dir)
    data_path = root / "preview_data" / "manual_gt_timeline_clips_v3.json"
    errors: list[str] = []
    warnings: list[str] = []
    key_counts: dict[str, dict[str, int]] = {}
    if not data_path.exists():
        errors.append(f"missing preview data: {data_path}")
        data = {"clips": []}
    else:
        data = load_json(data_path)
    clips = list(data.get("clips") or [])
    if float(data.get("keyframe_rate") or 0.0) != 1.0:
        warnings.append(f"package keyframe_rate is {data.get('keyframe_rate')}; v3 default is 1.0")
    for clip in clips:
        _validate_clip(root, clip, errors, warnings)
        _validate_v2_rotation_rules(root, clip, errors, warnings, key_counts, allow_dense_export=allow_dense_export)
        _validate_v3_hip_rules(root, clip, errors, warnings, allow_high_key_density=allow_high_key_density, allow_dense_export=allow_dense_export, expected_schema="manual_gt_timeline_example_v3")
    status = "ok" if not errors else "failed"
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual GT Timeline Validation V3",
        "",
        f"- Status: `{status}`",
        f"- Clips: `{len(clips)}`",
        f"- Errors: `{len(errors)}`",
        f"- Warnings: `{len(warnings)}`",
        "- hipControl required: `true`",
        "- Cowgirl primary driver: `hipControl`",
        "- pelvisControl Cowgirl role: `secondary/follower or static`",
        "- Position tracks required: `true`",
        "- Rotation quaternion tracks required: `true`",
        "- Default keyframe rate: `1 fps`",
        "- Dense 60fps export allowed: `false`",
        "- Review-only metadata required: `true`",
        "- Person/root/world tracks allowed: `false`",
        "- ML training performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Keyframe Counts",
        "",
    ]
    for clip_id, counts in sorted(key_counts.items()):
        lines.append(f"- `{clip_id}`: min `{counts.get('min')}`, max `{counts.get('max')}`, controllers `{counts.get('controllers')}`")
    lines.append("")
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {msg}" for msg in errors)
        lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {msg}" for msg in warnings)
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": status,
        "out": str(target),
        "clips": len(clips),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors,
        "warning_messages": warnings,
        "keyframe_counts": key_counts,
    }


def validate_manual_gt_timeline_examples_v4(
    preview_dir: str | Path,
    out: str | Path,
    *,
    allow_high_key_density: bool = False,
    allow_dense_export: bool = False,
) -> dict[str, Any]:
    root = Path(preview_dir)
    data_path = root / "preview_data" / "manual_gt_timeline_clips_v4.json"
    errors: list[str] = []
    warnings: list[str] = []
    key_counts: dict[str, dict[str, int]] = {}
    amplitude_profiles: dict[str, str] = {}
    if not data_path.exists():
        errors.append(f"missing preview data: {data_path}")
        data = {"clips": []}
    else:
        data = load_json(data_path)
    clips = list(data.get("clips") or [])
    if float(data.get("keyframe_rate") or 0.0) != 1.0:
        warnings.append(f"package keyframe_rate is {data.get('keyframe_rate')}; v4 default is 1.0")
    for clip in clips:
        _validate_clip(root, clip, errors, warnings)
        _validate_v2_rotation_rules(root, clip, errors, warnings, key_counts, allow_dense_export=allow_dense_export)
        _validate_v3_hip_rules(root, clip, errors, warnings, allow_high_key_density=allow_high_key_density, allow_dense_export=allow_dense_export, expected_schema="manual_gt_timeline_example_v4")
        key = str(clip.get("amplitude_profile_key") or "")
        amplitude_profiles[str(clip.get("clip_id") or "unknown")] = key
        if not key:
            errors.append(f"{clip.get('clip_id')}: missing amplitude_profile_key")
        if not isinstance(clip.get("amplitude_profile"), dict):
            errors.append(f"{clip.get('clip_id')}: missing amplitude_profile data")
    report_path = root / "reports" / "motion_amplitude_profile_report.md"
    if not report_path.exists():
        errors.append(f"missing amplitude profile report: {report_path}")
    status = "ok" if not errors else "failed"
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual GT Timeline Validation V4",
        "",
        f"- Status: `{status}`",
        f"- Clips: `{len(clips)}`",
        f"- Errors: `{len(errors)}`",
        f"- Warnings: `{len(warnings)}`",
        "- Baseline source unchanged from v3: `manual_pose_ground_truth_v1`",
        "- hipControl required: `true`",
        "- Cowgirl primary driver: `hipControl`",
        "- Amplitude profiles required: `true`",
        "- Position tracks required: `true`",
        "- Rotation quaternion tracks required: `true`",
        "- Default keyframe rate: `1 fps`",
        "- Dense 60fps export allowed: `false`",
        "- Review-only metadata required: `true`",
        "- Person/root/world tracks allowed: `false`",
        "- ML training performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Keyframe Counts",
        "",
    ]
    for clip_id, counts in sorted(key_counts.items()):
        lines.append(f"- `{clip_id}`: min `{counts.get('min')}`, max `{counts.get('max')}`, controllers `{counts.get('controllers')}`")
    lines.extend(["", "## Amplitude Profiles", ""])
    for clip_id, key in sorted(amplitude_profiles.items()):
        lines.append(f"- `{clip_id}`: `{key}`")
    lines.append("")
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {msg}" for msg in errors)
        lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {msg}" for msg in warnings)
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": status,
        "out": str(target),
        "clips": len(clips),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors,
        "warning_messages": warnings,
        "keyframe_counts": key_counts,
        "amplitude_profiles": amplitude_profiles,
    }


def _validate_clip(root: Path, clip: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    cid = str(clip.get("clip_id") or "unknown")
    family = str(clip.get("family") or "")
    if clip.get("review_only") is not True:
        errors.append(f"{cid}: review_only is not true")
    timeline = _timeline_path(root, clip)
    if not timeline.exists():
        errors.append(f"{cid}: timeline JSON missing")
    else:
        _validate_timeline_metadata(cid, timeline, errors)
    tracks = {str(track.get("controller_name")): track for track in (clip.get("controller_tracks") or [])}
    for name in tracks:
        if is_disallowed_timeline_track(name):
            errors.append(f"{cid}: disallowed Person/root/world track {name}")
    for name in clip.get("driver_controllers") or []:
        if _range(tracks.get(name)) <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: driver controller {name} has no meaningful motion")
    for name in clip.get("static_anchor_controllers") or []:
        if _range(tracks.get(name)) > STATIC_THRESHOLD:
            errors.append(f"{cid}: static anchor {name} moves too much ({_range(tracks.get(name)):.4f})")
    for track in tracks.values():
        if _range(track) > AMPLITUDE_MAX:
            warnings.append(f"{cid}: {track.get('controller_name')} amplitude is high for review-only clip ({_range(track):.4f})")
    if not _baseline_frame_exists(clip):
        errors.append(f"{cid}: missing baseline frame at t=0")
    _validate_family_rules(cid, family, tracks, errors)


def _validate_timeline_metadata(cid: str, timeline: Path, errors: list[str]) -> None:
    payload = load_json(timeline)
    meta = payload.get("VAMTimelineAIManualGTMetadata") or payload.get("VAMTimelineAIGeneratedMetadata") or {}
    if meta.get("review_only") is not True:
        errors.append(f"{cid}: timeline review_only metadata missing")
    if not meta.get("source_capture_id"):
        errors.append(f"{cid}: timeline missing source_capture_id")
    if meta.get("person_root_tracks_included") is not False:
        errors.append(f"{cid}: timeline does not explicitly disable Person/root tracks")
    clip = (payload.get("Clips") or [{}])[0]
    for controller in clip.get("Controllers") or []:
        name = str(controller.get("Controller") or "")
        if is_disallowed_timeline_track(name):
            errors.append(f"{cid}: timeline contains disallowed track {name}")


def _validate_v2_rotation_rules(
    root: Path,
    clip: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    key_counts: dict[str, dict[str, int]],
    *,
    allow_dense_export: bool,
) -> None:
    cid = str(clip.get("clip_id") or "unknown")
    tracks = {str(track.get("controller_name")): track for track in (clip.get("controller_tracks") or [])}
    keyframe_rate = float(clip.get("keyframe_rate") or 0.0)
    if keyframe_rate > 5.0:
        msg = f"{cid}: keyframe_rate {keyframe_rate} is dense; expected <= 5"
        if allow_dense_export:
            warnings.append(msg)
        else:
            errors.append(msg)
    timeline = _timeline_path(root, clip)
    if timeline.exists():
        payload = load_json(timeline)
        meta = payload.get("VAMTimelineAIManualGTMetadata") or payload.get("VAMTimelineAIGeneratedMetadata") or {}
        if meta.get("include_rotations") is not True:
            errors.append(f"{cid}: timeline metadata does not require rotations")
        if meta.get("identity_rotation_fallback_controllers"):
            errors.append(f"{cid}: identity rotation fallback used: {meta.get('identity_rotation_fallback_controllers')}")
        timeline_clip = (payload.get("Clips") or [{}])[0]
        counts = []
        for controller in timeline_clip.get("Controllers") or []:
            name = str(controller.get("Controller") or "")
            _validate_controller_has_position_and_rotation(cid, name, controller, errors)
            count = max(len(controller.get("X") or []), len(controller.get("RotX") or []))
            counts.append(count)
            if count > 25 and not allow_dense_export:
                errors.append(f"{cid}: {name} has dense keyframes ({count}); expected sparse semantic keys")
        if counts:
            key_counts[cid] = {"min": min(counts), "max": max(counts), "controllers": len(counts)}
    for name in clip.get("static_anchor_controllers") or []:
        if _rotation_range(tracks.get(name)) > STATIC_THRESHOLD:
            errors.append(f"{cid}: static anchor {name} rotation changes too much ({_rotation_range(tracks.get(name)):.4f})")
    if str(clip.get("family") or "") in {"bj_oral", "handjob"}:
        if _rotation_range(tracks.get("pelvisControl")) > STATIC_THRESHOLD:
            errors.append(f"{cid}: BJ/HJ pelvis rotation changes unexpectedly")
    if str(clip.get("family") or "") == "cowgirl":
        for name in ("lFootControl", "rFootControl"):
            if _rotation_range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: Cowgirl foot rotation changes unexpectedly: {name}")
    for track in tracks.values():
        if not track.get("rotations"):
            errors.append(f"{cid}: track missing rotations in preview data: {track.get('controller_name')}")
        if track.get("rotation_source") == "identity_missing_rotation_fallback":
            errors.append(f"{cid}: identity rotation fallback in preview data: {track.get('controller_name')}")


def _validate_v3_hip_rules(
    root: Path,
    clip: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    allow_high_key_density: bool,
    allow_dense_export: bool,
    expected_schema: str,
) -> None:
    cid = str(clip.get("clip_id") or "unknown")
    family = str(clip.get("family") or "")
    tracks = {str(track.get("controller_name")): track for track in (clip.get("controller_tracks") or [])}
    keyframe_rate = float(clip.get("keyframe_rate") or 0.0)
    if "hipControl" not in tracks:
        errors.append(f"{cid}: hipControl missing from exported clip")
    if keyframe_rate > 3.0:
        msg = f"{cid}: keyframe_rate {keyframe_rate} exceeds normal v3 maximum of 3 fps"
        if allow_high_key_density:
            warnings.append(msg)
        else:
            errors.append(msg)
    if keyframe_rate >= 30.0 and not allow_dense_export:
        errors.append(f"{cid}: dense 60fps-style keyframe export detected")
    timeline = _timeline_path(root, clip)
    if timeline.exists():
        payload = load_json(timeline)
        meta = payload.get("VAMTimelineAIManualGTMetadata") or payload.get("VAMTimelineAIGeneratedMetadata") or {}
        if meta.get("schema") != expected_schema:
            errors.append(f"{cid}: timeline schema is not {expected_schema}")
        if meta.get("keyframe_rate") and float(meta.get("keyframe_rate")) > 3.0 and not allow_high_key_density:
            errors.append(f"{cid}: timeline metadata keyframe_rate exceeds v3 maximum")
        timeline_clip = (payload.get("Clips") or [{}])[0]
        timeline_names = {str(controller.get("Controller") or "") for controller in timeline_clip.get("Controllers") or []}
        if "hipControl" not in timeline_names:
            errors.append(f"{cid}: hipControl missing from Timeline controller targets")
    if family == "cowgirl":
        if "hipControl" not in (clip.get("driver_controllers") or []):
            errors.append(f"{cid}: Cowgirl driver_controllers does not include hipControl")
        if _range(tracks.get("hipControl")) <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: Cowgirl hipControl primary driver has no meaningful motion")
        if _range(tracks.get("pelvisControl")) > _range(tracks.get("hipControl")) + STATIC_THRESHOLD:
            errors.append(f"{cid}: Cowgirl pelvisControl moves more than hipControl")
        if (clip.get("driver_controllers") or []) == ["pelvisControl"]:
            errors.append(f"{cid}: Cowgirl uses pelvisControl as sole primary driver")
        for name in ("lFootControl", "rFootControl"):
            if _range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: Cowgirl foot position changes unexpectedly: {name}")
            if _rotation_range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: Cowgirl foot rotation changes unexpectedly: {name}")
        for name in ("lHandControl", "rHandControl"):
            if _range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: Cowgirl hand support/controller moves unexpectedly: {name}")
    if family in {"bj_oral", "handjob"}:
        for name in ("hipControl", "pelvisControl"):
            if _range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: BJ/HJ {name} position changes unexpectedly")
            if _rotation_range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: BJ/HJ {name} rotation changes unexpectedly")
    if family == "bj_oral":
        if _range(tracks.get("headControl")) <= DRIVER_THRESHOLD and _range(tracks.get("chestControl")) <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: BJ lacks head/chest driver")
    if family == "handjob":
        hand_motion = max(_range(tracks.get("lHandControl")), _range(tracks.get("rHandControl")))
        if hand_motion <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: HJ lacks active hand driver")


def _validate_controller_has_position_and_rotation(cid: str, name: str, controller: dict[str, Any], errors: list[str]) -> None:
    for axis in ("X", "Y", "Z"):
        if not controller.get(axis):
            errors.append(f"{cid}: {name} missing position axis {axis}")
    for axis in ("RotX", "RotY", "RotZ", "RotW"):
        if not controller.get(axis):
            errors.append(f"{cid}: {name} missing rotation axis {axis}")
    if not controller.get("TargetsRotation") or not controller.get("ControlRotation"):
        errors.append(f"{cid}: {name} rotation is not enabled in Timeline controller payload")


def _validate_family_rules(cid: str, family: str, tracks: dict[str, dict[str, Any]], errors: list[str]) -> None:
    if family == "cowgirl":
        if _range(tracks.get("pelvisControl")) <= DRIVER_THRESHOLD and _range(tracks.get("hipControl")) <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: Cowgirl hip/pelvis driver missing")
        for name in ("lFootControl", "rFootControl", "lHandControl", "rHandControl"):
            if _range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: Cowgirl forbidden/static controller moves: {name}")
        if _range(tracks.get("headControl")) > 0.04:
            errors.append(f"{cid}: Cowgirl head appears to be primary driver")
    if family == "bj_oral":
        if _range(tracks.get("headControl")) <= DRIVER_THRESHOLD and _range(tracks.get("chestControl")) <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: BJ lacks head/chest driver")
        if _range(tracks.get("pelvisControl")) > STATIC_THRESHOLD:
            errors.append(f"{cid}: BJ pelvis moves as if riding")
    if family == "handjob":
        hand_motion = max(_range(tracks.get("lHandControl")), _range(tracks.get("rHandControl")))
        if hand_motion <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: HJ lacks hand driver")
        if _range(tracks.get("pelvisControl")) > STATIC_THRESHOLD:
            errors.append(f"{cid}: HJ pelvis moves as if Cowgirl")
    if family == "doggy":
        for name in ("lFootControl", "rFootControl", "lHandControl", "rHandControl", "lKneeControl", "rKneeControl"):
            if _range(tracks.get(name)) > STATIC_THRESHOLD:
                errors.append(f"{cid}: Doggy anchor moves too much: {name}")
        if _range(tracks.get("hipControl")) <= DRIVER_THRESHOLD and _range(tracks.get("pelvisControl")) <= DRIVER_THRESHOLD and _range(tracks.get("chestControl")) <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: Doggy receiver response missing")
    if family == "missionary":
        if _range(tracks.get("hipControl")) <= DRIVER_THRESHOLD and _range(tracks.get("pelvisControl")) <= DRIVER_THRESHOLD:
            errors.append(f"{cid}: Missionary counter hip/pelvis motion missing")
        if _range(tracks.get("headControl")) > STATIC_THRESHOLD or _range(tracks.get("chestControl")) > STATIC_THRESHOLD:
            errors.append(f"{cid}: Missionary chest/head should remain low/static in this review clip")


def _timeline_path(root: Path, clip: dict[str, Any]) -> Path:
    value = str(clip.get("timeline_json") or "")
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def _baseline_frame_exists(clip: dict[str, Any]) -> bool:
    tracks = clip.get("controller_tracks") or []
    if not tracks:
        return False
    for track in tracks:
        times = track.get("times") or []
        if not times or abs(float(times[0])) > 1e-6:
            return False
    return True


def _range(track: dict[str, Any] | None) -> float:
    if not track:
        return 0.0
    points = track.get("positions") or []
    if not points:
        return 0.0
    cols = list(zip(*points))
    spans = [max(float(v) for v in col) - min(float(v) for v in col) for col in cols[:3]]
    return max(spans) if spans else 0.0


def _rotation_range(track: dict[str, Any] | None) -> float:
    if not track:
        return 0.0
    points = track.get("rotations") or []
    if not points:
        return 0.0
    cols = list(zip(*points))
    spans = [max(float(v) for v in col) - min(float(v) for v in col) for col in cols[:4]]
    return max(spans) if spans else 0.0

"""Validate review-only VaM semantic preview packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

from vam_timeline_ai.generation.vam_semantic_preview import CORE_REVIEW_CONTROLLERS, is_disallowed_timeline_track
from vam_timeline_ai.io.json_utils import load_json
from vam_timeline_ai.timeline.codec import decode_keyframe_sequence


def validate_vam_semantic_preview_v0(preview_dir: str | Path, out: str | Path) -> dict[str, Any]:
    root = Path(preview_dir)
    preview_json = root / "preview_data" / "vam_semantic_preview_clips_v0.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not preview_json.exists():
        errors.append(f"missing preview data: {preview_json}")
        data = {"clips": []}
    else:
        data = load_json(preview_json)
    clips = list(data.get("clips") or [])
    exported = 0
    blocked = 0
    for clip in clips:
        cid = str(clip.get("clip_id") or "unknown")
        if clip.get("review_only") is not True:
            errors.append(f"{cid}: review_only metadata is not true")
        if clip.get("coordinate_space") != "synthetic_review_local":
            errors.append(f"{cid}: coordinate_space is not synthetic_review_local")
        for track in clip.get("controller_tracks") or []:
            name = str(track.get("controller_name") or "")
            if is_disallowed_timeline_track(name):
                errors.append(f"{cid}: disallowed Person/root/world/atom-like track {name}")
        if clip.get("export_status") == "exported":
            exported += 1
            _validate_exported_clip(root, clip, errors, warnings)
        else:
            blocked += 1
        _validate_semantic_clip_rules(clip, errors, warnings)

    status = "ok" if not errors else "failed"
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VaM Semantic Preview Validation V0",
        "",
        f"- Status: `{status}`",
        f"- Clips: `{len(clips)}`",
        f"- Exported clips: `{exported}`",
        f"- Blocked clips: `{blocked}`",
        f"- Errors: `{len(errors)}`",
        f"- Warnings: `{len(warnings)}`",
        "- Review-only: `true`",
        "- Source scene coordinates used: `false`",
        "- Person/root/world tracks included: `false`",
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
        "exported_clips": exported,
        "blocked_clips": blocked,
        "errors": len(errors),
        "warnings": len(warnings),
        "error_messages": errors,
        "warning_messages": warnings,
    }


def _validate_exported_clip(root: Path, clip: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    cid = str(clip.get("clip_id") or "unknown")
    timeline_path = Path(str(clip.get("timeline_json") or ""))
    if not timeline_path.is_absolute():
        if timeline_path.exists():
            timeline_path = timeline_path
        elif (root / timeline_path).exists():
            timeline_path = root / timeline_path
        else:
            timeline_path = root / timeline_path
    if not timeline_path.exists():
        errors.append(f"{cid}: timeline JSON missing: {timeline_path}")
        return
    payload = load_json(timeline_path)
    meta = payload.get("VAMTimelineAISemanticPreviewMetadata") or payload.get("VAMTimelineAIGeneratedMetadata") or {}
    if meta.get("review_only") is not True:
        errors.append(f"{cid}: timeline review_only metadata is not true")
    if meta.get("source_world_coords_used") is not False:
        errors.append(f"{cid}: timeline claims source world coordinates were used")
    if meta.get("person_root_tracks_included") is not False:
        errors.append(f"{cid}: timeline claims Person/root tracks are included")
    if meta.get("production_ready") is not False:
        errors.append(f"{cid}: timeline claims production-ready")
    timeline_clip = (payload.get("Clips") or [{}])[0]
    controllers = list(timeline_clip.get("Controllers") or [])
    controller_names = {str(c.get("Controller") or "") for c in controllers}
    for name in controller_names:
        if is_disallowed_timeline_track(name):
            errors.append(f"{cid}: timeline has disallowed controller track {name}")
    missing = sorted(CORE_REVIEW_CONTROLLERS - controller_names)
    if missing:
        warnings.append(f"{cid}: missing some core review controllers: {missing}")
    if not _has_baseline_frame_zero(controllers):
        errors.append(f"{cid}: missing baseline frame at t=0")


def _validate_semantic_clip_rules(clip: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    cid = str(clip.get("clip_id") or "unknown")
    family = str(clip.get("family") or "")
    if family in {"cowgirl", "reverse_cowgirl"}:
        if "partner_pelvis_target" not in (clip.get("target_points") or {}):
            errors.append(f"{cid}: missing partner_pelvis_target metadata")
        distance = _controller_target_max_distance(clip, "pelvisControl", "partner_pelvis_target")
        allowed = float((clip.get("alignment_validation") or {}).get("target_distance_max") or 0.0)
        if not math.isfinite(distance) or allowed <= 0 or distance > allowed + 1e-5:
            errors.append(f"{cid}: actor pelvis is outside partner pelvis target tolerance ({distance:.3f} > {allowed:.3f})")
    if family == "bj_oral":
        roles = _role_map(clip)
        if roles.get("headControl") != "driver" and roles.get("chestControl") != "driver":
            errors.append(f"{cid}: BJ/Oral preview lacks head/chest driver")
        pelvis_range = _track_motion_range(clip, "pelvisControl")
        if pelvis_range > 0.08:
            errors.append(f"{cid}: BJ/Oral pelvis is not static enough ({pelvis_range:.3f})")
    if family == "doggy":
        anchors = set((clip.get("labels") or {}).get("anchors") or [])
        if not {"hands", "knees"}.issubset(anchors):
            errors.append(f"{cid}: doggy preview lacks hands/knees support anchors")
    if family == "missionary":
        pose = str(clip.get("pose_subtype") or "")
        if "missionary" not in pose or "supine" not in pose and "flat" not in pose:
            errors.append(f"{cid}: missionary preview is not marked supine/flat")


def _role_map(clip: dict[str, Any]) -> dict[str, str]:
    return {str(t.get("controller_name") or ""): str(t.get("role") or "") for t in clip.get("controller_tracks") or []}


def _track_motion_range(clip: dict[str, Any], controller: str) -> float:
    for track in clip.get("controller_tracks") or []:
        if track.get("controller_name") == controller:
            points = track.get("positions") or []
            if not points:
                return 0.0
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            zs = [float(p[2]) for p in points]
            return max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return 0.0


def _controller_target_max_distance(clip: dict[str, Any], controller: str, target_name: str) -> float:
    target = (clip.get("target_points") or {}).get(target_name)
    if not isinstance(target, list) or len(target) < 3:
        return float("inf")
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    max_distance = 0.0
    found = False
    for track in clip.get("controller_tracks") or []:
        if track.get("controller_name") != controller:
            continue
        for point in track.get("positions") or []:
            found = True
            dx = float(point[0]) - tx
            dy = float(point[1]) - ty
            dz = float(point[2]) - tz
            max_distance = max(max_distance, math.sqrt(dx * dx + dy * dy + dz * dz))
    return max_distance if found else float("inf")


def _has_baseline_frame_zero(controllers: list[dict[str, Any]]) -> bool:
    if not controllers:
        return False
    for controller in controllers:
        curve = controller.get("X") or []
        if not curve:
            return False
        try:
            decoded = decode_keyframe_sequence(curve, version=283)
        except Exception:
            return False
        if not decoded or abs(float(decoded[0].time)) > 1e-6:
            return False
    return True

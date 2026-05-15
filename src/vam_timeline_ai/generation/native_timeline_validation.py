"""Validation for generated native Timeline JSON exports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.generated_motion import is_allowed_generated_controller
from vam_timeline_ai.io.json_utils import load_json
from vam_timeline_ai.timeline.codec import decode_keyframe_sequence
from vam_timeline_ai.timeline.parser import POSITION_AXES, ROTATION_AXES


ROOT_TOKENS = ("person", "root", "world", "atom")
REQUIRED_CLIP_FIELDS = ("AnimationName", "AnimationLength", "Controllers")
REQUIRED_CONTROLLER_FIELDS = ("Controller", "TargetsPosition", "ControlPosition")


def validate_native_timeline_export_v0(timeline: str | Path | dict[str, Any], report: str | Path | None = None) -> dict[str, Any]:
    data = load_json(timeline) if not isinstance(timeline, dict) else timeline
    checks: list[dict[str, Any]] = []
    _check(checks, "json_is_object", isinstance(data, dict), type(data).__name__)
    _check(checks, "no_custom_review_schema", "schema" not in data, data.get("schema") if isinstance(data, dict) else None)
    _check(checks, "serialize_version_present", bool(data.get("SerializeVersion")), data.get("SerializeVersion"))
    _check(checks, "atom_type_person", data.get("AtomType") == "Person", data.get("AtomType"))
    clips = data.get("Clips") if isinstance(data.get("Clips"), list) else []
    _check(checks, "animation_count_positive", len(clips) > 0, len(clips))
    metadata = data.get("VAMTimelineAIGeneratedMetadata") if isinstance(data.get("VAMTimelineAIGeneratedMetadata"), dict) else {}
    _check(checks, "generated_metadata_present", bool(metadata), sorted(metadata.keys()))
    _check(checks, "generated_from_relative_flow", metadata.get("generated_from_relative_flow") is True, metadata.get("generated_from_relative_flow"))
    _check(checks, "source_world_coords_not_used", metadata.get("source_world_coords_used") is False, metadata.get("source_world_coords_used"))
    _check(checks, "person_root_tracks_not_included", metadata.get("person_root_tracks_included") is False, metadata.get("person_root_tracks_included"))
    _check(checks, "review_player_not_required", metadata.get("review_player_not_required") is True, metadata.get("review_player_not_required"))

    version = int(float(data.get("SerializeVersion") or 283))
    controller_count = 0
    for clip_idx, clip in enumerate(clips):
        _validate_clip(checks, clip, clip_idx, version)
        controller_count += len(clip.get("Controllers") or []) if isinstance(clip, dict) else 0

    passed = all(bool(c.get("passed")) for c in checks)
    # Static schema validation cannot guarantee VaM's importer will accept the file.
    expected_importable = "unknown" if passed else "no"
    summary = {
        "schema": "native_timeline_export_validation_v0",
        "passed": passed,
        "expected_importable": expected_importable,
        "controller_count": controller_count,
        "checks": checks,
        "manual_test_required": passed,
        "warnings": [
            "Static validation cannot guarantee VaM Timeline import success.",
            "Manual VaM import must confirm controller tracks and playback behavior.",
        ] if passed else ["Validation failed; do not attempt to treat this as importable."],
    }
    if report is not None:
        _write_report(summary, report)
    return summary


def validate_native_timeline_export_v1(
    timeline: str | Path | dict[str, Any],
    baseline_pose: str | Path | dict[str, Any],
    report: str | Path | None = None,
) -> dict[str, Any]:
    data = load_json(timeline) if not isinstance(timeline, dict) else timeline
    baseline = load_json(baseline_pose) if not isinstance(baseline_pose, dict) else baseline_pose
    summary = validate_native_timeline_export_v0(data, None)
    checks = list(summary.get("checks", []) or [])
    metadata = data.get("VAMTimelineAIGeneratedMetadata") if isinstance(data.get("VAMTimelineAIGeneratedMetadata"), dict) else {}
    clips = data.get("Clips") if isinstance(data.get("Clips"), list) else []
    clip = clips[0] if clips and isinstance(clips[0], dict) else {}
    controllers = clip.get("Controllers") if isinstance(clip.get("Controllers"), list) else []
    controller_map = {row.get("Controller"): row for row in controllers if isinstance(row, dict)}
    baseline_map = {row.get("controller_name"): row for row in baseline.get("controller_poses", []) or []}
    required = {"lFootControl", "rFootControl", "lKneeControl", "rKneeControl"}

    _check(checks, "includes_baseline_keyframe", metadata.get("includes_baseline_keyframe") is True, metadata.get("includes_baseline_keyframe"))
    _check(checks, "includes_rotation_tracks", metadata.get("includes_rotation_tracks") is True, metadata.get("includes_rotation_tracks"))
    _check(checks, "generated_baseline_pose", metadata.get("generated_baseline_pose") is True, metadata.get("generated_baseline_pose"))
    _check(checks, "baseline_style_cowgirl", str(metadata.get("baseline_style") or baseline.get("baseline_style") or baseline.get("style")).startswith("cowgirl") or baseline.get("style") == "kneeling_forward", {"metadata": metadata.get("baseline_style"), "baseline": baseline.get("style")})
    _check(checks, "required_anchor_tracks_present", required.issubset(set(controller_map)), sorted(required - set(controller_map)))

    for name, base in baseline_map.items():
        if name not in controller_map:
            continue
        controller = controller_map[name]
        base_pos = [float(v) for v in (base.get("baseline_position") or [])]
        if len(base_pos) == 3:
            actual = [_first_key_value(controller.get(axis)) for axis in POSITION_AXES]
            ok = all(value is not None and abs(float(value) - base_pos[idx]) <= 1e-4 for idx, value in enumerate(actual))
            _check(checks, f"{name}:t0_matches_baseline_position", ok, {"expected": base_pos, "actual": actual})
        if metadata.get("includes_rotation_tracks") is True:
            rot_counts = [_key_count(controller.get(axis)) for axis in ROTATION_AXES]
            _check(checks, f"{name}:rotation_tracks_exist", all(count > 0 for count in rot_counts), dict(zip(ROTATION_AXES, rot_counts)))
        if name in required:
            stable = _anchor_stable(controller)
            _check(checks, f"{name}:anchor_stable", stable, name)

    passed = all(bool(c.get("passed")) for c in checks)
    expected_importable = "unknown" if passed else "no"
    summary.update({
        "schema": "native_timeline_export_validation_v1",
        "passed": passed,
        "expected_importable": expected_importable,
        "expected_pose_context": "cowgirl_kneeling_forward",
        "checks": checks,
        "manual_test_required": passed,
    })
    if report is not None:
        _write_report(summary, report)
    return summary


def _validate_clip(checks: list[dict[str, Any]], clip: Any, clip_idx: int, version: int) -> None:
    prefix = f"clip_{clip_idx}"
    _check(checks, f"{prefix}:is_object", isinstance(clip, dict), type(clip).__name__)
    if not isinstance(clip, dict):
        return
    for field in REQUIRED_CLIP_FIELDS:
        detail = len(clip.get(field) or []) if field == "Controllers" and isinstance(clip.get(field), list) else clip.get(field)
        _check(checks, f"{prefix}:has_{field}", field in clip, detail)
    duration = _as_float(clip.get("AnimationLength"), 0.0)
    _check(checks, f"{prefix}:duration_positive", duration > 0, duration)
    controllers = clip.get("Controllers") if isinstance(clip.get("Controllers"), list) else []
    _check(checks, f"{prefix}:controller_targets_exist", len(controllers) > 0, len(controllers))
    for idx, controller in enumerate(controllers):
        _validate_controller(checks, controller, f"{prefix}:controller_{idx}", duration, version)


def _validate_controller(checks: list[dict[str, Any]], controller: Any, prefix: str, duration: float, version: int) -> None:
    _check(checks, f"{prefix}:is_object", isinstance(controller, dict), type(controller).__name__)
    if not isinstance(controller, dict):
        return
    for field in REQUIRED_CONTROLLER_FIELDS:
        _check(checks, f"{prefix}:has_{field}", field in controller, controller.get(field))
    name = str(controller.get("Controller") or "")
    lower = name.lower()
    _check(checks, f"{prefix}:allowed_controller", is_allowed_generated_controller(name), name)
    _check(checks, f"{prefix}:not_person_root_world", not any(token in lower for token in ROOT_TOKENS), name)
    for axis in POSITION_AXES:
        _validate_axis(checks, controller, prefix, axis, duration, version, required=True)
    for axis in ROTATION_AXES:
        _validate_axis(checks, controller, prefix, axis, duration, version, required=False)


def _validate_axis(
    checks: list[dict[str, Any]],
    controller: dict[str, Any],
    prefix: str,
    axis: str,
    duration: float,
    version: int,
    *,
    required: bool,
) -> None:
    raw = controller.get(axis)
    _check(checks, f"{prefix}:{axis}:present", (not required) or isinstance(raw, list), "present" if isinstance(raw, list) else type(raw).__name__)
    if not isinstance(raw, list):
        return
    try:
        keys = decode_keyframe_sequence(raw, version)
    except Exception as exc:
        _check(checks, f"{prefix}:{axis}:decodable", False, f"{type(exc).__name__}: {exc}")
        return
    _check(checks, f"{prefix}:{axis}:decodable", True, len(keys))
    _check(checks, f"{prefix}:{axis}:key_count_positive", len(keys) > 0, len(keys))
    times = [float(k.time) for k in keys]
    values = [float(k.value) for k in keys]
    _check(checks, f"{prefix}:{axis}:times_sorted", times == sorted(times), times[:5])
    _check(checks, f"{prefix}:{axis}:times_in_duration", all(-1e-6 <= t <= duration + 1e-4 for t in times), {"first": times[:1], "last": times[-1:]})
    _check(checks, f"{prefix}:{axis}:finite_values", all(math.isfinite(v) for v in values), values[:5])


def _first_key_value(raw: Any) -> float | None:
    if not isinstance(raw, list):
        return None
    keys = decode_keyframe_sequence(raw, 283)
    if not keys:
        return None
    return float(keys[0].value)


def _key_count(raw: Any) -> int:
    if not isinstance(raw, list):
        return 0
    try:
        return len(decode_keyframe_sequence(raw, 283))
    except Exception:
        return 0


def _anchor_stable(controller: dict[str, Any]) -> bool:
    for axis in POSITION_AXES:
        raw = controller.get(axis)
        if not isinstance(raw, list):
            return False
        keys = decode_keyframe_sequence(raw, 283)
        if not keys:
            return False
        values = [float(k.value) for k in keys]
        if max(values) - min(values) > 1e-4:
            return False
    return True


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Native Timeline Export Validation V1" if summary.get("schema") == "native_timeline_export_validation_v1" else "# Native Timeline Export Validation V0",
        "",
        f"- Passed: `{summary.get('passed')}`",
        f"- Expected importable: `{summary.get('expected_importable')}`",
        f"- Controller count: `{summary.get('controller_count')}`",
        f"- Manual test required: `{summary.get('manual_test_required')}`",
        "",
        "## Checks",
        "",
    ]
    for check in summary.get("checks", []) or []:
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- {mark}: `{check.get('name')}` detail=`{check.get('detail')}`")
    lines.extend(["", "## Warnings", ""])
    for warning in summary.get("warnings", []) or []:
        lines.append(f"- {warning}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

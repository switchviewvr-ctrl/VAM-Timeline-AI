"""Validation for synthesized relative motion flows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.generation.generated_motion import is_allowed_generated_controller
from vam_timeline_ai.io.json_utils import load_json


MAX_REASONABLE_DELTA = 0.75
MAX_REASONABLE_FRAME_JUMP = 0.18
ROOT_TRACK_TOKENS = ("person", "root", "world", "atom")


def validate_generated_motion_flow_v0(flow: str | Path | dict[str, Any], out: str | Path | None = None) -> dict[str, Any]:
    data = load_json(flow) if not isinstance(flow, dict) else flow
    checks: list[dict[str, Any]] = []
    _check(checks, "coordinate_space_relative", data.get("coordinate_space") == "relative_body_motion", data.get("coordinate_space"))
    _check(checks, "no_world_coordinates_flag", data.get("no_world_coordinates") is True, data.get("no_world_coordinates"))
    _check(checks, "no_person_root_tracks_flag", data.get("no_person_root_tracks") is True, data.get("no_person_root_tracks"))
    _check(checks, "clip_stitching_not_used", data.get("clip_stitching_used") is False, data.get("clip_stitching_used"))
    _check(checks, "export_ready_false", data.get("export_ready") is False, data.get("export_ready"))
    _check(checks, "timeline_export_not_performed", data.get("timeline_export_performed") is False, data.get("timeline_export_performed"))

    track_results = [_validate_track(track) for track in data.get("controller_tracks", []) or []]
    for result in track_results:
        checks.extend(result["checks"])
    passed = all(bool(check.get("passed")) for check in checks)
    summary = {
        "schema": "generated_motion_flow_validation_v0",
        "flow_id": data.get("flow_id"),
        "passed": passed,
        "safe_for_timeline_export": False,
        "track_count": len(track_results),
        "checks": checks,
        "track_summaries": [{k: v for k, v in result.items() if k != "checks"} for result in track_results],
        "warnings": [
            "Validation is for relative-flow sanity only.",
            "safe_for_timeline_export is false until retargeting and Timeline export safety exist.",
        ],
    }
    if out is not None:
        _write_report(summary, out)
    return summary


def _validate_track(track: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    name = str(track.get("controller_name") or "")
    lower = name.lower()
    positions = np.asarray(track.get("position_deltas") or [], dtype=float)
    times = np.asarray(track.get("times") or [], dtype=float)
    role = str(track.get("role") or "unknown")
    _check(checks, f"{name}:allowed_controller", is_allowed_generated_controller(name), name)
    _check(checks, f"{name}:not_root_world_person", not any(token in lower for token in ROOT_TRACK_TOKENS), name)
    _check(checks, f"{name}:relative_track_space", track.get("coordinate_space") == "relative_body_motion", track.get("coordinate_space"))
    _check(checks, f"{name}:position_shape", positions.ndim == 2 and (positions.shape[1] if positions.ndim == 2 else 0) == 3, list(positions.shape))
    _check(checks, f"{name}:finite_values", bool(np.all(np.isfinite(positions))) if positions.size else False, "finite" if positions.size else "empty")
    if times.size and positions.ndim == 2:
        _check(checks, f"{name}:time_length_matches", len(times) == positions.shape[0], {"times": len(times), "positions": positions.shape[0]})
    max_delta = float(np.max(np.abs(positions))) if positions.size else math.inf
    _check(checks, f"{name}:max_delta_reasonable", max_delta <= MAX_REASONABLE_DELTA, round(max_delta, 6))
    diffs = np.diff(positions, axis=0) if positions.ndim == 2 and len(positions) > 1 else np.zeros((0, 3))
    max_jump = float(np.max(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0
    _check(checks, f"{name}:no_violent_jumps", max_jump <= MAX_REASONABLE_FRAME_JUMP, round(max_jump, 6))
    if role == "anchor":
        anchor_max = float(np.max(np.linalg.norm(positions, axis=1))) if positions.size else math.inf
        _check(checks, f"{name}:anchor_stable", anchor_max <= 1e-5, round(anchor_max, 8))
    safety = track.get("safety_flags", {}) or {}
    _check(checks, f"{name}:track_relative_flag", safety.get("relative_deltas_only") is True, safety)
    _check(checks, f"{name}:track_not_source_copied", safety.get("source_timeline_keyframes_copied") is False, safety)
    return {
        "controller_name": name,
        "role": role,
        "max_abs_delta": round(max_delta, 6) if math.isfinite(max_delta) else None,
        "max_frame_jump": round(max_jump, 6),
        "checks": checks,
    }


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated Motion Flow V0 Validation",
        "",
        f"- Flow: `{summary.get('flow_id')}`",
        f"- Passed: `{summary.get('passed')}`",
        f"- Safe for Timeline export: `{summary.get('safe_for_timeline_export')}`",
        f"- Track count: `{summary.get('track_count')}`",
        "",
        "## Checks",
        "",
    ]
    for check in summary.get("checks", []) or []:
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- {mark}: `{check.get('name')}` detail=`{check.get('detail')}`")
    lines.extend(["", "## Track Summary", ""])
    for track in summary.get("track_summaries", []) or []:
        lines.append(f"- `{track.get('controller_name')}` role={track.get('role')} max_delta={track.get('max_abs_delta')} max_jump={track.get('max_frame_jump')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

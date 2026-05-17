"""Validation for baseline-retargeted generated motion flows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.generation.generated_motion import is_allowed_generated_controller
from vam_timeline_ai.io.json_utils import load_json


REQUIRED_ANCHORS = {"lFootControl", "rFootControl", "lKneeControl", "rKneeControl"}
ROOT_TOKENS = ("person", "root", "world", "atom")


def validate_retargeted_motion_flow_v0(retargeted_flow: str | Path | dict[str, Any], out: str | Path | None = None) -> dict[str, Any]:
    data = load_json(retargeted_flow) if not isinstance(retargeted_flow, dict) else retargeted_flow
    checks: list[dict[str, Any]] = []
    tracks = data.get("controller_tracks", []) or []
    by_name = {track.get("controller_name"): track for track in tracks}
    _check(checks, "coordinate_space_retargeted", data.get("coordinate_space") == "retargeted_to_baseline_pose", data.get("coordinate_space"))
    _check(checks, "no_person_root_included", data.get("person_root_included") is False, data.get("person_root_included"))
    _check(checks, "no_source_world_coords", data.get("source_world_coords_used") is False and data.get("world_coords_source") == "none", data.get("world_coords_source"))
    _check(checks, "no_clip_stitching", data.get("clip_stitching_used") is False, data.get("clip_stitching_used"))
    _check(checks, "required_anchors_present", REQUIRED_ANCHORS.issubset(set(by_name)), sorted(REQUIRED_ANCHORS - set(by_name)))
    for track in tracks:
        checks.extend(_validate_track(track))
    checks.extend(_distance_checks(by_name))
    passed = all(bool(check.get("passed")) for check in checks)
    review_safe = passed
    generation_candidate = bool(review_safe and data.get("source_world_coords_used") is False)
    summary = {
        "schema": "retargeted_motion_flow_validation_v0",
        "flow_id": data.get("flow_id"),
        "passed": passed,
        "export_review_safe_candidate": review_safe,
        "generation_template_candidate": generation_candidate,
        "checks": checks,
        "warnings": [
            "Generation-template candidate only means synthetic retarget sanity passed; production export remains future work.",
            "Review export can still be marked review-only by Timeline exporter.",
        ],
    }
    if out is not None:
        _write_report(summary, out)
    return summary


def validate_retargeted_motion_flow_v1(retargeted_flow: str | Path | dict[str, Any], out: str | Path | None = None) -> dict[str, Any]:
    data = load_json(retargeted_flow) if not isinstance(retargeted_flow, dict) else retargeted_flow
    summary = validate_retargeted_motion_flow_v0(data, None)
    checks = list(summary.get("checks", []) or [])
    by_name = {track.get("controller_name"): track for track in data.get("controller_tracks", []) or []}
    _check(checks, "cowgirl_baseline_style", data.get("baseline_style") in {"kneeling_forward", None} and data.get("intended_family") in {"cowgirl", None}, {"style": data.get("baseline_style"), "family": data.get("intended_family")})
    pelvis = _positions(by_name.get("pelvisControl") or by_name.get("hipControl"))
    if pelvis is not None:
        centered = pelvis - np.mean(pelvis, axis=0, keepdims=True)
        lateral = float(np.max(centered[:, 0]) - np.min(centered[:, 0]))
        vertical = float(np.max(centered[:, 1]) - np.min(centered[:, 1]))
        forward = float(np.max(centered[:, 2]) - np.min(centered[:, 2]))
        denom = max(vertical + forward, 1e-6)
        _check(checks, "not_excessive_hula_hoop_lateral_dominance", (lateral / denom) <= 0.95, {"lateral": round(lateral, 6), "vertical": round(vertical, 6), "forward_back": round(forward, 6)})
        _check(checks, "meaningful_forward_or_vertical_component", max(vertical, forward) >= 0.05, {"vertical": round(vertical, 6), "forward_back": round(forward, 6)})
    chest = _positions(by_name.get("chestControl"))
    if pelvis is not None and chest is not None:
        pelvis_span = float(np.max(np.linalg.norm(pelvis - np.mean(pelvis, axis=0), axis=1)))
        chest_span = float(np.max(np.linalg.norm(chest - np.mean(chest, axis=0), axis=1)))
        _check(checks, "chest_follower_damped_vs_pelvis", chest_span <= pelvis_span * 0.75, {"pelvis_span": round(pelvis_span, 6), "chest_span": round(chest_span, 6)})
    passed = all(bool(check.get("passed")) for check in checks)
    summary.update({
        "schema": "retargeted_motion_flow_validation_v1",
        "passed": passed,
        "export_review_safe_candidate": passed,
        "generation_template_candidate": False,
        "checks": checks,
    })
    if out is not None:
        _write_report(summary, out)
    return summary


def validation_markdown_allows_export(validation: str | Path) -> bool:
    text = Path(validation).read_text(encoding="utf-8", errors="ignore")
    return "Passed: `True`" in text and "Export review safe candidate: `True`" in text


def _validate_track(track: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    name = str(track.get("controller_name") or "")
    lower = name.lower()
    positions = np.asarray(track.get("retargeted_positions") or [], dtype=float)
    times = np.asarray(track.get("times") or [], dtype=float)
    _check(checks, f"{name}:allowed_controller", is_allowed_generated_controller(name), name)
    _check(checks, f"{name}:not_person_root_world", not any(token in lower for token in ROOT_TOKENS), name)
    _check(checks, f"{name}:retargeted_space", track.get("coordinate_space") == "retargeted_to_baseline_pose", track.get("coordinate_space"))
    _check(checks, f"{name}:finite_positions", bool(np.all(np.isfinite(positions))) if positions.size else False, "finite" if positions.size else "empty")
    _check(checks, f"{name}:shape", positions.ndim == 2 and (positions.shape[1] if positions.ndim == 2 else 0) == 3, list(positions.shape))
    if times.size and positions.ndim == 2:
        _check(checks, f"{name}:time_length_matches", len(times) == positions.shape[0], {"times": len(times), "positions": positions.shape[0]})
    diffs = np.diff(positions, axis=0) if positions.ndim == 2 and len(positions) > 1 else np.zeros((0, 3))
    max_jump = float(np.max(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0
    _check(checks, f"{name}:no_violent_jumps", max_jump <= 0.20, round(max_jump, 6))
    if track.get("role") == "anchor":
        anchor_motion = float(np.max(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0
        _check(checks, f"{name}:anchor_stable", anchor_motion <= 1e-5, round(anchor_motion, 8))
    if name in {"pelvisControl", "hipControl"}:
        baseline = np.asarray(track.get("baseline_position") or [0, 0, 0], dtype=float)
        deltas = positions - baseline.reshape(1, 3)
        max_delta = float(np.max(np.linalg.norm(deltas, axis=1))) if positions.size else math.inf
        _check(checks, f"{name}:pelvis_motion_amplitude_reasonable", max_delta <= 0.45, round(max_delta, 6))
    return checks


def _distance_checks(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    pelvis = _positions(by_name.get("pelvisControl") or by_name.get("hipControl"))
    chest = _positions(by_name.get("chestControl"))
    if pelvis is not None and chest is not None:
        dist = _max_dist(pelvis, chest)
        _check(checks, "chest_not_too_far_from_pelvis", dist <= 0.75, round(dist, 6))
    for foot, knee in [("lFootControl", "lKneeControl"), ("rFootControl", "rKneeControl")]:
        foot_pos = _positions(by_name.get(foot))
        knee_pos = _positions(by_name.get(knee))
        if foot_pos is not None and knee_pos is not None:
            dist = _max_dist(foot_pos, knee_pos)
            _check(checks, f"{foot}:not_too_far_from_knee", dist <= 0.70, round(dist, 6))
        if foot_pos is not None and pelvis is not None:
            dist = _max_dist(foot_pos, pelvis)
            _check(checks, f"{foot}:not_too_far_from_pelvis", dist <= 1.25, round(dist, 6))
    return checks


def _positions(track: dict[str, Any] | None) -> np.ndarray | None:
    if not track:
        return None
    arr = np.asarray(track.get("retargeted_positions") or [], dtype=float)
    return arr if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) else None


def _max_dist(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return math.inf
    return float(np.max(np.linalg.norm(a[:n] - b[:n], axis=1)))


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    version = "V1" if str(summary.get("schema")) == "retargeted_motion_flow_validation_v1" else "V0"
    lines = [
        f"# Retargeted Motion Flow {version} Validation",
        "",
        f"- Flow: `{summary.get('flow_id')}`",
        f"- Passed: `{summary.get('passed')}`",
        f"- Export review safe candidate: `{summary.get('export_review_safe_candidate')}`",
        f"- Generation template candidate: `{summary.get('generation_template_candidate')}`",
        "",
        "## Checks",
        "",
    ]
    for check in summary.get("checks", []) or []:
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- {mark}: `{check.get('name')}` detail=`{check.get('detail')}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Extract abstract motion primitives from curated candidates."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.motion_primitives import MotionPrimitive, normalize_family, normalize_subtype
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


ALLOWED_COWGIRL_CATEGORIES = {
    "semantic_cowgirl_generation_safe",
    "semantic_cowgirl_core_soft_fail_generation_safe",
}


def extract_cowgirl_motion_primitives_v0(
    candidate_db: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    relative_index: str | Path,
    out_jsonl: str | Path,
    out_report: str | Path,
) -> list[dict[str, Any]]:
    rel = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    traj = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    index = {r.get("window_id"): r for r in load_jsonl(relative_index) if r.get("window_id")}
    rows: list[dict[str, Any]] = []
    for candidate in load_jsonl(candidate_db):
        if not _candidate_allowed(candidate):
            continue
        primitive = _primitive_from_candidate(candidate, rel.get(candidate.get("window_id"), {}), traj.get(candidate.get("window_id"), {}), index.get(candidate.get("window_id"), {}))
        rows.append(primitive.to_dict())
    rows.sort(key=lambda r: (-float(r.get("generation_parameters", {}).get("source_generation_score") or 0.0), r.get("primitive_id")))
    write_jsonl(out_jsonl, rows)
    _write_report(rows, out_report)
    return rows


def _candidate_allowed(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("category") in ALLOWED_COWGIRL_CATEGORIES
        and candidate.get("generation_safe") is True
        and candidate.get("semantic_family") == "cowgirl"
        and not candidate.get("excluded_from_cowgirl")
    )


def _primitive_from_candidate(candidate: dict[str, Any], relative: dict[str, Any], trajectory: dict[str, Any], index: dict[str, Any]) -> MotionPrimitive:
    wid = str(candidate.get("window_id"))
    rel_values = relative.get("feature_values", {}) or {}
    traj_values = trajectory.get("feature_values", {}) or {}
    trajectory_shape = trajectory.get("trajectory_shape_classification") or candidate.get("trajectory_shape") or "unknown"
    subtype = normalize_subtype(candidate.get("cowgirl_subtype"), trajectory_shape)
    amplitude = {
        "vertical": _num(rel_values.get("relative_pelvis_vertical_amplitude")),
        "forward_back": _num(rel_values.get("relative_pelvis_forward_back_amplitude")),
        "lateral": _num(rel_values.get("relative_pelvis_lateral_amplitude")),
        "local_path_length": _num(rel_values.get("local_path_length")),
        "local_motion_energy": _num(rel_values.get("local_motion_energy")),
    }
    rhythm = {
        "tempo_proxy": _num(rel_values.get("local_velocity_mean")),
        "velocity_max": _num(rel_values.get("local_velocity_max")),
        "regularity": _num(rel_values.get("local_rhythm_regularity")),
        "cycle_count_estimate": _num(traj_values.get("cycle_count_estimate")),
        "rhythm_repeat_score": _num(traj_values.get("rhythm_repeat_score")),
    }
    controllers = [str(c) for c in (index.get("controllers") or index.get("controller_names") or [])]
    controller_roles = _controller_roles(controllers)
    warnings = [
        "MotionPrimitive is an abstract relative pattern, not a Timeline clip.",
        "No absolute world-space controller positions are stored as generation targets.",
    ]
    if candidate.get("category") == "semantic_cowgirl_core_soft_fail_generation_safe":
        warnings.append("Core gate was a soft-fail accepted by other motion/pose evidence; generation must validate anchors again.")
    return MotionPrimitive(
        primitive_id=f"cowgirl_primitive_v0::{wid}",
        semantic_family=normalize_family(candidate.get("semantic_family")),
        subtype=subtype,
        source_window_ids=[wid],
        source_candidate_ids=[str(candidate.get("candidate_id") or wid)],
        learned_from_dataset="cowgirl_candidate_db_v3",
        duration_seconds=float(candidate.get("duration_seconds") or 0.0),
        relative_motion_summary={
            "coordinate_space": "relative_body_motion",
            "safe_for_learning": bool(candidate.get("safe_for_learning", True)),
            "relative_feature_window_id": relative.get("window_id"),
            "root_world_motion_removed": bool(rel_values.get("root_world_motion_removed") or True),
        },
        trajectory_shape={
            "classification": trajectory_shape,
            "oval_path_score": _num(traj_values.get("oval_path_score")),
            "ellipse_fit_score": _num(traj_values.get("ellipse_fit_score")),
            "closed_loop_ratio": _num(traj_values.get("closed_loop_ratio")),
            "linearity_score": _num(traj_values.get("linearity_score")),
            "dominant_motion_plane": trajectory.get("dominant_motion_plane"),
        },
        rhythm_profile=rhythm,
        amplitude_profile=amplitude,
        controller_role_map=controller_roles,
        anchor_requirements={
            "required": ["hip_or_pelvis_driver", "torso_reference", "lower_body_anchors"],
            "source_pose_anchor_status": candidate.get("pose_anchor_status"),
            "core_gate_status": candidate.get("core_gate_status"),
            "core_gate_can_be_overridden": bool(candidate.get("core_gate_can_be_overridden")),
        },
        safety_requirements={
            "no_world_coords": True,
            "no_person_root_tracks": True,
            "anchor_completeness_required": True,
            "controller_validity_required": True,
            "timeline_clip_stitching_allowed": False,
        },
        generation_parameters={
            "source_generation_score": candidate.get("generation_candidate_score"),
            "semantic_cowgirl_score": candidate.get("semantic_cowgirl_score"),
            "clean_motion_score": candidate.get("clean_motion_score"),
            "suggested_duration_seconds": candidate.get("duration_seconds"),
            "amplitude_units": "normalized_relative_controller_delta",
        },
        warnings=warnings,
    )


def _controller_roles(controllers: list[str]) -> dict[str, Any]:
    lower = {name.lower(): name for name in controllers}
    def names(tokens: list[str]) -> list[str]:
        return sorted(original for text, original in lower.items() if any(token in text for token in tokens))
    return {
        "driver_controllers": names(["hipcontrol", "pelviscontrol", "abdomencontrol"]),
        "anchor_controllers": names(["footcontrol", "kneecontrol", "thighcontrol"]),
        "follower_controllers": names(["chestcontrol", "headcontrol", "handcontrol"]),
        "all_source_controllers_seen": controllers,
    }


def _write_report(rows: list[dict[str, Any]], out_report: str | Path) -> None:
    target = Path(out_report)
    target.parent.mkdir(parents=True, exist_ok=True)
    subtypes = Counter(r.get("subtype") for r in rows)
    shapes = Counter((r.get("trajectory_shape") or {}).get("classification") for r in rows)
    lines = [
        "# Cowgirl Motion Primitives V0 Report",
        "",
        "These primitives are abstract relative motion patterns. They are not Timeline clips and not final generation output.",
        "",
        f"- Primitive count: {len(rows)}",
        "- Source categories: `semantic_cowgirl_generation_safe`, `semantic_cowgirl_core_soft_fail_generation_safe` when generation-safe",
        "- Absolute/world coordinates stored as generation targets: no",
        "",
        "## Subtypes",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in subtypes.most_common()) if subtypes else lines.append("- None")
    lines.extend(["", "## Trajectory Shapes", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in shapes.most_common()) if shapes else lines.append("- None")
    lines.extend(["", "## Example Primitives", ""])
    for row in rows[:10]:
        amp = row.get("amplitude_profile", {})
        lines.append(f"- `{row.get('primitive_id')}` subtype={row.get('subtype')} shape={(row.get('trajectory_shape') or {}).get('classification')} amp(v/f/l)={amp.get('vertical')}/{amp.get('forward_back')}/{amp.get('lateral')}")
    if not rows:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        return round(float(value or 0.0), 6)
    except Exception:
        return 0.0

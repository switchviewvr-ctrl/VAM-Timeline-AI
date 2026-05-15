"""Pose anchor controller completeness audit.

Cowgirl/kneeling poses can be semantically correct even when feet or knees do
not move much.  Those static controllers are still pose-critical anchors for
review exports and any future generation template.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


HIP_PARTS = {"hip", "pelvis", "abdomen"}
CHEST_PARTS = {"chest", "abdomen"}
FOOT_PARTS = {"left_foot", "right_foot"}
KNEE_PARTS = {"left_knee", "right_knee"}
THIGH_PARTS = {"left_thigh", "right_thigh"}
HAND_PARTS = {"left_hand", "right_hand"}
REQUIRED_COWGIRL_ANCHORS = ["hip_or_pelvis", "chest_or_abdomen", "left_knee", "right_knee", "left_foot", "right_foot"]


def audit_pose_anchor_completeness(
    run_dir: str | Path,
    relative_index: str | Path,
    sample_index: str | Path,
    controller_map: str | Path,
    body_quality: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    rows = [
        pose_anchor_completeness_for_window(row, samples.get(row.get("sample_id"), {}), body.get(row.get("window_id"), {}))
        for row in load_jsonl(relative_index)
    ]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def pose_anchor_completeness_for_window(relative_row: dict[str, Any], sample: dict[str, Any] | None = None, body_quality: dict[str, Any] | None = None) -> dict[str, Any]:
    sample = sample or {}
    body_quality = body_quality or {}
    controllers = [str(c) for c in (relative_row.get("controllers") or sample.get("controller_names") or [])]
    bodyparts = [_normalize_part(x) for x in (relative_row.get("bodyparts") or _parts_from_names(controllers))]
    part_to_controller = _part_to_controller(bodyparts, controllers)
    hip_present = bool(HIP_PARTS & set(bodyparts))
    chest_present = bool(CHEST_PARTS & set(bodyparts))
    left_foot = "left_foot" in bodyparts
    right_foot = "right_foot" in bodyparts
    left_knee = "left_knee" in bodyparts
    right_knee = "right_knee" in bodyparts
    left_thigh = "left_thigh" in bodyparts
    right_thigh = "right_thigh" in bodyparts
    hands = bool(HAND_PARTS & set(bodyparts))
    profile = _anchor_profile(relative_row, body_quality, bodyparts)
    required = list(REQUIRED_COWGIRL_ANCHORS if profile.startswith("cowgirl") else ["hip_or_pelvis", "chest_or_abdomen"])
    present_names = []
    missing = []
    checks = {
        "hip_or_pelvis": hip_present,
        "chest_or_abdomen": chest_present,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_foot": left_foot,
        "right_foot": right_foot,
    }
    for key in required:
        if checks.get(key):
            present_names.extend(_controllers_for_anchor(key, part_to_controller))
        else:
            missing.append(key)
    foot_present = left_foot and right_foot
    knee_present = left_knee and right_knee
    thigh_present: bool | str = True if left_thigh and right_thigh else ("unknown" if not (left_thigh or right_thigh) else False)
    lower_complete = bool(hip_present and foot_present and knee_present)
    score = len([x for x in required if checks.get(x)]) / max(len(required), 1)
    if not controllers and not bodyparts:
        status = "unknown"
        safe: bool | str = "unknown"
    elif score >= 0.99:
        status = "complete"
        safe = True
    elif score >= 0.50:
        status = "partial"
        safe = False
    else:
        status = "incomplete"
        safe = False
    moving = _moving_controllers(relative_row, body_quality, controllers)
    static_available = [c for c in present_names if c not in moving]
    warnings = []
    if not foot_present:
        warnings.append("Missing foot anchors; feet may twist or drift in review/generation export.")
    if not knee_present:
        warnings.append("Missing knee anchors; lower-body pose is incomplete.")
    if status != "complete":
        warnings.append("Pose anchor set is not complete for a generation-safe Cowgirl candidate.")
    return {
        "window_id": relative_row.get("window_id"),
        "sample_id": relative_row.get("sample_id"),
        "source_id": relative_row.get("source_id"),
        "source_scene_file": relative_row.get("source_scene_file"),
        "technical_atom_id": relative_row.get("technical_atom_id"),
        "required_anchor_profile": profile,
        "required_anchor_controllers": required,
        "present_anchor_controllers": _dedupe(present_names),
        "missing_anchor_controllers": missing,
        "missing_required_anchor_controllers": missing,
        "moving_controllers": moving,
        "static_anchor_controllers_available": static_available,
        "pose_anchor_completeness_score": round(float(score), 6),
        "lower_body_anchor_complete": lower_complete,
        "foot_controllers_present": foot_present,
        "knee_controllers_present": knee_present,
        "thigh_controllers_present": thigh_present,
        "hip_or_pelvis_present": hip_present,
        "chest_or_abdomen_present": chest_present,
        "hand_support_controllers_present": True if hands else "unknown",
        "generation_pose_anchor_status": status,
        "generation_pose_anchor_safe": safe,
        "missing_foot_controllers": not foot_present,
        "missing_knee_controllers": not knee_present,
        "pose_anchor_incomplete": status in {"partial", "incomplete"},
        "pose_anchor_controllers_present": status == "complete",
        "pose_anchor_controllers_missing": status in {"partial", "incomplete"},
        "warnings": warnings,
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def pose_anchor_completeness_for_parts(bodyparts: list[str], controllers: list[str] | None = None, profile: str = "cowgirl_kneeling") -> dict[str, Any]:
    row = {
        "window_id": "synthetic",
        "controllers": controllers or bodyparts,
        "bodyparts": bodyparts,
        "required_anchor_profile": profile,
    }
    return pose_anchor_completeness_for_window(row, {}, {})


def _anchor_profile(relative_row: dict[str, Any], body_quality: dict[str, Any], bodyparts: list[str]) -> str:
    explicit = relative_row.get("required_anchor_profile")
    if explicit:
        return str(explicit)
    parts = set(bodyparts)
    if {"left_foot", "right_foot", "left_knee", "right_knee"} & parts:
        return "cowgirl_kneeling"
    if {"left_knee", "right_knee"} & parts:
        return "cowgirl_squat"
    if body_quality.get("body_motion_quality") in {"good_body_motion", "partial_body_motion"}:
        return "generic_body_motion"
    return "cowgirl_unknown"


def _part_to_controller(bodyparts: list[str], controllers: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for idx, part in enumerate(bodyparts):
        name = controllers[idx] if idx < len(controllers) else part
        out.setdefault(part, []).append(name)
    return out


def _controllers_for_anchor(anchor: str, part_to_controller: dict[str, list[str]]) -> list[str]:
    if anchor == "hip_or_pelvis":
        parts = HIP_PARTS
    elif anchor == "chest_or_abdomen":
        parts = CHEST_PARTS
    else:
        parts = {anchor}
    out = []
    for part in parts:
        out.extend(part_to_controller.get(part, []))
    return out


def _moving_controllers(relative_row: dict[str, Any], body_quality: dict[str, Any], controllers: list[str]) -> list[str]:
    moving = relative_row.get("moving_controllers")
    if isinstance(moving, list):
        return [str(x) for x in moving]
    count = int(relative_row.get("moving_controller_count_relative") or body_quality.get("moving_controller_count") or 0)
    return controllers[: max(0, min(count, len(controllers)))]


def _parts_from_names(names: list[str]) -> list[str]:
    return [_normalize_part(name) for name in names]


def _normalize_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("control", "").replace("_", "").replace(" ", "")
    aliases = {
        "hip": "hip",
        "pelvis": "pelvis",
        "abdomen": "abdomen",
        "chest": "chest",
        "head": "head",
        "lfoot": "left_foot",
        "leftfoot": "left_foot",
        "rfoot": "right_foot",
        "rightfoot": "right_foot",
        "lknee": "left_knee",
        "leftknee": "left_knee",
        "rknee": "right_knee",
        "rightknee": "right_knee",
        "lthigh": "left_thigh",
        "leftthigh": "left_thigh",
        "rthigh": "right_thigh",
        "rightthigh": "right_thigh",
        "lhand": "left_hand",
        "lefthand": "left_hand",
        "rhand": "right_hand",
        "righthand": "right_hand",
    }
    return aliases.get(text, str(value or "").strip().lower())


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(r.get("generation_pose_anchor_status") for r in rows)
    foot_missing = sum(1 for r in rows if r.get("missing_foot_controllers"))
    knee_missing = sum(1 for r in rows if r.get("missing_knee_controllers"))
    lower_complete = sum(1 for r in rows if r.get("lower_body_anchor_complete") is True)
    safe = sum(1 for r in rows if r.get("generation_pose_anchor_safe") is True)
    lines = [
        "# Pose Anchor Controller Completeness Report",
        "",
        "Static feet/knees can be pose-critical anchors. This audit is generation/export safety, not semantic truth.",
        "",
        f"- Windows audited: {len(rows)}",
        f"- Foot missing count: {foot_missing}",
        f"- Knee missing count: {knee_missing}",
        f"- Lower-body anchor complete count: {lower_complete}",
        f"- Generation pose anchor safe count: {safe}",
        "",
        "## Anchor Status",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in status_counts.most_common()) if status_counts else lines.append("- None")
    lines.extend(["", "## Anchor-Incomplete Examples", ""])
    examples = [r for r in rows if r.get("pose_anchor_incomplete")]
    for row in examples[:25]:
        lines.append(
            f"- `{row.get('window_id')}` status=`{row.get('generation_pose_anchor_status')}` "
            f"missing={row.get('missing_required_anchor_controllers')} scene=`{row.get('source_scene_file')}`"
        )
    if not examples:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Group abstract motion primitives into coarse generation families."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.motion_primitives import MotionPrimitiveSet
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl


GROUP_ORDER = [
    "cowgirl_oval_grind",
    "cowgirl_circular_grind",
    "cowgirl_forward_back_rock",
    "cowgirl_vertical_bounce",
    "cowgirl_riding_general",
    "cowgirl_transition_or_context",
]


def group_cowgirl_motion_primitives_v0(primitives: str | Path, out_json: str | Path, report: str | Path) -> dict[str, Any]:
    rows = load_jsonl(primitives)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_group_key(row)].append(row)
    groups = []
    for key in GROUP_ORDER:
        groups.append(_make_group(key, buckets.get(key, [])))
    extra_keys = sorted(set(buckets) - set(GROUP_ORDER))
    for key in extra_keys:
        groups.append(_make_group(key, buckets[key]))
    data = {
        "schema": "cowgirl_motion_primitive_groups_v0",
        "is_timeline_clip_library": False,
        "groups": [g.to_dict() for g in groups],
    }
    dump_json(out_json, data)
    _write_report(groups, report)
    return data


def _group_key(row: dict[str, Any]) -> str:
    subtype = str(row.get("subtype") or "unknown")
    shape = str((row.get("trajectory_shape") or {}).get("classification") or "").lower()
    if subtype == "oval_grind" or "oval" in shape or "ellipse" in shape:
        return "cowgirl_oval_grind"
    if subtype == "circular_grind" or "circular" in shape:
        return "cowgirl_circular_grind"
    if subtype == "forward_back_rock" or "forward" in shape:
        return "cowgirl_forward_back_rock"
    if subtype == "vertical_bounce" or "bounce" in shape:
        return "cowgirl_vertical_bounce"
    if subtype in {"riding", "grinding"}:
        return "cowgirl_riding_general" if subtype == "riding" else "cowgirl_oval_grind"
    if subtype in {"intro_align", "transition", "hold"}:
        return "cowgirl_transition_or_context"
    return "cowgirl_riding_general"


def _make_group(key: str, rows: list[dict[str, Any]]) -> MotionPrimitiveSet:
    amp = [r.get("amplitude_profile", {}) or {} for r in rows]
    rhythm = [r.get("rhythm_profile", {}) or {} for r in rows]
    duration = [float(r.get("duration_seconds") or 0.0) for r in rows]
    roles = Counter()
    for row in rows:
        role_map = row.get("controller_role_map", {}) or {}
        for role in ["driver_controllers", "anchor_controllers", "follower_controllers"]:
            if role_map.get(role):
                roles[role] += 1
    subtype = key.replace("cowgirl_", "")
    return MotionPrimitiveSet(
        primitive_set_id=key,
        family="cowgirl",
        subtype=subtype,
        primitives=[str(r.get("primitive_id")) for r in rows],
        cluster_summary={
            "count": len(rows),
            "average_duration_seconds": _avg(duration),
            "trajectory_shapes": dict(Counter((r.get("trajectory_shape") or {}).get("classification") for r in rows)),
            "typical_controller_roles": dict(roles),
        },
        variation_ranges={
            "vertical_amplitude": _range([a.get("vertical") for a in amp]),
            "forward_back_amplitude": _range([a.get("forward_back") for a in amp]),
            "lateral_amplitude": _range([a.get("lateral") for a in amp]),
            "tempo_proxy": _range([r.get("tempo_proxy") for r in rhythm]),
            "regularity": _range([r.get("regularity") for r in rhythm]),
        },
        recommended_generation_use=(
            "clean_motion_seed" if rows and key != "cowgirl_transition_or_context" else "excluded_from_clean_motion_or_insufficient_examples"
        ),
        warnings=["Primitive group is statistical/abstract; do not stitch source clips."],
    )


def _write_report(groups: list[MotionPrimitiveSet], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cowgirl Motion Primitive Groups V0 Report",
        "",
        "Groups summarize abstract relative motion primitives. They are not clip bins for Timeline stitching.",
        "",
    ]
    for group in groups:
        data = group.to_dict()
        lines.extend([
            f"## {group.primitive_set_id}",
            "",
            f"- Count: {data['cluster_summary']['count']}",
            f"- Average duration: {data['cluster_summary']['average_duration_seconds']}",
            f"- Use: {group.recommended_generation_use}",
            f"- Variation ranges: `{data['variation_ranges']}`",
            f"- Example primitive IDs: `{group.primitives[:5]}`",
            "",
        ])
    target.write_text("\n".join(lines), encoding="utf-8")


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _range(values: list[Any]) -> dict[str, float | None]:
    nums = []
    for value in values:
        try:
            nums.append(float(value or 0.0))
        except Exception:
            pass
    if not nums:
        return {"min": None, "max": None}
    return {"min": round(min(nums), 6), "max": round(max(nums), 6)}

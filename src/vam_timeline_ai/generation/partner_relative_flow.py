"""Partner-relative flow synthesis prototype.

This generates review/prototype relative tracks in a partner-local frame. It
does not export Timeline and does not use source-scene world coordinates.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json, load_json


def synthesize_partner_relative_flow_v0(
    plan: str | Path | dict[str, Any],
    primitive_groups: str | Path | dict[str, Any],
    baseline: str | Path | dict[str, Any],
    out_json: str | Path,
    report: str | Path,
    duration: float = 4.0,
    fps: float = 60.0,
) -> dict[str, Any]:
    plan_data = load_json(plan) if not isinstance(plan, dict) else plan
    group_data = load_json(primitive_groups) if not isinstance(primitive_groups, dict) else primitive_groups
    baseline_data = load_json(baseline) if not isinstance(baseline, dict) else baseline
    phase = (plan_data.get("sequence") or [{}])[0]
    interaction = phase.get("interaction") or {}
    support = interaction.get("support_mode") or "hands_free"
    frame_count = max(2, int(round(duration * fps)) + 1)
    times = [round(i / fps, 6) for i in range(frame_count)]
    selected_group = _select_group(plan_data, group_data)
    pelvis = _oval_path(times, duration)
    chest = [[round(x * 0.25, 6), round(y * 0.2, 6), round(z * 0.25 - 0.03, 6)] for x, y, z in pelvis]
    tracks = [
        _track("pelvisControl", "driver", "partner_pelvis_local", times, pelvis),
        _track("chestControl", "follower", "partner_pelvis_local", times, chest),
    ]
    zero = [[0.0, 0.0, 0.0] for _ in times]
    for name in ["lKneeControl", "rKneeControl", "lFootControl", "rFootControl"]:
        tracks.append(_track(name, "anchor", "rider_baseline_local", times, zero))
    if support == "hands_on_partner_chest":
        for name in ["lHandControl", "rHandControl"]:
            tracks.append(_track(name, "support", "partner_chest_target", times, zero, {"target": "partner.chest"}))
    flow = {
        "flow_id": f"partner_relative_flow_v0::{uuid.uuid4().hex[:12]}",
        "semantic_plan": plan_data,
        "selected_primitive_group": selected_group,
        "baseline_pose_id": baseline_data.get("baseline_id"),
        "partner_references": baseline_data.get("partner_references") or {},
        "duration_seconds": duration,
        "fps": fps,
        "coordinate_space": "partner_relative_motion",
        "interaction_frame": interaction.get("coordinate_frame") or "partner_pelvis_local",
        "support_mode": support,
        "controller_tracks": tracks,
        "generated_from_relative_primitives": True,
        "source_world_coords_used": False,
        "person_root_tracks_included": False,
        "clip_stitching_used": False,
        "export_ready": False,
        "warnings": [
            "Partner-relative flow v0 is a review prototype, not final Timeline generation.",
            "Contact/support constraints must pass validation before native export.",
        ],
    }
    dump_json(out_json, flow)
    _write_report(flow, report)
    return flow


def _select_group(plan: dict[str, Any], group_data: dict[str, Any]) -> str:
    subtype = str((plan.get("requested_subtypes") or [""])[0]).lower()
    groups = {g.get("primitive_set_id"): g for g in (group_data.get("groups") or [])}
    if "grind" in subtype and "cowgirl_oval_grind" in groups:
        return "cowgirl_oval_grind"
    if "bounce" in subtype and "cowgirl_vertical_bounce" in groups:
        return "cowgirl_vertical_bounce"
    return next(iter(groups), "cowgirl_riding_general")


def _oval_path(times: list[float], duration: float) -> list[list[float]]:
    cycles = max(1.0, duration / 4.0)
    out: list[list[float]] = []
    for t in times:
        phase = 2.0 * math.pi * cycles * (t / max(duration, 0.001))
        lateral = 0.035 * math.sin(phase)
        vertical = 0.025 * (1.0 - math.cos(phase))
        forward = 0.06 * math.cos(phase)
        out.append([round(lateral, 6), round(vertical, 6), round(forward, 6)])
    return out


def _track(name: str, role: str, frame: str, times: list[float], deltas: list[list[float]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "controller_name": name,
        "role": role,
        "coordinate_space": frame,
        "times": times,
        "position_deltas": deltas,
        "rotation_deltas": None,
        "source_world_coords_used": False,
    }
    if extra:
        data.update(extra)
    return data


def _write_report(flow: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Partner Relative Flow V0 Report",
        "",
        "This is a generated partner-relative motion flow for inspection. It is not a Timeline export.",
        "",
        f"- Flow: `{flow.get('flow_id')}`",
        f"- Selected primitive group: `{flow.get('selected_primitive_group')}`",
        f"- Interaction frame: `{flow.get('interaction_frame')}`",
        f"- Support mode: `{flow.get('support_mode')}`",
        f"- Controller tracks: {len(flow.get('controller_tracks') or [])}",
        "- Source world coordinates used: false",
        "- Person/root tracks included: false",
        "- Clip stitching used: false",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

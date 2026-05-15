"""Skeleton for future generated relative motion flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

from vam_timeline_ai.io.json_utils import dump_json, load_json


def generate_motion_flow_skeleton_v0(plan: str | Path, retrieved_primitives: str | Path, out: str | Path, report: str | Path) -> dict[str, Any]:
    plan_data = load_json(plan)
    retrieved = load_json(retrieved_primitives)
    primitive_ids = []
    for match in retrieved.get("matches", []) or []:
        primitive_ids.extend(match.get("candidate_primitive_ids", [])[:5])
    flow = {
        "flow_id": f"flow_skeleton_v0::{uuid.uuid4().hex[:12]}",
        "semantic_plan": plan_data,
        "primitive_references": primitive_ids,
        "generated_relative_controller_tracks": _track_placeholders(plan_data),
        "generation_method": "placeholder",
        "coordinate_space": "relative_body_motion",
        "export_ready": False,
        "timeline_export_performed": False,
        "clip_stitching_performed": False,
        "warnings": [
            "Skeleton only. No final controller curves have been generated.",
            "No raw Timeline coordinates were copied.",
            "Future work must synthesize relative tracks, retarget to current pose, validate anchors, then export Timeline.",
        ],
    }
    dump_json(out, flow)
    _write_report(flow, report)
    return flow


def _track_placeholders(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = []
    for phase in plan.get("sequence", []) or []:
        query = phase.get("primitive_query", {}) or {}
        tracks.append({
            "phase_id": phase.get("phase_id"),
            "controllers": ["hipControl", "pelvis_or_abdomen_reference", "chestControl", "knee_foot_anchors"],
            "track_type": "relative_delta_placeholder",
            "intended_subtype": query.get("subtype"),
            "duration_seconds": phase.get("duration"),
            "requires_future_synthesis": True,
        })
    return tracks


def _write_report(flow: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated Motion Flow Skeleton V0 Report",
        "",
        "This is a skeleton for future relative motion generation. It is not Timeline output.",
        "",
        f"- Flow: `{flow.get('flow_id')}`",
        f"- Coordinate space: `{flow.get('coordinate_space')}`",
        f"- Export ready: `{flow.get('export_ready')}`",
        f"- Primitive references: `{flow.get('primitive_references')}`",
        f"- Placeholder tracks: `{flow.get('generated_relative_controller_tracks')}`",
        "",
        "## Missing Pieces",
        "",
        "- Relative curve synthesis from primitive parameters",
        "- Retargeting to the current VaM pose",
        "- Anchor/contact constraint solving",
        "- Controller validity validation after synthesis",
        "- Timeline export safety validation",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

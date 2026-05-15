"""One-command first generated motion review pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.baseline_pose import create_synthetic_baseline_pose_v0
from vam_timeline_ai.generation.baseline_pose import create_cowgirl_review_baseline_pose_v1
from vam_timeline_ai.generation.generated_motion_validation import validate_generated_motion_flow_v0
from vam_timeline_ai.generation.motion_flow_synthesis import synthesize_motion_flow_v0
from vam_timeline_ai.generation.motion_flow_synthesis import synthesize_motion_flow_v1
from vam_timeline_ai.generation.relative_flow_retargeter import retarget_motion_flow_v0
from vam_timeline_ai.generation.relative_flow_retargeter import retarget_motion_flow_v1
from vam_timeline_ai.generation.retarget_validation import validate_retargeted_motion_flow_v0
from vam_timeline_ai.generation.retarget_validation import validate_retargeted_motion_flow_v1
from vam_timeline_ai.generation.timeline_from_retargeted_flow import export_retargeted_flow_timeline_v0
from vam_timeline_ai.generation.review_player_export import prepare_vam_review_player_v1
from vam_timeline_ai.io.json_utils import dump_json
from vam_timeline_ai.visualization.generated_motion_preview import render_generated_motion_preview_v0
from vam_timeline_ai.visualization.retargeted_motion_preview import render_retargeted_motion_preview_v0
from vam_timeline_ai.visualization.retargeted_motion_preview import render_retargeted_motion_preview_v1


def run_first_generated_motion_review_v0(
    plan: str | Path,
    primitive_groups: str | Path,
    primitives: str | Path,
    out_dir: str | Path,
    duration: float = 4.0,
    fps: float = 60.0,
    seed: int = 42,
) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    flow_json = target / "generated_motion_flow_v0.json"
    flow_npz = target / "generated_motion_flow_v0.npz"
    flow_report = target / "generated_motion_flow_v0_report.md"
    flow_validation = target / "generated_motion_flow_v0_validation.md"
    baseline_json = target / "synthetic_baseline_pose_v0.json"
    retarget_json = target / "retargeted_motion_flow_v0.json"
    retarget_npz = target / "retargeted_motion_flow_v0.npz"
    retarget_report = target / "retargeted_motion_flow_v0_report.md"
    retarget_validation = target / "retargeted_motion_flow_v0_validation.md"
    generated_preview = target / "preview_generated_motion_v0"
    retarget_preview = target / "preview_retargeted_motion_v0"
    timeline_dir = target / "timeline_export_v0"

    flow = synthesize_motion_flow_v0(plan, primitive_groups, primitives, flow_json, flow_npz, flow_report, duration, fps, seed)
    generated_validation = validate_generated_motion_flow_v0(flow_json, flow_validation)
    baseline = create_synthetic_baseline_pose_v0(baseline_json)
    retargeted = retarget_motion_flow_v0(flow_json, baseline_json, retarget_json, retarget_npz, retarget_report)
    retargeted_validation = validate_retargeted_motion_flow_v0(retarget_json, retarget_validation)
    generated_preview_manifest = render_generated_motion_preview_v0(flow_json, generated_preview)
    retarget_preview_manifest = render_retargeted_motion_preview_v0(retarget_json, retarget_preview)
    timeline_export = export_retargeted_flow_timeline_v0(retarget_json, retarget_validation, timeline_dir)
    summary = {
        "schema": "first_generated_motion_review_v0",
        "out_dir": str(target),
        "generated_flow": str(flow_json),
        "generated_flow_validation_passed": generated_validation.get("passed"),
        "baseline_pose": str(baseline_json),
        "retargeted_flow": str(retarget_json),
        "retarget_validation_passed": retargeted_validation.get("passed"),
        "generated_preview": str(generated_preview),
        "retargeted_preview": str(retarget_preview),
        "timeline_export": timeline_export,
        "selected_primitive_group": flow.get("selected_primitive_group"),
        "retargeted_controller_count": len(retargeted.get("controller_tracks", []) or []),
        "baseline_controller_count": len(baseline.get("controller_poses", []) or []),
        "no_source_world_coords_used": True,
        "clip_stitching_used": False,
        "ml_training_run": False,
        "warnings": [
            "First generated motion review is a prototype.",
            "Timeline export, if written, is review-only and not production-ready.",
        ],
        "preview_manifests": {
            "generated": generated_preview_manifest,
            "retargeted": retarget_preview_manifest,
        },
    }
    dump_json(target / "first_generated_motion_review_v0_summary.json", summary)
    _write_report(summary, target / "first_generated_motion_review_v0_report.md")
    return summary


def run_cowgirl_motion_flow_v1_review(
    plan: str | Path,
    primitive_groups: str | Path,
    primitives: str | Path,
    out_dir: str | Path,
    duration: float = 4.0,
    fps: float = 60.0,
    seed: int = 42,
) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    baseline = target / "cowgirl_review_baseline_pose_v1.json"
    flow_json = target / "generated_motion_flow_v1.json"
    flow_npz = target / "generated_motion_flow_v1.npz"
    flow_report = target / "generated_motion_flow_v1_report.md"
    retarget_json = target / "retargeted_motion_flow_v1.json"
    retarget_npz = target / "retargeted_motion_flow_v1.npz"
    retarget_report = target / "retargeted_motion_flow_v1_report.md"
    validation = target / "retargeted_motion_flow_v1_validation.md"
    preview = target / "preview_retargeted_motion_v1"
    player_dir = target / "vam_review_player"
    create_cowgirl_review_baseline_pose_v1(baseline, "kneeling_forward")
    flow = synthesize_motion_flow_v1(
        plan, primitive_groups, primitives, "cowgirl_oval_grind_v1",
        flow_json, flow_npz, flow_report,
        duration=duration, fps=fps, seed=seed, tempo="slow",
        vertical_scale=1.25, lateral_scale=0.70, forward_back_scale=1.0, chest_follower_scale=0.35,
    )
    retargeted = retarget_motion_flow_v1(flow_json, baseline, retarget_json, retarget_npz, retarget_report)
    valid = validate_retargeted_motion_flow_v1(retarget_json, validation)
    preview_manifest = render_retargeted_motion_preview_v1(retarget_json, preview)
    player = prepare_vam_review_player_v1(retarget_json, player_dir)
    instructions = player_dir / "VAM_REVIEW_PLAYER_V1_INSTRUCTIONS.md"
    summary = {
        "schema": "cowgirl_motion_flow_v1_review",
        "out_dir": str(target),
        "baseline_pose": str(baseline),
        "generated_flow": str(flow_json),
        "retargeted_flow": str(retarget_json),
        "validation": str(validation),
        "validation_passed": valid.get("passed"),
        "preview": str(preview),
        "review_player_json": player.get("review_player_json"),
        "review_player_secure_path": player.get("vam_secure_json_path"),
        "script_copied_to": player.get("script_copied_to"),
        "instructions": str(instructions),
        "controllers": [t.get("controller_name") for t in flow.get("controller_tracks", [])],
        "axis_scales": flow.get("axis_scales"),
        "preview_manifest": preview_manifest,
        "no_source_world_coords_used": True,
        "clip_stitching_used": False,
        "ml_training_run": False,
    }
    dump_json(target / "cowgirl_motion_flow_v1_review_summary.json", summary)
    _write_v1_report(summary, target / "cowgirl_motion_flow_v1_review_report.md")
    return summary


def _write_v1_report(summary: dict[str, Any], report: Path) -> None:
    lines = [
        "# Cowgirl Motion Flow V1 Review",
        "",
        "V1 adds a Cowgirl review baseline, coordinated pelvis/chest/abdomen motion, and review-player axis controls.",
        "",
        f"- Validation passed: `{summary.get('validation_passed')}`",
        f"- Controllers: `{summary.get('controllers')}`",
        f"- Axis scales: `{summary.get('axis_scales')}`",
        f"- Preview: `{summary.get('preview')}`",
        f"- Review player JSON: `{summary.get('review_player_json')}`",
        f"- VaM secure JSON path: `{summary.get('review_player_secure_path')}`",
        f"- C# script copied to: `{summary.get('script_copied_to')}`",
        "",
        "Still review-only. No source world coordinates, no Person/root tracks, no clip stitching, no ML training.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(summary: dict[str, Any], report: Path) -> None:
    lines = [
        "# First Generated Motion Review V0",
        "",
        "This pipeline synthesizes relative curves, retargets them to a synthetic baseline, validates them, renders previews, and attempts a review-only export.",
        "",
        f"- Output directory: `{summary.get('out_dir')}`",
        f"- Generated flow validation passed: `{summary.get('generated_flow_validation_passed')}`",
        f"- Retarget validation passed: `{summary.get('retarget_validation_passed')}`",
        f"- Selected primitive group: `{summary.get('selected_primitive_group')}`",
        f"- Timeline export status: `{(summary.get('timeline_export') or {}).get('status')}`",
        f"- No source world coordinates used: `{summary.get('no_source_world_coords_used')}`",
        f"- Clip stitching used: `{summary.get('clip_stitching_used')}`",
        "",
        "## Paths",
        "",
        f"- Generated flow: `{summary.get('generated_flow')}`",
        f"- Baseline pose: `{summary.get('baseline_pose')}`",
        f"- Retargeted flow: `{summary.get('retargeted_flow')}`",
        f"- Generated preview: `{summary.get('generated_preview')}`",
        f"- Retargeted preview: `{summary.get('retargeted_preview')}`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

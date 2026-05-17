"""Review-flow JSON export from a retargeted generated flow.

This is not native VaM Timeline plugin JSON and is not importable through
Timeline. It remains only as a review-flow JSON artifact for diagnostics or a
future converter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.generation.retarget_validation import validation_markdown_allows_export
from vam_timeline_ai.io.json_utils import dump_json, load_json


def export_retargeted_flow_timeline_v0(retargeted_flow: str | Path, validation: str | Path, out_dir: str | Path) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    if not validation_markdown_allows_export(validation):
        reason = target / "export_unavailable.md"
        reason.write_text(
            "# Retargeted Review Flow Export Unavailable\n\n"
            "Validation did not pass review-flow gating, so no review-flow JSON was written.\n",
            encoding="utf-8",
        )
        return {"status": "export_unavailable", "timeline_json": None, "reason": str(reason)}
    data = load_json(retargeted_flow)
    timeline = {
        "schema": "review_only_retargeted_flow_timeline_v0",
        "review_only": True,
        "production_ready": False,
        "native_timeline_importable": False,
        "recommended_test_method": "VaM Generated Motion Review Player",
        "generated_from_relative_flow": True,
        "source_world_coords_used": False,
        "clip_stitching_used": False,
        "baseline_pose": data.get("baseline_source"),
        "generation_template_candidate": False,
        "warnings": [
            "Review-flow JSON only; not native Timeline JSON and not importable through Timeline.",
            "Use GeneratedMotionReviewPlayer.cs for VaM review playback.",
            "No Person/root/world tracks are included.",
        ],
        "controller_tracks": [
            {
                "controller_name": track.get("controller_name"),
                "bodypart": track.get("bodypart"),
                "times": track.get("times"),
                "positions": track.get("retargeted_positions"),
                "coordinate_space": "retargeted_to_synthetic_baseline_review",
            }
            for track in data.get("controller_tracks", []) or []
        ],
    }
    out_json = target / "review_only_timeline_v0.json"
    dump_json(out_json, timeline)
    (target / "README.md").write_text(
        "# Review-Only Retargeted Flow JSON V0\n\n"
        "This JSON is not native VaM Timeline plugin JSON and is not importable through Timeline. "
        "Use the VaM Generated Motion Review Player or a future converter. It is generated from relative motion "
        "retargeted to a synthetic baseline and does not use source world coordinates.\n",
        encoding="utf-8",
    )
    (target / "review_export_status.md").write_text(
        "# Review Flow JSON Status\n\n"
        "- native_timeline_importable: false\n"
        "- schema: review_only_retargeted_flow_timeline_v0\n"
        "- recommended_test_method: VaM Generated Motion Review Player\n",
        encoding="utf-8",
    )
    return {"status": "review_flow_json_written", "timeline_json": str(out_json), "track_count": len(timeline["controller_tracks"]), "native_timeline_importable": False}

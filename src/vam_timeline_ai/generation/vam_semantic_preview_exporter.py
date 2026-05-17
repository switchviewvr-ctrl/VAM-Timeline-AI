"""Export contact-aware semantic examples as review-only native Timeline clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.generation.native_timeline_exporter import _controller_payload, _timeline_payload
from vam_timeline_ai.generation.stickman_to_vam_preview_mapper import map_stickman_to_vam_preview_v0
from vam_timeline_ai.generation.vam_semantic_preview import is_disallowed_timeline_track
from vam_timeline_ai.io.json_utils import dump_json, load_json, safe_id_for_path, write_jsonl


def export_vam_semantic_preview_v0(
    motion_examples: str | Path,
    out_dir: str | Path,
    *,
    duration: float = 4.0,
    fps: int = 60,
) -> dict[str, Any]:
    target = Path(out_dir)
    clips_dir = target / "clips"
    preview_dir = target / "preview_data"
    reports_dir = target / "reports"
    partner_dir = target / "partner_reference"
    for folder in (clips_dir, preview_dir, reports_dir, partner_dir):
        folder.mkdir(parents=True, exist_ok=True)

    preview_json = preview_dir / "vam_semantic_preview_clips_v0.json"
    map_summary = map_stickman_to_vam_preview_v0(motion_examples, preview_json, duration_seconds=duration, fps=fps)
    preview_data = load_json(preview_json)
    manifest_rows: list[dict[str, Any]] = []
    exported = 0
    blocked = 0

    for clip in preview_data.get("clips") or []:
        clip_id = safe_id_for_path(str(clip.get("clip_id") or "semantic_preview"))
        row = {
            "clip_id": clip_id,
            "family": clip.get("family"),
            "pose_subtype": clip.get("pose_subtype"),
            "motion_subtype": clip.get("motion_subtype"),
            "review_only": True,
            "coordinate_space": "synthetic_review_local",
            "export_status": clip.get("export_status"),
            "timeline_json": None,
            "partner_pelvis_target": (clip.get("target_points") or {}).get("partner_pelvis_target"),
            "alignment_validation": clip.get("alignment_validation") or {},
        }
        if clip.get("export_status") == "exported":
            timeline_path = clips_dir / f"{clip_id}.timeline.json"
            payload = _timeline_payload_for_clip(clip, timeline_path)
            dump_json(timeline_path, payload)
            row["timeline_json"] = str(timeline_path)
            clip["timeline_json"] = str(timeline_path)
            exported += 1
        else:
            blocked += 1
        manifest_rows.append(row)

    preview_data["clips"] = preview_data.get("clips") or []
    dump_json(preview_json, preview_data)
    write_jsonl(target / "manifest.jsonl", manifest_rows)
    _write_partner_reference_files(preview_data, partner_dir)
    _write_import_instructions(target)
    _write_index_md(preview_data, manifest_rows, target / "index.md")
    _write_index_html(preview_data, manifest_rows, target / "index.html")
    _write_export_report(target / "reports" / "vam_semantic_preview_export_v0.md", map_summary, exported, blocked, target)

    return {
        "status": "ok",
        "out_dir": str(target),
        "manifest": str(target / "manifest.jsonl"),
        "preview_data": str(preview_json),
        "clip_count": len(manifest_rows),
        "exported_clips": exported,
        "blocked_clips": blocked,
        "clips_dir": str(clips_dir),
        "import_instructions": str(target / "import_instructions.md"),
    }


def _timeline_payload_for_clip(clip: dict[str, Any], timeline_path: Path) -> dict[str, Any]:
    duration = float(clip.get("duration_seconds") or 4.0)
    controllers = []
    skipped = []
    for track in clip.get("controller_tracks") or []:
        name = str(track.get("controller_name") or "")
        if is_disallowed_timeline_track(name):
            skipped.append({"controller_name": name, "reason": "person_root_world_or_atom_track"})
            continue
        times = np.asarray(track.get("times") or [], dtype=np.float32)
        positions = np.asarray(track.get("positions") or [], dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(times) != len(positions):
            skipped.append({"controller_name": name, "reason": "invalid_position_track"})
            continue
        controllers.append(_controller_payload(name, times, positions, duration, key_stride=1, target_rotation=False))
    animation_name = f"SemanticPreview_{safe_id_for_path(str(clip.get('clip_id') or timeline_path.stem))}"
    metadata = {
        "schema": "vam_semantic_preview_timeline_v0",
        "review_only": True,
        "production_ready": False,
        "timeline_generation_final": False,
        "synthetic_review_local": True,
        "source_world_coords_used": False,
        "old_raw_timeline_curves_copied": False,
        "person_root_tracks_included": False,
        "generated_from": "semantic_motion_examples_v2_contact_aware",
        "coordinate_space": "synthetic_review_local",
        "clip_id": clip.get("clip_id"),
        "family": clip.get("family"),
        "pose_subtype": clip.get("pose_subtype"),
        "motion_subtype": clip.get("motion_subtype"),
        "partner_reference": clip.get("partner_reference"),
        "target_points": clip.get("target_points"),
        "contact_zone": clip.get("contact_zone"),
        "alignment_validation": clip.get("alignment_validation"),
        "controller_roles": {t.get("controller_name"): t.get("role") for t in clip.get("controller_tracks") or []},
        "skipped_tracks": skipped,
        "warning": "Review-only synthetic semantic preview. Manual VaM inspection is required; do not use as production animation.",
    }
    payload = _timeline_payload(animation_name=animation_name, duration=duration, controllers=controllers, metadata=metadata)
    payload["VAMTimelineAISemanticPreviewMetadata"] = metadata
    payload["VAMTimelineAIGeneratedMetadata"] = metadata
    clip_payload = (payload.get("Clips") or [{}])[0]
    clip_payload["AnimationSegment"] = "SemanticPreviewReviewOnly"
    return payload


def _write_partner_reference_files(preview_data: dict[str, Any], out_dir: Path) -> None:
    markers: dict[str, Any] = {}
    poses: dict[str, Any] = {}
    for clip in preview_data.get("clips") or []:
        cid = str(clip.get("clip_id") or "unknown")
        markers[cid] = {
            "family": clip.get("family"),
            "target_points": clip.get("target_points") or {},
            "contact_zone": clip.get("contact_zone") or {},
            "alignment_validation": clip.get("alignment_validation") or {},
        }
        poses[cid] = {
            "partner_reference_points": (clip.get("partner_reference") or {}).get("partner_reference_points") or {},
            "partner_setup_required_in_vam": True,
        }
    dump_json(out_dir / "partner_reference_markers.json", markers)
    dump_json(out_dir / "partner_reference_pose.json", poses)
    (out_dir / "partner_reference_instructions.md").write_text(
        "# Partner Reference Instructions\n\n"
        "The v0 semantic preview clips import onto one actor Person atom. They do not create or animate a partner atom.\n\n"
        "Use `partner_reference_markers.json` and `partner_reference_pose.json` as neutral alignment guides:\n\n"
        "1. Load or create a simple second Person as the partner/reference body if the concept needs one.\n"
        "2. Align the actor/rider pelvis visually to the `partner_pelvis_target` marker for Cowgirl/Reverse Cowgirl concepts.\n"
        "3. For BJ/Oral concepts, check that head/chest motion points toward the partner pelvis target while actor pelvis stays mostly static.\n"
        "4. For Doggy concepts, check the partner-behind relation and front support anchors.\n"
        "5. Do not treat this package as production animation; it is only a semantic reality-check.\n\n"
        "No partner static Timeline clip is exported in v0 because second-Person scene setup is scene-dependent.\n",
        encoding="utf-8",
    )


def _write_import_instructions(out_dir: Path) -> None:
    (out_dir / "import_instructions.md").write_text(
        "# VaM Semantic Preview Import Instructions\n\n"
        "These clips are synthetic review-only Timeline previews. They are rough controller sketches for checking semantic meaning inside VaM.\n\n"
        "1. Open VaM.\n"
        "2. Load a simple scene with a Person atom as the actor/rider.\n"
        "3. Add AcidBubbles Timeline to that Person atom.\n"
        "4. Import one `.timeline.json` file from `clips/`.\n"
        "5. Select the `SemanticPreview_*` animation and play from `t=0`.\n"
        "6. If a partner reference is needed, load or pose a second Person manually and use `partner_reference/` as an alignment guide.\n"
        "7. Evaluate: does the pose resemble the stickman concept, does it float, is the pelvis/head/hand driver correct, are anchors sensible, and is the motion family visually correct?\n"
        "8. Write the result down manually. Do not promote these clips as production animations.\n\n"
        "Known limitations:\n"
        "- VaM IK may interpret these synthetic controller positions differently than the schematic stickman.\n"
        "- The partner is metadata/reference only in v0.\n"
        "- Rotations are identity placeholders; this pass focuses on controller position semantics.\n"
        "- No source scene coordinates or old raw Timeline curves are used.\n",
        encoding="utf-8",
    )


def _write_index_md(preview_data: dict[str, Any], manifest_rows: list[dict[str, Any]], out: Path) -> None:
    lines = [
        "# VaM Semantic Preview Package V0",
        "",
        "Review-only synthetic Timeline clips derived from contact-aware semantic stickman examples.",
        "",
        "- Production-ready: false",
        "- Source scene coordinates used: false",
        "- Person/root/world tracks included: false",
        "- ML training performed: false",
        "",
        "## Clips",
        "",
    ]
    for row in manifest_rows:
        timeline = Path(str(row.get("timeline_json") or "")).name if row.get("timeline_json") else "blocked"
        lines.append(f"- `{row['clip_id']}`: `{timeline}` | family `{row.get('family')}` | status `{row.get('export_status')}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index_html(preview_data: dict[str, Any], manifest_rows: list[dict[str, Any]], out: Path) -> None:
    rows = []
    by_id = {str(row["clip_id"]): row for row in manifest_rows}
    for clip in preview_data.get("clips") or []:
        cid = safe_id_for_path(str(clip.get("clip_id") or "unknown"))
        row = by_id.get(cid, {})
        timeline_name = Path(str(row.get("timeline_json") or "")).name if row.get("timeline_json") else ""
        stickman_gif = f"../semantic_stickman_previews_v3/{cid}/preview.gif"
        stickman_sheet = f"../semantic_stickman_previews_v3/{cid}/contact_sheet.png"
        rows.append(
            "<article class='card'>"
            f"<h2>{cid}</h2>"
            f"<p><b>Family:</b> {clip.get('family')} | <b>Pose:</b> {clip.get('pose_subtype')} | <b>Motion:</b> {clip.get('motion_subtype')}</p>"
            f"<p><b>Status:</b> {row.get('export_status')} | <b>Timeline:</b> <a href='clips/{timeline_name}'>{timeline_name or 'blocked'}</a></p>"
            f"<p><a href='{stickman_gif}'>stickman GIF</a> | <a href='{stickman_sheet}'>stickman contact sheet</a></p>"
            f"<p><b>Partner target:</b> {(clip.get('target_points') or {}).get('partner_pelvis_target', '-')}</p>"
            f"<p><b>Alignment:</b> {(clip.get('alignment_validation') or {}).get('valid', '-')} "
            f"max {(clip.get('alignment_validation') or {}).get('max_distance', '-')} / "
            f"{(clip.get('alignment_validation') or {}).get('target_distance_max', '-')}</p>"
            "<ul>"
            "<li>Does the pose read as the intended semantic family?</li>"
            "<li>Does the primary driver match the concept?</li>"
            "<li>Does the actor float or drift away from the partner target?</li>"
            "<li>Are hands/knees/feet plausible as support?</li>"
            "</ul>"
            "</article>"
        )
    out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>VaM Semantic Preview V0</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f6f8fb;color:#101828}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}"
        ".card{background:white;border:1px solid #d0d7e2;border-radius:8px;padding:16px}"
        "h1{margin-top:0} h2{font-size:18px}</style></head><body>"
        "<h1>VaM Semantic Preview Package V0</h1>"
        "<p>Review-only synthetic Timeline clips. Not production generation.</p>"
        "<p><a href='import_instructions.md'>Import instructions</a> | <a href='manifest.jsonl'>manifest.jsonl</a></p>"
        "<section class='grid'>"
        + "\n".join(rows)
        + "</section></body></html>\n",
        encoding="utf-8",
    )


def _write_export_report(out: Path, map_summary: dict[str, Any], exported: int, blocked: int, target: Path) -> None:
    out.write_text(
        "# VaM Semantic Preview Export V0\n\n"
        f"- Package: `{target}`\n"
        f"- Clips mapped: `{map_summary.get('clip_count')}`\n"
        f"- Timeline clips exported: `{exported}`\n"
        f"- Blocked clips: `{blocked}`\n"
        "- Review-only: `true`\n"
        "- Source scene coordinates used: `false`\n"
        "- Person/root/world tracks included: `false`\n"
        "- Final Timeline generation: `false`\n",
        encoding="utf-8",
    )

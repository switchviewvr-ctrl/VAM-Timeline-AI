"""Build practical VaM review packages for semantic review batches."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import html

import yaml

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl


EVIDENCE_FIELDS = [
    "rider_above_partner_score",
    "pelvis_alignment_score",
    "hands_on_partner_chest_score",
    "hands_on_partner_hips_score",
    "partner_lying_score",
]


def build_vam_review_package(
    review: str | Path,
    run_dir: str | Path,
    source_run: str | Path | None,
    out_dir: str | Path,
    attempt_timeline_segments: bool = True,
) -> dict[str, Any]:
    """Create a human-reviewable VaM package from a semantic review JSONL."""

    review_path = Path(review)
    run = Path(run_dir)
    source = Path(source_run) if source_run else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = _load_review_context(run, source)
    rows = load_jsonl(review_path)
    manifest: list[dict[str, Any]] = []
    export_results: list[dict[str, Any]] = []
    timeline_root = out / "timeline_segments"
    timeline_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        item = _enrich_review_item(row, data)
        segment_dir = timeline_root / item["review_id"]
        export_result = _maybe_export_timeline_segment(item, data, segment_dir, attempt_timeline_segments)
        item["can_export_timeline_segment"] = bool(export_result.get("success"))
        item["timeline_export_status"] = _export_status(export_result)
        item["timeline_export_path"] = export_result.get("timeline_export_path")
        item["timeline_export_warnings"] = export_result.get("warnings", [])
        if item["can_export_timeline_segment"]:
            item["review_method"] = "timeline_segment_import"
        elif item["can_review_in_original_scene"]:
            item["review_method"] = "original_scene_time_range"
        else:
            item["review_method"] = "unavailable"
        manifest.append(item)
        export_results.append(export_result)
        _write_item_folder(out / "items" / item["review_id"], item)

    write_jsonl(out / "vam_review_manifest.jsonl", manifest)
    _write_manifest_csv(out / "vam_review_manifest.csv", manifest)
    _write_scene_list(out / "vam_review_scene_list.md", manifest)
    _write_answer_sheet(out / "vam_review_answer_sheet.yaml", manifest)
    _write_index_html(out / "vam_review_index.html", manifest)
    _write_timeline_status(out / "timeline_export_status.md", manifest, export_results)
    _write_package_instructions(out / "VAM_REVIEW_PACKAGE_INSTRUCTIONS.md", manifest)

    return {
        "status": "ok",
        "review_items": len(manifest),
        "scene_count": len({m.get("source_scene_path") or m.get("source_scene_file") or "unknown" for m in manifest}),
        "timeline_segments_attempted": sum(1 for r in export_results if r.get("attempted")),
        "timeline_segments_successful": sum(1 for r in export_results if r.get("success")),
        "timeline_segments_unavailable": sum(1 for r in export_results if _export_status(r) == "unavailable"),
        "timeline_segments_failed": sum(1 for r in export_results if _export_status(r) == "failed"),
        "out_dir": str(out),
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def _load_review_context(run: Path, source_run: Path | None) -> dict[str, Any]:
    source_runs = [run]
    if source_run and source_run != run:
        source_runs.append(source_run)
    windows = _merge_by_key([r / "semantic" / "movement_windows.jsonl" for r in source_runs], "window_id")
    samples = _merge_by_key([r / "baked" / "motion_sample_index.jsonl" for r in source_runs], "sample_id")
    sources = _merge_by_key([r / "semantic" / "motion_source_index.jsonl" for r in source_runs], "source_id")
    pair_windows = _merge_by_key([r / "semantic" / "pair_windows_v1.jsonl" for r in source_runs], "pair_window_id")
    partner_features = _merge_by_key([r / "interaction_semantics" / "partner_relative_features_v0.jsonl" for r in source_runs], "pair_window_id")
    interaction = _merge_by_key([r / "interaction_semantics" / "interaction_semantics_v0.jsonl" for r in source_runs], "pair_window_id")
    partner_features_by_window = _merge_by_key([r / "interaction_semantics" / "partner_relative_features_v0.jsonl" for r in source_runs], "window_id")
    interaction_by_window = _merge_by_key([r / "interaction_semantics" / "interaction_semantics_v0.jsonl" for r in source_runs], "window_id")
    return {
        "run_dir": run,
        "source_run": source_run,
        "windows": windows,
        "samples": samples,
        "sources": sources,
        "pair_windows": pair_windows,
        "partner_features": partner_features,
        "partner_features_by_window": partner_features_by_window,
        "interaction": interaction,
        "interaction_by_window": interaction_by_window,
    }


def _merge_by_key(paths: list[Path], key: str) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            value = row.get(key)
            if value and value not in merged:
                merged[str(value)] = row
    return merged


def _enrich_review_item(row: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    window = data["windows"].get(row.get("window_id"), {})
    sample = data["samples"].get(window.get("sample_id") or row.get("sample_id"), {})
    source = data["sources"].get(window.get("source_id") or sample.get("source_id") or row.get("source_id"), {})
    pair = data["pair_windows"].get(row.get("pair_window_id"), {})
    partner_features = data["partner_features"].get(row.get("pair_window_id")) or data["partner_features_by_window"].get(row.get("window_id"), {})
    interaction = data["interaction"].get(row.get("pair_window_id")) or data["interaction_by_window"].get(row.get("window_id"), {})

    pose = row.get("pose_semantics") if isinstance(row.get("pose_semantics"), dict) else {}
    motion = row.get("motion_semantics") if isinstance(row.get("motion_semantics"), dict) else {}
    start = _first_not_none(window.get("start_seconds"), row.get("start_seconds"))
    end = _first_not_none(window.get("end_seconds"), row.get("end_seconds"))
    duration = _first_not_none(window.get("duration_seconds"), row.get("duration_seconds"), _duration(start, end))
    source_scene_path = _first_text(window.get("source_scene_path"), sample.get("source_scene_path"), source.get("source_scene_path"), row.get("source_scene_path"))
    source_scene_file = _first_text(window.get("source_scene_file"), sample.get("source_scene_file"), source.get("source_scene_file"), row.get("source_scene_file"))
    technical_atom = _first_text(window.get("technical_atom_id"), sample.get("technical_atom_id"), source.get("technical_atom_id"), row.get("technical_atom_id"))
    pair_actor = _first_text(partner_features.get("partner_actor_id"), interaction.get("partner_actor_id"), pair.get("technical_atom_id_b"))

    warnings = []
    if not source_scene_path and not source_scene_file:
        warnings.append("source scene path unavailable")
    if start is None or end is None:
        warnings.append("time range unavailable")
    if not technical_atom:
        warnings.append("technical actor unavailable")
    if row.get("is_human_ground_truth"):
        warnings.append("unexpected human-ground-truth flag present; treat package as audit review only")
    warnings.extend(_as_list(window.get("warnings")))
    warnings.extend(_as_list(sample.get("warnings")))
    warnings.extend(_as_list(partner_features.get("warnings")))

    evidence = {field: partner_features.get(field) for field in EVIDENCE_FIELDS}
    manifest = {
        "review_id": row.get("review_id") or "review_unknown",
        "semantic_family": row.get("semantic_family") or "unknown",
        "pose_family": pose.get("family") or row.get("pose_family") or "unknown",
        "pose_subtype": pose.get("subtype") or row.get("pose_subtype") or "unknown",
        "motion_subtype": motion.get("subtype") or row.get("motion_subtype") or "unknown",
        "partner_relation": row.get("partner_relation") or interaction.get("partner_relation") or [],
        "contact_support": row.get("contact_support") or interaction.get("support_context") or "unknown",
        "interaction_family": interaction.get("interaction_family") or row.get("interaction_family") or "unknown",
        "generation_safe": bool(row.get("generation_safe")),
        "why_selected": row.get("why_selected") or row.get("category") or "",
        "source_scene_file": source_scene_file,
        "source_scene_path": source_scene_path,
        "technical_atom_id": technical_atom,
        "pair_actor_id": pair_actor,
        "source_id": window.get("source_id") or sample.get("source_id") or source.get("source_id") or "",
        "sample_id": window.get("sample_id") or sample.get("sample_id") or "",
        "window_id": row.get("window_id") or window.get("window_id") or "",
        "pair_window_id": row.get("pair_window_id") or "",
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": duration,
        "frame_start": window.get("frame_start"),
        "frame_end": window.get("frame_end"),
        "source_type": sample.get("source_type") or source.get("source_type") or "unknown",
        "clip_name": sample.get("clip_name") or source.get("clip_name") or "",
        "clip_index": sample.get("clip_index") if sample.get("clip_index") is not None else source.get("clip_index"),
        "plugin_id": source.get("plugin_id") or "",
        "storable_id": sample.get("storable_id") or source.get("storable_id") or "",
        "can_review_in_original_scene": bool(source_scene_path or source_scene_file) and start is not None and end is not None,
        "can_export_timeline_segment": False,
        "review_method": "unavailable",
        "warnings": _dedupe(warnings),
        "evidence_scores": evidence,
        "rider_above_partner_score": evidence["rider_above_partner_score"],
        "pelvis_alignment_score": evidence["pelvis_alignment_score"],
        "hands_on_partner_chest_score": evidence["hands_on_partner_chest_score"],
        "hands_on_partner_hips_score": evidence["hands_on_partner_hips_score"],
        "partner_lying_score": evidence["partner_lying_score"],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }
    return manifest


def _maybe_export_timeline_segment(
    item: dict[str, Any],
    data: dict[str, Any],
    out_dir: Path,
    attempt: bool,
) -> dict[str, Any]:
    if not attempt:
        return _write_unavailable(out_dir, item, "timeline segment export disabled by command")
    if item.get("source_type") != "timeline_controller_motion":
        return _write_unavailable(out_dir, item, f"source_type is {item.get('source_type')}; only timeline_controller_motion is attempted")
    sample = data["samples"].get(item.get("sample_id"))
    if not sample:
        return _write_unavailable(out_dir, item, "sample record not found")
    if not _baked_npz_exists(sample, data["run_dir"], data.get("source_run")):
        return _write_unavailable(out_dir, item, "baked NPZ missing; source segment cannot be traced safely")
    try:
        from vam_timeline_ai.audits.semantic_review import _attempt_timeline_export

        export_row = dict(item)
        export_row["system_semantic_guess"] = {
            "safe_for_learning": False,
            "generation_pose_anchor_safe": False,
            "export_pose_validity": "review_source_segment_only",
            "missing_required_anchor_controllers": [],
        }
        result = _attempt_timeline_export(
            export_row,
            {"samples": data["samples"], "run_dir": data["run_dir"]},
            out_dir,
            export_mode="motion_plus_static_anchors",
        )
        if result.get("success"):
            result["warnings"] = _dedupe(
                list(result.get("warnings") or [])
                + [
                    "This is an optional original-source review segment.",
                    "It is not generated motion and must not be used as a learning target.",
                ]
            )
        return result
    except Exception as exc:  # noqa: BLE001
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "export_failed.md").write_text(
            "\n".join(
                [
                    f"# Timeline Segment Export Failed for {item['review_id']}",
                    "",
                    f"- Reason: {exc}",
                    "- No substitute segment was created.",
                    "- Review this item in the original scene/time range instead.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {"attempted": True, "success": False, "validation_status": "failed", "warnings": [str(exc)], "timeline_export_path": None}


def _write_unavailable(out_dir: Path, item: dict[str, Any], reason: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Timeline Segment Export Unavailable for {item['review_id']}",
        "",
        f"- Scene: `{item.get('source_scene_path') or item.get('source_scene_file')}`",
        f"- Actor: `{item.get('technical_atom_id')}`",
        f"- Time: `{item.get('start_seconds')}` to `{item.get('end_seconds')}`",
        f"- Reason: {reason}",
        "",
        "No fake Timeline segment was created. Review this item in the original VaM scene/time range.",
    ]
    (out_dir / "export_unavailable.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"attempted": True, "success": False, "validation_status": "unavailable", "warnings": [reason], "timeline_export_path": None}


def _baked_npz_exists(sample: dict[str, Any], run_dir: Path, source_run: Path | None) -> bool:
    text = str(sample.get("baked_npz_path") or "")
    if not text:
        return False
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        project_root = run_dir.parents[2] if len(run_dir.parents) > 2 else Path.cwd()
        candidates.extend([project_root / path, run_dir / path])
        if source_run:
            candidates.append(source_run / path)
    return any(candidate.exists() for candidate in candidates)


def _write_item_folder(out: Path, item: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "item_metadata.json", item)
    (out / "item_review.md").write_text(_item_review_text(item), encoding="utf-8")
    snippet = {"reviews": {item["review_id"]: _answer_template()}}
    (out / "answer_snippet.yaml").write_text(yaml.safe_dump(snippet, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _item_review_text(item: dict[str, Any]) -> str:
    evidence = item.get("evidence_scores") or {}
    lines = [
        f"# {item['review_id']} VaM Review",
        "",
        "This is an audit review item, not a training label.",
        "",
        "## Load",
        "",
        f"- VaM scene: `{item.get('source_scene_path') or item.get('source_scene_file') or 'unknown'}`",
        f"- Atom / technical actor: `{item.get('technical_atom_id') or 'unknown'}`",
        f"- Clip/source: `{item.get('clip_name') or 'unknown'}` (index `{item.get('clip_index')}`)",
        f"- Time range: `{item.get('start_seconds')}` to `{item.get('end_seconds')}` seconds",
        "",
        "## System Guess",
        "",
        f"- Family: `{item.get('semantic_family')}`",
        f"- Pose: `{item.get('pose_family')}` / `{item.get('pose_subtype')}`",
        f"- Motion: `{item.get('motion_subtype')}`",
        f"- Partner relation: `{_join(item.get('partner_relation'))}`",
        f"- Contact/support: `{item.get('contact_support')}`",
        f"- Interaction family: `{item.get('interaction_family')}`",
        f"- Generation safe: `{item.get('generation_safe')}`",
        f"- Why selected: `{item.get('why_selected')}`",
        "",
        "## Evidence Scores",
        "",
    ]
    for field in EVIDENCE_FIELDS:
        lines.append(f"- `{field}`: `{evidence.get(field)}`")
    lines.extend(
        [
            "",
            "## Timeline Segment",
            "",
            f"- Status: `{item.get('timeline_export_status')}`",
            f"- Path: `{item.get('timeline_export_path') or 'not available'}`",
            "",
            "If no segment exists, review the original scene/time range only.",
            "",
            "## What To Answer",
            "",
            "1. Is the semantic family correct?",
            "2. Is the pose correct?",
            "3. Is the motion correct?",
            "4. Is the partner relation correct?",
            "5. Is the contact/support correct?",
            "6. Is generation_safe correct?",
            "",
        ]
    )
    if item.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in item["warnings"])
        lines.append("")
    return "\n".join(lines)


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_id",
        "semantic_family",
        "pose_family",
        "pose_subtype",
        "motion_subtype",
        "partner_relation",
        "contact_support",
        "interaction_family",
        "generation_safe",
        "why_selected",
        "source_scene_path",
        "technical_atom_id",
        "pair_actor_id",
        "source_id",
        "sample_id",
        "window_id",
        "pair_window_id",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "source_type",
        "clip_name",
        "clip_index",
        "storable_id",
        "can_review_in_original_scene",
        "can_export_timeline_segment",
        "review_method",
        "timeline_export_status",
        "timeline_export_path",
        *EVIDENCE_FIELDS,
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["partner_relation"] = _join(row.get("partner_relation"))
            flat["warnings"] = " | ".join(_as_list(row.get("warnings")))
            writer.writerow(flat)


def _write_scene_list(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("source_scene_path") or row.get("source_scene_file") or "unknown"].append(row)
    lines = [
        "# VaM Review Scene List",
        "",
        "Load each scene once and review all listed time ranges before moving to the next scene.",
        "",
    ]
    for scene, items in sorted(groups.items()):
        lines.extend([f"## Scene: `{scene}`", ""])
        for item in sorted(items, key=lambda r: (float(r.get("start_seconds") or 0), r.get("review_id") or "")):
            guess = f"{item.get('semantic_family')} / {item.get('pose_subtype')} / {item.get('contact_support')}"
            lines.append(
                f"- `{item['review_id']}`: actor `{item.get('technical_atom_id')}`, "
                f"{item.get('start_seconds')} to {item.get('end_seconds')}s, guess `{guess}`, "
                f"Timeline segment `{item.get('timeline_export_status')}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_answer_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    data = {
        "allowed_values": ["true", "false", "unknown", "not_applicable"],
        "reviews": {row["review_id"]: _answer_template() for row in rows},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _answer_template() -> dict[str, Any]:
    return {
        "original_scene_opened": "unknown",
        "source_scene_loads": "unknown",
        "actor_found": "unknown",
        "time_range_found": "unknown",
        "semantic_family_correct": "unknown",
        "pose_correct": "unknown",
        "motion_correct": "unknown",
        "partner_relation_correct": "unknown",
        "contact_support_correct": "unknown",
        "generation_safe_correct": "unknown",
        "actual_semantic_family": "",
        "actual_pose": "",
        "actual_motion": "",
        "actual_partner_relation": "",
        "actual_contact_support": "",
        "notes": "",
    }


def _write_index_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        item_link = f"items/{html.escape(row['review_id'])}/item_review.md"
        segment_link = _segment_link(row)
        evidence = "".join(
            f"<li><code>{html.escape(field)}</code>: <code>{html.escape(str(row.get(field)))}</code></li>"
            for field in EVIDENCE_FIELDS
        )
        snippet = yaml.safe_dump({"reviews": {row["review_id"]: _answer_template()}}, sort_keys=False, allow_unicode=True)
        cards.append(
            f"""
<section>
  <h2>{html.escape(row['review_id'])}</h2>
  <p><strong>Scene:</strong> <code>{html.escape(str(row.get('source_scene_path') or row.get('source_scene_file') or 'unknown'))}</code></p>
  <p><strong>Actor:</strong> <code>{html.escape(str(row.get('technical_atom_id') or 'unknown'))}</code>
     <strong>Time:</strong> <code>{html.escape(str(row.get('start_seconds')))} - {html.escape(str(row.get('end_seconds')))}s</code></p>
  <p><strong>Guess:</strong>
     family <code>{html.escape(str(row.get('semantic_family')))}</code>,
     pose <code>{html.escape(str(row.get('pose_family')))} / {html.escape(str(row.get('pose_subtype')))}</code>,
     motion <code>{html.escape(str(row.get('motion_subtype')))}</code>,
     relation <code>{html.escape(_join(row.get('partner_relation')))}</code>,
     support <code>{html.escape(str(row.get('contact_support')))}</code>,
     generation_safe <code>{html.escape(str(row.get('generation_safe')))}</code>
  </p>
  <p><strong>Why selected:</strong> <code>{html.escape(str(row.get('why_selected')))}</code></p>
  <ul>{evidence}</ul>
  <p><a href="{item_link}">Per-item instructions</a> | {segment_link}</p>
  <details><summary>Answer snippet</summary><pre>{html.escape(snippet)}</pre></details>
</section>
"""
        )
    html_text = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>clean_v3 VaM Review Package v15</title>
<style>
body{font-family:system-ui,Segoe UI,sans-serif;margin:1.5rem;background:#f7f7f5;color:#202020;line-height:1.45}
section,.panel{background:#fff;border:1px solid #d8d8d0;border-radius:6px;padding:1rem;margin:1rem 0}
code,pre{background:#f0f0ea;border-radius:4px}
code{padding:.1rem .25rem}
pre{padding:.75rem;white-space:pre-wrap}
a{color:#064f8a}
</style>
</head>
<body>
<h1>clean_v3 Semantic Review v15 - VaM Review Package</h1>
<div class="panel">
<h2>How to test this in VaM</h2>
<ol>
<li>Open <a href="vam_review_scene_list.md">the scene list</a>.</li>
<li>Load the listed VaM scene.</li>
<li>Find the listed actor/atom.</li>
<li>Play or scrub the exact time range.</li>
<li>Fill <code>vam_review_answer_sheet.yaml</code>.</li>
<li>Optional: if a Timeline segment link exists, import it as a review convenience segment only.</li>
</ol>
<p>This package is for human validation only. It does not train ML and does not create generation targets.</p>
</div>
""" + "\n".join(cards) + "\n</body>\n</html>\n"
    path.write_text(html_text, encoding="utf-8")


def _segment_link(row: dict[str, Any]) -> str:
    if row.get("timeline_export_path"):
        rel = f"timeline_segments/{row['review_id']}/{row['review_id']}.timeline.json"
        return f'<a href="{html.escape(rel)}">Timeline segment</a>'
    return f'<a href="timeline_segments/{html.escape(row["review_id"])}/export_unavailable.md">Timeline unavailable</a>'


def _write_timeline_status(path: Path, rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    statuses = Counter(_export_status(result) for result in results)
    reason_counts = Counter((result.get("warnings") or ["unknown"])[0] for result in results if not result.get("success"))
    lines = [
        "# Timeline Segment Export Status",
        "",
        "These are optional source-scene review segments only. They are not generated motion and are not training labels.",
        "",
        f"- Review items: {len(rows)}",
        f"- Attempted: {sum(1 for result in results if result.get('attempted'))}",
        f"- Successful: {statuses.get('successful', 0)}",
        f"- Unavailable: {statuses.get('unavailable', 0)}",
        f"- Failed: {statuses.get('failed', 0)}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in reason_counts.most_common()) if reason_counts else lines.append("- None")
    lines.extend(["", "## Per Item", ""])
    for row in rows:
        lines.append(f"- `{row['review_id']}`: `{row.get('timeline_export_status')}` - `{row.get('timeline_export_path') or 'no file'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_package_instructions(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# VaM Review Package Instructions",
        "",
        "Open `vam_review_index.html` first.",
        "",
        "## Workflow",
        "",
        "1. Open `vam_review_scene_list.md`.",
        "2. Load the listed VaM scene.",
        "3. Find the listed atom / technical actor.",
        "4. Inspect the exact start/end time range.",
        "5. Fill `vam_review_answer_sheet.yaml`.",
        "6. If a Timeline segment exists, you may import it as a review convenience segment. If not, review the original scene only.",
        "",
        "## Important",
        "",
        "- This package is for human validation of clean_v3 interaction semantics.",
        "- It does not train ML.",
        "- It does not modify `manual_labels.yaml`.",
        "- Optional Timeline segments are source review clips only, not generation targets.",
        "",
        f"Review items: {len(rows)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_status(result: dict[str, Any]) -> str:
    if result.get("success"):
        return "successful"
    if result.get("validation_status") == "failed":
        return "failed"
    return "unavailable"


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        if value is not None and str(value):
            return str(value)
    return ""


def _duration(start: Any, end: Any) -> float | None:
    try:
        if start is None or end is None:
            return None
        return float(end) - float(start)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _join(value: Any) -> str:
    return ", ".join(str(v) for v in _as_list(value)) if value is not None else ""


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result

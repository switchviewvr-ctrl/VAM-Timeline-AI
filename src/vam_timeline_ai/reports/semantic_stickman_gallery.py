"""Build an HTML/Markdown gallery for semantic stickman previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import html

from vam_timeline_ai.io.json_utils import load_json


def build_semantic_stickman_gallery_v1(preview_dir: str | Path, out_html: str | Path, out_md: str | Path) -> dict[str, Any]:
    return _build_semantic_stickman_gallery(preview_dir, out_html, out_md, version="v1")


def build_semantic_stickman_gallery_v2(preview_dir: str | Path, out_html: str | Path, out_md: str | Path) -> dict[str, Any]:
    return _build_semantic_stickman_gallery(preview_dir, out_html, out_md, version="v2")


def build_semantic_stickman_gallery_v3(preview_dir: str | Path, out_html: str | Path, out_md: str | Path) -> dict[str, Any]:
    return _build_semantic_stickman_gallery(preview_dir, out_html, out_md, version="v3")


def _build_semantic_stickman_gallery(preview_dir: str | Path, out_html: str | Path, out_md: str | Path, version: str) -> dict[str, Any]:
    root = Path(preview_dir)
    manifest_path = root / f"semantic_stickman_preview_manifest_{version}.json"
    manifest = load_json(manifest_path).get("items", []) if manifest_path.exists() else []
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in manifest:
        groups.setdefault(str(item.get("family") or "unknown"), []).append(item)
    html_text = _html(groups, root, version)
    md_text = _md(groups, version)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html_text, encoding="utf-8")
    Path(out_md).write_text(md_text, encoding="utf-8")
    return {"status": "ok", "items": len(manifest), "families": {k: len(v) for k, v in groups.items()}, "out_html": str(out_html), "out_md": str(out_md)}


def _html(groups: dict[str, list[dict[str, Any]]], root: Path, version: str) -> str:
    cards = []
    if version in {"v2", "v3"}:
        cards.append(
            "<section class='legend'>"
            "<h2>Legend</h2>"
            "<p><span class='dot driver'></span>driver bodypart "
            "<span class='dot follower'></span>follower bodypart "
            "<span class='dot anchor'></span>anchor/support "
            "<span class='dot partner'></span>partner reference "
            "<span class='dot contact'></span>contact/support target</p>"
            "<p>Blue arrows mark rider pelvis to partner pelvis or family-specific target/alignment paths. "
            "Gray boxes mark partner reference body context. This is ontology visualization only.</p>"
            + ("<p>v3 additionally treats the partner target/contact zone as a constraint. Alignment validity and target distances are shown per card.</p>" if version == "v3" else "")
            + "</section>"
        )
    for family, items in sorted(groups.items()):
        cards.append(f"<h2>{html.escape(family)}</h2><div class='grid'>")
        for item in items:
            gif = _rel(root, item.get("gif_path"))
            sheet = _rel(root, item.get("contact_sheet_path"))
            media = f"<img src='{html.escape(gif)}' alt='{html.escape(str(item.get('concept_id')))}'>" if gif else ""
            ctx = item.get("semantic_context") or {}
            warnings = list(item.get("warnings") or []) + list(ctx.get("warnings") or [])
            warning_html = ""
            if warnings:
                warning_html = "<p class='warn'><b>Warnings:</b> " + html.escape("; ".join(sorted(set(map(str, warnings))))) + "</p>"
            context_html = ""
            if version in {"v2", "v3"}:
                context_html = (
                    "<p class='context'>"
                    f"partner: {html.escape(str(ctx.get('has_partner_reference')))} | "
                    f"alignment: {html.escape(str(ctx.get('has_alignment_target')))} | "
                    f"support targets: {html.escape(str(ctx.get('has_support_targets')))} | "
                    f"floating warning: {html.escape(str(ctx.get('appears_floating_warning')))}"
                    "</p>"
                )
            if version == "v3":
                valid = ctx.get("interaction_alignment_valid")
                distance = ctx.get("alignment_distance_max")
                limit = ctx.get("target_distance_max")
                failed = ctx.get("failed_constraints") or []
                context_html += (
                    "<p class='context strong'>"
                    f"contact-valid: {html.escape(str(valid))} | "
                    f"alignment distance max: {html.escape(str(distance))} / {html.escape(str(limit))}<br>"
                    f"failed constraints: {html.escape(', '.join(map(str, failed)) if failed else 'none')}"
                    "</p>"
                )
            cards.append(
                "<article class='card'>"
                f"<h3>{html.escape(str(item.get('concept_id')))}</h3>"
                f"{media}"
                f"<p><b>Pose:</b> {html.escape(str(item.get('pose_subtype')))}<br>"
                f"<b>Motion:</b> {html.escape(str(item.get('motion_subtype')))}</p>"
                f"{context_html}"
                f"{warning_html}"
                f"<p><a href='{html.escape(sheet)}'>contact sheet</a> | <a href='{html.escape(_rel(root, item.get('metadata_path')))}'>metadata</a></p>"
                "</article>"
            )
        cards.append("</div>")
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Semantic Stickman Previews</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f8fafc;color:#0f172a}
header{background:#0f172a;color:white;padding:18px 24px}
main{padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
.card{background:white;border:1px solid #d8dee9;border-radius:8px;padding:14px}
.card img{width:100%;border-radius:6px;border:1px solid #e2e8f0;background:#fff}
.legend{background:white;border:1px solid #d8dee9;border-radius:8px;padding:14px;margin-bottom:18px}
.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin:0 5px 0 14px;vertical-align:middle}
.driver{background:#dc372d}.follower{background:#2d69dc}.anchor{background:#239146}.partner{background:#919191}.contact{background:#9641be}
.warn{color:#b42318}.context{color:#475569;font-size:13px}.strong{font-weight:600}
a{color:#0f5ea8}
</style></head><body><header><h1>Semantic Stickman Previews</h1><p>Ontology sanity check only. Not VaM Timeline generation.</p></header><main>
""" + "\n".join(cards) + "</main></body></html>\n"


def _md(groups: dict[str, list[dict[str, Any]]], version: str) -> str:
    lines = [
        f"# Semantic Stickman Gallery {version.upper()}",
        "",
        "Schematic ontology sanity previews. These are not VaM production controller targets.",
        "",
    ]
    if version in {"v2", "v3"}:
        lines.extend([
            "## Legend",
            "",
            "- Driver bodyparts, followers, anchors/supports, partner references, contact targets, and alignment axes are rendered explicitly.",
            "- Warnings flag missing partner targets, support targets, floating/unsupported poses, or missing alignment context.",
            "",
        ])
    for family, items in sorted(groups.items()):
        lines.extend([f"## {family}", ""])
        for item in items:
            ctx = item.get("semantic_context") or {}
            detail = ""
            if version in {"v2", "v3"}:
                detail = f", partner `{ctx.get('has_partner_reference')}`, alignment `{ctx.get('has_alignment_target')}`, support `{ctx.get('has_support_targets')}`"
            if version == "v3":
                detail += f", contact-valid `{ctx.get('interaction_alignment_valid')}`, distance `{ctx.get('alignment_distance_max')}`"
            lines.append(f"- `{item.get('concept_id')}`: pose `{item.get('pose_subtype')}`, motion `{item.get('motion_subtype')}`{detail}")
        lines.append("")
    return "\n".join(lines)


def _rel(root: Path, path: Any) -> str:
    if not path:
        return ""
    p = Path(str(path))
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()

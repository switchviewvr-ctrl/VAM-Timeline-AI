"""clean_v3 semantic QA dashboard."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import html

from vam_timeline_ai.io.json_utils import load_json, load_jsonl


def write_clean_v3_dashboard(run_dir: str | Path, out_md: str | Path, out_html: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    manifest = _load_json_optional(run / "run_manifest.json")
    pose = load_jsonl(run / "pose_semantics" / "pose_semantics_v0.jsonl")
    interaction = load_jsonl(run / "interaction_semantics" / "interaction_semantics_v0.jsonl")
    actions = _load_first(run / "semantic_actions" / "semantic_actions_v1.jsonl", run / "semantic_actions" / "semantic_actions_v0.jsonl")
    semantic_db = _load_first(run / "datasets" / "semantic_candidate_db_v1.jsonl", run / "datasets" / "semantic_candidate_db_v0.jsonl")
    cowgirl_db = _load_first(run / "datasets" / "cowgirl_candidate_db_v6.jsonl", run / "datasets" / "cowgirl_candidate_db_v5.jsonl")
    lines = _dashboard_lines(run, manifest, pose, interaction, actions, semantic_db, cowgirl_db)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_html(lines, out_html)
    return {
        "status": "ok",
        "out_md": str(out_md),
        "out_html": str(out_html),
        "semantic_records": len(semantic_db),
        "cowgirl_records": len(cowgirl_db),
        "generation_safe": sum(1 for r in semantic_db if r.get("generation_safe")),
    }


def _dashboard_lines(
    run: Path,
    manifest: dict[str, Any],
    pose: list[dict[str, Any]],
    interaction: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    semantic_db: list[dict[str, Any]],
    cowgirl_db: list[dict[str, Any]],
) -> list[str]:
    families = Counter(r.get("semantic_family") for r in semantic_db)
    cowcats = Counter(r.get("category") for r in cowgirl_db)
    poses = Counter(r.get("pose_family") for r in pose)
    interactions = Counter(r.get("interaction_family") for r in interaction)
    phases = Counter(r.get("phase") for r in actions)
    contacts = Counter(r.get("contact_support") for r in actions)
    top_scenes = Counter(r.get("source_scene_file") for r in semantic_db)
    questionable = Counter(r.get("source_scene_file") for r in cowgirl_db if not r.get("generation_safe") or r.get("category") != "cowgirl_clean_motion_generation_safe")
    warnings = sum(len(r.get("warnings") or []) for r in actions)
    lines = [
        "# clean_v3 Semantic QA Dashboard",
        "",
        f"- Run: `{run}`",
        f"- Manifest purpose: `{manifest.get('purpose', 'unknown')}`",
        f"- Semantic DB records: {len(semantic_db)}",
        f"- Cowgirl DB records: {len(cowgirl_db)}",
        f"- Generation-safe semantic records: {sum(1 for r in semantic_db if r.get('generation_safe'))}",
        f"- Warning entries across semantic actions: {warnings}",
        "",
        "## Pose Semantic Counts",
        "",
    ]
    lines.extend(_counter_lines(poses))
    lines.extend(["", "## Interaction Semantic Counts", ""])
    lines.extend(_counter_lines(interactions))
    lines.extend(["", "## Semantic Family Counts", ""])
    lines.extend(_counter_lines(families))
    lines.extend(["", "## Cowgirl DB Category Counts", ""])
    lines.extend(_counter_lines(cowcats))
    lines.extend(["", "## Phase Counts", ""])
    lines.extend(_counter_lines(phases))
    lines.extend(["", "## Contact/Support Counts", ""])
    lines.extend(_counter_lines(contacts))
    lines.extend(
        [
            "",
            "## Key Buckets",
            "",
            f"- BJ/oral candidates: {families.get('bj_oral', 0)}",
            f"- Receiver response candidates: {families.get('receiver_response', 0)}",
            f"- Standing/hand/head candidates: {families.get('hand_gesture', 0) + families.get('head_gesture', 0)}",
            f"- Unknown/unusable candidates: {families.get('unknown', 0)}",
            f"- Low-motion/context actions: {phases.get('low_motion_hold', 0) + phases.get('pose_context_only', 0)}",
            f"- Missing partner context Cowgirl rows: {cowcats.get('cowgirl_missing_partner_context', 0)}",
            "",
            "## Top Scenes By Candidate Count",
            "",
        ]
    )
    lines.extend(_counter_lines(top_scenes, 10))
    lines.extend(["", "## Top Scenes By Questionable Count", ""])
    lines.extend(_counter_lines(questionable, 10))
    lines.extend(
        [
            "",
            "## Current Confidence Level",
            "",
            "- Cowgirl motion detection: medium, pending v16 review.",
            "- Contact/support detection: low/unknown until v16 review confirms it.",
            "- Partner relation detection: low/medium.",
            "- Generation-safe classification: experimental.",
            "- Text-to-Timeline generation: not ready.",
        ]
    )
    return lines


def _counter_lines(counter: Counter[Any], limit: int | None = None) -> list[str]:
    items = counter.most_common(limit)
    return [f"- `{k}`: {v}" for k, v in items] if items else ["- None"]


def _load_first(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def _load_json_optional(path: Path) -> dict[str, Any]:
    try:
        return load_json(path) if path.exists() else {}
    except Exception:
        return {}


def _write_html(lines: list[str], out_html: str | Path) -> None:
    body = "\n".join(f"<p>{html.escape(line)}</p>" if line and not line.startswith("#") else f"<h1>{html.escape(line.lstrip('# '))}</h1>" for line in lines)
    text = f"<!doctype html><meta charset='utf-8'><title>clean_v3 Dashboard</title><body>{body}</body>"
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(text, encoding="utf-8")

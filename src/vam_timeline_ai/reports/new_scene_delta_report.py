"""Compare a new-scene delta run against the clean_v3 reference run."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_json, load_jsonl


def compare_new_scenes_to_clean_v3(base_run: str | Path, new_run: str | Path, out: str | Path) -> dict[str, Any]:
    base = Path(base_run)
    new = Path(new_run)
    base_sem = _load_first(base / "datasets" / "semantic_candidate_db_v2.jsonl", base / "datasets" / "semantic_candidate_db_v1.jsonl", base / "datasets" / "semantic_candidate_db_v0.jsonl")
    base_cow = _load_first(base / "datasets" / "cowgirl_candidate_db_v7.jsonl", base / "datasets" / "cowgirl_candidate_db_v6.jsonl", base / "datasets" / "cowgirl_candidate_db_v5.jsonl")
    new_sem = _load_first(new / "datasets" / "semantic_candidate_db_v2.jsonl", new / "datasets" / "semantic_candidate_db_v0.jsonl")
    new_cow = _load_first(new / "datasets" / "cowgirl_candidate_db_v7.jsonl", new / "datasets" / "cowgirl_candidate_db_v0.jsonl")
    sources = load_jsonl(new / "semantic" / "motion_source_index.jsonl")
    samples = load_jsonl(new / "baked" / "motion_sample_index.jsonl")
    windows = load_jsonl(new / "semantic" / "movement_windows.jsonl")
    rel = load_jsonl(new / "relative_motion" / "relative_motion_window_index.jsonl")
    pose = load_jsonl(new / "pose_semantics" / "pose_semantics_v0.jsonl")
    interactions = load_jsonl(new / "interaction_semantics" / "interaction_semantics_v0.jsonl")
    manifest = _load_json_optional(new / "run_manifest.json")

    useful_categories = {
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_low_confidence_short",
        "cowgirl_hands_on_partner_chest",
        "cowgirl_hands_on_partner_hips",
        "cowgirl_ambiguous_partner_contact",
    }
    contact_counts = Counter(str(r.get("contact_support") or "unknown") for r in new_sem)
    new_families = Counter(r.get("semantic_family") for r in new_sem)
    base_families = Counter(r.get("semantic_family") for r in base_sem)
    new_cats = Counter(r.get("category") for r in new_cow)
    base_cats = Counter(r.get("category") for r in base_cow)
    scenes_useful = Counter(r.get("source_scene_file") for r in new_cow if r.get("category") in useful_categories)
    scenes_unknown = Counter(r.get("source_scene_file") for r in new_sem if r.get("semantic_family") == "unknown")
    scene_count = len({r.get("source_scene_path") or r.get("source_scene_file") for r in sources if r.get("source_scene_file")})
    raw_scene_count = int(manifest.get("scene_count_found_at_manifest") or scene_count)
    summary = {
        "status": "ok",
        "raw_scene_files": raw_scene_count,
        "new_scenes": scene_count,
        "new_sources": len(sources),
        "new_samples": len(samples),
        "new_baked_ok": sum(1 for r in samples if r.get("bake_status") == "ok"),
        "new_windows": len(windows),
        "new_relative_safe": sum(1 for r in rel if r.get("safe_for_learning")),
        "new_semantic_families": dict(new_families),
        "new_cowgirl_categories": dict(new_cats),
        "base_semantic_records": len(base_sem),
        "base_cowgirl_records": len(base_cow),
        "generation_safe_new": sum(1 for r in new_sem if r.get("generation_safe")),
        "generation_safe_base": sum(1 for r in base_sem if r.get("generation_safe")),
        "contact_support_counts": dict(contact_counts),
        "output": str(out),
    }
    _write_report(
        Path(out),
        manifest,
        summary,
        base_families,
        new_families,
        base_cats,
        new_cats,
        Counter(r.get("pose_family") for r in pose),
        Counter(r.get("interaction_family") for r in interactions),
        contact_counts,
        scenes_useful,
        scenes_unknown,
    )
    return summary


def _write_report(
    out: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    base_families: Counter[Any],
    new_families: Counter[Any],
    base_cats: Counter[Any],
    new_cats: Counter[Any],
    pose_counts: Counter[Any],
    interaction_counts: Counter[Any],
    contact_counts: Counter[Any],
    scenes_useful: Counter[Any],
    scenes_unknown: Counter[Any],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# New Scene Delta Report",
        "",
        f"- New run: `{manifest.get('run_name', 'unknown')}`",
        f"- Source folder: `{manifest.get('source_folder', 'unknown')}`",
        f"- Parent reference run: `{manifest.get('parent_reference_run', 'unknown')}`",
        f"- Raw scene JSON files found: {summary.get('raw_scene_files', summary['new_scenes'])}",
        f"- Scenes with motion sources: {summary['new_scenes']}",
        f"- New motion sources: {summary['new_sources']}",
        f"- Baked samples: {summary['new_samples']} total / {summary['new_baked_ok']} ok",
        f"- Movement windows: {summary['new_windows']}",
        f"- Relative-safe windows: {summary['new_relative_safe']}",
        f"- New generation-safe semantic candidates: {summary['generation_safe_new']}",
        f"- Base generation-safe semantic candidates: {summary['generation_safe_base']}",
        "",
        "## Semantic Family Counts",
        "",
        "### New Scenes",
        "",
    ]
    lines.extend(_counter_lines(new_families))
    lines.extend(["", "### clean_v3 Reference", ""])
    lines.extend(_counter_lines(base_families))
    lines.extend(["", "## Cowgirl Category Counts", "", "### New Scenes", ""])
    lines.extend(_counter_lines(new_cats))
    lines.extend(["", "### clean_v3 Reference", ""])
    lines.extend(_counter_lines(base_cats))
    lines.extend(["", "## Pose Family Counts", ""])
    lines.extend(_counter_lines(pose_counts))
    lines.extend(["", "## Interaction Family Counts", ""])
    lines.extend(_counter_lines(interaction_counts))
    lines.extend(["", "## Contact/Support Counts", ""])
    for key in ["hands_on_partner_chest", "hands_on_partner_hips", "hands_on_floor_or_bed", "hands_free", "ambiguous_partner_contact", "unknown_contact", "unknown"]:
        lines.append(f"- `{key}`: {contact_counts.get(key, 0)}")
    lines.extend(["", "## Most Useful New Scenes", ""])
    lines.extend(_counter_lines(scenes_useful, 12))
    lines.extend(["", "## Scenes With Many Unknown/Unusable Candidates", ""])
    lines.extend(_counter_lines(scenes_unknown, 12))
    lines.extend(["", "## Coverage Note", ""])
    if sum(new_cats.get(k, 0) for k in ["cowgirl_clean_motion_generation_safe", "cowgirl_clean_motion_low_confidence_short"]) > 0:
        lines.append("- The new scenes appear to add usable Cowgirl motion candidates for review.")
    else:
        lines.append("- No strong Cowgirl clean-motion candidates were found yet; review unknown/conflict rows before merging anything.")
    lines.append("- These are machine/audit candidates only, not manual training truth.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

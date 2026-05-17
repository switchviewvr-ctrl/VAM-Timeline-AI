"""Lineage checks from source scenes to semantic candidates and review items."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl


def write_candidate_lineage_report(run_dir: str | Path, out: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    sources = _by_id(_load_first(run / "semantic" / "motion_source_index.jsonl", run / "references" / "motion_source_index.jsonl"), "source_id")
    samples = _by_id(_load_first(run / "baked" / "motion_sample_index.jsonl"), "sample_id")
    windows = _by_id(_load_first(run / "relative_motion" / "relative_motion_window_index.jsonl", run / "semantic" / "movement_windows.jsonl"), "window_id")
    pair_windows = _by_id(_load_first(run / "semantic" / "pair_windows_v1.jsonl"), "pair_window_id")
    pose = _by_id(_load_first(run / "pose_semantics" / "pose_semantics_v0.jsonl"), "window_id")
    interactions = _group_by(_load_first(run / "interaction_semantics" / "interaction_semantics_v0.jsonl"), "window_id")
    actions = _by_id(_load_first(run / "semantic_actions" / "semantic_actions_v1.jsonl", run / "semantic_actions" / "semantic_actions_v0.jsonl"), "window_id")
    semantic = _load_first(run / "datasets" / "semantic_candidate_db_v1.jsonl", run / "datasets" / "semantic_candidate_db_v0.jsonl")
    cowgirl = _load_first(run / "datasets" / "cowgirl_candidate_db_v6.jsonl", run / "datasets" / "cowgirl_candidate_db_v5.jsonl")
    primitives = _load_first(run / "generation" / "cowgirl_motion_primitives_v1.jsonl", run / "generation" / "cowgirl_motion_primitives_v0.jsonl")
    review_items = _review_items(run)

    semantic_by_window = _group_by(semantic, "window_id")
    cowgirl_by_window = _group_by(cowgirl, "window_id")
    primitive_by_window = _primitive_windows(primitives)
    review_by_window = _group_by(review_items, "window_id")

    orphan_windows = sorted(set(windows) - set(actions) - set(semantic_by_window))[:200]
    missing_pose = []
    missing_interaction = []
    missing_action = []
    family_complete = Counter()
    family_incomplete = Counter()
    duplicate_candidates = _duplicate_values([r.get("candidate_id") for r in semantic if r.get("candidate_id")])
    duplicate_windows = _duplicate_values([r.get("window_id") for r in semantic if r.get("window_id")])

    for row in semantic:
        wid = row.get("window_id")
        family = str(row.get("semantic_family") or "unknown")
        missing = []
        if wid not in pose:
            missing.append("pose_semantics")
            missing_pose.append(wid)
        if wid not in interactions:
            missing.append("interaction_semantics")
            missing_interaction.append(wid)
        if wid not in actions:
            missing.append("semantic_action")
            missing_action.append(wid)
        if missing:
            family_incomplete[family] += 1
        else:
            family_complete[family] += 1

    review_missing_source_time = [
        r.get("review_id") or r.get("window_id") or "unknown"
        for r in review_items
        if not (r.get("source_scene_path") or r.get("source_scene_file")) or r.get("start_seconds") is None or r.get("end_seconds") is None
    ]
    lineage_rows = _sample_lineage_rows(
        semantic[:50],
        sources,
        samples,
        windows,
        pair_windows,
        pose,
        interactions,
        actions,
        cowgirl_by_window,
        primitive_by_window,
        review_by_window,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        out_path,
        counts={
            "sources": len(sources),
            "samples": len(samples),
            "windows": len(windows),
            "pose_records": len(pose),
            "interaction_windows": len(interactions),
            "semantic_actions": len(actions),
            "semantic_candidates": len(semantic),
            "cowgirl_candidates": len(cowgirl),
            "motion_primitives": len(primitives),
            "review_items": len(review_items),
        },
        orphan_windows=orphan_windows,
        missing_pose=missing_pose,
        missing_interaction=missing_interaction,
        missing_action=missing_action,
        duplicate_candidates=duplicate_candidates,
        duplicate_windows=duplicate_windows,
        review_missing_source_time=review_missing_source_time,
        family_complete=family_complete,
        family_incomplete=family_incomplete,
        lineage_rows=lineage_rows,
    )
    return {
        "status": "ok",
        "orphan_windows": len(orphan_windows),
        "missing_pose": len(set(missing_pose)),
        "missing_interaction": len(set(missing_interaction)),
        "review_missing_source_time": len(review_missing_source_time),
        "duplicate_candidate_ids": len(duplicate_candidates),
        "out": str(out_path),
    }


def _load_first(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def _by_id(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {row.get(key): row for row in rows if row.get(key)}


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key):
            grouped[row.get(key)].append(row)
    return dict(grouped)


def _duplicate_values(values: list[Any]) -> dict[str, int]:
    counts = Counter(str(v) for v in values if v)
    return {k: v for k, v in counts.items() if v > 1}


def _primitive_windows(primitives: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for primitive in primitives:
        for wid in primitive.get("source_window_ids") or []:
            grouped[str(wid)].append(primitive)
    return dict(grouped)


def _review_items(run: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    audits = run / "audits"
    if not audits.exists():
        return items
    for review_dir in sorted(audits.glob("semantic_review*")):
        if not review_dir.is_dir():
            continue
        manifest = review_dir / "vam_review_package" / "vam_review_manifest.jsonl"
        rows = load_jsonl(manifest) if manifest.exists() else load_jsonl(review_dir / "semantic_review_010.jsonl")
        for row in rows:
            out = dict(row)
            out["source_review_folder"] = str(review_dir)
            items.append(out)
    return items


def _sample_lineage_rows(
    semantic_rows: list[dict[str, Any]],
    sources: dict[Any, dict[str, Any]],
    samples: dict[Any, dict[str, Any]],
    windows: dict[Any, dict[str, Any]],
    pair_windows: dict[Any, dict[str, Any]],
    pose: dict[Any, dict[str, Any]],
    interactions: dict[Any, list[dict[str, Any]]],
    actions: dict[Any, dict[str, Any]],
    cowgirl_by_window: dict[Any, list[dict[str, Any]]],
    primitive_by_window: dict[str, list[dict[str, Any]]],
    review_by_window: dict[Any, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for row in semantic_rows:
        wid = row.get("window_id")
        window = windows.get(wid) or {}
        sample_id = row.get("sample_id") or window.get("sample_id")
        sample = samples.get(sample_id) or {}
        source_id = row.get("source_id") or sample.get("source_id") or window.get("source_id")
        source = sources.get(source_id) or {}
        pair_window_id = row.get("pair_window_id") or window.get("pair_window_id")
        rows.append(
            {
                "source_scene": row.get("source_scene_file") or source.get("source_scene_file") or source.get("path") or "",
                "source_id": source_id or "",
                "sample_id": sample_id or "",
                "window_id": wid or "",
                "pair_window_id": pair_window_id or "",
                "has_pair_window": bool(pair_window_id and pair_window_id in pair_windows),
                "has_pose": wid in pose,
                "has_interaction": wid in interactions,
                "has_semantic_action": wid in actions,
                "has_semantic_candidate": True,
                "has_cowgirl_candidate": wid in cowgirl_by_window,
                "has_motion_primitive": str(wid) in primitive_by_window,
                "has_review_item": wid in review_by_window,
                "semantic_family": row.get("semantic_family") or "",
            }
        )
    return rows


def _write_report(
    out: Path,
    counts: dict[str, int],
    orphan_windows: list[str],
    missing_pose: list[Any],
    missing_interaction: list[Any],
    missing_action: list[Any],
    duplicate_candidates: dict[str, int],
    duplicate_windows: dict[str, int],
    review_missing_source_time: list[str],
    family_complete: Counter[str],
    family_incomplete: Counter[str],
    lineage_rows: list[dict[str, Any]],
) -> None:
    lines = ["# Candidate Lineage Report", "", "Lineage is audit-only and is used to keep future generation abstractions traceable.", ""]
    lines.extend(["## Artifact Counts", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in counts.items())
    lines.extend(
        [
            "",
            "## Lineage Warnings",
            "",
            f"- Orphan windows: {len(orphan_windows)}",
            f"- Candidate records missing pose semantics: {len(set(missing_pose))}",
            f"- Candidate records missing interaction semantics: {len(set(missing_interaction))}",
            f"- Candidate records missing semantic action: {len(set(missing_action))}",
            f"- Review items missing source path/time: {len(review_missing_source_time)}",
            f"- Duplicate candidate IDs: {len(duplicate_candidates)}",
            f"- Duplicate semantic candidate window IDs: {len(duplicate_windows)}",
            "",
            "## Family Counts With Complete Lineage",
            "",
        ]
    )
    lines.extend(f"- `{k}`: {v}" for k, v in family_complete.most_common()) if family_complete else lines.append("- None")
    lines.extend(["", "## Family Counts With Incomplete Lineage", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in family_incomplete.most_common()) if family_incomplete else lines.append("- None")
    lines.extend(["", "## Example Orphan Windows", ""])
    lines.extend(f"- `{wid}`" for wid in orphan_windows[:20]) if orphan_windows else lines.append("- None")
    lines.extend(["", "## Review Items Missing Source/Time", ""])
    lines.extend(f"- `{rid}`" for rid in review_missing_source_time[:20]) if review_missing_source_time else lines.append("- None")
    lines.extend(["", "## Duplicate Candidate IDs", ""])
    lines.extend(f"- `{cid}`: {count}" for cid, count in list(duplicate_candidates.items())[:20]) if duplicate_candidates else lines.append("- None")
    lines.extend(["", "## Sample Lineage Rows", ""])
    for row in lineage_rows[:20]:
        lines.append(
            "- "
            f"window `{row['window_id']}` family `{row['semantic_family']}` "
            f"source `{row['source_scene']}` "
            f"pose={row['has_pose']} interaction={row['has_interaction']} action={row['has_semantic_action']} "
            f"cowgirl_db={row['has_cowgirl_candidate']} primitive={row['has_motion_primitive']} review={row['has_review_item']}"
        )
    if not lineage_rows:
        lines.append("- None")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

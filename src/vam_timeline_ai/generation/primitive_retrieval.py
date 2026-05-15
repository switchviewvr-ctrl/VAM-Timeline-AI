"""Retrieve abstract primitives for a semantic motion plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl


def retrieve_primitives_for_plan_v0(
    plan: str | Path,
    primitive_groups: str | Path,
    primitives: str | Path,
    out: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    plan_data = load_json(plan)
    group_data = load_json(primitive_groups)
    primitive_rows = load_jsonl(primitives)
    requested = _requested_queries(plan_data)
    matches = []
    for query in requested:
        group_matches = _match_groups(query, group_data.get("groups", []))
        primitive_matches = _match_primitives(query, primitive_rows)
        matches.append({
            "query": query,
            "matching_primitive_groups": group_matches,
            "candidate_primitive_ids": [row.get("primitive_id") for row in primitive_matches[:20]],
            "candidate_count": len(primitive_matches),
            "note": "Retrieved for primitive-space inspection only; not clip stitching.",
        })
    result = {
        "schema": "retrieved_primitives_v0",
        "source_prompt": plan_data.get("source_prompt"),
        "plan_id": plan_data.get("plan_id"),
        "matches": matches,
        "timeline_export_performed": False,
        "clip_stitching_performed": False,
        "warnings": [
            "Retrieval inspects learned primitive space. It does not concatenate Timeline clips.",
            "A future generator must synthesize relative controller tracks from primitive parameters.",
        ],
    }
    dump_json(out, result)
    _write_report(result, report)
    return result


def _requested_queries(plan: dict[str, Any]) -> list[dict[str, Any]]:
    queries = []
    for phase in plan.get("sequence", []) or []:
        query = dict(phase.get("primitive_query") or {})
        query["phase_id"] = phase.get("phase_id")
        query["phase_type"] = phase.get("phase_type")
        queries.append(query)
    return queries


def _match_groups(query: dict[str, Any], groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family = query.get("family")
    subtype = str(query.get("subtype") or "")
    shape = str(query.get("trajectory_shape") or "")
    scored = []
    for group in groups:
        if group.get("family") != family:
            continue
        score = 0
        gid = str(group.get("primitive_set_id") or "")
        if subtype and subtype in gid:
            score += 3
        if "grind" in subtype and "grind" in gid:
            score += 2
        if "bounce" in subtype and "bounce" in gid:
            score += 2
        if "forward" in subtype and "forward" in gid:
            score += 2
        if shape and shape in gid:
            score += 1
        if group.get("cluster_summary", {}).get("count", 0):
            score += 1
        if score:
            scored.append({"primitive_set_id": group.get("primitive_set_id"), "score": score, "count": group.get("cluster_summary", {}).get("count"), "recommended_generation_use": group.get("recommended_generation_use")})
    return sorted(scored, key=lambda r: r["score"], reverse=True)


def _match_primitives(query: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family = query.get("family")
    subtype = str(query.get("subtype") or "")
    scored = []
    for row in rows:
        if row.get("semantic_family") != family:
            continue
        score = 0.0
        row_subtype = str(row.get("subtype") or "")
        if row_subtype == subtype:
            score += 3.0
        elif "grind" in subtype and "grind" in row_subtype:
            score += 2.0
        elif subtype == "riding" and row_subtype in {"riding", "forward_back_rock", "vertical_bounce"}:
            score += 1.5
        score += float((row.get("generation_parameters") or {}).get("source_generation_score") or 0.0)
        if score > 0:
            copy = dict(row)
            copy["_retrieval_score"] = round(score, 6)
            scored.append(copy)
    return sorted(scored, key=lambda r: r.get("_retrieval_score", 0.0), reverse=True)


def _write_report(result: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieved Primitives V0 Report",
        "",
        f"- Prompt: `{result.get('source_prompt')}`",
        f"- Plan: `{result.get('plan_id')}`",
        "- Timeline export performed: false",
        "- Clip stitching performed: false",
        "",
        "## Matches",
        "",
    ]
    for match in result.get("matches", []):
        lines.append(f"- Query `{match.get('query')}`")
        lines.append(f"  - Groups: `{match.get('matching_primitive_groups')}`")
        lines.append(f"  - Candidate primitives: `{match.get('candidate_primitive_ids')[:10]}`")
        lines.append("  - Missing capability: relative track synthesis and retarget-safe Timeline export")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

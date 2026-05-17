"""Align bottom-up candidates with the top-down motion ontology."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.ontology_loader import latest_existing, load_motion_families


def align_candidates_to_motion_ontology_v1(
    run_dir: str | Path,
    ontology: str | Path,
    semantic_db: str | Path,
    cowgirl_db: str | Path,
    resolved: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    semantic_path = Path(semantic_db)
    cowgirl_path = Path(cowgirl_db)
    if not semantic_path.exists():
        semantic_path = latest_existing([run / "datasets" / "semantic_candidate_db_v3.jsonl", run / "datasets" / "semantic_candidate_db_v2.jsonl", run / "datasets" / "semantic_candidate_db_v1.jsonl"]) or semantic_path
    if not cowgirl_path.exists():
        cowgirl_path = latest_existing([run / "datasets" / "cowgirl_candidate_db_v8.jsonl", run / "datasets" / "cowgirl_candidate_db_v7.jsonl", run / "datasets" / "cowgirl_candidate_db_v6.jsonl"]) or cowgirl_path

    families = load_motion_families(ontology)
    resolved_by_window = {str(r.get("window_id")): r for r in load_jsonl(resolved)}
    cowgirl_by_window = {str(r.get("window_id")): r for r in load_jsonl(cowgirl_path)}
    rows = []
    for candidate in load_jsonl(semantic_path):
        wid = str(candidate.get("window_id") or "")
        res = resolved_by_window.get(wid) or {}
        cow = cowgirl_by_window.get(wid) or {}
        rows.append(_align_one(candidate, cow, res, families))

    write_jsonl(out_jsonl, rows)
    counts = Counter(r["ontology_match"] for r in rows)
    conflicts = Counter(flag for r in rows for flag in r.get("ontology_conflict", []))
    priority = Counter(r["recommended_review_priority"] for r in rows)
    lines = [
        "# Ontology Alignment Report V1",
        "",
        "Bottom-up labels are compared against top-down ontology rules. This is not ground truth.",
        "",
        f"- Semantic DB: `{semantic_path}`",
        f"- Cowgirl DB: `{cowgirl_path}`",
        f"- Records: {len(rows)}",
        f"- Ontology match counts: {dict(counts)}",
        f"- Review priority counts: {dict(priority)}",
        f"- Top conflicts: {dict(conflicts.most_common(15))}",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "records": len(rows), "match_counts": dict(counts), "conflict_counts": dict(conflicts), "priority_counts": dict(priority), "out_jsonl": str(out_jsonl), "report": str(report)}


def _align_one(candidate: dict[str, Any], cowgirl: dict[str, Any], resolved: dict[str, Any], families: dict[str, Any]) -> dict[str, Any]:
    source_family = str(candidate.get("semantic_family") or "unknown")
    resolved_family = str(resolved.get("resolved_semantic_family") or "unknown")
    primary = str(resolved.get("primary_motion_center") or "unknown")
    gate = str(resolved.get("clean_motion_gate") or "unknown")
    pose = str(resolved.get("pose_subtype") or candidate.get("pose_subtype") or "unknown")
    category = str(cowgirl.get("category") or cowgirl.get("cowgirl_category") or candidate.get("category") or "")
    conflicts = list(resolved.get("conflict_flags") or [])
    missing = list(resolved.get("missing_requirements") or [])

    if source_family == resolved_family or (source_family == "cowgirl" and resolved_family in {"cowgirl", "pose_context_hold", "transition_setup"}):
        match = "match_or_refined"
    elif resolved_family == "unknown":
        match = "incomplete"
    else:
        match = "conflict"
        conflicts.append(f"source_{source_family}_resolved_{resolved_family}")

    if source_family == "cowgirl" and primary in {"head_neck", "hands"}:
        conflicts.append(f"cowgirl_label_but_{primary}_driver")
    if source_family == "cowgirl" and gate in {"fail_low_motion", "fail_no_driver", "fail_wrong_driver"}:
        conflicts.append(f"cowgirl_label_but_gate_{gate}")
    if "generation_safe" in category and (conflicts or missing):
        conflicts.append("generation_safe_candidate_missing_ontology_requirements")
    if resolved_family == "doggy" and pose not in {"doggy_all_fours", "doggy_bent_forward", "doggy_elevated_support"} and "partner_behind" not in (resolved.get("partner_relation") or []):
        conflicts.append("doggy_without_required_pose_or_partner_behind")
    if resolved_family == "bj_oral" and primary not in {"head_neck", "chest_abdomen"}:
        conflicts.append("bj_without_head_chest_driver")

    requirements_satisfied = not conflicts and not missing and resolved_family != "unknown"
    priority = "review_conflict" if conflicts else "review_missing_requirements" if missing else "spot_check" if requirements_satisfied else "review_unknown"
    return {
        "window_id": candidate.get("window_id"),
        "candidate_id": candidate.get("candidate_id"),
        "source_scene_file": candidate.get("source_scene_file"),
        "source_semantic_family": source_family,
        "resolved_semantic_family": resolved_family,
        "ontology_match": match,
        "ontology_conflict": sorted(set(conflicts)),
        "missing_requirements": missing,
        "generation_requirements_satisfied": requirements_satisfied,
        "recommended_review_priority": priority,
        "primary_motion_center": primary,
        "clean_motion_gate": gate,
        "pose_subtype": pose,
        "not_labels": resolved.get("not_labels") or [],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }

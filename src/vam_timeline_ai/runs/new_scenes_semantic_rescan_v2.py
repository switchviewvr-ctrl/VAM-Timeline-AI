"""Semantic rescan v2 for the clean_v3 new-scene delta run.

This layer is analysis-only. It does not train ML, write manual labels, or
export generated Timeline animation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import html

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl, safe_id_for_path, write_jsonl
from vam_timeline_ai.semantics.ontology_loader import latest_existing, load_motion_families
from vam_timeline_ai.semantics.pose_first_resolver import resolve_candidate
from vam_timeline_ai.ui.review_ui import build_static_review_ui


REQUIRED_BASE_INPUTS = [
    "data/ontology/motion_families_v2.yaml",
    "data/ontology/pose_first_motion_rules_v2.yaml",
    "data/config/manual_gt_motion_amplitude_profiles_v1.yaml",
    "data/runs/clean_v3/manual_pose_ground_truth_v1/manual_pose_ground_truth_v1.jsonl",
    "data/runs/clean_v3/generation/manual_gt_timeline_examples_v4/ACCEPTED_BASELINE_REFERENCE.md",
]

REVIEW_TARGETS = [
    ("cowgirl_clean_motion_candidate", 4),
    ("cowgirl_pose_context_low_motion", 2),
    ("cowgirl_transition_setup", 2),
    ("reverse_cowgirl_candidate", 2),
    ("doggy_classic_candidate", 2),
    ("standing_doggy_candidate", 1),
    ("bj_oral_candidate", 2),
    ("bj_oral_cowgirl_like_pose", 1),
    ("handjob_candidate", 2),
    ("missionary_candidate", 2),
    ("ontology_conflict", 2),
    ("unknown_or_unusable", 2),
    ("standing_hand_head_negative", 2),
]


def resolve_new_scenes_pose_first_semantics_v2(
    new_run: str | Path,
    base_run: str | Path,
    ontology: str | Path,
    rules: str | Path,
    manual_gt: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    new = Path(new_run)
    out_root = Path(out_jsonl).parent
    blocked = _verify_inputs(new, Path(base_run), Path(ontology), Path(rules), Path(manual_gt), out_root)
    if blocked:
        return blocked

    artifacts = _artifact_paths(new)
    candidates = load_jsonl(artifacts["candidate_db"])
    pose_by_window = {str(r.get("window_id")): r for r in load_jsonl(artifacts["pose_semantics"])}
    rel_by_window = {str(r.get("window_id")): r for r in load_jsonl(artifacts["relative_features"])}
    interaction_by_window, interaction_by_pair = _load_best_interactions(artifacts["interaction_semantics"])
    context = _load_run_context(new)
    manual_hints = _manual_gt_hint_index(manual_gt)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        wid = str(candidate.get("window_id") or "")
        pair_id = str(candidate.get("pair_window_id") or "")
        pose = pose_by_window.get(wid) or {}
        rel = rel_by_window.get(wid) or {}
        inter = interaction_by_pair.get(pair_id) if pair_id else None
        inter = inter or interaction_by_window.get(wid) or {}
        resolved = resolve_candidate(candidate, pose, rel, inter)
        resolved = _apply_v2_semantic_corrections(resolved, candidate, pose, rel, inter)
        resolved.update(_source_context_fields(candidate, context))
        resolved["resolver_version"] = "new_scenes_pose_first_semantics_v2"
        resolved["ontology"] = str(ontology)
        resolved["rules"] = str(rules)
        resolved["accepted_manual_gt_reference"] = "manual_gt_timeline_examples_v4"
        resolved["manual_gt_reference_hint"] = _manual_hint_for(resolved, manual_hints)
        resolved["vam_generation_mapping_hint"] = _vam_mapping_hint(resolved)
        resolved["is_human_ground_truth"] = False
        resolved["is_training_label"] = False
        rows.append(resolved)

    write_jsonl(out_jsonl, rows)
    counts = Counter(r.get("resolved_semantic_family", "unknown") for r in rows)
    gates = Counter(r.get("clean_motion_gate", "unknown") for r in rows)
    conflicts = sum(1 for r in rows if r.get("conflict_flags"))
    lines = [
        "# New Scenes Pose-First Semantic Rescan V2",
        "",
        "Analysis-only rescan using ontology v2, pose-first rules v2, and manual GT v4 controller mapping hints.",
        "",
        f"- New run: `{new}`",
        f"- Candidate DB: `{artifacts['candidate_db']}`",
        f"- Records: `{len(rows)}`",
        f"- Resolved family counts: `{dict(counts)}`",
        f"- Clean motion gate counts: `{dict(gates)}`",
        f"- Records with conflicts: `{conflicts}`",
        "- Manual labels modified: `false`",
        "- ML training performed: `false`",
        "- Timeline generation performed: `false`",
    ]
    _write_text(report, lines)
    return {
        "status": "ok",
        "records": len(rows),
        "family_counts": dict(counts),
        "gate_counts": dict(gates),
        "conflicts": conflicts,
        "out_jsonl": str(out_jsonl),
        "report": str(report),
    }


def build_new_scenes_ontology_candidate_db_v2(
    new_run: str | Path,
    resolved: str | Path,
    ontology: str | Path,
    manual_gt: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    families = load_motion_families(ontology)
    _ = families  # Loaded intentionally to validate ontology availability and future extension.
    old_by_window = {str(r.get("window_id")): r for r in load_jsonl(_artifact_paths(Path(new_run))["candidate_db"])}
    rows: list[dict[str, Any]] = []
    for res in load_jsonl(resolved):
        old = old_by_window.get(str(res.get("window_id") or "")) or {}
        category = _candidate_category(res)
        readiness = _generation_readiness(res, category)
        conflicts = list(res.get("conflict_flags") or [])
        if category == "ontology_conflict" and not conflicts:
            conflicts.append("ontology_conflict")
        rows.append(
            {
                **old,
                **res,
                "ontology_category": category,
                "generation_readiness": readiness,
                "ontology_conflict": conflicts,
                "generation_requirements_satisfied": readiness in {"pose_reference_candidate", "motion_reference_candidate"},
                "manual_gt_similarity_hint": res.get("manual_gt_reference_hint"),
                "is_human_ground_truth": False,
                "is_training_label": False,
                "manual_labels_yaml_modified": False,
                "ml_training_run": False,
            }
        )
    write_jsonl(out_jsonl, rows)
    _write_csv(out_csv, rows)
    counts = Counter(r["ontology_category"] for r in rows)
    readiness_counts = Counter(r["generation_readiness"] for r in rows)
    lines = [
        "# New Scenes Ontology-Aligned Candidate DB V2",
        "",
        f"- Records: `{len(rows)}`",
        f"- Category counts: `{dict(counts)}`",
        f"- Generation readiness counts: `{dict(readiness_counts)}`",
        f"- Ontology: `{ontology}`",
        f"- Manual GT reference: `{manual_gt}`",
        "- No candidate is a training label or auto-label.",
    ]
    _write_text(report, lines)
    return {"status": "ok", "records": len(rows), "category_counts": dict(counts), "generation_readiness_counts": dict(readiness_counts), "out_jsonl": str(out_jsonl), "out_csv": str(out_csv), "report": str(report)}


def write_new_scenes_family_reports_v2(new_run: str | Path, candidates: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(candidates)
    _write_overview_report(rows, out / "overview.md")
    report_specs = [
        ("cowgirl_candidates.md", {"cowgirl_clean_motion_candidate", "cowgirl_pose_context_low_motion", "cowgirl_transition_setup", "cowgirl_missing_partner_context"}),
        ("reverse_cowgirl_candidates.md", {"reverse_cowgirl_candidate"}),
        ("doggy_candidates.md", {"doggy_classic_candidate", "standing_doggy_candidate"}),
        ("bj_oral_candidates.md", {"bj_oral_candidate", "bj_oral_cowgirl_like_pose"}),
        ("handjob_candidates.md", {"handjob_candidate"}),
        ("missionary_candidates.md", {"missionary_candidate"}),
        ("receiver_response_candidates.md", {"receiver_response_candidate"}),
        ("conflicts_and_rejections.md", {"ontology_conflict", "unknown_or_unusable", "standing_hand_head_negative"}),
    ]
    for filename, cats in report_specs:
        _write_family_report(rows, cats, out / filename)
    _write_manual_gt_similarity_report(rows, out / "manual_gt_similarity_report.md")
    drift = _write_drift_report(Path(new_run), rows, out.parent / "new_scenes_semantic_drift_v1_to_v2.md")
    return {"status": "ok", "out_dir": str(out), "reports": len(report_specs) + 3, "drift_report": drift}


def export_new_scenes_semantic_review_v2(
    new_run: str | Path,
    candidates: str | Path,
    out_dir: str | Path,
    *,
    count: int = 20,
    build_vam_package: bool = True,
    build_static_ui: bool = True,
) -> dict[str, Any]:
    run = Path(new_run).resolve()
    out = Path(out_dir).resolve()
    _ensure_inside(run, out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(candidates)
    selected, selection = _select_review_rows(rows, count)
    cards = [_review_card(i, row) for i, row in enumerate(selected, start=1)]
    write_jsonl(out / "semantic_review_010.jsonl", cards)
    write_jsonl(out / "strict_novel_review_manifest.jsonl", cards)
    _write_review_answer_sheet(cards, out / "semantic_review_010_answer_sheet.yaml")
    _write_review_md(cards, out / "semantic_review_010.md")
    _write_review_html(cards, out / "semantic_review_010_index.html")
    _write_review_quality_report(cards, selection, out / "review_quality_report.md")
    package = None
    if build_vam_package:
        package = build_vam_review_package(out / "semantic_review_010.jsonl", run, run, out / "vam_review_package", attempt_timeline_segments=False)
    static = None
    if build_static_ui:
        static = build_static_review_ui(run, out, out / "review_ui_static")
    summary = {
        "status": "ok",
        "out_dir": str(out),
        "selected": len(cards),
        "selection": selection,
        "vam_review_package": package,
        "static_review_ui": static,
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }
    dump_json(out / "semantic_review_new_scenes_v2_summary.json", summary)
    return summary


def _verify_inputs(new_run: Path, base_run: Path, ontology: Path, rules: Path, manual_gt: Path, out_root: Path) -> dict[str, Any] | None:
    missing: list[str] = []
    for path in [Path(p) for p in REQUIRED_BASE_INPUTS]:
        if not path.exists():
            missing.append(str(path))
    for path in [new_run / "run_manifest.json", ontology, rules, manual_gt, base_run]:
        if not path.exists():
            missing.append(str(path))
    if new_run.exists():
        artifacts = _artifact_paths(new_run)
        for key, path in artifacts.items():
            if key != "cowgirl_db" and not path.exists():
                missing.append(str(path))
    else:
        missing.append(str(new_run))
    if not missing:
        return None
    out_root.mkdir(parents=True, exist_ok=True)
    blocked = out_root / "BLOCKED_MISSING_NEW_SCENES_ARTIFACTS.md"
    _write_text(
        blocked,
        [
            "# Blocked: Missing New Scenes Artifacts",
            "",
            "The semantic rescan v2 did not run because required inputs are missing.",
            "",
            "## Missing",
            *[f"- `{path}`" for path in sorted(set(missing))],
        ],
    )
    return {"status": "blocked", "missing": sorted(set(missing)), "blocked_report": str(blocked)}


def _artifact_paths(run: Path) -> dict[str, Path]:
    return {
        "candidate_db": latest_existing([run / "datasets" / "semantic_candidate_db_v3.jsonl", run / "datasets" / "semantic_candidate_db_v2.jsonl", run / "datasets" / "semantic_candidate_db_v0.jsonl"]) or run / "datasets" / "semantic_candidate_db_v3.jsonl",
        "cowgirl_db": latest_existing([run / "datasets" / "cowgirl_candidate_db_v8.jsonl", run / "datasets" / "cowgirl_candidate_db_v7.jsonl", run / "datasets" / "cowgirl_candidate_db_v0.jsonl"]) or run / "datasets" / "cowgirl_candidate_db_v8.jsonl",
        "pose_semantics": latest_existing([run / "pose_semantics" / "pose_semantics_lean_back_support_v1.jsonl", run / "pose_semantics" / "pose_semantics_v0.jsonl"]) or run / "pose_semantics" / "pose_semantics_v0.jsonl",
        "relative_features": run / "relative_motion" / "relative_motion_features.jsonl",
        "interaction_semantics": latest_existing([run / "interaction_semantics" / "interaction_semantics_lean_back_support_v1.jsonl", run / "interaction_semantics" / "interaction_semantics_v0.jsonl"]) or run / "interaction_semantics" / "interaction_semantics_v0.jsonl",
    }


def _load_best_interactions(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_window: dict[str, dict[str, Any]] = {}
    by_pair: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        score = _num(row.get("interaction_confidence"))
        wid = str(row.get("window_id") or "")
        pid = str(row.get("pair_window_id") or "")
        if wid and score >= _num((by_window.get(wid) or {}).get("interaction_confidence"), -1):
            by_window[wid] = row
        if pid and score >= _num((by_pair.get(pid) or {}).get("interaction_confidence"), -1):
            by_pair[pid] = row
    return by_window, by_pair


def _load_run_context(run: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "windows": {str(r.get("window_id")): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl")},
        "samples": {str(r.get("sample_id")): r for r in load_jsonl(run / "baked" / "motion_sample_index.jsonl")},
        "sources": {str(r.get("source_id")): r for r in load_jsonl(run / "semantic" / "motion_source_index.jsonl")},
    }


def _source_context_fields(candidate: dict[str, Any], context: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    wid = str(candidate.get("window_id") or "")
    window = context["windows"].get(wid) or {}
    sample_id = str(candidate.get("sample_id") or window.get("sample_id") or "")
    sample = context["samples"].get(sample_id) or {}
    source_id = str(window.get("source_id") or sample.get("source_id") or candidate.get("source_id") or "")
    source = context["sources"].get(source_id) or {}
    start = _first(candidate.get("start_seconds"), window.get("start_seconds"))
    end = _first(candidate.get("end_seconds"), window.get("end_seconds"))
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "source_scene_file": _first(candidate.get("source_scene_file"), window.get("source_scene_file"), sample.get("source_scene_file"), source.get("source_scene_file")),
        "source_scene_path": _first(candidate.get("source_scene_path"), window.get("source_scene_path"), sample.get("source_scene_path"), source.get("source_scene_path")),
        "technical_actor_id": _first(candidate.get("technical_actor_id"), candidate.get("technical_atom_id"), window.get("technical_atom_id"), sample.get("technical_atom_id"), source.get("technical_atom_id")),
        "clip_name": _first(sample.get("clip_name"), source.get("clip_name"), source.get("storable_id")),
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": _duration(start, end),
    }


def _apply_v2_semantic_corrections(resolved: dict[str, Any], candidate: dict[str, Any], pose: dict[str, Any], rel: dict[str, Any], inter: dict[str, Any]) -> dict[str, Any]:
    out = dict(resolved)
    pose_family = str(out.get("pose_family") or candidate.get("pose_family") or pose.get("pose_family") or "")
    pose_subtype = str(out.get("pose_subtype") or "")
    primary = str(out.get("primary_motion_center") or "")
    semantic_family = str(candidate.get("semantic_family") or "")
    phase = str(candidate.get("phase") or "")
    partner_relation = set(out.get("partner_relation") or [])
    conflicts = list(out.get("conflict_flags") or [])
    not_labels = set(out.get("not_labels") or [])

    if semantic_family == "bj_oral" or primary == "head_neck":
        out["resolved_semantic_family"] = "bj_oral"
        out["resolved_motion_subtype"] = "bj_head_bob"
        out["clean_motion_gate"] = "pass" if primary in {"head_neck", "chest_abdomen"} else "fail_wrong_driver"
        not_labels.update(["cowgirl_clean_motion", "doggy"])
    if semantic_family in {"handjob", "hand_gesture"} and primary == "hands":
        if pose_family in {"standing", "hand_head_gesture"}:
            out["resolved_semantic_family"] = "standing_hand_head"
            out["clean_motion_gate"] = "fail_wrong_pose"
            not_labels.update(["cowgirl", "doggy"])
        else:
            out["resolved_semantic_family"] = "handjob"
            out["resolved_motion_subtype"] = "hand_repetitive_up_down"
            out["clean_motion_gate"] = "pass"
            not_labels.add("cowgirl_clean_motion")
    if pose_family == "hand_head_gesture" and out.get("resolved_semantic_family") in {"unknown", "cowgirl"}:
        out["resolved_semantic_family"] = "standing_hand_head"
        out["clean_motion_gate"] = "fail_wrong_pose"
        not_labels.update(["cowgirl", "doggy"])
    if out.get("resolved_semantic_family") == "doggy" and pose_subtype not in {"doggy_all_fours", "doggy_bent_forward", "doggy_elevated_support", "doggy_drop_flat", "doggy_arched_upright"} and not ({"partner_behind", "behind_receiver"} & partner_relation):
        conflicts.append("doggy_requires_support_or_partner_behind_v2")
        out["clean_motion_gate"] = "fail_missing_partner_context"
    if out.get("resolved_semantic_family") == "cowgirl":
        if primary not in {"pelvis_hip", "thighs"}:
            conflicts.append("cowgirl_v2_requires_pelvis_hip_or_thigh_driver")
        if not ({"rider_over_receiver", "pelvis_aligned", "rider_facing_partner"} & partner_relation):
            out["clean_motion_gate"] = "fail_missing_partner_context"
            out.setdefault("missing_requirements", []).append("partner_relation_from_v2")
        if phase in {"low_motion_hold", "pose_context_only"} or primary == "static_pose":
            out["resolved_semantic_family"] = "pose_context_hold"
            out["resolved_motion_subtype"] = "cowgirl_pose_context_low_motion"
            out["clean_motion_gate"] = "fail_low_motion"
            not_labels.add("cowgirl_clean_motion")
    out["conflict_flags"] = sorted(set(conflicts))
    out["not_labels"] = sorted(not_labels)
    out["explanation"] = str(out.get("explanation") or "") + "; v2 uses manual_gt_v4 hipControl mapping for generation expectations"
    return out


def _candidate_category(row: dict[str, Any]) -> str:
    fam = str(row.get("resolved_semantic_family") or "unknown")
    gate = str(row.get("clean_motion_gate") or "")
    pose = str(row.get("pose_subtype") or "")
    phase = str(row.get("phase") or "")
    conflicts = row.get("conflict_flags") or []
    missing = row.get("missing_requirements") or []
    if conflicts:
        return "ontology_conflict"
    if fam == "cowgirl":
        if gate == "fail_missing_partner_context" or missing:
            return "cowgirl_missing_partner_context"
        if gate == "fail_low_motion":
            return "cowgirl_pose_context_low_motion"
        if phase == "transition_setup":
            return "cowgirl_transition_setup"
        return "cowgirl_clean_motion_candidate" if gate in {"pass", "soft_pass_short"} else "cowgirl_pose_context_low_motion"
    if fam == "pose_context_hold":
        return "cowgirl_pose_context_low_motion" if "cowgirl" in pose else "unknown_or_unusable"
    if fam == "transition_setup":
        return "cowgirl_transition_setup" if "cowgirl" in pose else "unknown_or_unusable"
    if fam == "reverse_cowgirl":
        return "reverse_cowgirl_candidate"
    if fam == "doggy":
        return "standing_doggy_candidate" if "standing" in pose else "doggy_classic_candidate"
    if fam == "bj_oral":
        return "bj_oral_cowgirl_like_pose" if "cowgirl" in pose else "bj_oral_candidate"
    if fam == "handjob":
        return "handjob_candidate"
    if fam == "missionary":
        return "missionary_candidate"
    if fam == "receiver_response":
        return "receiver_response_candidate"
    if fam == "standing_hand_head":
        return "standing_hand_head_negative"
    return "unknown_or_unusable"


def _generation_readiness(row: dict[str, Any], category: str) -> str:
    if category == "ontology_conflict":
        return "needs_human_review"
    if row.get("clean_motion_gate") in {"fail_wrong_pose", "fail_wrong_driver", "fail_missing_partner_context"}:
        return "needs_human_review"
    if category.endswith("_candidate") and row.get("clean_motion_gate") in {"pass", "soft_pass_short"}:
        return "motion_reference_candidate"
    if "pose_context" in category or "transition" in category:
        return "pose_reference_candidate"
    if category in {"unknown_or_unusable", "standing_hand_head_negative"}:
        return "rejected"
    return "not_ready"


def _manual_gt_hint_index(manual_gt: str | Path) -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in load_jsonl(manual_gt):
        labels = row.get("human_labels") or {}
        counts[(str(labels.get("family") or "unknown"), str(labels.get("pose_subtype") or "unknown"))] += 1
    return dict(counts)


def _manual_hint_for(row: dict[str, Any], hints: dict[tuple[str, str], int]) -> dict[str, Any]:
    family = str(row.get("resolved_semantic_family") or "unknown")
    subtype = str(row.get("pose_subtype") or "unknown")
    exact = hints.get((family, subtype), 0)
    family_count = sum(count for (fam, _), count in hints.items() if fam == family)
    return {"family": family, "pose_subtype": subtype, "exact_manual_gt_count": exact, "family_manual_gt_count": family_count, "accepted_reference": "manual_gt_timeline_examples_v4"}


def _vam_mapping_hint(row: dict[str, Any]) -> dict[str, Any]:
    family = str(row.get("resolved_semantic_family") or "")
    if family in {"cowgirl", "reverse_cowgirl"}:
        return {"semantic_center": "pelvis_hip", "vam_primary_driver": "hipControl", "pelvisControl": "light_follower_or_static", "anchors_static": True}
    if family == "bj_oral":
        return {"vam_primary_driver": "headControl", "secondary": "chestControl", "hipControl": "static", "pelvisControl": "static"}
    if family == "handjob":
        return {"vam_primary_driver": "handControl", "hipControl": "static", "pelvisControl": "static"}
    return {}


def _select_review_rows(rows: list[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("ontology_category") or "unknown_or_unusable")].append(row)
    for cat_rows in by_category.values():
        cat_rows.sort(key=lambda r: (-_num(r.get("confidence")), str(r.get("source_scene_file") or ""), _num(r.get("start_seconds"))))
    selected: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_sources: set[str] = set()
    scene_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()

    def add(row: dict[str, Any]) -> bool:
        scene = str(row.get("source_scene_file") or "unknown")
        sample = str(row.get("sample_id") or "")
        source = str(row.get("source_id") or "")
        if scene_counts[scene] >= 2:
            return False
        if sample and sample in seen_samples:
            return False
        if source and source in seen_sources:
            return False
        selected.append(row)
        scene_counts[scene] += 1
        selected_counts[str(row.get("ontology_category") or "unknown")] += 1
        if sample:
            seen_samples.add(sample)
        if source:
            seen_sources.add(source)
        return True

    for category, target in REVIEW_TARGETS:
        for row in by_category.get(category, []):
            if selected_counts[category] >= target or len(selected) >= count:
                break
            add(row)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for row in sorted(rows, key=lambda r: (-_num(r.get("confidence")), str(r.get("ontology_category") or ""))):
            if len(selected) >= count:
                break
            if row not in selected:
                add(row)
    return selected, {"selected_counts": dict(selected_counts), "scene_counts": dict(scene_counts), "target_counts": dict(REVIEW_TARGETS), "selected": len(selected)}


def _review_card(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("ontology_category") or "unknown")
    why_not = _why_not_cowgirl(row)
    return {
        "review_id": f"review_{idx:03d}",
        "review_label": f"{idx:03d}_{safe_id_for_path(category)}",
        "why_selected": category,
        "semantic_family": row.get("resolved_semantic_family"),
        "resolved_semantic_family_v2": row.get("resolved_semantic_family"),
        "old_system_guess": row.get("source_semantic_family") or row.get("semantic_family"),
        "pose_family": row.get("pose_family"),
        "pose_subtype": row.get("pose_subtype"),
        "primary_motion_center": row.get("primary_motion_center"),
        "target_region": row.get("target_region"),
        "contact_support": row.get("contact_support"),
        "role_context": row.get("role_context"),
        "clean_motion_gate": row.get("clean_motion_gate"),
        "conflict_flags": row.get("conflict_flags") or row.get("ontology_conflict") or [],
        "not_labels": row.get("not_labels") or [],
        "explanation": row.get("explanation"),
        "why_not_cowgirl": why_not,
        "manual_gt_similarity_hint": row.get("manual_gt_similarity_hint") or row.get("manual_gt_reference_hint"),
        "source_scene_file": row.get("source_scene_file"),
        "source_scene_path": row.get("source_scene_path"),
        "source_id": row.get("source_id"),
        "sample_id": row.get("sample_id"),
        "technical_actor_id": row.get("technical_actor_id"),
        "window_id": row.get("window_id"),
        "start_seconds": row.get("start_seconds"),
        "end_seconds": row.get("end_seconds"),
        "motion_subtype": row.get("resolved_motion_subtype") or row.get("motion_subtype"),
        "generation_readiness": row.get("generation_readiness"),
        "generation_safe": False,
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _why_not_cowgirl(row: dict[str, Any]) -> str:
    fam = str(row.get("resolved_semantic_family") or "")
    if fam == "cowgirl":
        return ""
    if fam == "bj_oral":
        return "Head/chest driver or partner-pelvis target indicates BJ/oral, not Cowgirl."
    if fam == "handjob":
        return "Hand driver near partner target indicates handjob/hand interaction, not Cowgirl."
    if fam == "doggy":
        return "Doggy requires support/partner-behind relation and is not rider-over-receiver Cowgirl."
    if fam == "standing_hand_head":
        return "Standing hand/head gesture lacks rider hip-driven relation."
    if row.get("clean_motion_gate") == "fail_low_motion":
        return "Pose/hold or low motion is not clean Cowgirl motion."
    return "Ontology v2 did not find enough Cowgirl driver + relation evidence."


def _write_review_answer_sheet(cards: list[dict[str, Any]], path: Path) -> None:
    data = {
        "allowed_review_labels": [
            "correct_family",
            "wrong_family",
            "correct_pose_context",
            "wrong_driver",
            "wrong_partner_relation",
            "low_motion_or_transition",
            "usable_pose_reference",
            "usable_motion_reference",
            "unknown_unclear",
        ],
        "reviews": {card["review_id"]: {"family_correct": "unknown", "pose_correct": "unknown", "motion_driver_correct": "unknown", "review_labels": [], "notes": ""} for card in cards},
        "audit_only": True,
        "manual_labels_yaml_modified": False,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_review_md(cards: list[dict[str, Any]], path: Path) -> None:
    lines = ["# New Scenes Semantic Review V2", "", "Audit-only review batch. Not training truth.", ""]
    for card in cards:
        lines.extend([f"## {card['review_label']}", "", f"- Family v2: `{card.get('resolved_semantic_family_v2')}`", f"- Old guess: `{card.get('old_system_guess')}`", f"- Scene/time: `{card.get('source_scene_file')}` `{card.get('start_seconds')}`-`{card.get('end_seconds')}`", f"- Driver: `{card.get('primary_motion_center')}`", f"- Gate: `{card.get('clean_motion_gate')}`", f"- Why not Cowgirl: {card.get('why_not_cowgirl')}", f"- Explanation: {card.get('explanation')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_html(cards: list[dict[str, Any]], path: Path) -> None:
    body = ["<!doctype html><meta charset='utf-8'><title>New Scenes Semantic Review V2</title>", "<style>body{font-family:system-ui;margin:24px;background:#f6f6f3}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:14px;margin:12px 0}code{background:#eee;padding:2px 4px;border-radius:4px}</style>", "<h1>New Scenes Semantic Review V2</h1>"]
    for c in cards:
        body.append(f"<section class='card'><h2>{html.escape(c['review_label'])}</h2><p><b>V2:</b> <code>{html.escape(str(c.get('resolved_semantic_family_v2')))}</code> <b>Old:</b> <code>{html.escape(str(c.get('old_system_guess')))}</code></p><p><b>Pose:</b> {html.escape(str(c.get('pose_subtype')))} <b>Driver:</b> {html.escape(str(c.get('primary_motion_center')))} <b>Gate:</b> {html.escape(str(c.get('clean_motion_gate')))}</p><p><b>Scene:</b> <code>{html.escape(str(c.get('source_scene_file')))}</code> {html.escape(str(c.get('start_seconds')))}-{html.escape(str(c.get('end_seconds')))}s</p><p><b>Why not Cowgirl:</b> {html.escape(str(c.get('why_not_cowgirl')))}</p><p>{html.escape(str(c.get('explanation')))}</p></section>")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_review_quality_report(cards: list[dict[str, Any]], selection: dict[str, Any], path: Path) -> None:
    lines = ["# Review Quality Report V2", "", f"- Selected: `{len(cards)}`", f"- Selected counts: `{selection.get('selected_counts')}`", f"- Scene counts: `{selection.get('scene_counts')}`", "- Duplicate padding used: `false`", "- Timeline generation performed: `false`"]
    _write_text(path, lines)


def _write_overview_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# New Scenes Semantic Rescan V2 Overview", "", f"- Records: `{len(rows)}`", f"- Family counts: `{dict(Counter(r.get('resolved_semantic_family') for r in rows))}`", f"- Category counts: `{dict(Counter(r.get('ontology_category') for r in rows))}`", f"- Readiness counts: `{dict(Counter(r.get('generation_readiness') for r in rows))}`"]
    _write_text(path, lines)


def _write_family_report(rows: list[dict[str, Any]], categories: set[str], path: Path) -> None:
    subset = [r for r in rows if r.get("ontology_category") in categories]
    lines = [f"# {path.stem.replace('_', ' ').title()}", "", f"- Records: `{len(subset)}`", f"- Category counts: `{dict(Counter(r.get('ontology_category') for r in subset))}`", f"- Scene counts top 12: `{dict(Counter(r.get('source_scene_file') for r in subset).most_common(12))}`", "", "## Top Candidates", ""]
    for row in sorted(subset, key=lambda r: -_num(r.get("confidence")))[:25]:
        lines.append(f"- `{row.get('window_id')}` `{row.get('resolved_semantic_family')}` `{row.get('clean_motion_gate')}` conf `{row.get('confidence')}` scene `{row.get('source_scene_file')}`: {row.get('explanation')}")
    _write_text(path, lines)


def _write_manual_gt_similarity_report(rows: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(str((r.get("manual_gt_similarity_hint") or {}).get("family")) for r in rows)
    exact = sum(1 for r in rows if (r.get("manual_gt_similarity_hint") or {}).get("exact_manual_gt_count"))
    lines = ["# Manual GT Similarity Report", "", f"- Records with exact family/subtype hint: `{exact}`", f"- Family hint counts: `{dict(counts)}`", "- Reference package: `manual_gt_timeline_examples_v4`"]
    _write_text(path, lines)


def _write_drift_report(new_run: Path, rows: list[dict[str, Any]], path: Path) -> str:
    old = {str(r.get("window_id")): r for r in load_jsonl(_artifact_paths(new_run)["candidate_db"])}
    transitions = Counter()
    category_transitions = Counter()
    for row in rows:
        old_row = old.get(str(row.get("window_id") or "")) or {}
        transitions[(str(old_row.get("semantic_family") or "unknown"), str(row.get("resolved_semantic_family") or "unknown"))] += 1
        category_transitions[(str(old_row.get("category") or old_row.get("semantic_family") or "unknown"), str(row.get("ontology_category") or "unknown"))] += 1
    lines = ["# New Scenes Semantic Drift V1 To V2", "", f"- Compared records: `{len(rows)}`", "## Family Transitions", ""]
    lines.extend(f"- `{a}` -> `{b}`: {n}" for (a, b), n in transitions.most_common(30))
    lines.extend(["", "## Category Transitions", ""])
    lines.extend(f"- `{a}` -> `{b}`: {n}" for (a, b), n in category_transitions.most_common(30))
    _write_text(path, lines)
    return str(path)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = ["window_id", "candidate_id", "source_scene_file", "source_id", "sample_id", "technical_actor_id", "resolved_semantic_family", "ontology_category", "generation_readiness", "pose_subtype", "primary_motion_center", "target_region", "contact_support", "clean_motion_gate", "confidence", "conflict_flags", "not_labels", "explanation"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    return value


def _ensure_inside(run: Path, out: Path) -> None:
    try:
        out.relative_to(run)
    except ValueError as exc:
        raise ValueError(f"output must stay inside {run}") from exc
    if run.name == "clean_v3":
        raise ValueError("new-scene rescan must not write inside clean_v3")


def _duration(start: Any, end: Any) -> float | None:
    try:
        return float(end) - float(start)
    except Exception:
        return None


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_text(path: str | Path, lines: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Ontology-v2 semantic rescan for the clean_v3 new-scene delta run.

This is an analysis/review-assist layer only. It does not train ML, does not
write manual labels, and does not generate Timeline animations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import html
import json

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl, safe_id_for_path, write_jsonl
from vam_timeline_ai.semantics.pose_first_resolver import resolve_candidate
from vam_timeline_ai.ui.review_ui import build_static_review_ui


RESCAN_SCHEMA = "new_scenes_semantic_rescan_v2"
ACCEPTED_BASELINE_NAME = "manual_gt_timeline_examples_v4"
VAM_DRIVER_MAPPING_HINT = {
    "semantic_motion_center": "pelvis_hip",
    "vam_primary_visible_driver": "hipControl",
    "vam_secondary_or_follower": ["pelvisControl", "abdomenControl", "abdomen2Control"],
    "note": "Analysis hint only: future VaM exports should not use pelvisControl as sole Cowgirl driver.",
}


def resolve_new_scenes_pose_first_semantics_v2(
    new_run: str | Path,
    base_run: str | Path,
    ontology: str | Path,
    rules: str | Path,
    manual_gt: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    new_run_path = Path(new_run)
    base_run_path = Path(base_run)
    out_path = Path(out_jsonl)
    blocked = _verify_inputs(new_run_path, base_run_path, Path(ontology), Path(rules), Path(manual_gt))
    if blocked:
        blocked_report = out_path.parent / "BLOCKED_MISSING_NEW_SCENES_ARTIFACTS.md"
        _write_blocked_report(blocked_report, blocked)
        return {"status": "blocked", "missing": blocked, "blocked_report": str(blocked_report)}

    candidates_path = _latest_candidate_db(new_run_path)
    candidates = load_jsonl(candidates_path)
    pose_by_window = _by_key(load_jsonl(new_run_path / "pose_semantics" / "pose_semantics_v0.jsonl"), "window_id")
    rel_by_window = _by_key(load_jsonl(new_run_path / "relative_motion" / "relative_motion_features.jsonl"), "window_id")
    interaction_by_window = _best_interaction_by_window(new_run_path / "interaction_semantics" / "interaction_semantics_v0.jsonl")
    windows = _by_key(load_jsonl(new_run_path / "semantic" / "movement_windows.jsonl"), "window_id")
    samples = _by_key(load_jsonl(new_run_path / "baked" / "motion_sample_index.jsonl"), "sample_id")
    manual_gt_hints = _manual_gt_hints(load_jsonl(manual_gt))

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        wid = str(candidate.get("window_id") or "")
        sample_id = str(candidate.get("sample_id") or "")
        base = resolve_candidate(
            candidate,
            pose_by_window.get(wid) or {},
            rel_by_window.get(wid) or {},
            interaction_by_window.get(wid) or {},
        )
        context = {
            "candidate": candidate,
            "pose": pose_by_window.get(wid) or {},
            "relative": rel_by_window.get(wid) or {},
            "interaction": interaction_by_window.get(wid) or {},
            "window": windows.get(wid) or {},
            "sample": samples.get(sample_id) or {},
        }
        resolved = _apply_v2_rules(base, context)
        _add_context_fields(resolved, context)
        _add_manual_gt_hint(resolved, manual_gt_hints)
        resolved.update(
            {
                "schema": RESCAN_SCHEMA,
                "ontology_path": str(ontology),
                "rules_path": str(rules),
                "accepted_manual_gt_reference": ACCEPTED_BASELINE_NAME,
                "vam_driver_mapping_hint": VAM_DRIVER_MAPPING_HINT,
                "manual_labels_modified": False,
                "ml_training_performed": False,
                "timeline_generation_performed": False,
                "is_human_ground_truth": False,
                "is_training_label": False,
                "auto_label": False,
            }
        )
        rows.append(resolved)

    write_jsonl(out_path, rows)
    summary = _resolved_summary(rows)
    _write_resolver_report(Path(report), rows, candidates_path, summary)
    return {"status": "ok", "records": len(rows), "out_jsonl": str(out_path), "report": str(report), **summary}


def build_new_scenes_ontology_candidate_db_v2(
    new_run: str | Path,
    resolved: str | Path,
    ontology: str | Path,
    manual_gt: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    rows = load_jsonl(resolved)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        category = _candidate_category(row)
        out = dict(row)
        out.update(
            {
                "schema": "new_scenes_ontology_aligned_candidate_v2",
                "category": category,
                "ontology_match": _ontology_match(row, category),
                "ontology_conflict": bool(row.get("conflict_flags")),
                "generation_readiness": _generation_readiness(row, category),
                "recommended_review_priority": _review_priority(row, category),
                "candidate_not_ground_truth": True,
                "manual_labels_modified": False,
                "ml_training_performed": False,
                "timeline_generation_performed": False,
            }
        )
        out_rows.append(out)

    write_jsonl(out_jsonl, out_rows)
    _write_csv(out_csv, out_rows)
    counts = Counter(r["category"] for r in out_rows)
    readiness = Counter(r["generation_readiness"] for r in out_rows)
    _write_alignment_report(Path(report), out_rows, Path(resolved), Path(ontology), Path(manual_gt), counts, readiness)
    return {
        "status": "ok",
        "records": len(out_rows),
        "category_counts": dict(counts),
        "generation_readiness_counts": dict(readiness),
        "out_jsonl": str(out_jsonl),
        "out_csv": str(out_csv),
        "report": str(report),
    }


def write_new_scenes_family_reports_v2(
    new_run: str | Path,
    candidates: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    rows = load_jsonl(candidates)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_overview_report(out / "overview.md", rows)
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
    for name, cats in report_specs:
        _write_category_report(out / name, rows, cats)
    _write_manual_gt_similarity_report(out / "manual_gt_similarity_report.md", rows)
    drift = _write_drift_report(Path(new_run), rows, out.parent / "new_scenes_semantic_drift_v1_to_v2.md")
    return {"status": "ok", "reports_dir": str(out), "reports": len(report_specs) + 2, "drift_report": str(drift)}


def export_new_scenes_semantic_review_v2(
    new_run: str | Path,
    candidates: str | Path,
    out_dir: str | Path,
    count: int = 20,
    build_vam_package: bool = True,
    build_static_ui: bool = True,
) -> dict[str, Any]:
    new_run_path = Path(new_run)
    rows = load_jsonl(candidates)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reviewed = _previously_reviewed_window_ids(new_run_path, out)
    selected = _select_review_items(rows, count, reviewed)
    review_rows = [_review_row(row, i + 1) for i, row in enumerate(selected)]

    review_jsonl = out / "semantic_review_010.jsonl"
    write_jsonl(review_jsonl, review_rows)
    _write_review_markdown(out / "semantic_review_010.md", review_rows)
    _write_answer_sheet(out / "semantic_review_010_answer_sheet.yaml", review_rows)
    _write_review_index(out / "semantic_review_010_index.html", review_rows)
    _write_review_quality_report(out / "review_quality_report.md", rows, review_rows, reviewed, count)

    package_summary = None
    if build_vam_package:
        package_summary = build_vam_review_package(
            review_jsonl,
            new_run_path,
            new_run_path,
            out / "vam_review_package",
            attempt_timeline_segments=False,
        )
    ui_summary = None
    if build_static_ui:
        ui_summary = build_static_review_ui(new_run_path, out, out / "review_ui_static")
    return {
        "status": "ok",
        "review_items": len(review_rows),
        "out_dir": str(out),
        "review_jsonl": str(review_jsonl),
        "vam_package": package_summary,
        "static_ui": ui_summary,
        "timeline_generation_performed": False,
        "ml_training_performed": False,
        "manual_labels_modified": False,
    }


def _apply_v2_rules(base: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    candidate = context["candidate"]
    interaction = context["interaction"]
    pose_family = str(row.get("pose_family") or "unknown")
    pose_subtype = str(row.get("pose_subtype") or "unknown")
    primary = str(row.get("primary_motion_center") or "unknown")
    target = str(row.get("target_region") or "unknown")
    relation = set(_as_list(row.get("partner_relation") or candidate.get("partner_relation") or interaction.get("partner_relation")))
    old_family = str(candidate.get("semantic_family") or "")
    phase = str(candidate.get("phase") or "")
    conflicts = list(row.get("conflict_flags") or [])
    not_labels = set(row.get("not_labels") or [])
    missing = list(row.get("missing_requirements") or [])

    cowgirl_like = pose_family == "cowgirl" or "cowgirl" in pose_subtype or old_family == "cowgirl"
    standing_like = pose_family in {"standing", "hand_head_gesture"} or "standing" in pose_subtype or old_family in {"hand_gesture", "standing_hand_head"}
    doggy_pose = "doggy" in pose_subtype or pose_family == "doggy"
    doggy_relation = bool(relation & {"partner_behind", "behind_receiver"})
    low_motion = phase in {"low_motion_hold", "pose_context_only", "intro_alignment", "transition_setup"} or row.get("clean_motion_gate") in {"fail_low_motion", "fail_no_driver"}

    if cowgirl_like and primary == "head_neck":
        _set_resolution(row, "bj_oral", "bj_head_bob", "fail_wrong_driver", 0.66)
        conflicts.append("v2_cowgirl_like_pose_with_head_driver_is_bj_oral_candidate")
        not_labels.add("cowgirl_clean_motion")
    elif cowgirl_like and primary == "hands":
        family = "handjob" if target in {"partner_pelvis_or_genital_area", "partner_hips", "partner_unknown", "unknown"} else "hand_touching"
        _set_resolution(row, family, "hand_repetitive_up_down" if family == "handjob" else "hand_touching", "fail_wrong_driver", 0.64)
        conflicts.append("v2_cowgirl_like_pose_with_hand_driver_not_cowgirl")
        not_labels.add("cowgirl_clean_motion")
    elif standing_like and primary in {"hands", "head_neck"}:
        if primary == "hands" and target in {"partner_pelvis_or_genital_area", "partner_hips", "partner_unknown"}:
            _set_resolution(row, "handjob", "hand_repetitive_up_down", "pass", 0.68)
        else:
            _set_resolution(row, "standing_hand_head", "standing_hand_head_gesture", "fail_wrong_pose", 0.72)
        not_labels.update({"cowgirl", "cowgirl_clean_motion"})
    elif doggy_pose or doggy_relation:
        if doggy_pose or doggy_relation:
            if pose_subtype in {"cowgirl_kneeling", "cowgirl_upright"} and not doggy_relation:
                _set_resolution(row, "unknown", "unknown", "fail_wrong_pose", 0.35)
                conflicts.append("v2_kneeling_alone_does_not_imply_doggy")
                not_labels.add("doggy")
            else:
                _set_resolution(row, "doggy", "doggy_forward_back", row.get("clean_motion_gate") or "pass", max(float(row.get("confidence") or 0.0), 0.66))
    elif row.get("resolved_semantic_family") == "cowgirl":
        if low_motion:
            _set_resolution(row, "pose_context_hold", "cowgirl_pose_context_low_motion", "fail_low_motion", 0.58)
            not_labels.add("cowgirl_clean_motion")
        elif "rider_over_receiver" not in relation and "pelvis_aligned" not in relation:
            row["clean_motion_gate"] = "fail_missing_partner_context"
            missing.append("rider_over_receiver_or_pelvis_aligned")
            conflicts.append("v2_cowgirl_requires_partner_relation_for_clean_motion")
        if primary == "pelvis_hip":
            row["vam_primary_driver_hint"] = "hipControl"
    elif row.get("resolved_semantic_family") == "reverse_cowgirl":
        if not (relation & {"back_to_partner", "facing_away_from_partner"}) and "reverse" not in str(candidate.get("facing_context") or ""):
            row["clean_motion_gate"] = "fail_missing_partner_context"
            missing.append("back_to_partner_or_facing_away")
            conflicts.append("v2_reverse_cowgirl_requires_back_to_partner_evidence")
        if primary == "pelvis_hip":
            row["vam_primary_driver_hint"] = "hipControl"

    if row.get("resolved_semantic_family") == "cowgirl" and primary not in {"pelvis_hip", "thighs"}:
        conflicts.append("v2_cowgirl_clean_requires_pelvis_hip_or_thigh_driver")
        not_labels.add("cowgirl_clean_motion")
    if row.get("resolved_semantic_family") == "doggy" and not (doggy_pose or doggy_relation):
        conflicts.append("v2_doggy_requires_support_pose_or_partner_behind")
        not_labels.add("doggy")

    row["conflict_flags"] = sorted(set(str(c) for c in conflicts if c))
    row["not_labels"] = sorted(str(n) for n in not_labels if n)
    row["missing_requirements"] = sorted(set(str(m) for m in missing if m))
    row["explanation"] = _v2_explanation(row)
    row["why_not_cowgirl"] = _why_not_cowgirl(row)
    return row


def _set_resolution(row: dict[str, Any], family: str, subtype: str, gate: str, confidence: float) -> None:
    row["resolved_semantic_family"] = family
    row["resolved_motion_subtype"] = subtype
    row["clean_motion_gate"] = gate
    row["confidence"] = round(confidence, 3)


def _add_context_fields(row: dict[str, Any], context: dict[str, Any]) -> None:
    candidate = context["candidate"]
    window = context["window"]
    sample = context["sample"]
    for key in ["sample_id", "source_id", "source_scene_path", "source_scene_file", "technical_atom_id", "start_seconds", "end_seconds", "duration_seconds"]:
        row[key] = _first(row.get(key), candidate.get(key), window.get(key), sample.get(key))
    row["old_system_guess"] = candidate.get("semantic_family")
    row["old_motion_subtype"] = candidate.get("motion_subtype")
    row["old_clean_motion_gate"] = candidate.get("clean_motion_gate")
    row["semantic_score"] = candidate.get("semantic_score")
    row["motion_score"] = candidate.get("motion_score")
    row["pose_score"] = candidate.get("pose_score")
    row["interaction_score"] = candidate.get("interaction_score")


def _add_manual_gt_hint(row: dict[str, Any], hints: dict[str, Any]) -> None:
    family = str(row.get("resolved_semantic_family") or "unknown")
    subtype = str(row.get("pose_subtype") or "unknown")
    row["manual_gt_reference_hint"] = {
        "accepted_reference": ACCEPTED_BASELINE_NAME,
        "matching_family_examples": hints["families"].get(family, 0),
        "matching_subtype_examples": hints["subtypes"].get(subtype, 0),
        "controller_mapping_note": "For VaM, pelvis_hip motion should be checked against hipControl-primary manual GT v4.",
    }


def _candidate_category(row: dict[str, Any]) -> str:
    family = str(row.get("resolved_semantic_family") or "unknown")
    gate = str(row.get("clean_motion_gate") or "")
    subtype = str(row.get("pose_subtype") or "")
    conflicts = row.get("conflict_flags") or []
    if conflicts and family in {"unknown", "cowgirl"} and gate.startswith("fail"):
        return "ontology_conflict"
    if family == "cowgirl":
        if gate == "pass" or gate == "soft_pass_short":
            return "cowgirl_clean_motion_candidate"
        if gate == "fail_low_motion":
            return "cowgirl_pose_context_low_motion"
        if gate == "fail_missing_partner_context":
            return "cowgirl_missing_partner_context"
        return "cowgirl_transition_setup"
    if family == "pose_context_hold" and "cowgirl" in str(row.get("resolved_motion_subtype") or ""):
        return "cowgirl_pose_context_low_motion"
    if family == "transition_setup":
        return "cowgirl_transition_setup"
    if family == "reverse_cowgirl":
        return "reverse_cowgirl_candidate"
    if family == "doggy":
        return "standing_doggy_candidate" if "standing" in subtype else "doggy_classic_candidate"
    if family == "bj_oral":
        return "bj_oral_cowgirl_like_pose" if "cowgirl" in subtype or row.get("source_semantic_family") == "cowgirl" else "bj_oral_candidate"
    if family in {"handjob", "hand_touching"}:
        return "handjob_candidate"
    if family == "missionary":
        return "missionary_candidate"
    if family == "receiver_response":
        return "receiver_response_candidate"
    if family == "standing_hand_head":
        return "standing_hand_head_negative"
    return "unknown_or_unusable"


def _generation_readiness(row: dict[str, Any], category: str) -> str:
    if category in {"unknown_or_unusable", "ontology_conflict", "standing_hand_head_negative"}:
        return "rejected" if category == "unknown_or_unusable" else "needs_human_review"
    if category.endswith("low_motion") or "pose_context" in category:
        return "pose_reference_candidate"
    if category in {"cowgirl_clean_motion_candidate", "reverse_cowgirl_candidate", "doggy_classic_candidate", "standing_doggy_candidate", "bj_oral_candidate", "bj_oral_cowgirl_like_pose", "handjob_candidate", "missionary_candidate"}:
        return "motion_reference_candidate" if not row.get("conflict_flags") else "needs_human_review"
    return "not_ready"


def _review_priority(row: dict[str, Any], category: str) -> str:
    if category == "ontology_conflict" or row.get("conflict_flags"):
        return "must_review"
    if row.get("clean_motion_gate") in {"pass", "soft_pass_short"} and float(row.get("confidence") or 0.0) >= 0.65:
        return "high"
    if category == "unknown_or_unusable":
        return "low"
    return "medium"


def _ontology_match(row: dict[str, Any], category: str) -> bool:
    return category not in {"ontology_conflict", "unknown_or_unusable"} and not bool(row.get("conflict_flags"))


def _select_review_items(rows: list[dict[str, Any]], count: int, reviewed: set[str]) -> list[dict[str, Any]]:
    quotas = [
        (4, {"cowgirl_clean_motion_candidate"}),
        (2, {"cowgirl_pose_context_low_motion", "cowgirl_transition_setup", "cowgirl_missing_partner_context"}),
        (2, {"reverse_cowgirl_candidate"}),
        (3, {"doggy_classic_candidate", "standing_doggy_candidate"}),
        (3, {"bj_oral_candidate", "bj_oral_cowgirl_like_pose"}),
        (2, {"handjob_candidate"}),
        (2, {"missionary_candidate"}),
        (2, {"ontology_conflict", "unknown_or_unusable", "standing_hand_head_negative"}),
    ]
    selected: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    used_samples: set[str] = set()
    used_windows: set[str] = set()
    priority_order = {"must_review": 0, "high": 1, "medium": 2, "low": 3}
    ordered = sorted(
        rows,
        key=lambda r: (
            priority_order.get(str(r.get("recommended_review_priority") or ""), 9),
            -float(r.get("confidence") or 0.0),
            str(r.get("window_id") or ""),
        ),
    )
    for quota, cats in quotas:
        for row in ordered:
            if len([r for r in selected if r.get("category") in cats]) >= quota or len(selected) >= count:
                break
            if row.get("category") not in cats or not _eligible_review_row(row, reviewed, scene_counts, used_samples, used_windows):
                continue
            _add_selection(row, selected, scene_counts, used_samples, used_windows)
    if len(selected) < count:
        for row in ordered:
            if len(selected) >= count:
                break
            if _eligible_review_row(row, reviewed, scene_counts, used_samples, used_windows):
                _add_selection(row, selected, scene_counts, used_samples, used_windows)
    return selected


def _eligible_review_row(row: dict[str, Any], reviewed: set[str], scene_counts: Counter[str], used_samples: set[str], used_windows: set[str]) -> bool:
    wid = str(row.get("window_id") or "")
    sample = str(row.get("sample_id") or row.get("source_id") or wid)
    scene = str(row.get("source_scene_file") or row.get("source_scene_path") or "unknown")
    if not wid or wid in reviewed or wid in used_windows:
        return False
    if sample in used_samples:
        return False
    if scene_counts[scene] >= 2:
        return False
    return True


def _add_selection(row: dict[str, Any], selected: list[dict[str, Any]], scene_counts: Counter[str], used_samples: set[str], used_windows: set[str]) -> None:
    selected.append(row)
    scene_counts[str(row.get("source_scene_file") or row.get("source_scene_path") or "unknown")] += 1
    used_samples.add(str(row.get("sample_id") or row.get("source_id") or row.get("window_id")))
    used_windows.add(str(row.get("window_id")))


def _review_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    rid = f"newscenes_v2_{index:03d}_{safe_id_for_path(str(row.get('category') or 'candidate'))}"
    out = dict(row)
    out.update(
        {
            "review_id": rid,
            "review_index": index,
            "semantic_family": row.get("resolved_semantic_family"),
            "motion_subtype": row.get("resolved_motion_subtype"),
            "resolved_semantic_family_v2": row.get("resolved_semantic_family"),
            "review_question": "Human semantic review: is this v2 pose-first interpretation correct?",
            "selection_reason": row.get("category"),
            "timeline_generation_performed": False,
            "ml_training_performed": False,
            "manual_labels_modified": False,
        }
    )
    return out


def _previously_reviewed_window_ids(run: Path, out: Path) -> set[str]:
    reviewed: set[str] = set()
    audits = run / "audits"
    if not audits.exists():
        return reviewed
    for path in audits.rglob("semantic_review_010.jsonl"):
        if out in path.parents:
            continue
        for row in load_jsonl(path):
            if row.get("window_id"):
                reviewed.add(str(row["window_id"]))
    return reviewed


def _verify_inputs(new_run: Path, base_run: Path, ontology: Path, rules: Path, manual_gt: Path) -> list[str]:
    required = [
        ontology,
        rules,
        Path("data/config/manual_gt_motion_amplitude_profiles_v1.yaml"),
        manual_gt,
        base_run / "generation" / "manual_gt_timeline_examples_v4" / "ACCEPTED_BASELINE_REFERENCE.md",
        new_run / "run_manifest.json",
        new_run / "pose_semantics" / "pose_semantics_v0.jsonl",
        new_run / "relative_motion" / "relative_motion_features.jsonl",
        new_run / "interaction_semantics" / "interaction_semantics_v0.jsonl",
        new_run / "semantic" / "movement_windows.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if _latest_candidate_db(new_run, required=False) is None:
        missing.append(str(new_run / "datasets" / "semantic_candidate_db_v3.jsonl"))
    return missing


def _write_blocked_report(path: Path, missing: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Blocked: Missing New-Scenes Artifacts", "", "The v2 semantic rescan did not invent replacement data.", "", "Missing inputs:"]
    lines.extend(f"- `{m}`" for m in missing)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latest_candidate_db(run: Path, required: bool = True) -> Path | None:
    candidates = [
        run / "datasets" / "semantic_candidate_db_v3.jsonl",
        run / "datasets" / "semantic_candidate_db_v2.jsonl",
        run / "datasets" / "semantic_candidate_db_v1.jsonl",
        run / "datasets" / "semantic_candidate_db_v0.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"No semantic candidate DB found under {run / 'datasets'}")
    return None


def _by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows if row.get(key) is not None}


def _best_interaction_by_window(path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        wid = str(row.get("window_id") or "")
        if not wid:
            continue
        if wid not in best or float(row.get("interaction_confidence") or 0.0) > float(best[wid].get("interaction_confidence") or 0.0):
            best[wid] = row
    return best


def _manual_gt_hints(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    families: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    for row in rows:
        labels = row.get("human_labels") if isinstance(row.get("human_labels"), dict) else {}
        family = labels.get("family") or labels.get("pose_family") or labels.get("pose_family_normalized")
        subtype = labels.get("pose_subtype")
        if family:
            families[str(family)] += 1
        if subtype:
            subtypes[str(subtype)] += 1
    return {"families": families, "subtypes": subtypes}


def _resolved_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "family_counts": dict(Counter(str(r.get("resolved_semantic_family") or "unknown") for r in rows)),
        "gate_counts": dict(Counter(str(r.get("clean_motion_gate") or "unknown") for r in rows)),
        "conflict_count": sum(1 for r in rows if r.get("conflict_flags")),
    }


def _write_resolver_report(path: Path, rows: list[dict[str, Any]], candidate_path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# New Scenes Pose-First Semantic Resolver V2",
        "",
        "Analysis-only rescan using ontology v2, pose-first rules v2, and manual GT v4 mapping notes.",
        "",
        f"- Candidate source: `{candidate_path}`",
        f"- Records: {len(rows)}",
        f"- Resolved families: `{summary['family_counts']}`",
        f"- Clean-motion gates: `{summary['gate_counts']}`",
        f"- Conflict rows: {summary['conflict_count']}",
        "- Cowgirl VaM mapping: semantic pelvis_hip => hipControl primary visible driver.",
        "- ML training performed: false",
        "- Timeline generation performed: false",
        "- manual_labels.yaml modified: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_id", "sample_id", "source_scene_file", "technical_actor_id", "old_system_guess",
        "resolved_semantic_family", "resolved_motion_subtype", "category", "generation_readiness",
        "pose_family", "pose_subtype", "primary_motion_center", "target_region", "contact_support",
        "clean_motion_gate", "confidence", "conflict_flags", "not_labels", "explanation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_alignment_report(path: Path, rows: list[dict[str, Any]], resolved: Path, ontology: Path, manual_gt: Path, counts: Counter[str], readiness: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# New Scenes Ontology Alignment V2",
        "",
        f"- Resolved input: `{resolved}`",
        f"- Ontology: `{ontology}`",
        f"- Manual GT reference: `{manual_gt}`",
        f"- Records: {len(rows)}",
        f"- Category counts: `{dict(counts)}`",
        f"- Generation readiness counts: `{dict(readiness)}`",
        "",
        "No row is ground truth; all rows are review-assist candidates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overview_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# New Scenes Semantic Rescan V2 Overview",
        "",
        f"- Records: {len(rows)}",
        f"- Category counts: `{dict(Counter(r.get('category') for r in rows))}`",
        f"- Resolved family counts: `{dict(Counter(r.get('resolved_semantic_family') for r in rows))}`",
        f"- Readiness counts: `{dict(Counter(r.get('generation_readiness') for r in rows))}`",
        f"- Conflicts: {sum(1 for r in rows if r.get('ontology_conflict'))}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_category_report(path: Path, rows: list[dict[str, Any]], categories: set[str]) -> None:
    subset = [r for r in rows if r.get("category") in categories or (path.name == "conflicts_and_rejections.md" and r.get("ontology_conflict"))]
    top = sorted(subset, key=lambda r: (-float(r.get("confidence") or 0.0), str(r.get("source_scene_file") or "")))[:20]
    lines = [f"# {path.stem.replace('_', ' ').title()}", "", f"- Records: {len(subset)}", f"- Categories: `{dict(Counter(r.get('category') for r in subset))}`", ""]
    lines.append("## Top Examples")
    for row in top:
        lines.append(
            f"- `{row.get('window_id')}` | {row.get('source_scene_file')} | {row.get('category')} | "
            f"{row.get('resolved_semantic_family')} | driver={row.get('primary_motion_center')} | gate={row.get('clean_motion_gate')} | "
            f"why={row.get('explanation')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manual_gt_similarity_report(path: Path, rows: list[dict[str, Any]]) -> None:
    with_family = [r for r in rows if (r.get("manual_gt_reference_hint") or {}).get("matching_family_examples")]
    lines = [
        "# Manual GT Similarity Hints",
        "",
        f"- Rows with same-family manual GT examples: {len(with_family)}",
        f"- Families with manual GT hints: `{dict(Counter(r.get('resolved_semantic_family') for r in with_family))}`",
        "",
        "These are coarse hints, not ground-truth matches.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_drift_report(new_run: Path, rows: list[dict[str, Any]], path: Path) -> Path:
    old_path = _latest_candidate_db(new_run, required=False)
    old_by_window = _by_key(load_jsonl(old_path), "window_id") if old_path else {}
    transitions: Counter[str] = Counter()
    old_cowgirl_total = 0
    old_cowgirl_remain = 0
    old_cowgirl_to_other: Counter[str] = Counter()
    for row in rows:
        old = old_by_window.get(str(row.get("window_id") or ""), {})
        old_family = str(old.get("semantic_family") or "unknown")
        new_family = str(row.get("resolved_semantic_family") or "unknown")
        transitions[f"{old_family} -> {new_family}"] += 1
        if old_family == "cowgirl":
            old_cowgirl_total += 1
            if new_family == "cowgirl":
                old_cowgirl_remain += 1
            else:
                old_cowgirl_to_other[new_family] += 1
    lines = [
        "# New Scenes Semantic Drift V1 To V2",
        "",
        f"- Previous candidate source: `{old_path}`",
        f"- Compared rows: {len(rows)}",
        f"- Old Cowgirl candidates: {old_cowgirl_total}",
        f"- Old Cowgirl still Cowgirl: {old_cowgirl_remain}",
        f"- Old Cowgirl reclassified: `{dict(old_cowgirl_to_other)}`",
        f"- Top transitions: `{dict(transitions.most_common(20))}`",
        "",
        "V2 is stricter about driver, partner relation, low motion, and Manual-GT-derived VaM controller mapping.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_review_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# New Scenes Semantic Review V2", "", f"Items: {len(rows)}", ""]
    for row in rows:
        lines.extend([
            f"## {row['review_id']}",
            f"- Scene: `{row.get('source_scene_file')}`",
            f"- Time: {row.get('start_seconds')} - {row.get('end_seconds')}",
            f"- V2 family: `{row.get('resolved_semantic_family')}`",
            f"- Category: `{row.get('category')}`",
            f"- Driver: `{row.get('primary_motion_center')}`",
            f"- Gate: `{row.get('clean_motion_gate')}`",
            f"- Why: {row.get('explanation')}",
            f"- Why not Cowgirl: {row.get('why_not_cowgirl')}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_answer_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    answers = {
        "schema": "new_scenes_semantic_review_v2_answer_sheet",
        "instructions": "Human review remains final truth. Fill in manually; do not auto-label.",
        "items": [
            {
                "review_id": r["review_id"],
                "window_id": r.get("window_id"),
                "v2_family_correct": "unknown",
                "actual_family": "",
                "actual_motion": "",
                "notes": "",
            }
            for r in rows
        ],
    }
    path.write_text(yaml.safe_dump(answers, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_review_index(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<section><h2>{rid}</h2><dl>"
            "<dt>Scene</dt><dd>{scene}</dd><dt>V2 Family</dt><dd>{family}</dd>"
            "<dt>Category</dt><dd>{cat}</dd><dt>Driver</dt><dd>{driver}</dd>"
            "<dt>Gate</dt><dd>{gate}</dd><dt>Why</dt><dd>{why}</dd>"
            "<dt>Why not Cowgirl</dt><dd>{notcow}</dd></dl></section>".format(
                rid=html.escape(str(row.get("review_id"))),
                scene=html.escape(str(row.get("source_scene_file"))),
                family=html.escape(str(row.get("resolved_semantic_family"))),
                cat=html.escape(str(row.get("category"))),
                driver=html.escape(str(row.get("primary_motion_center"))),
                gate=html.escape(str(row.get("clean_motion_gate"))),
                why=html.escape(str(row.get("explanation"))),
                notcow=html.escape(str(row.get("why_not_cowgirl"))),
            )
        )
    page = "<!doctype html><meta charset='utf-8'><title>New Scenes Semantic Review V2</title><style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto}section{border:1px solid #ddd;padding:12px;margin:12px 0;border-radius:6px}dt{font-weight:bold}</style><h1>New Scenes Semantic Review V2</h1>" + "\n".join(cards)
    path.write_text(page, encoding="utf-8")


def _write_review_quality_report(path: Path, all_rows: list[dict[str, Any]], selected: list[dict[str, Any]], reviewed: set[str], requested: int) -> None:
    lines = [
        "# Review Quality Report",
        "",
        f"- Requested: {requested}",
        f"- Exported: {len(selected)}",
        f"- Available candidates: {len(all_rows)}",
        f"- Previously reviewed windows excluded: {len(reviewed)}",
        f"- Selected categories: `{dict(Counter(r.get('category') for r in selected))}`",
        f"- Max per scene respected: true",
        f"- Duplicate sample/window padding used: false",
        f"- Timeline generation performed: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _v2_explanation(row: dict[str, Any]) -> str:
    bits = [
        f"{row.get('resolved_semantic_family')}: pose={row.get('pose_subtype')}",
        f"driver={row.get('primary_motion_center')}",
        f"target={row.get('target_region')}",
        f"gate={row.get('clean_motion_gate')}",
    ]
    if row.get("vam_primary_driver_hint"):
        bits.append(f"VaM driver hint={row.get('vam_primary_driver_hint')}")
    if row.get("conflict_flags"):
        bits.append("conflicts=" + ",".join(row.get("conflict_flags")[:3]))
    if row.get("missing_requirements"):
        bits.append("missing=" + ",".join(row.get("missing_requirements")[:3]))
    return "; ".join(str(b) for b in bits)


def _why_not_cowgirl(row: dict[str, Any]) -> str:
    family = row.get("resolved_semantic_family")
    if family == "cowgirl":
        if row.get("clean_motion_gate") == "fail_missing_partner_context":
            return "Cowgirl candidate but missing rider-over-receiver / pelvis-aligned partner context."
        return "Resolved as Cowgirl candidate under v2 rules."
    primary = row.get("primary_motion_center")
    if primary == "head_neck":
        return "Primary motion appears head/chest driven, so this is BJ/oral or standing/head motion, not clean Cowgirl."
    if primary == "hands":
        return "Primary motion appears hand driven, so this is hand interaction/HJ, not clean Cowgirl."
    if family == "doggy":
        return "Doggy requires bent/all-fours/partner-behind relation; not rider-over-receiver Cowgirl."
    if family == "pose_context_hold":
        return "Pose/motion appears too low-motion for clean Cowgirl motion."
    return "V2 did not find Cowgirl driver + partner relation requirements."


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)

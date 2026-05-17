"""Motion Semantics v1: cycle-aware and gate-aware semantic resolver."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv
import html
import json

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import load_jsonl, safe_id_for_path, write_jsonl
from vam_timeline_ai.semantics.biomechanical_motion_gates import evaluate_biomechanical_gates, evaluate_male_active_thrust_gates
from vam_timeline_ai.ui.review_ui import build_static_review_ui


def resolve_new_scenes_motion_semantics_v1(
    new_run: str | Path,
    pose_resolved: str | Path,
    cycle_features: str | Path,
    ontology: str | Path,
    cycle_rules: str | Path,
    manual_gt: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
    relational_features: str | Path | None = None,
) -> dict[str, Any]:
    pose_rows = load_jsonl(pose_resolved)
    cycle_by_window = {str(r.get("window_id")): r for r in load_jsonl(cycle_features)}
    relational_by_window = _load_relational_features(relational_features)
    rows: list[dict[str, Any]] = []
    for pose in pose_rows:
        cycle = cycle_by_window.get(str(pose.get("window_id") or ""), {})
        relational = relational_by_window.get(str(pose.get("window_id") or ""), {})
        rows.append(resolve_motion_candidate_v1(pose, cycle, relational))
    write_jsonl(out_jsonl, rows)
    _write_motion_report(Path(report), rows, Path(pose_resolved), Path(cycle_features), Path(cycle_rules), Path(relational_features) if relational_features else None)
    return {
        "status": "ok",
        "records": len(rows),
        "motion_state_counts": dict(Counter(r.get("motion_state") for r in rows)),
        "relational_feature_rows": len(relational_by_window),
        "out_jsonl": str(out_jsonl),
        "report": str(report),
    }


def resolve_motion_candidate_v1(pose: dict[str, Any], cycle: dict[str, Any], relational: dict[str, Any] | None = None) -> dict[str, Any]:
    family = str(pose.get("resolved_semantic_family") or "unknown")
    relational = relational or {}
    enriched_pose = _merge_relational_context(pose, relational)
    male_gates = evaluate_male_active_thrust_gates(enriched_pose, cycle, relational)
    if male_gates.get("male_active_candidate") and male_gates.get("final_clean_motion_gate") in {"pass", "soft_pass"}:
        family = "male_active_thrust"
        gates = male_gates
    elif male_gates.get("male_active_candidate") and family in {"cowgirl", "reverse_cowgirl", "doggy", "missionary", "unknown"}:
        family = "male_active_thrust"
        gates = male_gates
    else:
        gates = evaluate_biomechanical_gates(enriched_pose, cycle, relational)
    driver_controller, driver_metric = _primary_driver(family, cycle)
    final_gate = gates["final_clean_motion_gate"]
    motion_family = family
    motion_state = _motion_state(family, final_gate, gates, pose)
    if final_gate not in {"pass", "soft_pass"}:
        motion_family = _downgraded_family(family, motion_state)
    subtype = _motion_subtype(motion_family, family, motion_state, driver_metric, pose)
    out = dict(enriched_pose)
    out.update(
        {
            "schema": "motion_semantic_resolved_v1",
            "resolved_motion_family": motion_family,
            "resolved_motion_subtype": subtype,
            "motion_state": motion_state,
            "primary_driver_controller": driver_controller,
            "semantic_driver": _semantic_driver(family),
            "dominant_axis": (driver_metric or {}).get("dominant_axis"),
            "cycle_count": (driver_metric or {}).get("estimated_cycle_count", 0.0),
            "frequency_hz": (driver_metric or {}).get("estimated_frequency_hz", 0.0),
            "cyclicity_score": (driver_metric or {}).get("cyclicity_score", 0.0),
            "transition_score": (driver_metric or {}).get("transition_score", 0.0),
            "anchor_stability_status": _anchor_status(cycle),
            "break_state_reasons": gates.get("gate_failure_reasons", []),
            "confidence": _motion_confidence(pose, driver_metric, gates),
            "explanation": _motion_explanation(family, motion_family, motion_state, driver_controller, gates),
            **gates,
            "manual_labels_modified": False,
            "ml_training_performed": False,
            "timeline_generation_performed": False,
            "is_training_label": False,
            "auto_label": False,
        }
    )
    return out


def _load_relational_features(relational_features: str | Path | None) -> dict[str, dict[str, Any]]:
    if not relational_features:
        return {}
    path = Path(relational_features)
    if not path.exists():
        return {}
    by_window: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        window_id = str(row.get("window_id") or "")
        if not window_id:
            continue
        if row.get("schema") == "relational_semantic_features_v1":
            by_window[window_id] = row
    return by_window


def _merge_relational_context(pose: dict[str, Any], relational: dict[str, Any]) -> dict[str, Any]:
    if not relational:
        return dict(pose)
    out = dict(pose)
    relation_hints = list(relational.get("partner_relation_hints") or [])
    existing_relation = list(out.get("partner_relation") or [])
    merged_relation = sorted(set(str(x) for x in existing_relation + relation_hints if x))
    if merged_relation:
        out["partner_relation"] = merged_relation
    if relational.get("facing_context_hint"):
        out["facing_context_relational_hint"] = relational.get("facing_context_hint")
    out["relational_context_available"] = True
    out["relational_partner_local_frame_quality"] = relational.get("partner_local_frame_quality")
    out["relational_actor_reference_points_present"] = relational.get("actor_reference_points_present", [])
    out["relational_partner_reference_points_present"] = relational.get("partner_reference_points_present", [])
    out["relational_actor_pelvis_partner_alignment_distance_mean"] = relational.get("actor_pelvis_partner_alignment_distance_mean")
    out["relational_actor_pelvis_partner_alignment_score"] = relational.get("actor_pelvis_partner_alignment_score")
    out["relational_actor_above_partner_score"] = relational.get("actor_above_partner_score")
    out["relational_actor_pelvis_to_partner_local"] = relational.get("actor_pelvis_to_partner_pelvis_partner_local")
    out["relational_head_to_partner_pelvis_distance_mean"] = relational.get("head_to_partner_pelvis_distance_mean")
    out["relational_head_to_partner_pelvis_target_score"] = relational.get("head_to_partner_pelvis_target_score")
    out["relational_chest_to_partner_pelvis_target_score"] = relational.get("chest_to_partner_pelvis_target_score")
    out["relational_hand_partner_targets"] = relational.get("hand_partner_targets")
    out["relational_support_contact_hints"] = relational.get("support_contact_hints", [])
    out["relational_best_lHand_partner_target"] = relational.get("best_lHand_partner_target")
    out["relational_best_rHand_partner_target"] = relational.get("best_rHand_partner_target")
    out["relational_best_lHand_own_target"] = relational.get("best_lHand_own_target")
    out["relational_best_rHand_own_target"] = relational.get("best_rHand_own_target")
    out["relational_hip_motion_partner_axes"] = relational.get("hip_motion_partner_axes")
    out["relational_pelvis_motion_partner_axes"] = relational.get("pelvis_motion_partner_axes")
    out["relational_hip_motion_contact_axis"] = relational.get("hip_motion_contact_axis")
    out["relational_pelvis_motion_contact_axis"] = relational.get("pelvis_motion_contact_axis")
    out["relational_interaction_zone"] = relational.get("interaction_zone")
    return out


def build_new_scenes_motion_candidate_db_v1(
    new_run: str | Path,
    motion_resolved: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    rows = load_jsonl(motion_resolved)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        category = _motion_category(row)
        item = dict(row)
        item.update(
            {
                "schema": "ontology_motion_candidate_v1",
                "category": category,
                "old_v2_category": row.get("category"),
                "candidate_not_ground_truth": True,
                "generation_readiness": _motion_readiness(category),
            }
        )
        out_rows.append(item)
    write_jsonl(out_jsonl, out_rows)
    _write_csv(out_csv, out_rows)
    _write_candidate_report(Path(report), out_rows)
    _write_drift_report(Path(new_run), out_rows, Path(out_jsonl).parent / "motion_semantic_drift_report_v1.md")
    return {"status": "ok", "records": len(out_rows), "category_counts": dict(Counter(r["category"] for r in out_rows)), "out_jsonl": str(out_jsonl), "out_csv": str(out_csv), "report": str(report)}


def export_motion_semantics_review_v1(
    new_run: str | Path,
    candidates: str | Path,
    out_dir: str | Path,
    count: int = 20,
    build_static_ui: bool = True,
    build_vam_package: bool = True,
) -> dict[str, Any]:
    rows = load_jsonl(candidates)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = _select_motion_review(rows, count)
    review_rows = [_review_row(r, i + 1, "motion_cycle_review") for i, r in enumerate(selected)]
    review_jsonl = out / "semantic_review_010.jsonl"
    write_jsonl(review_jsonl, review_rows)
    _write_review_quality(out / "review_quality_report.md", rows, review_rows)
    _write_simple_html(out / "semantic_review_010_index.html", review_rows)
    package_summary = build_vam_review_package(review_jsonl, new_run, new_run, out / "vam_review_package", attempt_timeline_segments=False) if build_vam_package else None
    ui_summary = build_static_review_ui(new_run, out, out / "review_ui_static") if build_static_ui else None
    return {"status": "ok", "review_items": len(review_rows), "out_dir": str(out), "static_ui": ui_summary, "vam_package": package_summary}


def export_strict_cowgirl_motion_batch_v1(
    new_run: str | Path,
    candidates: str | Path,
    out_dir: str | Path,
    count: int = 10,
) -> dict[str, Any]:
    rows = [r for r in load_jsonl(candidates) if r.get("category") == "cowgirl_clean_cyclic_motion"]
    rows.sort(key=lambda r: (-float(r.get("cyclicity_score") or 0.0), -float(r.get("cycle_count") or 0.0), str(r.get("source_scene_file") or ""), float(r.get("start_seconds") or 0.0)))
    selected: list[dict[str, Any]] = []
    used_windows: set[str] = set()
    for row in rows:
        if len(selected) >= count:
            break
        if row.get("window_id") in used_windows:
            continue
        selected.append(row)
        used_windows.add(str(row.get("window_id")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    review_rows = [_review_row(r, i + 1, "cowgirl_motion_cycle_strict") for i, r in enumerate(selected)]
    write_jsonl(out / "semantic_review_010.jsonl", review_rows)
    _write_review_quality(out / "review_quality_report.md", rows, review_rows)
    return {"status": "ok", "review_items": len(review_rows), "out_dir": str(out)}


def _primary_driver(family: str, cycle: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    metrics = cycle.get("controller_metrics") if isinstance(cycle.get("controller_metrics"), dict) else {}
    choices = {
        "cowgirl": ["hipControl", "pelvisControl"],
        "reverse_cowgirl": ["hipControl", "pelvisControl"],
        "doggy": ["hipControl", "pelvisControl"],
        "bj_oral": ["headControl"],
        "handjob": ["lHandControl", "rHandControl"],
        "hand_touching": ["lHandControl", "rHandControl"],
        "missionary": ["hipControl", "pelvisControl"],
        "male_active_thrust": ["hipControl", "pelvisControl"],
    }.get(family, ["hipControl", "pelvisControl", "headControl", "lHandControl", "rHandControl"])
    best_name = ""
    best_metric = None
    best_score = -1.0
    for name in choices:
        metric = metrics.get(name)
        if not metric:
            continue
        score = float(metric.get("cyclicity_score") or 0.0) + float(metric.get("max_displacement_range") or 0.0)
        if score > best_score:
            best_name, best_metric, best_score = name, metric, score
    return best_name, best_metric


def _motion_state(family: str, final_gate: str, gates: dict[str, Any], pose: dict[str, Any]) -> str:
    if final_gate == "pass":
        return "clean_motion"
    if final_gate == "soft_pass":
        return "short_cycle_candidate"
    reasons = " ".join(gates.get("gate_failure_reasons") or [])
    if "pose_hold" in final_gate:
        return "pose_context_hold"
    if "locomotion" in final_gate or "crawling" in reasons:
        return "crawling_locomotion"
    if "wrong_driver" in final_gate and family == "bj_oral":
        return "unknown"
    if "transition" in final_gate or "monotonic" in final_gate or "transition" in reasons or "one-way" in reasons:
        if family in {"handjob", "hand_touching"}:
            return "reaching_acquisition"
        return "intro_transition"
    if "anchor_lost" in final_gate or "pose_broken" in final_gate or "alignment" in final_gate:
        return "intro_transition"
    return "unknown"


def _downgraded_family(family: str, motion_state: str) -> str:
    if family == "cowgirl" and motion_state == "pose_context_hold":
        return "cowgirl_pose_context_hold"
    if family == "cowgirl":
        return "cowgirl_transition_setup"
    if family == "doggy" and motion_state == "crawling_locomotion":
        return "doggy_crawling_or_transition"
    if family == "bj_oral" and motion_state != "clean_motion":
        return "bj_oral_reaching_or_alignment"
    if family in {"handjob", "hand_touching"} and motion_state != "clean_motion":
        return "hand_reaching_or_touching"
    if family == "missionary" and motion_state != "clean_motion":
        return "missionary_getting_up_or_transition"
    if family == "male_active_thrust" and motion_state != "clean_motion":
        return "male_active_thrust_transition_or_invalid"
    return family


def _motion_subtype(motion_family: str, original_family: str, state: str, metric: dict[str, Any] | None, pose: dict[str, Any]) -> str:
    if state != "clean_motion":
        return motion_family
    axis = str((metric or {}).get("dominant_axis") or "")
    if original_family == "cowgirl":
        if axis == "y":
            return "cowgirl_vertical_bounce"
        if axis in {"x", "z"}:
            return "cowgirl_grinding"
    if original_family == "bj_oral":
        return "bj_head_cycle"
    if original_family in {"handjob", "hand_touching"}:
        return "hand_cycle"
    if original_family == "doggy":
        return "doggy_forward_back_cycle"
    if original_family == "missionary":
        return "missionary_counter_motion"
    if original_family == "male_active_thrust":
        return "male_active_penetration_thrust"
    return str(pose.get("resolved_motion_subtype") or motion_family)


def _semantic_driver(family: str) -> str:
    return {
        "cowgirl": "pelvis_hip",
        "reverse_cowgirl": "pelvis_hip",
        "doggy": "pelvis_hip",
        "bj_oral": "head_neck",
        "handjob": "hands",
        "hand_touching": "hands",
        "missionary": "pelvis_counter_driver",
        "male_active_thrust": "male_pelvis_hip_thrust",
    }.get(family, "unknown")


def _anchor_status(cycle: dict[str, Any]) -> str:
    anchor = cycle.get("anchor_summary") or {}
    bad = [name for name in ["feet_stable", "knees_stable", "hands_stable"] if anchor.get(name) is False]
    return "unstable:" + ",".join(bad) if bad else "stable_or_unknown"


def _motion_confidence(pose: dict[str, Any], metric: dict[str, Any] | None, gates: dict[str, Any]) -> float:
    base = float(pose.get("confidence") or 0.4)
    cyc = float((metric or {}).get("cyclicity_score") or 0.0)
    if gates["final_clean_motion_gate"] == "pass":
        return round(min(0.95, base * 0.55 + cyc * 0.45 + 0.15), 3)
    if gates["final_clean_motion_gate"] == "soft_pass":
        return round(min(0.72, base * 0.5 + cyc * 0.35), 3)
    return round(min(0.45, base * 0.35 + cyc * 0.2), 3)


def _motion_explanation(original: str, resolved: str, state: str, driver: str, gates: dict[str, Any]) -> str:
    reasons = gates.get("gate_failure_reasons") or []
    return f"{original} -> {resolved}; state={state}; driver={driver or 'unknown'}; final_gate={gates.get('final_clean_motion_gate')}; reasons={','.join(reasons[:4])}"


def _motion_category(row: dict[str, Any]) -> str:
    family = str(row.get("resolved_motion_family") or "")
    state = str(row.get("motion_state") or "")
    if row.get("final_clean_motion_gate") not in {"pass", "soft_pass"} and row.get("break_state_reasons"):
        if str(row.get("resolved_semantic_family")) in {"cowgirl", "doggy", "bj_oral", "handjob", "missionary"}:
            conflict = False
        else:
            conflict = True
    else:
        conflict = False
    if family == "cowgirl" and state == "clean_motion":
        return "cowgirl_clean_cyclic_motion"
    if str(row.get("resolved_semantic_family")) == "cowgirl" and state == "short_cycle_candidate":
        return "cowgirl_short_cycle_candidate"
    if family == "cowgirl_pose_context_hold" or state == "pose_context_hold":
        return "cowgirl_pose_context_hold"
    if family == "cowgirl_transition_setup" or (str(row.get("resolved_semantic_family")) == "cowgirl" and state == "intro_transition"):
        return "cowgirl_transition_or_getting_up"
    if family == "doggy" and state == "clean_motion":
        return "doggy_clean_cyclic_motion"
    if family == "doggy_crawling_or_transition":
        return "doggy_crawling_or_transition"
    if family == "bj_oral" and state == "clean_motion":
        return "bj_oral_clean_head_cycle"
    if family == "bj_oral_reaching_or_alignment":
        return "bj_oral_reaching_or_alignment"
    if family == "handjob" and state == "clean_motion":
        return "handjob_clean_hand_cycle"
    if family == "hand_reaching_or_touching":
        return "hand_reaching_or_touching"
    if family == "missionary" and state == "clean_motion":
        return "missionary_counter_motion"
    if family == "missionary_getting_up_or_transition":
        return "missionary_getting_up_or_transition"
    if family == "male_active_thrust" and state == "clean_motion":
        return "male_active_thrust_clean_motion"
    if family == "male_active_thrust_transition_or_invalid":
        return "male_active_thrust_transition_or_invalid"
    if str(row.get("resolved_semantic_family")) == "standing_hand_head":
        return "standing_hand_head_negative"
    if conflict:
        return "ontology_motion_conflict"
    return "unknown_or_unusable"


def _motion_readiness(category: str) -> str:
    if "clean" in category or category == "missionary_counter_motion":
        return "motion_reference_candidate"
    if "short_cycle" in category:
        return "needs_human_review"
    if "pose_context" in category:
        return "pose_reference_candidate"
    return "not_ready"


def _select_motion_review(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    quotas = [
        (4, {"cowgirl_clean_cyclic_motion"}),
        (3, {"cowgirl_short_cycle_candidate"}),
        (2, {"cowgirl_transition_or_getting_up"}),
        (2, {"bj_oral_clean_head_cycle"}),
        (2, {"handjob_clean_hand_cycle"}),
        (2, {"doggy_clean_cyclic_motion", "doggy_crawling_or_transition"}),
        (2, {"missionary_counter_motion", "missionary_getting_up_or_transition"}),
        (3, {"ontology_motion_conflict", "unknown_or_unusable", "standing_hand_head_negative"}),
    ]
    ordered = sorted(rows, key=lambda r: (-float(r.get("cyclicity_score") or 0.0), -float(r.get("confidence") or 0.0), str(r.get("window_id") or "")))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for quota, cats in quotas:
        taken = 0
        for row in ordered:
            if taken >= quota or len(selected) >= count:
                break
            if row.get("category") not in cats or row.get("window_id") in used:
                continue
            selected.append(row)
            used.add(row.get("window_id"))
            taken += 1
    for row in ordered:
        if len(selected) >= count:
            break
        if row.get("window_id") not in used:
            selected.append(row)
            used.add(row.get("window_id"))
    return selected


def _review_row(row: dict[str, Any], index: int, prefix: str) -> dict[str, Any]:
    item = dict(row)
    rid = f"{prefix}_{index:03d}"
    item.update(
        {
            "review_id": rid,
            "review_index": index,
            "review_label": f"{rid}_{safe_id_for_path(str(row.get('category') or 'candidate'))}",
            "semantic_family": row.get("resolved_motion_family"),
            "motion_subtype": row.get("resolved_motion_subtype"),
            "why_selected": row.get("category"),
            "review_only": True,
            "not_training_truth": True,
            "manual_labels_modified": False,
            "ml_training_performed": False,
        }
    )
    return item


def _write_motion_report(path: Path, rows: list[dict[str, Any]], pose_path: Path, cycle_path: Path, rules_path: Path, relational_path: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Motion Semantic Resolver V1",
        "",
        f"- Pose input: `{pose_path}`",
        f"- Cycle input: `{cycle_path}`",
        f"- Relational input: `{relational_path}`",
        f"- Rules: `{rules_path}`",
        f"- Records: {len(rows)}",
        f"- Motion states: `{dict(Counter(r.get('motion_state') for r in rows))}`",
        f"- Final gates: `{dict(Counter(r.get('final_clean_motion_gate') for r in rows))}`",
        f"- Relational context rows used: {sum(1 for r in rows if r.get('relational_context_available'))}",
        f"- Partner frame quality: `{dict(Counter(r.get('relational_partner_local_frame_quality') for r in rows if r.get('relational_context_available')))}`",
        f"- Relational hand partner targets: `{dict(Counter(t for r in rows for t in [r.get('relational_best_lHand_partner_target'), r.get('relational_best_rHand_partner_target')] if t))}`",
        f"- Gate failures: `{dict(Counter(reason for r in rows for reason in (r.get('gate_failure_reasons') or [])))}`",
        "- ML training performed: false",
        "- Timeline generation performed: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_candidate_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Ontology Motion Candidate Report V1",
        "",
        f"- Records: {len(rows)}",
        f"- Category counts: `{dict(Counter(r.get('category') for r in rows))}`",
        f"- Motion state counts: `{dict(Counter(r.get('motion_state') for r in rows))}`",
        "",
        "Clean motion categories require cycle + biomechanical gate pass/soft-pass.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_drift_report(new_run: Path, rows: list[dict[str, Any]], path: Path) -> None:
    old = load_jsonl(new_run / "semantic_rescan_v2" / "ontology_aligned_candidates_v2.jsonl")
    old_by_window = {str(r.get("window_id")): r for r in old}
    old_cowgirl = [r for r in old if r.get("category") == "cowgirl_clean_motion_candidate"]
    transitions = Counter()
    for row in rows:
        old_cat = (old_by_window.get(str(row.get("window_id") or "")) or {}).get("category", "missing")
        transitions[f"{old_cat} -> {row.get('category')}"] += 1
    lines = [
        "# Motion Semantic Drift Report V1",
        "",
        f"- Previous v2 cowgirl_clean_motion_candidate: {len(old_cowgirl)}",
        f"- Remain cowgirl_clean_cyclic_motion: {sum(1 for r in rows if old_by_window.get(str(r.get('window_id') or ''), {}).get('category') == 'cowgirl_clean_motion_candidate' and r.get('category') == 'cowgirl_clean_cyclic_motion')}",
        f"- Become cowgirl_short_cycle_candidate: {sum(1 for r in rows if old_by_window.get(str(r.get('window_id') or ''), {}).get('category') == 'cowgirl_clean_motion_candidate' and r.get('category') == 'cowgirl_short_cycle_candidate')}",
        f"- Become cowgirl_transition_or_getting_up: {sum(1 for r in rows if old_by_window.get(str(r.get('window_id') or ''), {}).get('category') == 'cowgirl_clean_motion_candidate' and r.get('category') == 'cowgirl_transition_or_getting_up')}",
        f"- Become pose_context_hold: {sum(1 for r in rows if old_by_window.get(str(r.get('window_id') or ''), {}).get('category') == 'cowgirl_clean_motion_candidate' and r.get('category') == 'cowgirl_pose_context_hold')}",
        f"- Top transitions: `{dict(transitions.most_common(20))}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = ["window_id", "source_scene_file", "technical_atom_id", "category", "resolved_motion_family", "motion_state", "primary_driver_controller", "dominant_axis", "cycle_count", "frequency_hz", "cyclicity_score", "transition_score", "final_clean_motion_gate", "gate_failure_reasons"]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in fields})


def _write_review_quality(path: Path, all_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Motion Semantics Review Quality",
                "",
                f"- Selected: {len(rows)}",
                f"- Source candidates: {len(all_rows)}",
                f"- Selected categories: `{dict(Counter(r.get('category') for r in rows))}`",
                "- Timeline generation performed: false",
                "- ML training performed: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_simple_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for r in rows:
        cards.append(f"<section><h2>{html.escape(str(r.get('review_id')))}</h2><p>{html.escape(str(r.get('source_scene_file')))} {r.get('start_seconds')}-{r.get('end_seconds')}</p><p>category={html.escape(str(r.get('category')))} state={html.escape(str(r.get('motion_state')))} driver={html.escape(str(r.get('primary_driver_controller')))} cycles={r.get('cycle_count')} gate={html.escape(str(r.get('final_clean_motion_gate')))}</p><p>{html.escape(str(r.get('explanation')))}</p></section>")
    path.write_text("<!doctype html><meta charset='utf-8'><style>body{font-family:Arial;margin:24px auto;max-width:1100px}section{border:1px solid #ccc;margin:12px;padding:12px}</style><h1>Motion Semantics Review V1</h1>" + "\n".join(cards), encoding="utf-8")

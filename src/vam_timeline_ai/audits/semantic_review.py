"""Small VaM semantic review batches with guarded Timeline exports."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import html
import math

import numpy as np
import yaml

from vam_timeline_ai.io.identity import stable_hash
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, safe_id_for_path, write_jsonl
from vam_timeline_ai.references.handmade_parser import classify_timeline_target
from vam_timeline_ai.semantics.domain_guards import evaluate_domain_guards
from vam_timeline_ai.semantics.motion_phase_classifier import classify_motion_phase
from vam_timeline_ai.timeline.codec import TimelineKeyframe, decode_keyframe_sequence, encode_keyframe_sequence


MOVEMENT_LABELS = {
    "cowgirl_vertical_bounce",
    "cowgirl_forward_back_rock",
    "cowgirl_lateral_sway",
    "cowgirl_circular_grind",
    "cowgirl_fast_shallow",
    "cowgirl_deep_slow",
    "cowgirl_pause_hold",
    "cowgirl_adjustment_transition",
    "cowgirl_irregular_human_motion",
}
CONTACT_LABELS = {
    "cowgirl_hand_supported_on_partner",
    "cowgirl_hand_supported_on_partner_chest",
    "cowgirl_hand_supported_on_partner_hips",
    "rider_active",
    "partner_context_static",
}
ROLE_LABELS = {"rider_active", "partner_context_static", "receiver_passive"}
HIGH_RISK_EXPORT_SOURCE_TYPES = {"vam_native_motion_animation", "native_motion_animation"}
LINEAR = 2


def export_semantic_review_010(
    run_dir: str | Path,
    out_dir: str | Path,
    count: int = 10,
    attempt_timeline_export: bool = True,
    use_body_motion_quality: bool = False,
    prefer_clean_body_motion: bool = False,
    use_handmade_reference_matches: bool = False,
    prefer_longer_cowgirl_windows: bool = False,
    min_cowgirl_window_seconds: float = 4.0,
    use_cowgirl_candidate_score_v2: bool = False,
    use_cowgirl_candidate_score_v3: bool = False,
    use_cowgirl_candidate_score_v4: bool = False,
    use_cowgirl_candidate_score_v5: bool = False,
    use_cowgirl_candidate_score_v6: bool = False,
    use_rider_receiver_discrimination: bool = False,
    use_relative_motion_features: bool = False,
    use_trajectory_shape_features: bool = False,
    use_relative_reference_matches: bool = False,
    use_pose_export_validity: bool = False,
    use_controller_validity: bool = False,
) -> dict[str, Any]:
    if count != 10:
        raise ValueError("semantic review MVP expects exactly 10 items")
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = _load_data(run)
    data["selection_flags"] = {
        "use_body_motion_quality": use_body_motion_quality,
        "prefer_clean_body_motion": prefer_clean_body_motion,
        "use_handmade_reference_matches": use_handmade_reference_matches,
        "prefer_longer_cowgirl_windows": prefer_longer_cowgirl_windows,
        "min_cowgirl_window_seconds": min_cowgirl_window_seconds,
        "use_cowgirl_candidate_score_v2": use_cowgirl_candidate_score_v2,
        "use_cowgirl_candidate_score_v3": use_cowgirl_candidate_score_v3,
        "use_cowgirl_candidate_score_v4": use_cowgirl_candidate_score_v4,
        "use_cowgirl_candidate_score_v5": use_cowgirl_candidate_score_v5,
        "use_cowgirl_candidate_score_v6": use_cowgirl_candidate_score_v6,
        "use_rider_receiver_discrimination": use_rider_receiver_discrimination,
        "use_relative_motion_features": use_relative_motion_features,
        "use_trajectory_shape_features": use_trajectory_shape_features,
        "use_relative_reference_matches": use_relative_reference_matches,
        "use_pose_export_validity": use_pose_export_validity,
        "use_controller_validity": use_controller_validity,
    }
    if use_cowgirl_candidate_score_v6 or use_controller_validity:
        selected = _select_10_v8(data)
    elif use_cowgirl_candidate_score_v5 or use_pose_export_validity:
        selected = _select_10_v7(data)
    elif use_cowgirl_candidate_score_v4 or use_relative_motion_features or use_trajectory_shape_features or use_relative_reference_matches:
        selected = _select_10_v6(data)
    elif use_cowgirl_candidate_score_v3 or use_rider_receiver_discrimination:
        selected = _select_10_v5(data)
    elif use_cowgirl_candidate_score_v2:
        selected = _select_10_v4(data)
    else:
        selected = _select_10_v3(data) if (use_body_motion_quality or use_handmade_reference_matches or prefer_clean_body_motion) else _select_10(data)
    rows: list[dict[str, Any]] = []
    export_results: list[dict[str, Any]] = []
    timeline_root = out / "timeline_segments"
    timeline_root.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(selected, start=1):
        row = _make_review_row(idx, item, data)
        if attempt_timeline_export:
            export_result = _attempt_timeline_export(row, data, timeline_root / row["review_id"])
        else:
            export_result = _write_export_unavailable(timeline_root / row["review_id"], row, "timeline export disabled by command")
        row["has_timeline_export"] = bool(export_result.get("success"))
        row["timeline_export_path"] = export_result.get("timeline_export_path")
        row["timeline_export_validation_status"] = export_result.get("validation_status", "unavailable")
        row["timeline_export_warnings"] = export_result.get("warnings", [])
        rows.append(row)
        export_results.append(export_result)
        _write_per_item_guess(out / row["review_id"], row)
    write_jsonl(out / "semantic_review_010.jsonl", rows)
    _write_markdown(rows, out / "semantic_review_010.md")
    _write_csv(rows, out / "semantic_review_010.csv")
    _write_answer_sheet_md(rows, out / "semantic_review_010_answer_sheet.md")
    _write_answer_sheet_yaml(rows, out / "semantic_review_010_answer_sheet.yaml")
    _write_index_html(rows, out / "semantic_review_010_index.html")
    _write_timeline_export_status(rows, export_results, out / "timeline_segment_export_status.md")
    summarize_semantic_review_010(out / "semantic_review_010_answer_sheet.yaml", out / "semantic_review_010.jsonl", out / "semantic_review_010_result.md")
    return {
        "status": "ok",
        "review_items": len(rows),
        "category_distribution": dict(Counter(r["category"] for r in rows)),
        "pair_examples": sum(1 for r in rows if r.get("pair_window_id")),
        "timeline_exports_attempted": sum(1 for r in export_results if r.get("attempted")),
        "timeline_exports_successful": sum(1 for r in export_results if r.get("success")),
        "timeline_exports_unavailable": sum(1 for r in export_results if not r.get("success")),
        "manual_labels_modified": False,
    }


def summarize_semantic_review_010(answers: str | Path, review: str | Path, out: str | Path) -> dict[str, Any]:
    answer_path = Path(answers)
    rows = load_jsonl(review)
    data = yaml.safe_load(answer_path.read_text(encoding="utf-8")) if answer_path.exists() else {}
    reviews = (data or {}).get("reviews", {}) or {}
    fields = [
        "user_verdict",
        "timeline_import_worked",
        "original_scene_review_worked",
        "active_rider_correct",
        "movement_correct",
        "contact_correct",
        "timing_correct",
        "timeline_export_correct",
    ]
    counts = {field: Counter() for field in fields}
    false_labels: Counter[str] = Counter()
    all_unknown = True
    for rid, item in reviews.items():
        if not isinstance(item, dict):
            continue
        for field in fields:
            value = str(item.get(field, "unknown"))
            counts[field][value] += 1
            if value != "unknown":
                all_unknown = False
        false_labels.update(str(v) for v in item.get("false_system_labels", []) or [])
        if item.get("actual_labels") or item.get("notes"):
            all_unknown = False
    status = "not_completed" if all_unknown or not reviews else "completed"
    verdict = {
        "data_extraction_trusted": _semantic_verdict(counts["original_scene_review_worked"], counts["timing_correct"]),
        "timeline_export_trusted": _semantic_verdict(counts["timeline_import_worked"], counts["timeline_export_correct"]),
        "feature_semantics_trusted": _semantic_verdict(counts["movement_correct"]),
        "pair_contact_trusted": _semantic_verdict(counts["contact_correct"]),
        "machine_labels_trusted_for_proxy_ml": _semantic_verdict(counts["user_verdict"]),
    }
    _write_semantic_result(Path(out), rows, status, counts, false_labels, verdict)
    return {"status": status, "review_items": len(rows), "counts": {k: dict(v) for k, v in counts.items()}, "verdict": verdict}


def _load_data(run: Path) -> dict[str, Any]:
    return {
        "run_dir": run,
        "windows": {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")},
        "features": {r.get("window_id"): r for r in load_jsonl(run / "features" / "cowgirl_window_features_v1.jsonl") if r.get("window_id")},
        "samples": {r.get("sample_id"): r for r in load_jsonl(run / "baked" / "motion_sample_index.jsonl") if r.get("sample_id")},
        "weak": {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "weak_labels_v2.jsonl") if r.get("window_id")},
        "pair_windows": {r.get("pair_window_id"): r for r in load_jsonl(run / "semantic" / "pair_windows_v1.jsonl") if r.get("pair_window_id")},
        "pair_by_window": _pair_by_window(load_jsonl(run / "semantic" / "pair_windows_v1.jsonl")),
        "pair_features": {r.get("pair_window_id"): r for r in load_jsonl(run / "features" / "cowgirl_pair_features_v0.jsonl") if r.get("pair_window_id")},
        "window_scores": _group_by(load_jsonl(run / "labels" / "machine_proposals" / "machine_window_label_scores_v2.jsonl"), "window_id"),
        "pair_scores": _group_by(load_jsonl(run / "labels" / "machine_proposals" / "machine_pair_label_scores_v2.jsonl"), "pair_window_id"),
        "silver_windows": {r.get("window_id"): r for r in load_jsonl(run / "labels" / "machine_proposals" / "silver_window_labels_v2.jsonl") if r.get("window_id")},
        "silver_pairs": {r.get("pair_window_id"): r for r in load_jsonl(run / "labels" / "machine_proposals" / "silver_pair_labels_v2.jsonl") if r.get("pair_window_id")},
        "baked_audit": {r.get("sample_id"): r for r in load_jsonl(run / "audits" / "baked_sample_audit.jsonl") if r.get("sample_id")},
        "body_quality": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "body_motion_quality.jsonl") if r.get("window_id")},
        "reference_matches": {r.get("window_id"): r for r in load_jsonl(run / "references" / "handmade_animations" / "wild_reference_matches.jsonl") if r.get("window_id")},
        "cowgirl_scores_v2": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "cowgirl_candidate_scores_v2.jsonl") if r.get("window_id")},
        "rider_receiver_scores": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "rider_receiver_scores_v1.jsonl") if r.get("window_id")},
        "cowgirl_scores_v3": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "cowgirl_candidate_scores_v3.jsonl") if r.get("window_id")},
        "relative_features": {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "relative_motion_features.jsonl") if r.get("window_id")},
        "trajectory_features": {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "trajectory_shape_features.jsonl") if r.get("window_id")},
        "relative_reference_matches": {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "relative_reference_matches.jsonl") if r.get("window_id")},
        "cowgirl_scores_v4": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "cowgirl_candidate_scores_v4.jsonl") if r.get("window_id")},
        "pose_export_validity": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "pose_export_validity.jsonl") if r.get("window_id")},
        "cowgirl_scores_v5": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "cowgirl_candidate_scores_v5.jsonl") if r.get("window_id")},
        "controller_validity": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "controller_validity.jsonl") if r.get("window_id")},
        "cowgirl_scores_v6": {r.get("window_id"): r for r in load_jsonl(run / "audits" / "cowgirl_candidate_scores_v6.jsonl") if r.get("window_id")},
    }


def _select_10(data: dict[str, Any]) -> list[dict[str, Any]]:
    pools = {
        "likely_positive": _positive_candidates(data),
        "pair_contact": _pair_contact_candidates(data),
        "suspicious_problem": _problem_candidates(data),
        "negative_control": _negative_candidates(data),
        "borderline_unclear": _borderline_candidates(data),
    }
    for rows in pools.values():
        _enrich_candidates(rows, data)
    quotas = {key: 2 for key in pools}
    selected: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    for category, quota in quotas.items():
        _take(pools[category], quota, selected, seen_windows, per_scene, per_sample, strict=True)
    if len(selected) < 10:
        _take([c for rows in pools.values() for c in rows], 10 - len(selected), selected, seen_windows, per_scene, per_sample, strict=False)
    if len(selected) != 10:
        raise ValueError(f"Could not select exactly 10 examples; selected {len(selected)}")
    return selected


def _select_10_v3(data: dict[str, Any]) -> list[dict[str, Any]]:
    quotas = {
        "likely_cowgirl_candidate": 3,
        "transition_realign": 2,
        "likely_head_bj_false_positive": 2,
        "doggy_other_confusion": 1,
        "isolated_gesture": 1,
        "unknown_mess": 1,
    }
    pools = {key: [] for key in quotas}
    for wid, frow in data["features"].items():
        bq = data["body_quality"].get(wid, {})
        match = data["reference_matches"].get(wid, {})
        phase = classify_motion_phase(frow, bq)["motion_phase_candidate"]
        guard = evaluate_domain_guards(frow, bq)
        status = match.get("recommended_review_status")
        cow_score = float(match.get("cowgirl_reference_score") or 0.0)
        head_score = max(float(match.get("bj_reference_score") or 0.0), float(match.get("head_reference_score") or 0.0))
        doggy_score = float(match.get("doggy_reference_score") or 0.0)
        quality = bq.get("body_motion_quality", "unknown")
        root_bad = quality in {"controller_only_whole_person_motion", "root_only_motion", "static_or_empty"}
        if status == "likely_cowgirl_candidate" and not root_bad and phase != "transition_adjustment_candidate":
            pools["likely_cowgirl_candidate"].append(_candidate("likely_cowgirl_candidate", wid, None, cow_score, ["high handmade cowgirl-reference score", f"body quality {quality}", f"phase {phase}"], ["cowgirl_reference_candidate"]))
        if status == "likely_transition_or_realign" or phase == "transition_adjustment_candidate":
            pools["transition_realign"].append(_candidate("transition_realign", wid, None, max(cow_score, 0.5), ["transition/realign-like motion", f"phase {phase}"], ["transition_adjustment_candidate"]))
        if status == "likely_not_cowgirl_head_or_bj" or guard.get("domain_guard_audit_labels") == ["possible_non_cowgirl_head_dominant_motion"]:
            pools["likely_head_bj_false_positive"].append(_candidate("likely_head_bj_false_positive", wid, None, head_score, ["head/BJ-domain guard candidate"], ["possible_non_cowgirl_head_dominant_motion"]))
        if status == "likely_doggy_or_other_hip_motion":
            pools["doggy_other_confusion"].append(_candidate("doggy_other_confusion", wid, None, doggy_score, ["doggy/other hip-motion confusion candidate"], ["likely_doggy_or_other_hip_motion"]))
        if status == "likely_isolated_gesture":
            pools["isolated_gesture"].append(_candidate("isolated_gesture", wid, None, max(head_score, float(match.get("hand_reference_score") or 0.0)), ["isolated hand/head gesture candidate"], ["likely_isolated_gesture"]))
        if status in {"root_or_controller_only_false_positive", "unknown_needs_review"} or root_bad:
            pools["unknown_mess"].append(_candidate("unknown_mess", wid, None, 1.0 if root_bad else 0.4, [f"quality/status needs audit: {quality}/{status}"], [status or quality]))
    for rows in pools.values():
        _enrich_candidates(rows, data)
        rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    for category, quota in quotas.items():
        _take(pools[category], quota, selected, seen, per_scene, per_sample, strict=True)
    if len(selected) < 10:
        fallback = [c for rows in pools.values() for c in rows]
        _take(fallback, 10 - len(selected), selected, seen, per_scene, per_sample, strict=False)
    if len(selected) < 10:
        fallback = _positive_candidates(data) + _borderline_candidates(data) + _negative_candidates(data)
        _enrich_candidates(fallback, data)
        _take(fallback, 10 - len(selected), selected, seen, per_scene, per_sample, strict=False)
    if len(selected) != 10:
        raise ValueError(f"Could not select exactly 10 examples; selected {len(selected)}")
    return selected


def _select_10_v4(data: dict[str, Any]) -> list[dict[str, Any]]:
    flags = data.get("selection_flags", {})
    min_duration = float(flags.get("min_cowgirl_window_seconds") or 4.0)
    quotas = {
        "likely_cowgirl_candidate": 4,
        "transition_realign": 2,
        "likely_head_bj_false_positive": 1,
        "isolated_gesture": 1,
        "doggy_other_confusion": 1,
        "unknown_mess": 1,
    }
    pools = {key: [] for key in quotas}
    for wid, score in data["cowgirl_scores_v2"].items():
        duration = float(score.get("duration_seconds") or 0.0)
        if score.get("clean_cowgirl_candidate") and duration >= min_duration:
            pools["likely_cowgirl_candidate"].append(
                _candidate(
                    "likely_cowgirl_candidate",
                    wid,
                    None,
                    float(score.get("final_clean_cowgirl_candidate_score") or 0.0),
                    ["clean Cowgirl score v2", f"duration {duration:.1f}s", "not static/micro/head-only/root-only"],
                    ["clean_cowgirl_candidate_v2"],
                )
            )
    # If the data does not have enough long clean candidates, use shorter but mark them explicitly.
    if len(pools["likely_cowgirl_candidate"]) < quotas["likely_cowgirl_candidate"]:
        for wid, score in data["cowgirl_scores_v2"].items():
            if score.get("clean_cowgirl_candidate") and wid not in {x["window_id"] for x in pools["likely_cowgirl_candidate"]}:
                duration = float(score.get("duration_seconds") or 0.0)
                pools["likely_cowgirl_candidate"].append(
                    _candidate(
                        "likely_cowgirl_candidate",
                        wid,
                        None,
                        float(score.get("final_clean_cowgirl_candidate_score") or 0.0) * 0.65,
                        ["clean Cowgirl score v2 but shorter than preferred", "too_short_for_semantic_judgment"],
                        ["clean_cowgirl_candidate_v2", "too_short_for_semantic_judgment"],
                    )
                )
    for wid, frow in data["features"].items():
        bq = data["body_quality"].get(wid, {})
        match = data["reference_matches"].get(wid, {})
        phase = classify_motion_phase(frow, bq)["motion_phase_candidate"]
        guard = evaluate_domain_guards(frow, bq)
        status = match.get("recommended_review_status")
        cow_score = float(match.get("cowgirl_reference_score") or 0.0)
        head_score = max(float(match.get("bj_reference_score") or 0.0), float(match.get("head_reference_score") or 0.0))
        doggy_score = float(match.get("doggy_reference_score") or 0.0)
        quality = bq.get("body_motion_quality", "unknown")
        if status == "likely_transition_or_realign" or phase == "transition_adjustment_candidate":
            pools["transition_realign"].append(_candidate("transition_realign", wid, None, max(cow_score, 0.5), ["transition/realign-like motion", f"phase {phase}"], ["transition_adjustment_candidate"]))
        if status == "likely_not_cowgirl_head_or_bj" or "possible_non_cowgirl_head_dominant_motion" in guard.get("domain_guard_audit_labels", []):
            pools["likely_head_bj_false_positive"].append(_candidate("likely_head_bj_false_positive", wid, None, head_score, ["head/BJ-domain guard candidate"], ["possible_non_cowgirl_head_dominant_motion"]))
        if status == "likely_doggy_or_other_hip_motion":
            pools["doggy_other_confusion"].append(_candidate("doggy_other_confusion", wid, None, doggy_score, ["doggy/other hip-motion confusion candidate"], ["likely_doggy_or_other_hip_motion"]))
        if status == "likely_isolated_gesture" or bq.get("static_or_micro_motion") or bq.get("minimal_head_motion_only") or bq.get("minimal_hand_jitter_only"):
            pools["isolated_gesture"].append(_candidate("isolated_gesture", wid, None, max(head_score, float(match.get("hand_reference_score") or 0.0), float(bq.get("micro_motion_score") or 0.0)), ["isolated/static micro-motion candidate"], ["static_or_micro_motion" if bq.get("static_or_micro_motion") else "likely_isolated_gesture"]))
        if status in {"root_or_controller_only_false_positive", "unknown_needs_review"} or quality in {"controller_only_whole_person_motion", "root_only_motion"}:
            pools["unknown_mess"].append(_candidate("unknown_mess", wid, None, 1.0 if quality in {"controller_only_whole_person_motion", "root_only_motion"} else 0.4, [f"quality/status needs audit: {quality}/{status}"], [status or quality]))
    for rows in pools.values():
        _enrich_candidates(rows, data)
        rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    for category, quota in quotas.items():
        _take(pools[category], quota, selected, seen, per_scene, per_sample, strict=True)
    if len(selected) < 10:
        _take([c for rows in pools.values() for c in rows], 10 - len(selected), selected, seen, per_scene, per_sample, strict=False)
    if len(selected) != 10:
        raise ValueError(f"Could not select exactly 10 examples; selected {len(selected)}")
    return selected


def _select_10_v5(data: dict[str, Any]) -> list[dict[str, Any]]:
    flags = data.get("selection_flags", {})
    min_duration = float(flags.get("min_cowgirl_window_seconds") or 4.0)
    quotas = {
        "likely_cowgirl_candidate": 4,
        "transition_realign": 2,
        "receiver_body_response": 1,
        "likely_head_bj_false_positive": 1,
        "isolated_gesture": 1,
        "unknown_mess": 1,
    }
    pools = {key: [] for key in quotas}
    grinding_wids: set[str] = set()
    for wid, score in data["cowgirl_scores_v3"].items():
        duration = float(score.get("duration_seconds") or 0.0)
        role_status = score.get("role_status")
        if score.get("likely_grinding_subtype"):
            grinding_wids.add(str(wid))
        if (
            score.get("clean_cowgirl_rider_candidate_v3")
            and duration >= min_duration
            and role_status != "likely_receiver_body_response"
            and not score.get("likely_receiver_false_positive")
        ):
            bonus = 0.03 if score.get("likely_grinding_subtype") else 0.0
            pools["likely_cowgirl_candidate"].append(
                _candidate(
                    "likely_cowgirl_candidate",
                    wid,
                    _pair_id_from_role_score(score),
                    float(score.get("final_clean_cowgirl_rider_score_v3") or 0.0) + bonus,
                    [
                        "clean Cowgirl rider score v3",
                        f"duration {duration:.1f}s",
                        f"role status {role_status}",
                        "receiver/body-response penalty did not trigger",
                    ],
                    ["clean_cowgirl_candidate_v3", "cowgirl_circular_grind" if score.get("likely_grinding_subtype") else "active_rider_candidate"],
                )
            )
    if grinding_wids and not any(item["window_id"] in grinding_wids for item in pools["likely_cowgirl_candidate"]):
        for wid in grinding_wids:
            score = data["cowgirl_scores_v3"].get(wid, {})
            if score.get("likely_receiver_false_positive"):
                continue
            pools["likely_cowgirl_candidate"].append(
                _candidate(
                    "likely_cowgirl_candidate",
                    wid,
                    _pair_id_from_role_score(score),
                    float(score.get("final_clean_cowgirl_rider_score_v3") or 0.0) * 0.8,
                    ["grinding subtype review candidate", "added to keep grinding represented"],
                    ["cowgirl_circular_grind", "clean_cowgirl_candidate_v3"],
                )
            )
            break

    for wid, score in data["rider_receiver_scores"].items():
        status = score.get("rider_receiver_status")
        receiver_score = float(score.get("receiver_body_response_score") or 0.0)
        bq = data["body_quality"].get(wid, {})
        quality = bq.get("body_motion_quality")
        root_or_static = quality in {"controller_only_whole_person_motion", "root_only_motion", "static_or_micro_motion", "static_or_empty"}
        if (status == "likely_receiver_body_response" or receiver_score >= 0.55) and not root_or_static:
            pools["receiver_body_response"].append(
                _candidate(
                    "receiver_body_response",
                    wid,
                    _pair_id_from_role_score(score),
                    receiver_score,
                    ["receiver/body-response false-positive candidate", f"role status {status}", "other actor appears more active in pair context"],
                    ["receiver_body_response", "not_active_rider"],
                )
            )

    for wid, frow in data["features"].items():
        bq = data["body_quality"].get(wid, {})
        match = data["reference_matches"].get(wid, {})
        phase = classify_motion_phase(frow, bq)["motion_phase_candidate"]
        guard = evaluate_domain_guards(frow, bq)
        status = match.get("recommended_review_status")
        cow_score = float(match.get("cowgirl_reference_score") or 0.0)
        head_score = max(float(match.get("bj_reference_score") or 0.0), float(match.get("head_reference_score") or 0.0))
        quality = bq.get("body_motion_quality", "unknown")
        doggy_score = float(match.get("doggy_reference_score") or 0.0)
        if status == "likely_transition_or_realign" or phase == "transition_adjustment_candidate":
            pools["transition_realign"].append(_candidate("transition_realign", wid, None, max(cow_score, 0.5), ["transition/realign-like motion", f"phase {phase}"], ["transition_adjustment_candidate"]))
        if status == "likely_not_cowgirl_head_or_bj" or "possible_non_cowgirl_head_dominant_motion" in guard.get("domain_guard_audit_labels", []):
            pools["likely_head_bj_false_positive"].append(_candidate("likely_head_bj_false_positive", wid, None, head_score, ["head/BJ-domain guard candidate"], ["possible_non_cowgirl_head_dominant_motion"]))
        if status == "likely_isolated_gesture" or bq.get("static_or_micro_motion") or bq.get("minimal_head_motion_only") or bq.get("minimal_hand_jitter_only"):
            pools["isolated_gesture"].append(_candidate("isolated_gesture", wid, None, max(head_score, float(match.get("hand_reference_score") or 0.0), float(bq.get("micro_motion_score") or 0.0)), ["isolated/static micro-motion candidate"], ["static_or_micro_motion" if bq.get("static_or_micro_motion") else "likely_isolated_gesture"]))
        if status == "likely_doggy_or_other_hip_motion" and len(pools["unknown_mess"]) < 50:
            pools["unknown_mess"].append(_candidate("unknown_mess", wid, None, doggy_score, ["doggy/other hip-motion confusion candidate kept as unknown/mess slot"], ["likely_doggy_or_other_hip_motion"]))
        if status in {"root_or_controller_only_false_positive", "unknown_needs_review"} or quality in {"controller_only_whole_person_motion", "root_only_motion", "static_or_empty"}:
            pools["unknown_mess"].append(_candidate("unknown_mess", wid, None, 1.0 if quality in {"controller_only_whole_person_motion", "root_only_motion"} else 0.4, [f"quality/status needs audit: {quality}/{status}"], [status or quality]))
    for rows in pools.values():
        _enrich_candidates(rows, data)
        rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    for category, quota in quotas.items():
        _take(pools[category], quota, selected, seen, per_scene, per_sample, strict=True)
    if len(selected) < 10:
        _take([c for rows in pools.values() for c in rows], 10 - len(selected), selected, seen, per_scene, per_sample, strict=False)
    if len(selected) != 10:
        raise ValueError(f"Could not select exactly 10 examples; selected {len(selected)}")
    return selected


def _select_10_v6(data: dict[str, Any]) -> list[dict[str, Any]]:
    flags = data.get("selection_flags", {})
    min_duration = float(flags.get("min_cowgirl_window_seconds") or 4.0)
    quotas = {
        "likely_cowgirl_candidate": 4,
        "transition_realign": 2,
        "receiver_body_response": 1,
        "likely_head_bj_false_positive": 1,
        "isolated_gesture": 1,
        "unknown_mess": 1,
    }
    pools = {key: [] for key in quotas}
    grind_present = False
    for wid, score in data["cowgirl_scores_v4"].items():
        duration = float(score.get("duration_seconds") or 0.0)
        safe = bool(score.get("safe_for_learning"))
        if (
            score.get("clean_cowgirl_candidate_v4")
            and safe
            and duration >= min_duration
            and not score.get("likely_receiver_response")
            and not score.get("likely_static_or_jitter")
            and not score.get("likely_head_or_bj_false_positive")
        ):
            bonus = 0.05 if score.get("likely_cowgirl_grinding") else 0.0
            grind_present = grind_present or bool(score.get("likely_cowgirl_grinding"))
            labels = ["clean_cowgirl_candidate_v4"]
            if score.get("likely_cowgirl_grinding"):
                labels.append("oval_or_circular_grind_trajectory")
            elif score.get("likely_cowgirl_vertical_bounce"):
                labels.append("vertical_bounce_trajectory")
            elif score.get("likely_cowgirl_forward_back_rock"):
                labels.append("forward_back_rock_trajectory")
            pools["likely_cowgirl_candidate"].append(
                _candidate(
                    "likely_cowgirl_candidate",
                    wid,
                    _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                    float(score.get("final_clean_cowgirl_score_v4") or 0.0) + bonus,
                    [
                        "relative/trajectory Cowgirl score v4",
                        f"duration {duration:.1f}s",
                        f"shape {score.get('trajectory_shape_classification')}",
                        "safe_for_learning true",
                    ],
                    labels,
                )
            )
    if not grind_present:
        for wid, score in data["cowgirl_scores_v4"].items():
            if score.get("likely_cowgirl_grinding") and score.get("safe_for_learning") and not score.get("likely_receiver_response"):
                pools["likely_cowgirl_candidate"].append(
                    _candidate(
                        "likely_cowgirl_candidate",
                        wid,
                        _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                        float(score.get("final_clean_cowgirl_score_v4") or 0.0) * 0.9,
                        ["grinding trajectory representative", "added to keep oval/circular grind evidence visible"],
                        ["oval_or_circular_grind_trajectory"],
                    )
                )
                break
    for wid, score in data["cowgirl_scores_v4"].items():
        rel = data["relative_reference_matches"].get(wid, {})
        if score.get("likely_transition_or_adjustment") or rel.get("recommended_review_status") == "likely_transition_or_realign":
            pools["transition_realign"].append(_candidate("transition_realign", wid, None, max(float(score.get("transition_penalty") or 0.0), float(rel.get("transition_trajectory_score") or 0.0)), ["transition/realign shape or relative-reference match"], ["transition_or_adjustment"]))
        if score.get("likely_head_or_bj_false_positive") or rel.get("recommended_review_status") == "likely_not_cowgirl_head_or_bj":
            pools["likely_head_bj_false_positive"].append(_candidate("likely_head_bj_false_positive", wid, None, max(float(rel.get("bj_relative_score") or 0.0), float(rel.get("head_relative_score") or 0.0), float(score.get("head_bj_penalty") or 0.0)), ["head/BJ false-positive guard in relative space"], ["head_or_bj_false_positive"]))
        if score.get("likely_static_or_jitter") or rel.get("recommended_review_status") == "likely_isolated_gesture":
            pools["isolated_gesture"].append(_candidate("isolated_gesture", wid, None, max(float(rel.get("jitter_static_score") or 0.0), float(score.get("jitter_penalty") or 0.0)), ["static/jitter or isolated gesture candidate in relative trajectory space"], ["static_or_jitter"]))
        if rel.get("recommended_review_status") in {"unsafe_relative_motion", "unknown_needs_review"} or not score.get("safe_for_learning"):
            pools["unknown_mess"].append(_candidate("unknown_mess", wid, None, 1.0 if not score.get("safe_for_learning") else 0.4, ["unsafe/unknown relative motion review case"], [rel.get("recommended_review_status") or "unknown_relative_motion"]))
    for wid, score in data["rider_receiver_scores"].items():
        status = score.get("rider_receiver_status")
        receiver_score = float(score.get("receiver_body_response_score") or 0.0)
        if status == "likely_receiver_body_response" or receiver_score >= 0.55:
            pools["receiver_body_response"].append(
                _candidate(
                    "receiver_body_response",
                    wid,
                    _pair_id_from_role_score(score),
                    receiver_score,
                    ["receiver/body-response false-positive candidate retained for audit"],
                    ["receiver_body_response", "not_active_rider"],
                )
            )
    for rows in pools.values():
        _enrich_candidates(rows, data)
        rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    for category, quota in quotas.items():
        _take(pools[category], quota, selected, seen, per_scene, per_sample, strict=True)
    if len(selected) < 10:
        _take([c for rows in pools.values() for c in rows], 10 - len(selected), selected, seen, per_scene, per_sample, strict=False)
    if len(selected) != 10:
        raise ValueError(f"Could not select exactly 10 examples; selected {len(selected)}")
    return selected


def _select_10_v7(data: dict[str, Any]) -> list[dict[str, Any]]:
    flags = data.get("selection_flags", {})
    min_duration = float(flags.get("min_cowgirl_window_seconds") or 4.0)
    quotas = {
        "semantic_cowgirl": 3,
        "generation_safe_cowgirl": 2,
        "cowgirl_context_intro": 1,
        "receiver_body_response": 1,
        "likely_head_bj_false_positive": 1,
        "transition_realign": 1,
        "export_or_pose_problem": 1,
    }
    pools = {key: [] for key in quotas}
    for wid, score in data["cowgirl_scores_v5"].items():
        duration = float(score.get("duration_seconds") or 0.0)
        if score.get("semantic_cowgirl_candidate_v5") and duration >= min_duration and not score.get("likely_receiver_response") and not score.get("likely_head_or_bj_false_positive"):
            pools["semantic_cowgirl"].append(
                _candidate(
                    "semantic_cowgirl",
                    wid,
                    _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                    float(score.get("final_semantic_cowgirl_score_v5") or 0.0),
                    ["high semantic Cowgirl score v5", "export validity is reported separately"],
                    ["semantic_cowgirl_candidate_v5"],
                )
            )
        if score.get("generation_candidate_v5") and duration >= min_duration and not score.get("semantically_good_but_not_generation_safe"):
            pools["generation_safe_cowgirl"].append(
                _candidate(
                    "generation_safe_cowgirl",
                    wid,
                    _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                    float(score.get("final_generation_candidate_score_v5") or 0.0),
                    ["higher generation candidate score v5", "no known broken-pose/export block"],
                    ["generation_candidate_v5"],
                )
            )
        if score.get("cowgirl_context_low_motion_intro") or score.get("too_short_penalty", 0) or score.get("intro_low_motion_penalty", 0):
            pools["cowgirl_context_intro"].append(
                _candidate(
                    "cowgirl_context_intro",
                    wid,
                    None,
                    max(float(score.get("final_semantic_cowgirl_score_v5") or 0.0), float(score.get("intro_low_motion_penalty") or 0.0)),
                    ["Cowgirl pose/context but low-motion intro or too short"],
                    ["cowgirl_context_low_motion_intro"],
                )
            )
        if score.get("likely_transition_or_adjustment"):
            pools["transition_realign"].append(
                _candidate("transition_realign", wid, None, float(score.get("transition_penalty") or 0.0), ["transition/realign candidate"], ["transition_or_adjustment"])
            )
        if score.get("likely_head_or_bj_false_positive"):
            pools["likely_head_bj_false_positive"].append(
                _candidate("likely_head_bj_false_positive", wid, None, float(score.get("head_bj_penalty") or 0.0), ["head/BJ or isolated false-positive guard"], ["head_or_bj_false_positive"])
            )
        if score.get("semantically_good_but_not_generation_safe") or score.get("export_pose_validity") in {"broken_pose", "export_unavailable"}:
            pools["export_or_pose_problem"].append(
                _candidate(
                    "export_or_pose_problem",
                    wid,
                    None,
                    max(float(score.get("final_semantic_cowgirl_score_v5") or 0.0), float(score.get("broken_pose_penalty_for_export_only") or 0.0)),
                    ["semantic/export split case: inspect pose validity separately"],
                    [score.get("export_pose_validity") or "export_pose_validity_unknown"],
                )
            )
    for wid, score in data["rider_receiver_scores"].items():
        receiver_score = float(score.get("receiver_body_response_score") or 0.0)
        if score.get("rider_receiver_status") == "likely_receiver_body_response" or receiver_score >= 0.55:
            pools["receiver_body_response"].append(
                _candidate(
                    "receiver_body_response",
                    wid,
                    _pair_id_from_role_score(score),
                    receiver_score,
                    ["receiver/body-response negative retained for audit"],
                    ["receiver_body_response", "not_active_rider"],
                )
            )
    for rows in pools.values():
        _enrich_candidates(rows, data)
        rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    for category, quota in quotas.items():
        _take(pools[category], quota, selected, seen, per_scene, per_sample, strict=True)
    if len(selected) < 10:
        _take([c for rows in pools.values() for c in rows], 10 - len(selected), selected, seen, per_scene, per_sample, strict=False)
    if len(selected) != 10:
        raise ValueError(f"Could not select exactly 10 examples; selected {len(selected)}")
    return selected


def _select_10_v8(data: dict[str, Any]) -> list[dict[str, Any]]:
    flags = data.get("selection_flags", {})
    min_duration = float(flags.get("min_cowgirl_window_seconds") or 4.0)
    quotas = {
        "semantic_cowgirl_controller_invalid": 2,
        "generation_safe_cowgirl": 2,
        "semantic_cowgirl": 3,
        "cowgirl_context_intro": 1,
        "receiver_body_response": 1,
        "unknown_or_export_unavailable": 1,
    }
    pools = {key: [] for key in quotas}
    for wid, score in data["cowgirl_scores_v6"].items():
        duration = float(score.get("duration_seconds") or 0.0)
        controller_status = str(score.get("controller_validity_status") or "unknown")
        foot_outlier = bool(score.get("foot_controller_outlier"))
        if score.get("semantic_cowgirl_candidate_v6") and duration >= min_duration and not score.get("semantically_cowgirl_but_controller_invalid") and not score.get("likely_receiver_response") and not score.get("likely_head_or_bj_false_positive"):
            pools["semantic_cowgirl"].append(
                _candidate(
                    "semantic_cowgirl",
                    wid,
                    _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                    float(score.get("final_semantic_cowgirl_score_v6") or 0.0),
                    ["high semantic Cowgirl score v6", "controller validity is reported separately"],
                    ["semantic_cowgirl_candidate_v6"],
                )
            )
        if score.get("generation_candidate_v6") and duration >= min_duration and controller_status == "valid" and not foot_outlier:
            pools["generation_safe_cowgirl"].append(
                _candidate(
                    "generation_safe_cowgirl",
                    wid,
                    _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                    float(score.get("final_generation_candidate_score_v6") or 0.0),
                    ["generation-safe Cowgirl candidate v6", "valid controller pose; no foot outlier"],
                    ["generation_candidate_v6", "generation_pose_valid"],
                )
            )
        if score.get("semantically_cowgirl_but_controller_invalid") or (score.get("semantic_cowgirl_candidate_v6") and foot_outlier):
            pools["semantic_cowgirl_controller_invalid"].append(
                _candidate(
                    "semantic_cowgirl_controller_invalid",
                    wid,
                    _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                    float(score.get("final_semantic_cowgirl_score_v6") or 0.0),
                    ["semantically Cowgirl but controller/pose invalid for generation", "inspect foot/controller outlier"],
                    ["foot_controller_outlier" if foot_outlier else "controller_outlier"],
                )
            )
        if score.get("cowgirl_context_low_motion_intro") or float(score.get("cowgirl_context_score") or 0.0) >= 0.25:
            pools["cowgirl_context_intro"].append(
                _candidate(
                    "cowgirl_context_intro",
                    wid,
                    None,
                    float(score.get("cowgirl_context_score") or 0.0),
                    ["Cowgirl pose/context or low-motion intro; not clean motion"],
                    ["cowgirl_context_low_motion_intro"],
                )
            )
        if score.get("export_unavailable_for_generation") or score.get("export_pose_validity") == "export_unavailable" or controller_status == "unknown":
            pools["unknown_or_export_unavailable"].append(
                _candidate(
                    "unknown_or_export_unavailable",
                    wid,
                    None,
                    max(float(score.get("final_semantic_cowgirl_score_v6") or 0.0), 0.25),
                    ["unknown/export-unavailable/controller-unknown audit case"],
                    [score.get("export_pose_validity") or controller_status],
                )
            )
    for wid, score in data["rider_receiver_scores"].items():
        receiver_score = float(score.get("receiver_body_response_score") or 0.0)
        if score.get("rider_receiver_status") == "likely_receiver_body_response" or receiver_score >= 0.55:
            pools["receiver_body_response"].append(
                _candidate(
                    "receiver_body_response",
                    wid,
                    _pair_id_from_role_score(score),
                    receiver_score,
                    ["receiver/body-response negative retained for audit"],
                    ["receiver_body_response", "not_active_rider"],
                )
            )
    # If controller-invalid semantic examples are rare, include high outlier windows
    # so the batch still checks the newly added plausibility gate.
    if len(pools["semantic_cowgirl_controller_invalid"]) < quotas["semantic_cowgirl_controller_invalid"]:
        for wid, controller in data["controller_validity"].items():
            if controller.get("foot_controller_outlier") or controller.get("controller_validity_status") == "invalid":
                score = data["cowgirl_scores_v6"].get(wid, {})
                pools["semantic_cowgirl_controller_invalid"].append(
                    _candidate(
                        "semantic_cowgirl_controller_invalid",
                        wid,
                        _pair_id_from_role_score(data["rider_receiver_scores"].get(wid, {})),
                        max(float(score.get("final_semantic_cowgirl_score_v6") or 0.0), float(controller.get("controller_validity_score") or 0.0)),
                        ["controller-invalid audit case; may or may not be Cowgirl semantics"],
                        ["foot_controller_outlier" if controller.get("foot_controller_outlier") else "controller_outlier"],
                    )
                )
    for rows in pools.values():
        _enrich_candidates(rows, data)
        rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    for category, quota in quotas.items():
        _take(pools[category], quota, selected, seen, per_scene, per_sample, strict=True)
    if len(selected) < 10:
        _take([c for rows in pools.values() for c in rows], 10 - len(selected), selected, seen, per_scene, per_sample, strict=False)
    if len(selected) != 10:
        raise ValueError(f"Could not select exactly 10 examples; selected {len(selected)}")
    return selected


def _positive_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    wanted = {"cowgirl_vertical_bounce", "cowgirl_forward_back_rock", "cowgirl_circular_grind"}
    for wid, rows in data["window_scores"].items():
        labels = {str(r.get("label")) for r in rows}
        if not labels & wanted:
            continue
        score = max(float(r.get("final_score") or 0.0) for r in rows if r.get("label") in wanted)
        out.append(_candidate("likely_positive", wid, None, score, ["high movement machine/silver candidate"], sorted(labels & wanted)))
    return _sort_with_timeline_preference(out, data)


def _pair_contact_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for pid, rows in data["pair_scores"].items():
        labels = {str(r.get("label")) for r in rows}
        if not labels & CONTACT_LABELS:
            continue
        pair = data["pair_windows"].get(pid, {})
        wid = _preferred_pair_window_id(pair, rows)
        score = max(float(r.get("final_score") or 0.0) for r in rows)
        out.append(_candidate("pair_contact", wid, pid, score, ["pair/contact or active-passive candidate"], sorted(labels & CONTACT_LABELS)))
    return _sort_with_timeline_preference(out, data)


def _problem_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    pair_context_counts = Counter()
    for rows in data["pair_scores"].values():
        for row in rows:
            for wid in row.get("window_ids", []) or []:
                pair_context_counts[str(wid)] += 1
    for wid, rows in data["window_scores"].items():
        labels = {str(r.get("label")) for r in rows}
        reasons = []
        if any(r.get("conflict_flags") or r.get("recommended_status") == "reject_conflict" for r in rows):
            reasons.append("contradictory machine labels")
        if pair_context_counts[wid] > 30:
            reasons.append("many pair contexts")
        frow = data["features"].get(wid, {})
        if frow.get("missing_controller_groups") or (frow.get("feature_quality", {}) or {}).get("root_mapping_confidence") not in {"high", None}:
            reasons.append("controller/axis feature ambiguity")
        sample_audit = data["baked_audit"].get(frow.get("sample_id"), {})
        if sample_audit.get("suspiciously_static") or sample_audit.get("suspiciously_huge_motion"):
            reasons.append("baked sample audit warning")
        if reasons:
            out.append(_candidate("suspicious_problem", wid, None, float(len(rows) + pair_context_counts[wid]), reasons, sorted(labels)))
    return _sort_with_timeline_preference(out, data)


def _negative_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for wid, frow in data["features"].items():
        values = frow.get("feature_values", {}) or {}
        energy = _num(values.get("pelvis_movement_energy"), 999.0)
        speed = _num(values.get("pelvis_mean_speed"), 999.0)
        pause = _num(values.get("pause_hold_score_proxy"), 0.0)
        score = (1.0 - min(energy * 20.0, 1.0)) + (1.0 - min(speed * 8.0, 1.0)) + pause
        if score > 1.0:
            reasons = ["low pelvis/root motion or passive/static context candidate"]
            if data["baked_audit"].get(frow.get("sample_id"), {}).get("suspiciously_static"):
                reasons.append("suspicious static sample")
            out.append(_candidate("negative_control", wid, None, score, reasons, []))
    return _sort_with_timeline_preference(out, data)


def _borderline_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for wid, rows in data["window_scores"].items():
        labels = {str(r.get("label")) for r in rows}
        if not ({"cowgirl_adjustment_transition", "cowgirl_pause_hold", "cowgirl_irregular_human_motion"} & labels):
            continue
        scores = [float(r.get("final_score") or 0.0) for r in rows]
        if any(0.55 <= s <= 0.82 for s in scores):
            out.append(_candidate("borderline_unclear", wid, None, max(scores), ["medium-confidence transition/pause/unclear movement candidate"], sorted(labels & MOVEMENT_LABELS)))
    return _sort_with_timeline_preference(out, data)


def _candidate(category: str, wid: str, pid: str | None, score: float, reasons: list[str], labels: list[str]) -> dict[str, Any]:
    return {"category": category, "window_id": wid, "pair_window_id": pid, "score": score, "why_selected": reasons, "labels": labels}


def _pair_id_from_role_score(score: dict[str, Any]) -> str | None:
    for item in score.get("pair_evidence", []) or []:
        pid = item.get("pair_window_id")
        if pid:
            return str(pid)
    return None


def _enrich_candidates(candidates: list[dict[str, Any]], data: dict[str, Any]) -> None:
    for item in candidates:
        wrow = data["windows"].get(item.get("window_id"), {})
        item["source_scene_file"] = wrow.get("source_scene_file")
        item["sample_id"] = wrow.get("sample_id")


def _sort_with_timeline_preference(items: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[int, float]:
        sample = _sample_for_window(item["window_id"], data)
        timeline_safe = 1 if sample and sample.get("source_type") == "timeline_controller_motion" else 0
        return (timeline_safe, float(item.get("score") or 0.0))
    return sorted(items, key=key, reverse=True)


def _take(candidates: list[dict[str, Any]], quota: int, selected: list[dict[str, Any]], seen: set[str], per_scene: Counter[str], per_sample: Counter[str], strict: bool) -> None:
    added = 0
    for item in candidates:
        if added >= quota:
            return
        wid = item["window_id"]
        if wid in seen:
            continue
        wrow = getattr(item, "_wrow", None)
        scene = _get_window_scene(wid, item, wrow)
        sample = _get_window_sample(wid, item, wrow)
        if strict and (per_scene[scene] >= 2 or per_sample[sample] >= 1):
            continue
        if not strict and (per_scene[scene] >= 3 or per_sample[sample] >= 2):
            continue
        selected.append(item)
        seen.add(wid)
        per_scene[scene] += 1
        per_sample[sample] += 1
        added += 1


def _make_review_row(idx: int, item: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    wid = item["window_id"]
    wrow = data["windows"].get(wid, {})
    frow = data["features"].get(wid, {})
    sample = data["samples"].get(wrow.get("sample_id") or frow.get("sample_id"), {})
    pair = data["pair_windows"].get(item.get("pair_window_id")) if item.get("pair_window_id") else data["pair_by_window"].get(wid, {})
    pair_feature = data["pair_features"].get(pair.get("pair_window_id"), {}) if pair else {}
    pair_scores = data["pair_scores"].get(pair.get("pair_window_id"), []) if pair else []
    window_scores = data["window_scores"].get(wid, [])
    silver_window = data["silver_windows"].get(wid, {})
    silver_pair = data["silver_pairs"].get(pair.get("pair_window_id"), {}) if pair else {}
    body_quality = data["body_quality"].get(wid, {})
    reference_match = data["reference_matches"].get(wid, {})
    cowgirl_score = data["cowgirl_scores_v2"].get(wid, {})
    rider_receiver = data["rider_receiver_scores"].get(wid, {})
    cowgirl_score_v3 = data["cowgirl_scores_v3"].get(wid, {})
    relative_feature = data["relative_features"].get(wid, {})
    trajectory_feature = data["trajectory_features"].get(wid, {})
    relative_match = data["relative_reference_matches"].get(wid, {})
    cowgirl_score_v4 = data["cowgirl_scores_v4"].get(wid, {})
    pose_export = data["pose_export_validity"].get(wid, {})
    cowgirl_score_v5 = data["cowgirl_scores_v5"].get(wid, {})
    controller_validity = data["controller_validity"].get(wid, {})
    cowgirl_score_v6 = data["cowgirl_scores_v6"].get(wid, {})
    phase = classify_motion_phase(frow, body_quality)
    guard = evaluate_domain_guards(frow, body_quality)
    review_id = f"review_{idx:03d}"
    guess = _semantic_guess(item, frow, window_scores, pair_feature, pair_scores, silver_window, silver_pair, body_quality, phase, guard, reference_match, rider_receiver, cowgirl_score_v3, relative_feature, trajectory_feature, relative_match, cowgirl_score_v4, pose_export, cowgirl_score_v5, controller_validity, cowgirl_score_v6)
    pair_actor = _pair_actor(wid, pair)
    return {
        "review_id": review_id,
        "source_scene_path": sample.get("source_scene_path") or wrow.get("source_scene_path"),
        "source_scene_file": wrow.get("source_scene_file") or frow.get("source_scene_file") or sample.get("source_scene_file"),
        "technical_atom_id": wrow.get("technical_atom_id") or frow.get("technical_atom_id") or sample.get("technical_atom_id"),
        "pair_technical_atom_id": pair_actor,
        "window_id": wid,
        "pair_window_id": pair.get("pair_window_id") if pair else None,
        "sample_id": wrow.get("sample_id") or frow.get("sample_id"),
        "source_id": wrow.get("source_id") or frow.get("source_id"),
        "start_seconds": wrow.get("start_seconds"),
        "end_seconds": wrow.get("end_seconds"),
        "duration_seconds": wrow.get("duration_seconds"),
        "frame_start": wrow.get("frame_start"),
        "frame_end": wrow.get("frame_end"),
        "category": item["category"],
        "has_timeline_export": False,
        "timeline_export_path": None,
        "timeline_export_validation_status": "not_attempted",
        "timeline_export_warnings": [],
        "system_semantic_guess": guess,
        "evidence": {
            "top_features": _top_features(frow.get("feature_values", {})),
            "weak_labels": _weak_labels(data["weak"].get(wid, {})),
            "machine_proposals": _score_hints(window_scores, 10),
            "silver_labels": _silver_hint(silver_window),
            "pair_feature_summary": _pair_summary(pair, pair_feature, pair_scores, silver_pair),
            "body_motion_quality": body_quality,
            "motion_phase_candidate": phase,
            "domain_guard_warnings": guard,
            "handmade_reference_match": reference_match,
            "clean_cowgirl_candidate_score_v2": cowgirl_score,
            "rider_receiver_discrimination": rider_receiver,
            "clean_cowgirl_candidate_score_v3": cowgirl_score_v3,
            "relative_motion_features": relative_feature,
            "trajectory_shape_features": trajectory_feature,
            "relative_reference_match": relative_match,
            "clean_cowgirl_candidate_score_v4": cowgirl_score_v4,
            "pose_export_validity": pose_export,
            "clean_cowgirl_candidate_score_v5": cowgirl_score_v5,
            "controller_validity": controller_validity,
            "clean_cowgirl_candidate_score_v6": cowgirl_score_v6,
        },
        "why_selected": item["why_selected"],
        "user_questions": _questions_for_item(bool(pair)),
        "answer_options": ["correct", "wrong", "unclear"],
        "is_human_ground_truth": False,
        "export_context_padding_seconds": 0.5 if (data.get("selection_flags", {}).get("use_cowgirl_candidate_score_v2") or data.get("selection_flags", {}).get("use_cowgirl_candidate_score_v3") or data.get("selection_flags", {}).get("use_cowgirl_candidate_score_v4") or data.get("selection_flags", {}).get("use_cowgirl_candidate_score_v5") or data.get("selection_flags", {}).get("use_cowgirl_candidate_score_v6")) else 0.0,
    }


def _semantic_guess(
    item: dict[str, Any],
    frow: dict[str, Any],
    scores: list[dict[str, Any]],
    pair_feature: dict[str, Any],
    pair_scores: list[dict[str, Any]],
    silver_window: dict[str, Any],
    silver_pair: dict[str, Any],
    body_quality: dict[str, Any] | None = None,
    phase: dict[str, Any] | None = None,
    guard: dict[str, Any] | None = None,
    reference_match: dict[str, Any] | None = None,
    rider_receiver: dict[str, Any] | None = None,
    cowgirl_score_v3: dict[str, Any] | None = None,
    relative_feature: dict[str, Any] | None = None,
    trajectory_feature: dict[str, Any] | None = None,
    relative_match: dict[str, Any] | None = None,
    cowgirl_score_v4: dict[str, Any] | None = None,
    pose_export: dict[str, Any] | None = None,
    cowgirl_score_v5: dict[str, Any] | None = None,
    controller_validity: dict[str, Any] | None = None,
    cowgirl_score_v6: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score_labels = [str(r.get("label")) for r in sorted(scores, key=lambda r: float(r.get("final_score") or 0.0), reverse=True)]
    silver_labels = list(silver_window.get("positive_labels", []) or [])
    movement_labels = [label for label in [*silver_labels, *score_labels, *item.get("labels", [])] if label in MOVEMENT_LABELS]
    movement_labels = _dedupe(movement_labels)[:3]
    pair_labels = [str(r.get("label")) for r in sorted(pair_scores, key=lambda r: float(r.get("final_score") or 0.0), reverse=True)]
    contact_labels = [label for label in [*(silver_pair.get("positive_labels", []) or []), *pair_labels] if label in CONTACT_LABELS and label not in ROLE_LABELS]
    role_labels = [label for label in [*score_labels, *pair_labels] if label in ROLE_LABELS]
    values = frow.get("feature_values", {}) or {}
    pair_q = pair_feature.get("feature_quality", {}) or {}
    role_conf = _num(pair_q.get("active_actor_confidence"), 0.0)
    move_conf = max([_num(r.get("final_score"), 0.0) for r in scores if r.get("label") in movement_labels] or [0.0])
    contact_conf = max([_num(r.get("final_score"), 0.0) for r in pair_scores if r.get("label") in contact_labels] or [0.0])
    active_candidate = "likely yes" if "rider_active" in role_labels or role_conf >= 0.65 else "unclear"
    passive_candidate = "possible paired context" if "partner_context_static" in role_labels else "unclear"
    posture = []
    if _num(values.get("torso_lean_forward_proxy"), 0.5) > 0.7:
        posture.append("lean_forward_proxy_uncertain")
    if _num(values.get("torso_lean_back_proxy"), 0.5) > 0.7:
        posture.append("lean_back_proxy_uncertain")
    if not movement_labels and item["category"] == "negative_control":
        movement_labels = ["low_motion_or_control_candidate"]
    body_quality = body_quality or {}
    phase = phase or {}
    guard = guard or {}
    reference_match = reference_match or {}
    rider_receiver = rider_receiver or {}
    cowgirl_score_v3 = cowgirl_score_v3 or {}
    relative_feature = relative_feature or {}
    trajectory_feature = trajectory_feature or {}
    relative_match = relative_match or {}
    cowgirl_score_v4 = cowgirl_score_v4 or {}
    pose_export = pose_export or {}
    cowgirl_score_v5 = cowgirl_score_v5 or {}
    controller_validity = controller_validity or {}
    cowgirl_score_v6 = cowgirl_score_v6 or {}
    multiplier = float(guard.get("cowgirl_confidence_multiplier") or 1.0)
    if body_quality.get("body_motion_quality") in {"controller_only_whole_person_motion", "root_only_motion"}:
        active_candidate = "unsafe/root-motion only; not valid rider output"
        movement_labels = ["root_only_motion_false_positive" if body_quality.get("body_motion_quality") == "root_only_motion" else "controller_only_whole_person_motion"]
        move_conf *= multiplier
        role_conf *= multiplier
    if "possible_non_cowgirl_head_dominant_motion" in guard.get("domain_guard_audit_labels", []):
        movement_labels = _dedupe(["possible_non_cowgirl_head_dominant_motion", *movement_labels])[:3]
        move_conf *= multiplier
    role_status = rider_receiver.get("rider_receiver_status")
    active_score = _num(rider_receiver.get("active_rider_score"), 0.0)
    receiver_score = _num(rider_receiver.get("receiver_body_response_score"), 0.0)
    role_conf = max(role_conf, active_score, receiver_score)
    if cowgirl_score_v3.get("likely_grinding_subtype"):
        movement_labels = _dedupe(["cowgirl_circular_grind", *movement_labels])[:3]
        move_conf = max(move_conf, _num(cowgirl_score_v3.get("cowgirl_grinding_score"), 0.0))
    if cowgirl_score_v4.get("likely_cowgirl_grinding"):
        movement_labels = _dedupe(["cowgirl_circular_grind", *movement_labels])[:4]
        move_conf = max(move_conf, _num(cowgirl_score_v4.get("final_clean_cowgirl_score_v4"), 0.0), _num(cowgirl_score_v4.get("trajectory_grind_score"), 0.0))
    elif cowgirl_score_v4.get("likely_cowgirl_vertical_bounce"):
        movement_labels = _dedupe(["cowgirl_vertical_bounce", *movement_labels])[:4]
        move_conf = max(move_conf, _num(cowgirl_score_v4.get("final_clean_cowgirl_score_v4"), 0.0))
    elif cowgirl_score_v4.get("likely_cowgirl_forward_back_rock"):
        movement_labels = _dedupe(["cowgirl_forward_back_rock", *movement_labels])[:4]
        move_conf = max(move_conf, _num(cowgirl_score_v4.get("final_clean_cowgirl_score_v4"), 0.0))
    if cowgirl_score_v4.get("likely_static_or_jitter"):
        movement_labels = _dedupe(["static_or_jitter_trajectory", *movement_labels])[:4]
    if cowgirl_score_v4.get("likely_head_or_bj_false_positive"):
        movement_labels = _dedupe(["possible_non_cowgirl_head_dominant_motion", *movement_labels])[:4]
    if cowgirl_score_v5.get("cowgirl_context_low_motion_intro"):
        movement_labels = _dedupe(["cowgirl_intro_or_start_pose", "not_clean_motion", *movement_labels])[:4]
    if cowgirl_score_v6.get("foot_controller_outlier"):
        movement_labels = _dedupe([*movement_labels, "foot_controller_outlier"])[:5]
    if role_status == "likely_active_rider":
        active_candidate = "likely yes (motion/pair evidence)"
        passive_candidate = "unlikely"
    elif role_status == "likely_receiver_body_response":
        active_candidate = "likely no - receiver/body-response candidate"
        passive_candidate = "likely receiver/body-response"
        movement_labels = _dedupe(["receiver_body_response", "not_active_rider", *movement_labels])[:4]
    elif role_status == "likely_passive_context":
        active_candidate = "likely no - passive context candidate"
        passive_candidate = "likely passive/context"
    elif role_status == "insufficient_pair_context" and item["category"] == "likely_cowgirl_candidate":
        active_candidate = "possible; pair context unavailable"
    elif role_status == "role_unclear":
        active_candidate = "unclear; rider/receiver evidence conflicts"
    return {
        "active_rider_candidate": active_candidate,
        "passive_receiver_candidate": passive_candidate,
        "movement_labels": movement_labels,
        "contact_labels": _dedupe(contact_labels)[:3] or ["unclear"],
        "posture_labels": posture or ["unclear"],
        "role_confidence": round(float(role_conf), 3),
        "movement_confidence": round(float(move_conf), 3),
        "contact_confidence": round(float(contact_conf), 3),
        "overall_confidence": round(float(max(role_conf, move_conf, contact_conf)), 3),
        "body_motion_quality": body_quality.get("body_motion_quality", "unknown"),
        "motion_phase_candidate": phase.get("motion_phase_candidate", "unknown_phase"),
        "domain_guard_warnings": guard.get("domain_guard_warnings", []),
        "reference_review_status": reference_match.get("recommended_review_status"),
        "nearest_handmade_reference_families": reference_match.get("nearest_reference_families", []),
        "rider_receiver_status": role_status,
        "active_rider_score": round(float(active_score), 3),
        "receiver_body_response_score": round(float(receiver_score), 3),
        "clean_cowgirl_rider_score_v3": cowgirl_score_v3.get("final_clean_cowgirl_rider_score_v3"),
        "cowgirl_grinding_score": cowgirl_score_v3.get("cowgirl_grinding_score"),
        "likely_grinding_subtype": cowgirl_score_v3.get("likely_grinding_subtype"),
        "relative_cowgirl_score": relative_match.get("cowgirl_relative_score"),
        "trajectory_shape_classification": trajectory_feature.get("trajectory_shape_classification"),
        "oval_path_score": (trajectory_feature.get("feature_values") or {}).get("oval_path_score"),
        "ellipse_fit_score": (trajectory_feature.get("feature_values") or {}).get("ellipse_fit_score"),
        "closed_loop_ratio": (trajectory_feature.get("feature_values") or {}).get("closed_loop_ratio"),
        "dominant_motion_plane": trajectory_feature.get("dominant_motion_plane"),
        "clean_cowgirl_score_v4": cowgirl_score_v4.get("final_clean_cowgirl_score_v4"),
        "semantic_cowgirl_score_v5": cowgirl_score_v5.get("final_semantic_cowgirl_score_v5"),
        "generation_candidate_score_v5": cowgirl_score_v5.get("final_generation_candidate_score_v5"),
        "semantic_cowgirl_score_v6": cowgirl_score_v6.get("final_semantic_cowgirl_score_v6"),
        "generation_candidate_score_v6": cowgirl_score_v6.get("final_generation_candidate_score_v6"),
        "clean_motion_score": cowgirl_score_v6.get("clean_motion_score"),
        "cowgirl_context_score": cowgirl_score_v6.get("cowgirl_context_score"),
        "controller_validity_score": cowgirl_score_v6.get("controller_validity_score") or controller_validity.get("controller_validity_score"),
        "controller_validity_status": cowgirl_score_v6.get("controller_validity_status") or controller_validity.get("controller_validity_status"),
        "foot_controller_outlier": bool(cowgirl_score_v6.get("foot_controller_outlier") or controller_validity.get("foot_controller_outlier")),
        "hand_controller_outlier": bool(cowgirl_score_v6.get("hand_controller_outlier") or controller_validity.get("hand_controller_outlier")),
        "controller_outlier_count": cowgirl_score_v6.get("controller_outlier_count") or controller_validity.get("controller_outlier_count"),
        "export_pose_validity": cowgirl_score_v5.get("export_pose_validity") or pose_export.get("export_pose_validity"),
        "generation_template_safe": bool(cowgirl_score_v6.get("generation_candidate_v6") or cowgirl_score_v5.get("generation_template_safe") or pose_export.get("generation_template_safe")),
        "semantically_good_but_not_generation_safe": bool(cowgirl_score_v5.get("semantically_good_but_not_generation_safe")),
        "semantically_cowgirl_but_controller_invalid": bool(cowgirl_score_v6.get("semantically_cowgirl_but_controller_invalid")),
        "low_motion_intro_candidate": bool(cowgirl_score_v5.get("cowgirl_context_low_motion_intro") or pose_export.get("low_motion_intro_candidate")),
        "safe_for_learning": bool((relative_feature.get("feature_values") or {}).get("safe_for_learning") or relative_match.get("safe_for_learning")),
        "teleport_risk": (relative_feature.get("feature_quality") or {}).get("teleport_risk"),
        "safe_for_generation_template": bool(cowgirl_score_v4.get("safe_for_learning")) and False,
        "relative_nearest_handmade_references": relative_match.get("nearest_handmade_references", []),
        "why_not_receiver_body_response": (
            "receiver/body-response penalty did not trigger"
            if role_status != "likely_receiver_body_response"
            else "receiver/body-response penalty triggered; do not treat as active rider without human confirmation"
        ),
        "warning": "Machine/weak/silver labels are hints only and not human truth.",
    }


def _attempt_timeline_export(row: dict[str, Any], data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = data["samples"].get(row.get("sample_id"))
    if not sample:
        return _write_export_unavailable(out_dir, row, "sample record not found")
    if sample.get("source_type") != "timeline_controller_motion":
        reason = f"source_type is {sample.get('source_type')}; only timeline_controller_motion is exported by this guarded review tool"
        return _write_export_unavailable(out_dir, row, reason)
    if sample.get("source_type") in HIGH_RISK_EXPORT_SOURCE_TYPES:
        return _write_export_unavailable(out_dir, row, "native/high-risk source type is not exported by default")
    loaded = _load_npz_window(sample, row, data["run_dir"])
    if loaded is None:
        return _write_export_unavailable(out_dir, row, "baked NPZ missing or unreadable")
    positions, rotations, names, times, export_start, export_end = loaded
    positions, rotations, names, safety = _filter_safe_export_controllers(positions, rotations, names)
    if not names:
        return _write_export_unavailable(out_dir, row, "no allowed bodypart controller tracks remain after stripping Person/root/world tracks")
    if positions.size == 0 or rotations.size == 0:
        return _write_export_unavailable(out_dir, row, "empty position/rotation arrays")
    validation = _validate_arrays(positions, rotations)
    if validation["status"] != "ok":
        result = _write_export_unavailable(out_dir, row, f"numeric validation failed: {validation['warnings']}")
        result["validation_status"] = validation["status"]
        return result
    duration = float(times[-1] if len(times) else row.get("duration_seconds") or 0.0)
    timeline = _build_timeline_json(row["review_id"], duration, names, positions, rotations)
    roundtrip = _validate_timeline_roundtrip(timeline, positions, rotations)
    if roundtrip["status"] != "ok":
        result = _write_export_unavailable(out_dir, row, f"roundtrip validation failed: {roundtrip['warnings']}")
        result["validation_status"] = roundtrip["status"]
        return result
    timeline_path = out_dir / f"{row['review_id']}.timeline.json"
    meta_path = out_dir / f"{row['review_id']}.timeline_meta.json"
    notes_path = out_dir / f"{row['review_id']}_import_notes.md"
    dump_json(timeline_path, timeline)
    meta = {
        "review_id": row["review_id"],
        "window_id": row["window_id"],
        "source_scene_file": row["source_scene_file"],
        "source_scene_path": row["source_scene_path"],
        "technical_atom_id": row["technical_atom_id"],
        "source_id": row["source_id"],
        "sample_id": row["sample_id"],
        "original_start_seconds": row["start_seconds"],
        "original_end_seconds": row["end_seconds"],
        "semantic_window_start_seconds": row["start_seconds"],
        "semantic_window_end_seconds": row["end_seconds"],
        "exported_with_context_start_seconds": export_start,
        "exported_with_context_end_seconds": export_end,
        "exported_duration_seconds": duration,
        "controller_names": names,
        "export_format": "AcidBubbles Timeline-style JSON, dense linear keys",
        "coordinate_space_assumption": "Original Timeline controller-space/control-space for the same source scene and atom; not retargeted.",
        "source_world_coords_stripped": safety["source_world_coords_stripped"],
        "exported_as_relative_motion": safety["exported_as_relative_motion"],
        "uses_absolute_review_coordinates": True,
        "review_export_safe_to_inspect": safety["safe_for_import"],
        "safe_for_import": safety["safe_for_import"],
        "safe_for_learning": bool(row.get("system_semantic_guess", {}).get("safe_for_learning")),
        "generation_template_safe": False,
        "safe_for_generation_template": safety["safe_for_generation_template"],
        "pose_broken_risk": row.get("system_semantic_guess", {}).get("export_pose_validity") in {"broken_pose", "unknown"},
        "controller_validity_status": row.get("system_semantic_guess", {}).get("controller_validity_status"),
        "foot_controller_outlier": row.get("system_semantic_guess", {}).get("foot_controller_outlier"),
        "hand_controller_outlier": row.get("system_semantic_guess", {}).get("hand_controller_outlier"),
        "controller_outlier_count": row.get("system_semantic_guess", {}).get("controller_outlier_count"),
        "generation_template_block_reasons": _generation_template_block_reasons(row, safety),
        "exported_controller_count": len(names),
        "stripped_world_transform_count": safety["stripped_world_transform_count"],
        "stripped_atom_root_count": safety["stripped_atom_root_count"],
        "teleport_risk": safety["teleport_risk"],
        "export_safe_for_import": safety["export_safe_for_import"],
        "timeline_export_safe_for_animation": safety["timeline_export_safe_for_animation"],
        "validation_status": "ok",
        "warnings": [
            "VaM visual import has not been tested.",
            "Use the original source scene/atom for safest review.",
            "This export is a convenience segment, not proof of semantic correctness.",
            "This review export uses source controller coordinates and is not a generation-safe relative template.",
        ],
        "import_instructions": "Open a copy of the source scene, select the listed technical atom, open AcidBubbles Timeline, and try importing the JSON segment if your Timeline version accepts this external format.",
        "roundtrip": roundtrip,
        "safety": safety,
    }
    dump_json(meta_path, meta)
    _write_import_notes(notes_path, row, meta)
    return {
        "attempted": True,
        "success": True,
        "timeline_export_path": str(timeline_path),
        "metadata_path": str(meta_path),
        "validation_status": "ok",
        "warnings": meta["warnings"],
    }


def _filter_safe_export_controllers(positions: np.ndarray, rotations: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    safe_indices: list[int] = []
    stripped_root = 0
    stripped_world = 0
    for idx, name in enumerate(names):
        kind = classify_timeline_target(name)
        if kind == "allowed_body_controller":
            safe_indices.append(idx)
        elif "world" in str(name).lower():
            stripped_world += 1
        else:
            stripped_root += 1
    safe_names = [names[i] for i in safe_indices]
    if safe_indices:
        safe_positions = positions[:, safe_indices, :]
        safe_rotations = rotations[:, safe_indices, :]
    else:
        safe_positions = positions[:, :0, :]
        safe_rotations = rotations[:, :0, :]
    moving = 0
    if safe_positions.size:
        ranges = np.nanmax(safe_positions, axis=0) - np.nanmin(safe_positions, axis=0)
        moving = int(np.sum(np.linalg.norm(ranges, axis=1) > 1e-5))
    teleport_risk = "low" if stripped_root == 0 and stripped_world == 0 else ("medium" if safe_names else "high")
    safety = {
        "exported_controller_count": len(safe_names),
        "stripped_world_transform_count": stripped_world,
        "stripped_atom_root_count": stripped_root,
        "coordinate_space_assumption": "Allowed bodypart controller tracks only; Person/root/world tracks stripped.",
        "teleport_risk": teleport_risk,
        "export_safe_for_import": bool(safe_names) and teleport_risk in {"low", "medium"},
        "safe_for_import": bool(safe_names) and teleport_risk in {"low", "medium"},
        "timeline_export_safe_for_animation": bool(safe_names) and moving >= 1,
        "source_world_coords_stripped": bool(stripped_root or stripped_world),
        "exported_as_relative_motion": False,
        "safe_for_generation_template": False,
        "review_export_warning": "This export is for VaM review only. It preserves source controller coordinates and is not a generative reuse template.",
        "stripped_controller_names": [name for idx, name in enumerate(names) if idx not in safe_indices],
    }
    return safe_positions, safe_rotations, safe_names, safety


def _generation_template_block_reasons(row: dict[str, Any], safety: dict[str, Any]) -> list[str]:
    reasons = ["absolute_review_coordinates"]
    guess = row.get("system_semantic_guess", {})
    validity = guess.get("export_pose_validity")
    if validity == "broken_pose":
        reasons.append("broken_pose")
    if validity == "export_unavailable":
        reasons.append("export_unavailable")
    if not safety.get("exported_controller_count"):
        reasons.append("no_safe_bodypart_controllers")
    if float(row.get("duration_seconds") or 0.0) < 4.0:
        reasons.append("too_short")
    if guess.get("low_motion_intro_candidate"):
        reasons.append("low_motion")
    if guess.get("rider_receiver_status") == "likely_receiver_body_response":
        reasons.append("receiver_body_response")
    if guess.get("foot_controller_outlier"):
        reasons.append("foot_controller_outlier")
    if guess.get("hand_controller_outlier"):
        reasons.append("hand_controller_outlier")
    if guess.get("controller_validity_status") in {"invalid", "unknown"}:
        reasons.append(f"controller_validity_{guess.get('controller_validity_status')}")
    return _dedupe(reasons)


def _write_export_unavailable(out_dir: Path, row: dict[str, Any], reason: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Timeline Export Unavailable for {row['review_id']}",
        "",
        f"- Window: `{row.get('window_id')}`",
        f"- Sample: `{row.get('sample_id')}`",
        f"- Scene: `{row.get('source_scene_path') or row.get('source_scene_file')}`",
        f"- Reason: {reason}",
        "",
        "Inspect this example in the original VaM scene/time instead. No substitute motion segment was created.",
    ]
    (out_dir / "export_unavailable.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"attempted": True, "success": False, "validation_status": "unavailable", "warnings": [reason], "timeline_export_path": None}


def _load_npz_window(sample: dict[str, Any], row: dict[str, Any], run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, float, float] | None:
    path = Path(str(sample.get("baked_npz_path") or ""))
    if not path.is_absolute():
        project_root = run_dir.parents[2] if len(run_dir.parents) > 2 else Path.cwd()
        path = project_root / path if str(path).startswith("data") else run_dir / path
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        rotations = np.asarray(data["rotations"], dtype=np.float32)
        times = np.asarray(data["times"], dtype=np.float32)
        names = [str(x) for x in data["controller_names"].tolist()]
    semantic_start = max(0, min(int(row.get("frame_start") or 0), len(times) - 1))
    semantic_end = max(semantic_start + 1, min(int(row.get("frame_end") or len(times)), len(times)))
    fps = float(sample.get("fps") or 60.0)
    pad_frames = int(round(float(row.get("export_context_padding_seconds") or 0.0) * fps))
    start = max(0, semantic_start - pad_frames)
    end = min(len(times), semantic_end + pad_frames)
    rel_times = times[start:end] - times[start]
    export_start = float(times[start]) if len(times) else float(row.get("start_seconds") or 0.0)
    export_end = float(times[end - 1]) if len(times) and end > start else float(row.get("end_seconds") or 0.0)
    return positions[start:end], _normalize_quat_continuity(rotations[start:end]), names, rel_times, export_start, export_end


def _build_timeline_json(review_id: str, duration: float, names: list[str], positions: np.ndarray, rotations: np.ndarray) -> dict[str, Any]:
    fps = 60.0
    frame_times = [idx / fps for idx in range(positions.shape[0])]
    duration = max(float(duration), frame_times[-1] if frame_times else 0.0)
    controllers = []
    for c_idx, name in enumerate(names):
        controller = {
            "Controller": name,
            "TargetsPosition": True,
            "TargetsRotation": True,
            "ControlPosition": True,
            "ControlRotation": True,
        }
        for axis_idx, axis in enumerate(["X", "Y", "Z"]):
            values = [float(v) for v in positions[:, c_idx, axis_idx]]
            controller[axis] = _encode_dense_values(frame_times, values, duration, values[-1] if values else 0.0)
        for axis_idx, axis in enumerate(["RotX", "RotY", "RotZ", "RotW"]):
            values = [float(v) for v in rotations[:, c_idx, axis_idx]]
            controller[axis] = _encode_dense_values(frame_times, values, duration, values[-1] if values else (1.0 if axis == "RotW" else 0.0))
        controllers.append(controller)
    return {
        "SerializeVersion": "283",
        "AtomType": "Person",
        "Clips": [
            {
                "AnimationName": review_id,
                "AnimationLength": duration,
                "BlendDuration": 0,
                "Loop": 0,
                "PreserveLastFrame": 1,
                "LoopSelfBlendDuration": 0,
                "NextAnimationRandomizeWeight": 1,
                "AutoTransitionPrevious": 0,
                "AutoTransitionNext": 0,
                "SyncTransitionTime": 1,
                "SyncTransitionTimeNL": 0,
                "EnsureQuaternionContinuity": 1,
                "AnimationLayer": "Main",
                "Speed": 1,
                "Weight": 1,
                "Uninterruptible": 0,
                "AnimationSegment": "SemanticReview",
                "NextAnimationName": "",
                "NextAnimationTime": duration,
                "Controllers": controllers,
            }
        ],
    }


def _encode_dense_values(frame_times: list[float], values: list[float], duration: float, endpoint_value: float) -> list[str]:
    keys = [TimelineKeyframe(float(t), float(v), LINEAR) for t, v in zip(frame_times, values)]
    if not keys or abs(keys[-1].time - duration) > 1e-6:
        keys.append(TimelineKeyframe(float(duration), float(endpoint_value), LINEAR))
    return encode_keyframe_sequence(keys)


def _validate_arrays(positions: np.ndarray, rotations: np.ndarray) -> dict[str, Any]:
    warnings = []
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        warnings.append("NaN/Inf values found")
    norms = np.linalg.norm(rotations, axis=-1)
    if np.nanmin(norms) < 0.95 or np.nanmax(norms) > 1.05:
        warnings.append("quaternion norms outside broad [0.95, 1.05] tolerance")
    return {"status": "error" if warnings else "ok", "warnings": warnings}


def _validate_timeline_roundtrip(timeline: dict[str, Any], positions: np.ndarray, rotations: np.ndarray) -> dict[str, Any]:
    clip = timeline["Clips"][0]
    version = int(timeline["SerializeVersion"])
    max_pos_error = 0.0
    min_rot_dot = 1.0
    frame_times = np.asarray([idx / 60.0 for idx in range(positions.shape[0])], dtype=np.float32)
    for c_idx, controller in enumerate(clip.get("Controllers", [])):
        for axis_idx, axis in enumerate(["X", "Y", "Z"]):
            values = _decode_values_at(controller[axis], version, frame_times)
            max_pos_error = max(max_pos_error, float(np.max(np.abs(values - positions[:, c_idx, axis_idx]))))
        rot_values = np.stack([_decode_values_at(controller[axis], version, frame_times) for axis in ["RotX", "RotY", "RotZ", "RotW"]], axis=1)
        rot_values = _normalize_quat_continuity(rot_values[:, None, :])[:, 0, :]
        dots = np.abs(np.sum(rot_values * rotations[:, c_idx, :], axis=1))
        min_rot_dot = min(min_rot_dot, float(np.min(dots)))
    warnings = []
    if max_pos_error > 1e-4:
        warnings.append(f"max position error {max_pos_error:.6g} > 1e-4")
    if min_rot_dot < 0.999:
        warnings.append(f"min rotation dot {min_rot_dot:.6g} < 0.999")
    return {"status": "warning" if warnings else "ok", "max_position_abs_error": max_pos_error, "min_rotation_dot": min_rot_dot, "warnings": warnings}


def _decode_values_at(keys: list[str], version: int, frame_times: np.ndarray) -> np.ndarray:
    decoded = decode_keyframe_sequence(keys, version=version)
    by_time = {round(k.time, 6): float(k.value) for k in decoded}
    return np.asarray([by_time.get(round(float(t), 6), decoded[min(i, len(decoded) - 1)].value) for i, t in enumerate(frame_times)], dtype=np.float32)


def _normalize_quat_continuity(rot: np.ndarray) -> np.ndarray:
    out = np.asarray(rot, dtype=np.float32).copy()
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    norms = np.where(norms <= 1e-8, 1.0, norms)
    out = out / norms
    if out.ndim == 3:
        for c_idx in range(out.shape[1]):
            for i in range(1, out.shape[0]):
                if float(np.dot(out[i - 1, c_idx], out[i, c_idx])) < 0:
                    out[i, c_idx] *= -1.0
    else:
        for i in range(1, out.shape[0]):
            if float(np.dot(out[i - 1], out[i])) < 0:
                out[i] *= -1.0
    return out


def _write_import_notes(path: Path, row: dict[str, Any], meta: dict[str, Any]) -> None:
    guess = row["system_semantic_guess"]
    lines = [
        f"# Import Notes for {row['review_id']}",
        "",
        f"- Source scene: `{row.get('source_scene_path')}`",
        f"- Target technical atom: `{row.get('technical_atom_id')}`",
        f"- Original time: {row.get('start_seconds')}s - {row.get('end_seconds')}s",
        f"- Exported duration: {meta['exported_duration_seconds']:.3f}s",
        f"- Timeline JSON: `{path.with_name(row['review_id'] + '.timeline.json')}`",
        "",
        "Use a copy of the source scene. Select the listed technical atom and use AcidBubbles Timeline if available.",
        "",
        "Timeline JSON was exported in project format, but manual VaM import steps may need verification. VaM visual import/playback has not been tested.",
        "",
        "## Expected System Guess",
        "",
        f"- Active rider candidate: {guess.get('active_rider_candidate')}",
        f"- Movement: {', '.join(guess.get('movement_labels', []))}",
        f"- Contact: {', '.join(guess.get('contact_labels', []))}",
        f"- Overall confidence: {guess.get('overall_confidence')}",
        "",
        "Compare the imported segment and/or original scene playback against the semantic guess. Treat the export as a convenience only; original-scene review is the safest default.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_per_item_guess(item_dir: Path, row: dict[str, Any]) -> None:
    item_dir.mkdir(parents=True, exist_ok=True)
    dump_json(item_dir / f"{row['review_id']}_semantic_guess.json", row)
    lines = _item_markdown(row)
    (item_dir / f"{row['review_id']}_semantic_guess.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# VaM Semantic Review 010", "", "Review these 10 examples inside VaM. Machine/weak/silver labels are hints only.", ""]
    for row in rows:
        lines.extend(_item_markdown(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _item_markdown(row: dict[str, Any]) -> list[str]:
    guess = row["system_semantic_guess"]
    evidence = row["evidence"]
    return [
        f"## {row['review_id']} - {row['category']}",
        "",
        f"- Scene: `{row.get('source_scene_path') or row.get('source_scene_file')}`",
        f"- Original time: {row.get('start_seconds')}s - {row.get('end_seconds')}s",
        f"- Technical actor: `{row.get('technical_atom_id')}`",
        f"- Pair actor if any: `{row.get('pair_technical_atom_id')}`",
        f"- Window: `{row.get('window_id')}`",
        f"- Pair window: `{row.get('pair_window_id')}`",
        f"- Timeline export: `{row.get('timeline_export_validation_status')}` {row.get('timeline_export_path') or ''}",
        "",
        "System's semantic guess:",
        f"- active rider: {guess.get('active_rider_candidate')}",
        f"- passive receiver/context: {guess.get('passive_receiver_candidate')}",
        f"- movement: {', '.join(guess.get('movement_labels', []))}",
        f"- contact: {', '.join(guess.get('contact_labels', []))}",
        f"- posture: {', '.join(guess.get('posture_labels', []))}",
        f"- body motion quality: {guess.get('body_motion_quality')}",
        f"- phase: {guess.get('motion_phase_candidate')}",
        f"- rider/receiver status: {guess.get('rider_receiver_status')} (active={guess.get('active_rider_score')}, receiver={guess.get('receiver_body_response_score')})",
        f"- clean Cowgirl rider score v3: {guess.get('clean_cowgirl_rider_score_v3')}",
        f"- grinding subtype score: {guess.get('cowgirl_grinding_score')}",
        f"- reference status: {guess.get('reference_review_status')}",
        f"- domain warnings: {', '.join(guess.get('domain_guard_warnings', [])) or 'none'}",
        f"- confidence: role={guess.get('role_confidence')}, movement={guess.get('movement_confidence')}, contact={guess.get('contact_confidence')}, overall={guess.get('overall_confidence')}",
        "",
        "Why the system thinks this:",
        f"- selected because: {'; '.join(row.get('why_selected', []))}",
        f"- top features: `{_compact(evidence.get('top_features', {}))}`",
        f"- weak hints: `{', '.join(item.get('label', '') for item in evidence.get('weak_labels', [])[:6])}`",
        f"- machine proposals: `{', '.join(item.get('label', '') for item in evidence.get('machine_proposals', [])[:6])}`",
        f"- handmade reference match: `{_compact(evidence.get('handmade_reference_match', {}), 6)}`",
        f"- clean Cowgirl score v2: `{_compact(evidence.get('clean_cowgirl_candidate_score_v2', {}), 8)}`",
        f"- rider/receiver evidence: `{_compact(evidence.get('rider_receiver_discrimination', {}), 8)}`",
        f"- clean Cowgirl score v3: `{_compact(evidence.get('clean_cowgirl_candidate_score_v3', {}), 8)}`",
        "",
        "What to check in VaM:",
        *[f"- {q}" for q in row.get("user_questions", [])],
        "",
    ]


def _write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "category", "source_scene_file", "technical_atom_id", "start_seconds", "end_seconds", "movement_guess", "contact_guess", "timeline_export_validation_status", "timeline_export_path"])
        writer.writeheader()
        for row in rows:
            guess = row["system_semantic_guess"]
            writer.writerow({
                "review_id": row["review_id"],
                "category": row["category"],
                "source_scene_file": row.get("source_scene_file"),
                "technical_atom_id": row.get("technical_atom_id"),
                "start_seconds": row.get("start_seconds"),
                "end_seconds": row.get("end_seconds"),
                "movement_guess": ", ".join(guess.get("movement_labels", [])),
                "contact_guess": ", ".join(guess.get("contact_labels", [])),
                "timeline_export_validation_status": row.get("timeline_export_validation_status"),
                "timeline_export_path": row.get("timeline_export_path"),
            })


def _write_answer_sheet_md(rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Semantic Review 010 Answer Sheet", ""]
    for row in rows:
        lines.extend([
            f"## {row['review_id']}",
            "",
            "- correct / wrong / unclear:",
            "- if wrong, what is wrong:",
            "- did Timeline import work:",
            "- did original scene review work:",
            "- what did you actually see:",
            "- notes:",
            "",
        ])
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_answer_sheet_yaml(rows: list[dict[str, Any]], out: Path) -> None:
    data = {"metadata": {"review_batch": "semantic_review_010", "is_training_label_file": False}, "reviews": {}}
    for row in rows:
        data["reviews"][row["review_id"]] = {
            "user_verdict": "unknown",
            "timeline_import_worked": "unknown",
            "original_scene_review_worked": "unknown",
            "active_rider_correct": "unknown",
            "movement_correct": "unknown",
            "contact_correct": "unknown",
            "timing_correct": "unknown",
            "timeline_export_correct": "unknown",
            "actual_labels": [],
            "false_system_labels": [],
            "trust_for_ml": "unknown",
            "notes": "",
        }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_index_html(rows: list[dict[str, Any]], out: Path) -> None:
    cards = []
    for row in rows:
        guess = row["system_semantic_guess"]
        evidence = html.escape(yaml.safe_dump(row["evidence"], sort_keys=False, allow_unicode=True))
        answer = html.escape(yaml.safe_dump({"reviews": {row["review_id"]: _answer_stub()}}, sort_keys=False, allow_unicode=True))
        export = row.get("timeline_export_path")
        export_rel = f"timeline_segments/{row['review_id']}/{row['review_id']}.timeline.json"
        export_link = f'<a href="{html.escape(export_rel)}">Timeline export</a>' if export else "Unavailable; inspect original scene"
        cards.append(
            f"<section><h2>{row['review_id']} - {html.escape(row['category'])}</h2>"
            f"<p><b>Scene:</b> {html.escape(str(row.get('source_scene_path') or row.get('source_scene_file')))}<br>"
            f"<b>Technical actor:</b> {html.escape(str(row.get('technical_atom_id')))}<br>"
            f"<b>Pair actor:</b> {html.escape(str(row.get('pair_technical_atom_id')))}<br>"
            f"<b>Time:</b> {row.get('start_seconds')}s - {row.get('end_seconds')}s<br>"
            f"<b>Export:</b> {html.escape(str(row.get('timeline_export_validation_status')))} - {export_link}</p>"
            f"<h3>System guess</h3><ul>"
            f"<li>Active rider: {html.escape(str(guess.get('active_rider_candidate')))}</li>"
            f"<li>Movement: {html.escape(', '.join(guess.get('movement_labels', [])))}</li>"
            f"<li>Contact: {html.escape(', '.join(guess.get('contact_labels', [])))}</li>"
            f"<li>Body quality: {html.escape(str(guess.get('body_motion_quality')))}</li>"
            f"<li>Phase: {html.escape(str(guess.get('motion_phase_candidate')))}</li>"
            f"<li>Rider/receiver status: {html.escape(str(guess.get('rider_receiver_status')))} "
            f"(active={guess.get('active_rider_score')}, receiver={guess.get('receiver_body_response_score')})</li>"
            f"<li>Clean Cowgirl rider score v3: {html.escape(str(guess.get('clean_cowgirl_rider_score_v3')))}</li>"
            f"<li>Grinding subtype score: {html.escape(str(guess.get('cowgirl_grinding_score')))}</li>"
            f"<li>Relative Cowgirl score: {html.escape(str(guess.get('relative_cowgirl_score')))}</li>"
            f"<li>Trajectory shape: {html.escape(str(guess.get('trajectory_shape_classification')))} "
            f"(oval={html.escape(str(guess.get('oval_path_score')))}, ellipse={html.escape(str(guess.get('ellipse_fit_score')))}, closed={html.escape(str(guess.get('closed_loop_ratio')))})</li>"
            f"<li>Clean Cowgirl score v4: {html.escape(str(guess.get('clean_cowgirl_score_v4')))}</li>"
            f"<li>Semantic Cowgirl score v5: {html.escape(str(guess.get('semantic_cowgirl_score_v5')))}</li>"
            f"<li>Generation candidate score v5: {html.escape(str(guess.get('generation_candidate_score_v5')))}</li>"
            f"<li>Semantic Cowgirl score v6: {html.escape(str(guess.get('semantic_cowgirl_score_v6')))}</li>"
            f"<li>Generation candidate score v6: {html.escape(str(guess.get('generation_candidate_score_v6')))}</li>"
            f"<li>Controller validity: {html.escape(str(guess.get('controller_validity_status')))} "
            f"(score={html.escape(str(guess.get('controller_validity_score')))}, foot_outlier={html.escape(str(guess.get('foot_controller_outlier')))}, hand_outlier={html.escape(str(guess.get('hand_controller_outlier')))})</li>"
            f"<li>Clean motion score: {html.escape(str(guess.get('clean_motion_score')))}; Cowgirl context score: {html.escape(str(guess.get('cowgirl_context_score')))}</li>"
            f"<li>Export pose validity: {html.escape(str(guess.get('export_pose_validity')))}; generation template safe: {html.escape(str(guess.get('generation_template_safe')))}</li>"
            f"<li>Safe for learning: {html.escape(str(guess.get('safe_for_learning')))}; teleport risk: {html.escape(str(guess.get('teleport_risk')))}</li>"
            f"<li>Reference status: {html.escape(str(guess.get('reference_review_status')))}</li>"
            f"<li>Overall confidence: {guess.get('overall_confidence')}</li></ul>"
            f"<p><b>Hints are not truth.</b> Machine/weak/silver labels must be checked in VaM.</p>"
            f"<details><summary>Evidence</summary><pre>{evidence}</pre></details>"
            f"<h3>What to check in VaM</h3><ul>{''.join('<li>' + html.escape(q) + '</li>' for q in row.get('user_questions', []))}</ul>"
            f"<details open><summary>Answer YAML</summary><pre>{answer}</pre></details></section>"
        )
    html_text = """<!doctype html><meta charset='utf-8'><title>Semantic Review 010</title>
<style>body{font-family:system-ui,Segoe UI,sans-serif;margin:1.5rem;background:#fafafa;color:#202020}section{background:white;border:1px solid #ddd;border-radius:6px;padding:1rem;margin:1rem 0}pre{white-space:pre-wrap;background:#f2f2f2;padding:.6rem;border-radius:4px}</style>
<h1>VaM Semantic Review 010</h1><p>Review inside VaM. Hints are not truth. No ML training is implied.</p>
""" + "\n".join(cards)
    out.write_text(html_text, encoding="utf-8")


def _write_timeline_export_status(rows: list[dict[str, Any]], results: list[dict[str, Any]], out: Path) -> None:
    attempted = sum(1 for r in results if r.get("attempted"))
    success = sum(1 for r in results if r.get("success"))
    unavailable = [r for r in results if not r.get("success")]
    reason_counts = Counter((r.get("warnings") or ["unknown"])[0] for r in unavailable)
    lines = [
        "# Timeline Segment Export Status",
        "",
        f"- Total review items: {len(rows)}",
        f"- Exports attempted: {attempted}",
        f"- Exports successful: {success}",
        f"- Exports unavailable: {len(unavailable)}",
        f"- Exports failed validation: {sum(1 for r in results if r.get('validation_status') not in {'ok', 'unavailable'})}",
        "",
        "## Unavailable Reasons",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in reason_counts.items()) if reason_counts else lines.append("- None")
    lines.extend(["", "## Inspect Original Scene Instead", ""])
    for row in rows:
        if not row.get("has_timeline_export"):
            lines.append(f"- `{row['review_id']}`: `{row.get('source_scene_path') or row.get('source_scene_file')}` at {row.get('start_seconds')}s - {row.get('end_seconds')}s")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_semantic_result(out: Path, rows: list[dict[str, Any]], status: str, counts: dict[str, Counter[str]], false_labels: Counter[str], verdict: dict[str, str]) -> None:
    lines = ["# Semantic Review 010 Result", "", f"- Status: `{status}`", f"- Review items: {len(rows)}", ""]
    if status == "not_completed":
        lines.append("Answers are still unknown. Review is not completed yet.")
    lines.extend(["", "## Counts", ""])
    for field, counter in counts.items():
        lines.append(f"### {field}")
        lines.extend(f"- `{key}`: {value}" for key, value in counter.most_common())
        lines.append("")
    lines.extend(["## Common False Labels", ""])
    lines.extend(f"- `{label}`: {count}" for label, count in false_labels.most_common()) if false_labels else lines.append("- None yet")
    labels = Counter()
    try:
        answer_data = yaml.safe_load(out.with_name("semantic_review_010_answer_sheet.yaml").read_text(encoding="utf-8")) or {}
        for item in (answer_data.get("reviews") or {}).values():
            labels.update(str(x) for x in item.get("actual_labels", []) or [])
    except Exception:
        labels = Counter()
    if labels:
        cowgirl_true = labels.get("cowgirl_true_segment", 0)
        possible_cowgirl = labels.get("possible_cowgirl_context", 0)
        transition = labels.get("transition_adjustment", 0)
        root_false = labels.get("controller_only_whole_person_motion", 0) + labels.get("root_only_motion_false_positive", 0)
        receiver_response = labels.get("receiver_body_response", 0) + labels.get("passive_receiver_motion", 0)
        broken_pose = labels.get("pose_broken", 0) + labels.get("export_pose_validity_issue", 0)
        clean_motion = labels.get("clean_cowgirl_motion", 0)
        low_intro = labels.get("low_motion_intro", 0) + labels.get("cowgirl_intro_or_start_pose", 0)
        export_unavailable = labels.get("export_unavailable", 0)
        wrong_or_unclear = counts.get("user_verdict", Counter()).get("wrong", 0) + counts.get("user_verdict", Counter()).get("unclear", 0)
        correct = counts.get("user_verdict", Counter()).get("correct", 0) + counts.get("user_verdict", Counter()).get("correct_or_unclear", 0) + counts.get("user_verdict", Counter()).get("partially_correct", 0)
        if (cowgirl_true <= 1 and len(rows) >= 10) or wrong_or_unclear > correct:
            verdict.update(
                {
                    "feature_semantics_trusted": "no" if cowgirl_true <= 1 else "uncertain",
                    "machine_labels_trusted_for_proxy_ml": "no",
                    "proceed_to_ml": "no",
                }
            )
        elif cowgirl_true >= 3:
            verdict.update(
                {
                    "feature_semantics_trusted": "uncertain_but_improving",
                    "machine_labels_trusted_for_proxy_ml": "no",
                    "proceed_to_ml": "no",
                }
            )
        lines.extend(
            [
                "## Human Review Interpretation",
                "",
                f"- Clear true Cowgirl positives: {cowgirl_true}/{len(rows)}",
                f"- Clean Cowgirl motion labels: {clean_motion}",
                f"- Possible Cowgirl context/ambiguous examples: {possible_cowgirl}",
                f"- Transition/adjustment/in-between examples: {transition}",
                f"- Whole-person/controller/root false positives: {root_false}",
                f"- Receiver/body-response false-positive audit labels: {receiver_response}",
                f"- Broken pose/export validity issues: {broken_pose}",
                f"- Cowgirl context but low-motion/intro examples: {low_intro}",
                f"- Export unavailable examples: {export_unavailable}",
                "- Machine/silver labels are still not training truth and ML remains stopped.",
                "- Current focus: separate semantic correctness from export/pose/generation-template usability.",
                "",
            ]
        )
    lines.extend(["", "## Verdict", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in verdict.items())
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _answer_stub() -> dict[str, Any]:
    return {
        "user_verdict": "unknown",
        "timeline_import_worked": "unknown",
        "original_scene_review_worked": "unknown",
        "active_rider_correct": "unknown",
        "movement_correct": "unknown",
        "contact_correct": "unknown",
        "timing_correct": "unknown",
        "timeline_export_correct": "unknown",
        "actual_labels": [],
        "false_system_labels": [],
        "trust_for_ml": "unknown",
        "notes": "",
    }


def _questions_for_item(has_pair: bool) -> list[str]:
    base = [
        "Is this actually the active rider or relevant actor?",
        "Is the movement label plausible?",
        "Is the original time window correct?",
        "Does the movement match the feature evidence?",
    ]
    if has_pair:
        base.extend([
            "Is the pair context plausible?",
            "Is the active/passive candidate plausible?",
            "Does the hand/contact proxy look plausible?",
        ])
    base.append("Should this item be trusted for later ML work?")
    return base


def _pair_by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        for key in ["window_id_a", "window_id_b"]:
            wid = row.get(key)
            if wid and wid not in out:
                out[wid] = row
    return out


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out = defaultdict(list)
    for row in rows:
        if row.get(key):
            out[str(row[key])].append(row)
    return out


def _preferred_pair_window_id(pair: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    score_window_ids = [wid for row in rows for wid in (row.get("window_ids") or [])]
    for wid in score_window_ids:
        if wid == pair.get("window_id_a"):
            return wid
    return pair.get("window_id_a") or (score_window_ids[0] if score_window_ids else "")


def _pair_actor(wid: str, pair: dict[str, Any]) -> str | None:
    if not pair:
        return None
    if pair.get("window_id_a") == wid:
        return pair.get("technical_atom_id_b")
    if pair.get("window_id_b") == wid:
        return pair.get("technical_atom_id_a")
    return None


def _sample_for_window(wid: str, data: dict[str, Any]) -> dict[str, Any] | None:
    wrow = data["windows"].get(wid, {})
    return data["samples"].get(wrow.get("sample_id"))


def _get_window_scene(wid: str, item: dict[str, Any], wrow: dict[str, Any] | None = None) -> str:
    return str((wrow or {}).get("source_scene_file") or item.get("source_scene_file") or wid.split("_")[0])


def _get_window_sample(wid: str, item: dict[str, Any], wrow: dict[str, Any] | None = None) -> str:
    return str((wrow or {}).get("sample_id") or item.get("sample_id") or wid)


def _top_features(values: dict[str, Any], limit: int = 10) -> dict[str, float]:
    out = []
    for key, value in values.items():
        val = _num(value)
        if val == val and not math.isinf(val):
            out.append((key, abs(val), round(float(val), 6)))
    return {key: val for key, _, val in sorted(out, key=lambda x: x[1], reverse=True)[:limit]}


def _weak_labels(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"label": x.get("label"), "confidence": x.get("confidence"), "is_human_ground_truth": False} for x in row.get("weak_labels", [])[:8]]


def _score_hints(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    hints = []
    seen = set()
    for row in sorted(rows, key=lambda r: float(r.get("final_score") or 0.0), reverse=True):
        label = row.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        hints.append({
            "label": label,
            "score": row.get("final_score"),
            "status": row.get("recommended_status"),
            "conflict_flags": row.get("conflict_flags", []),
            "is_human_ground_truth": False,
        })
        if len(hints) >= limit:
            break
    return hints


def _silver_hint(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "positive_labels": row.get("positive_labels", []),
        "review_only_labels": row.get("review_only_labels", []),
        "scores_by_label": row.get("scores_by_label", {}),
        "label_source": row.get("label_source", "silver_machine_v2"),
        "is_human_ground_truth": False,
    }


def _pair_summary(pair: dict[str, Any], feature: dict[str, Any], scores: list[dict[str, Any]], silver: dict[str, Any]) -> dict[str, Any]:
    if not pair:
        return {}
    return {
        "pair_window_id": pair.get("pair_window_id"),
        "pair_actor_a": pair.get("technical_atom_id_a"),
        "pair_actor_b": pair.get("technical_atom_id_b"),
        "pairing_reasons": pair.get("pairing_reasons", []),
        "feature_quality": feature.get("feature_quality", {}),
        "top_pair_features": _top_features(feature.get("feature_values", {}), 8),
        "machine_pair_proposals": _score_hints(scores, 8),
        "silver_pair_labels": _silver_hint(silver),
        "warning": "Pair features are context proxies only, not semantic truth.",
    }


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compact(data: dict[str, Any], limit: int = 4) -> str:
    return ", ".join(f"{k}={v}" for k, v in list(data.items())[:limit])


def _semantic_verdict(*counters: Counter[str]) -> str:
    yes = sum(c.get("correct", 0) + c.get("correct_or_unclear", 0) + c.get("partially_correct", 0) + c.get("true", 0) + c.get("yes", 0) for c in counters)
    no = sum(c.get("wrong", 0) + c.get("false", 0) + c.get("no", 0) for c in counters)
    unclear = sum(c.get("unclear", 0) + c.get("unknown", 0) for c in counters)
    if no > max(1, yes * 0.25):
        return "no"
    if yes >= max(3, no * 3) and yes > unclear:
        return "yes"
    return "uncertain"

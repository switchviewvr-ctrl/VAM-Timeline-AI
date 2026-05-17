"""Focused review planner for the clean_v3 new-scene delta run."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import html
import json

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import build_static_review_ui


TARGETS = {
    "possible_cowgirl_lean_back_supported_review": 8,
    "cowgirl_lean_back_supported_clean_motion": 5,
    "cowgirl_lean_back_supported_pose_context": 5,
    "cowgirl_hands_behind_support": 5,
    "cowgirl_hands_on_partner_legs_or_thighs": 5,
    "cowgirl_clean_motion_generation_safe": 20,
    "cowgirl_clean_motion_low_confidence_short": 15,
    "cowgirl_pose_context_low_motion": 15,
    "cowgirl_transition_setup": 15,
    "not_cowgirl_standing_hand_head": 15,
    "not_cowgirl_bj_oral": 10,
    "unknown_or_unusable_high_movement": 10,
    "contact_support_ambiguous": 10,
}

STRICT_COWGIRL_TARGETS = {
    "strict_cowgirl_clean_motion": 30,
    "strict_cowgirl_short_motion": 6,
    "strict_cowgirl_lean_back_check": 4,
}

FOREIGN_OR_SETUP_CLIP_TOKENS = {
    "intro",
    "mount",
    "switch",
    "insert",
    "dismount",
    "finish",
    "finished",
    "cum",
    "cums",
    "walking",
    "walk",
    "standing",
    "stand",
    "transition",
    "trans",
    "backdoor",
    "pullout",
    "reset",
    "start",
    "pose to",
    "to pose",
    "bj",
    "oral",
    "blow",
    "suck",
}

REVIEW_LABELS = [
    "correct_clean_cowgirl_motion",
    "correct_short_cowgirl_motion",
    "cowgirl_pose_only_low_motion",
    "cowgirl_transition_intro_alignment",
    "standing_hand_head_not_cowgirl",
    "bj_oral_not_cowgirl",
    "receiver_response_not_rider_motion",
    "wrong_partner_context",
    "wrong_contact_support",
    "correct_lean_back_supported_cowgirl",
    "hands_behind_support_correct",
    "hands_on_partner_legs_or_thighs_correct",
    "front_cowgirl_not_reverse",
    "wrongly_marked_reverse_cowgirl",
    "broken_pose_or_bad_data",
    "unknown_unclear",
]

REVIEW_QUESTIONS = [
    "Write one short free-text note about what is correct or wrong.",
]

DIAGNOSTIC_FIELDS = [
    "source_scene_file",
    "source_id",
    "technical_atom_id",
    "clip_name",
    "start_seconds",
    "end_seconds",
    "semantic_family",
    "cowgirl_bucket",
    "pose_family",
    "pose_subtype",
    "motion_subtype",
    "motion_metrics",
    "hip_motion_strength",
    "pelvis_trajectory_strength",
    "axis_breakdown",
    "partner_context_confidence",
    "contact_support",
    "torso_lean_direction",
    "facing_context",
    "support_context",
    "hands_behind_support_score",
    "hands_on_partner_legs_score",
    "hands_on_partner_thighs_score",
    "why_selected",
    "likely_failure_mode",
]


def build_focused_new_scenes_review(
    run_dir: str | Path,
    previous_review: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    previous = Path(previous_review).resolve()
    out = Path(out_dir).resolve()
    _ensure_inside(run, out)
    out.mkdir(parents=True, exist_ok=True)

    context = _load_context(run)
    cowgirl = _load_first(run / "datasets" / "cowgirl_candidate_db_v8.jsonl", run / "datasets" / "cowgirl_candidate_db_v7.jsonl")
    semantic = _load_first(run / "datasets" / "semantic_candidate_db_v3.jsonl", run / "datasets" / "semantic_candidate_db_v2.jsonl")
    selected, selection = select_focused_review_cards(cowgirl, semantic, context, TARGETS)
    cards = [_review_card(idx, row, context) for idx, row in enumerate(selected, start=1)]
    write_jsonl(out / "semantic_review_010.jsonl", cards)
    write_jsonl(out / "focused_review_manifest.jsonl", cards)
    _write_manifest_csv(cards, out / "focused_review_manifest.csv")
    _write_answer_sheet(cards, out / "semantic_review_010_answer_sheet.yaml")
    _write_review_md(cards, out / "semantic_review_010.md")
    _write_review_html(cards, out / "semantic_review_010_index.html")
    _write_selection_report(cards, selection, out / "focused_review_selection_report.md")
    _write_codex_prompt_template(out / "codex_summary_prompt_after_review.md")
    package = build_vam_review_package(out / "semantic_review_010.jsonl", run, run, out / "vam_review_package", attempt_timeline_segments=True)
    static_ui = build_static_review_ui(run, out, out / "review_ui_static")
    summary = {
        "status": "ok",
        "run_dir": str(run),
        "previous_review": str(previous),
        "out_dir": str(out),
        "target_counts": TARGETS,
        "selected": len(cards),
        "selected_counts": dict(Counter(card["why_selected"] for card in cards)),
        "scene_counts": dict(Counter(card.get("source_scene_file") for card in cards)),
        "sample_count": len({card.get("sample_id") for card in cards if card.get("sample_id")}),
        "source_id_count": len({card.get("source_id") for card in cards if card.get("source_id")}),
        "selection": selection,
        "vam_review_package": package,
        "static_review_ui": static_ui,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    dump_json(out / "focused_review_summary.json", summary)
    _write_summary(out / "focused_review_summary.md", summary)
    return summary


def build_strict_new_scenes_cowgirl_review(
    run_dir: str | Path,
    out_dir: str | Path,
    previous_review: str | Path | None = None,
    human_answers: str | Path | None = None,
    batch_size: int | None = None,
    batch_index: int = 1,
) -> dict[str, Any]:
    """Build a smaller strict Cowgirl review after the broad batch proved noisy."""

    run = Path(run_dir).resolve()
    out = Path(out_dir).resolve()
    _ensure_inside(run, out)
    out.mkdir(parents=True, exist_ok=True)

    context = _load_context(run)
    cowgirl = _load_first(run / "datasets" / "cowgirl_candidate_db_v8.jsonl", run / "datasets" / "cowgirl_candidate_db_v7.jsonl")
    previous_cards = _load_previous_review_cards(previous_review)
    answer_notes = _load_human_answer_notes(human_answers)
    human_exclusions = _human_exclusion_windows(previous_cards, answer_notes)
    human_positive_lean_back = _human_positive_lean_back_rows(previous_cards, answer_notes)
    selected, selection = select_strict_cowgirl_review_cards(cowgirl + human_positive_lean_back, context, human_exclusions)
    full_selected_count = len(selected)
    if batch_size:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_index <= 0:
            raise ValueError("batch_index must be positive")
        start = (batch_index - 1) * batch_size
        selected = selected[start : start + batch_size]
        selection = {
            **selection,
            "batch_size": batch_size,
            "batch_index": batch_index,
            "full_selected_count": full_selected_count,
            "batch_start_1_based": start + 1 if selected else None,
            "batch_end_1_based": start + len(selected) if selected else None,
        }
    cards = [_review_card(idx, row, context) for idx, row in enumerate(selected, start=1)]
    write_jsonl(out / "semantic_review_010.jsonl", cards)
    write_jsonl(out / "strict_cowgirl_review_manifest.jsonl", cards)
    write_jsonl(out / "focused_review_manifest.jsonl", cards)
    _write_manifest_csv(cards, out / "strict_cowgirl_review_manifest.csv")
    _write_manifest_csv(cards, out / "focused_review_manifest.csv")
    _write_answer_sheet(cards, out / "semantic_review_010_answer_sheet.yaml")
    _write_review_md(cards, out / "semantic_review_010.md")
    _write_review_html(cards, out / "semantic_review_010_index.html")
    _write_selection_report(cards, selection, out / "strict_cowgirl_selection_report.md", STRICT_COWGIRL_TARGETS)
    _write_human_answer_summary(answer_notes, out / "previous_human_answer_summary.md")
    _write_codex_prompt_template(out / "codex_summary_prompt_after_review.md")
    package = build_vam_review_package(out / "semantic_review_010.jsonl", run, run, out / "vam_review_package", attempt_timeline_segments=True)
    static_ui = build_static_review_ui(run, out, out / "review_ui_static")
    summary = {
        "status": "ok",
        "run_dir": str(run),
        "out_dir": str(out),
        "previous_review": str(previous_review) if previous_review else None,
        "human_answers": str(human_answers) if human_answers else None,
        "target_counts": STRICT_COWGIRL_TARGETS,
        "batch_size": batch_size,
        "batch_index": batch_index if batch_size else None,
        "full_selected_count": full_selected_count,
        "selected": len(cards),
        "selected_counts": dict(Counter(card["why_selected"] for card in cards)),
        "scene_counts": dict(Counter(card.get("source_scene_file") for card in cards)),
        "human_answer_count": len(answer_notes),
        "human_exclusion_windows": len(human_exclusions),
        "human_positive_lean_back_candidates": len(human_positive_lean_back),
        "selection": selection,
        "vam_review_package": package,
        "static_review_ui": static_ui,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    dump_json(out / "strict_cowgirl_review_summary.json", summary)
    _write_summary(out / "strict_cowgirl_review_summary.md", summary)
    return summary


def select_focused_review_cards(
    cowgirl: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    context: dict[str, Any],
    targets: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = targets or TARGETS
    pools = _build_pools(cowgirl, semantic, context)
    selected: list[dict[str, Any]] = []
    state = {
        "windows": set(),
        "samples": set(),
        "source_ids": set(),
        "near": set(),
        "scene_counts": Counter(),
        "motion_counts": Counter(),
        "contact_counts": Counter(),
        "rejected": Counter(),
    }
    for bucket, target in targets.items():
        _add_bucket(selected, pools.get(bucket, []), bucket, target, context, state, scene_cap=8, relax_scene_cap=False)
        missing = target - sum(1 for row in selected if row.get("_focused_bucket") == bucket)
        if missing > 0:
            _add_bucket(selected, pools.get(bucket, []), bucket, missing, context, state, scene_cap=999, relax_scene_cap=True)
    selection = {
        "target_counts": targets,
        "selected_counts": dict(Counter(row.get("_focused_bucket") for row in selected)),
        "selected": len(selected),
        "rejected_by_rule": dict(state["rejected"]),
        "scene_counts": dict(state["scene_counts"]),
        "motion_shape_counts": dict(state["motion_counts"]),
        "contact_counts": dict(state["contact_counts"]),
    }
    return selected, selection


def select_strict_cowgirl_review_cards(
    cowgirl: list[dict[str, Any]],
    context: dict[str, Any],
    human_exclusions: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    human_exclusions = human_exclusions or set()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for row in cowgirl:
        reason = _strict_reject_reason(row, context, human_exclusions)
        if reason:
            rejected[reason] += 1
            continue
        category = str(row.get("category") or "")
        bucket = "strict_cowgirl_clean_motion"
        if category == "cowgirl_clean_motion_low_confidence_short" or row.get("clean_motion_gate") == "soft_pass_short":
            bucket = "strict_cowgirl_short_motion"
        if _strict_lean_back_candidate(row, context):
            pools["strict_cowgirl_lean_back_check"].append(dict(row))
        pools[bucket].append(dict(row))

    for bucket, rows in pools.items():
        rows.sort(key=lambda r: _strict_sort_key(bucket, r, context))

    selected: list[dict[str, Any]] = []
    state = {
        "windows": set(),
        "samples": set(),
        "source_ids": set(),
        "near": set(),
        "scene_counts": Counter(),
        "motion_counts": Counter(),
        "contact_counts": Counter(),
        "rejected": rejected,
    }
    for bucket, target in STRICT_COWGIRL_TARGETS.items():
        _add_bucket(selected, pools.get(bucket, []), bucket, target, context, state, scene_cap=3, relax_scene_cap=False)
        missing = target - sum(1 for row in selected if row.get("_focused_bucket") == bucket)
        if missing:
            _add_bucket(selected, pools.get(bucket, []), bucket, missing, context, state, scene_cap=5, relax_scene_cap=False)
        missing = target - sum(1 for row in selected if row.get("_focused_bucket") == bucket)
        if missing:
            _add_bucket(selected, pools.get(bucket, []), bucket, missing, context, state, scene_cap=999, relax_scene_cap=True)

    selection = {
        "target_counts": STRICT_COWGIRL_TARGETS,
        "selected_counts": dict(Counter(row.get("_focused_bucket") for row in selected)),
        "selected": len(selected),
        "rejected_by_rule": dict(state["rejected"]),
        "scene_counts": dict(state["scene_counts"]),
        "motion_shape_counts": dict(state["motion_counts"]),
        "contact_counts": dict(state["contact_counts"]),
        "filter_policy": {
            "allowed_categories": [
                "cowgirl_clean_motion_generation_safe",
                "cowgirl_clean_motion_low_confidence_short",
                "cowgirl_lean_back_supported_clean_motion",
            ],
            "excluded_clip_tokens": sorted(FOREIGN_OR_SETUP_CLIP_TOKENS),
            "human_answer_exclusions": len(human_exclusions),
        },
    }
    return selected, selection


def _build_pools(cowgirl: list[dict[str, Any]], semantic: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cowgirl:
        category = str(row.get("category") or "unknown")
        if category in TARGETS:
            pools[category].append(dict(row))
        if category in {"cowgirl_hands_on_partner_legs_or_thighs", "cowgirl_hands_behind_support"}:
            pools["contact_support_ambiguous"].append(dict(row))
        if _possible_lean_back_review(row, context):
            rec = dict(row)
            rec["category"] = "possible_cowgirl_lean_back_supported_review"
            rec["pose_subtype"] = rec.get("pose_subtype") or "possible_cowgirl_lean_back_supported"
            rec["likely_review_focus"] = "front_lean_back_vs_reverse_cowgirl"
            pools["possible_cowgirl_lean_back_supported_review"].append(rec)
        if category == "unknown_or_unusable" and _num(row.get("motion_score")) >= 0.35:
            pools["unknown_or_unusable_high_movement"].append(dict(row))
        if row.get("contact_support") == "ambiguous_partner_contact" or row.get("contact_support_ambiguous"):
            pools["contact_support_ambiguous"].append(dict(row))
    seen_unknown = {r.get("window_id") for r in pools["unknown_or_unusable_high_movement"]}
    seen_contact = {r.get("window_id") for r in pools["contact_support_ambiguous"]}
    for row in semantic:
        if row.get("semantic_family") == "unknown" and _num(row.get("motion_score")) >= 0.35 and row.get("window_id") not in seen_unknown:
            rec = dict(row)
            rec["category"] = "unknown_or_unusable"
            pools["unknown_or_unusable_high_movement"].append(rec)
        if (row.get("contact_support") == "ambiguous_partner_contact" or row.get("contact_support_ambiguous")) and row.get("window_id") not in seen_contact:
            rec = dict(row)
            rec["category"] = row.get("category") or row.get("semantic_family") or "contact_support_ambiguous"
            pools["contact_support_ambiguous"].append(rec)
        if _possible_lean_back_review(row, context):
            rec = dict(row)
            rec["category"] = "possible_cowgirl_lean_back_supported_review"
            rec["pose_subtype"] = rec.get("pose_subtype") or "possible_cowgirl_lean_back_supported"
            rec["likely_review_focus"] = "front_lean_back_vs_reverse_cowgirl"
            pools["possible_cowgirl_lean_back_supported_review"].append(rec)
    for bucket, rows in pools.items():
        rows.sort(key=lambda r: _sort_key(bucket, r))
    return pools


def _strict_reject_reason(row: dict[str, Any], context: dict[str, Any], human_exclusions: set[str]) -> str | None:
    wid = str(row.get("window_id") or "")
    if wid in human_exclusions:
        return "rejected_by_human_notes_from_previous_batch"
    window = context["windows"].get(wid, {})
    sample = str(row.get("sample_id") or window.get("sample_id") or "")
    source_id = str(window.get("source_id") or context["samples"].get(sample, {}).get("source_id") or "")
    if sample and f"sample:{sample}" in human_exclusions:
        return "rejected_by_human_notes_from_previous_batch"
    if source_id and f"source:{source_id}" in human_exclusions:
        return "rejected_by_human_notes_from_previous_batch"
    category = str(row.get("category") or "")
    if category not in {
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_low_confidence_short",
        "cowgirl_lean_back_supported_clean_motion",
    }:
        return "not_strict_cowgirl_motion_category"
    if row.get("semantic_family") != "cowgirl":
        return "not_cowgirl_semantic_family"
    if row.get("pose_family") != "cowgirl":
        return "not_cowgirl_pose_family"
    if str(row.get("motion_subtype") or "unknown") == "unknown":
        return "unknown_motion_subtype"
    if str(row.get("phase") or "") not in {"clean_motion", ""}:
        return "not_clean_motion_phase"
    gate = str(row.get("clean_motion_gate") or "")
    if gate not in {"pass", "soft_pass_short"}:
        return "clean_motion_gate_not_pass"
    if _num(row.get("hip_motion_strength")) < 0.75:
        return "weak_hip_motion_strength"
    if _num(row.get("pelvis_trajectory_strength")) < 0.70:
        return "weak_pelvis_trajectory_strength"
    token = _foreign_clip_token(row, context)
    if token and not row.get("_human_positive_lean_back_hint"):
        return f"excluded_clip_token_{token}"
    return None


def _strict_lean_back_candidate(row: dict[str, Any], context: dict[str, Any]) -> bool:
    text = _row_context_text(row, context)
    return (
        row.get("pose_subtype") == "cowgirl_lean_back_supported"
        or row.get("torso_lean_direction") == "backward"
        or row.get("contact_support") in {"hands_behind_support", "hands_on_partner_legs_or_thighs"}
        or any(token in text for token in ("lean back", "lean_back", "hand back", "hands behind"))
    )


def _strict_sort_key(bucket: str, row: dict[str, Any], context: dict[str, Any]) -> tuple[Any, ...]:
    text = _row_context_text(row, context)
    lean_bonus = 1 if _strict_lean_back_candidate(row, context) else 0
    normal_clip_bonus = 1 if any(token in text for token in ("hard", "veryhard", "grind", "loop", "anim", "slow", "reit")) else 0
    return (
        -lean_bonus if bucket == "strict_cowgirl_lean_back_check" else 0,
        -normal_clip_bonus,
        -_num(row.get("semantic_score")),
        -_num(row.get("hip_motion_strength")),
        -_num(row.get("pelvis_trajectory_strength")),
        -_num(row.get("motion_score")),
        str(row.get("source_scene_file") or ""),
    )


def _foreign_clip_token(row: dict[str, Any], context: dict[str, Any]) -> str | None:
    text = _row_context_text(row, context)
    # "reverse" is intentionally not a rejection token: the user confirmed at least
    # one clip named Reverse Cowgirl is visually normal/front Cowgirl.
    for token in sorted(FOREIGN_OR_SETUP_CLIP_TOKENS, key=len, reverse=True):
        if token in text:
            return token.replace(" ", "_")
    return None


def _sort_key(bucket: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if bucket == "possible_cowgirl_lean_back_supported_review":
        return (-_num(row.get("semantic_score")), -_num(row.get("hip_motion_strength")), str(row.get("source_scene_file")))
    if bucket == "unknown_or_unusable_high_movement":
        return (-_num(row.get("motion_score")), -_num(row.get("hip_motion_strength")), str(row.get("source_scene_file")))
    if bucket == "contact_support_ambiguous":
        return (-_num(row.get("contact_support_confidence")), _num(row.get("contact_support_margin")), str(row.get("source_scene_file")))
    return (-_num(row.get("semantic_score")), -_num(row.get("hip_motion_strength")), -_num(row.get("motion_score")), str(row.get("source_scene_file")))


def _add_bucket(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    bucket: str,
    target: int,
    context: dict[str, Any],
    state: dict[str, Any],
    scene_cap: int,
    relax_scene_cap: bool,
) -> None:
    added = 0
    for row in _diverse_order(candidates, context):
        if added >= target:
            return
        if _try_add(row, bucket, context, state, scene_cap, relax_scene_cap):
            selected.append(row)
            added += 1


def _try_add(
    row: dict[str, Any],
    bucket: str,
    context: dict[str, Any],
    state: dict[str, Any],
    scene_cap: int,
    relax_scene_cap: bool,
) -> bool:
    wid = str(row.get("window_id") or "")
    if not wid or wid in state["windows"]:
        state["rejected"]["same_window"] += 1
        return False
    window = context["windows"].get(wid, {})
    sample = str(row.get("sample_id") or window.get("sample_id") or "")
    source_id = str(window.get("source_id") or context["samples"].get(sample, {}).get("source_id") or "")
    scene = str(row.get("source_scene_file") or window.get("source_scene_file") or "unknown")
    if sample and sample in state["samples"]:
        state["rejected"]["sample_cap"] += 1
        return False
    if source_id and source_id in state["source_ids"]:
        state["rejected"]["source_cap"] += 1
        return False
    if not relax_scene_cap and state["scene_counts"][scene] >= scene_cap:
        state["rejected"]["scene_cap"] += 1
        return False
    near = _near_duplicate_group(row, window, bucket)
    if near in state["near"]:
        state["rejected"]["near_duplicate"] += 1
        return False
    row["_focused_bucket"] = bucket
    state["windows"].add(wid)
    if sample:
        state["samples"].add(sample)
    if source_id:
        state["source_ids"].add(source_id)
    state["scene_counts"][scene] += 1
    state["motion_counts"][str(row.get("motion_subtype") or "unknown")] += 1
    state["contact_counts"][str(row.get("contact_support") or "unknown")] += 1
    state["near"].add(near)
    return True


def _diverse_order(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        window = context["windows"].get(row.get("window_id"), {})
        key = (
            str(row.get("source_scene_file") or window.get("source_scene_file") or "unknown"),
            str(row.get("motion_subtype") or "unknown"),
            str(row.get("contact_support") or "unknown"),
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    while groups:
        for key in sorted(list(groups), key=lambda k: (len(groups[k]), k), reverse=True):
            out.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    return out


def _review_card(idx: int, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    window = context["windows"].get(row.get("window_id"), {})
    sample = context["samples"].get(row.get("sample_id") or window.get("sample_id"), {})
    source = context["sources"].get(window.get("source_id") or sample.get("source_id"), {})
    rel = context["relative"].get(row.get("window_id"), {})
    fv = rel.get("feature_values") or {}
    bucket = row.get("_focused_bucket") or row.get("category") or "unknown"
    simple_label = _simple_review_label(idx, bucket)
    start = _first(row.get("start_seconds"), window.get("start_seconds"))
    end = _first(row.get("end_seconds"), window.get("end_seconds"))
    card = {
        "review_id": f"review_{idx:03d}",
        "review_label": simple_label,
        "semantic_review_label": simple_label,
        "window_id": row.get("window_id"),
        "pair_window_id": row.get("pair_window_id"),
        "semantic_family": row.get("semantic_family") or "unknown",
        "cowgirl_bucket": row.get("category") or bucket,
        "category": row.get("category") or bucket,
        "pose_semantics": {"family": row.get("pose_family"), "subtype": row.get("pose_subtype")},
        "pose_family": row.get("pose_family"),
        "pose_subtype": row.get("pose_subtype"),
        "motion_semantics": {"subtype": row.get("motion_subtype"), "phase": row.get("phase")},
        "motion_subtype": row.get("motion_subtype"),
        "phase": row.get("phase"),
        "partner_relation": row.get("partner_relation") or ["unknown"],
        "contact_support": row.get("contact_support") or "unknown",
        "torso_lean_direction": row.get("torso_lean_direction"),
        "facing_context": row.get("facing_context"),
        "support_context": row.get("support_context"),
        "hands_behind_support_score": row.get("hands_behind_support_score"),
        "hands_on_partner_legs_score": row.get("hands_on_partner_legs_score"),
        "hands_on_partner_thighs_score": row.get("hands_on_partner_thighs_score"),
        "partner_leg_support_confidence": row.get("partner_leg_support_confidence"),
        "facing_confidence": row.get("facing_confidence"),
        "interaction_family": row.get("interaction_family") or ("cowgirl" if row.get("semantic_family") == "cowgirl" else row.get("semantic_family")),
        "generation_safe": bool(row.get("generation_safe")),
        "why_selected": bucket,
        "likely_failure_mode": _likely_failure_mode(bucket, row),
        "source_scene_file": row.get("source_scene_file") or window.get("source_scene_file"),
        "source_scene_path": row.get("source_scene_path") or window.get("source_scene_path"),
        "source_id": window.get("source_id") or sample.get("source_id") or source.get("source_id"),
        "sample_id": row.get("sample_id") or window.get("sample_id"),
        "technical_atom_id": row.get("technical_actor_id") or window.get("technical_atom_id") or sample.get("technical_atom_id"),
        "storable_id": source.get("storable_id") or sample.get("storable_id"),
        "plugin_id": source.get("plugin_id"),
        "clip_name": source.get("clip_name") or sample.get("clip_name"),
        "clip_index": source.get("clip_index") if source.get("clip_index") is not None else sample.get("clip_index"),
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": _first(row.get("duration_seconds"), window.get("duration_seconds"), _duration(start, end)),
        "motion_metrics": {
            "motion_score": row.get("motion_score"),
            "motion_content_strength": row.get("motion_content_strength"),
            "clean_motion_score": row.get("clean_motion_score"),
            "low_motion_hold_score": row.get("low_motion_hold_score"),
            "intro_alignment_score": row.get("intro_alignment_score"),
            "local_path_length": fv.get("local_path_length"),
            "local_motion_energy": fv.get("local_motion_energy"),
            "local_velocity_mean": fv.get("local_velocity_mean"),
        },
        "hip_motion_strength": row.get("hip_motion_strength"),
        "pelvis_trajectory_strength": row.get("pelvis_trajectory_strength"),
        "axis_breakdown": {
            "vertical": fv.get("relative_pelvis_vertical_amplitude"),
            "forward_back": fv.get("relative_pelvis_forward_back_amplitude"),
            "lateral": fv.get("relative_pelvis_lateral_amplitude"),
        },
        "vertical_movement": fv.get("relative_pelvis_vertical_amplitude"),
        "forward_back_movement": fv.get("relative_pelvis_forward_back_amplitude"),
        "lateral_movement": fv.get("relative_pelvis_lateral_amplitude"),
        "partner_context_confidence": row.get("partner_context_confidence"),
        "contact_support_confidence": row.get("contact_support_confidence"),
        "contact_support_margin": row.get("contact_support_margin"),
        "contact_support_ambiguous": row.get("contact_support_ambiguous"),
        "best_contact_target": row.get("best_contact_target") or row.get("hand_contact_target"),
        "second_best_contact_target": row.get("second_best_contact_target"),
        "clean_motion_gate": row.get("clean_motion_gate"),
        "clean_motion_gate_reason": row.get("clean_motion_gate_reason"),
        "semantic_score": row.get("semantic_score"),
        "pose_score": row.get("pose_score"),
        "motion_score": row.get("motion_score"),
        "interaction_score": row.get("interaction_score"),
        "rider_above_partner_score": None,
        "pelvis_alignment_score": row.get("partner_relative_alignment_score") or row.get("interaction_score"),
        "hands_on_partner_chest_score": row.get("hands_on_partner_chest_score"),
        "hands_on_partner_hips_score": None,
        "hands_on_partner_legs_score": row.get("hands_on_partner_legs_score"),
        "hands_on_partner_thighs_score": row.get("hands_on_partner_thighs_score"),
        "hands_behind_support_score": row.get("hands_behind_support_score"),
        "partner_lying_score": None,
        "diagnostic_fields_present": DIAGNOSTIC_FIELDS,
        "review_questions": REVIEW_QUESTIONS,
        "allowed_review_labels": REVIEW_LABELS,
        "warnings": row.get("warnings") or [],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }
    return card


def _load_context(run: Path) -> dict[str, Any]:
    return {
        "windows": {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")},
        "samples": {r.get("sample_id"): r for r in load_jsonl(run / "baked" / "motion_sample_index.jsonl") if r.get("sample_id")},
        "sources": {r.get("source_id"): r for r in load_jsonl(run / "semantic" / "motion_source_index.jsonl") if r.get("source_id")},
        "relative": {r.get("window_id"): r for r in load_jsonl(run / "relative_motion" / "relative_motion_features.jsonl") if r.get("window_id")},
    }


def _load_first(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def _load_previous_review_cards(previous_review: str | Path | None) -> dict[str, dict[str, Any]]:
    if not previous_review:
        return {}
    path = Path(previous_review)
    if path.is_dir():
        path = path / "semantic_review_010.jsonl"
    if not path.exists():
        return {}
    return {str(row.get("review_id")): row for row in load_jsonl(path) if row.get("review_id")}


def _load_human_answer_notes(human_answers: str | Path | None) -> list[dict[str, Any]]:
    if not human_answers:
        return []
    path = Path(human_answers)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(row)
    return rows


def _human_exclusion_windows(previous_cards: dict[str, dict[str, Any]], answers: list[dict[str, Any]]) -> set[str]:
    bad_terms = {
        "steht",
        "stehen",
        "walking",
        "walk",
        "transition",
        "aufsteht",
        "aufstehen",
        "hand",
        "hände",
        "unklar",
        "broken",
        "nix mit cowgirl",
        "kein cowgirl",
        "nicht zuzuweisen",
        "penis ein",
    }
    good_terms = {
        "normal cowgirl",
        "passende grinding",
        "passender grinding",
        "passende reit",
        "passender reit",
        "passender slow grinding",
        "passende slow grinding",
        "animation stimmt",
    }
    out: set[str] = set()
    for answer in answers:
        review_id = str(answer.get("review_id") or "")
        note = str(answer.get("notes") or "").lower()
        card = previous_cards.get(review_id)
        if card:
            _add_card_exclusion_keys(card, out)
        if any(term in note for term in good_terms) and not any(term in note for term in {"transition", "broken", "unklar"}):
            continue
        if not any(term in note for term in bad_terms):
            continue
        if card:
            _add_card_exclusion_keys(card, out)
    return out


def _add_card_exclusion_keys(card: dict[str, Any], out: set[str]) -> None:
    if card.get("window_id"):
        out.add(str(card["window_id"]))
    if card.get("sample_id"):
        out.add(f"sample:{card['sample_id']}")
    if card.get("source_id"):
        out.add(f"source:{card['source_id']}")


def _human_positive_lean_back_rows(previous_cards: dict[str, dict[str, Any]], answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive_terms = {
        "lean back cowgirl",
        "passende grinding",
        "passender grinding",
        "passender slow grinding",
        "passende slow grinding",
        "passende reit",
        "passender reit",
        "animation stimmt",
    }
    negative_terms = {"transition", "walking", "broken", "unklar", "steht", "stehen", "bj", "oral"}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for answer in answers:
        note = str(answer.get("notes") or "").lower()
        if "lean back" not in note and "nach hinten" not in note:
            continue
        if not any(term in note for term in positive_terms):
            continue
        if any(term in note for term in negative_terms):
            continue
        card = previous_cards.get(str(answer.get("review_id") or ""))
        if not card or card.get("semantic_family") != "cowgirl" or not card.get("window_id"):
            continue
        wid = str(card["window_id"])
        if wid in seen:
            continue
        seen.add(wid)
        rows.append(
            {
                **card,
                "category": "cowgirl_lean_back_supported_clean_motion",
                "pose_family": "cowgirl",
                "pose_subtype": card.get("pose_subtype") or (card.get("pose_semantics") or {}).get("subtype") or "cowgirl_lean_back_supported",
                "semantic_family": "cowgirl",
                "phase": card.get("phase") or (card.get("motion_semantics") or {}).get("phase") or "clean_motion",
                "clean_motion_gate": card.get("clean_motion_gate") or "pass",
                "hip_motion_strength": _first(card.get("hip_motion_strength"), 1.0),
                "pelvis_trajectory_strength": _first(card.get("pelvis_trajectory_strength"), 1.0),
                "motion_score": _first(card.get("motion_score"), 1.0),
                "semantic_score": _first(card.get("semantic_score"), 1.0),
                "_human_positive_lean_back_hint": True,
            }
        )
    return rows


def _write_human_answer_summary(answers: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Previous Human Answer Summary",
        "",
        "These are audit notes used only to avoid repeating obvious bad windows. They are not training labels.",
        "",
        f"- Answer records found: {len(answers)}",
        f"- Screenshot records found: {sum(len(row.get('screenshots') or []) for row in answers)}",
        "",
    ]
    for row in answers:
        note = str(row.get("notes") or "").replace("\n", " ").strip()
        if not note:
            continue
        lines.append(f"- `{row.get('review_id')}`: {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _likely_failure_mode(bucket: str, row: dict[str, Any]) -> str:
    if bucket == "cowgirl_clean_motion_generation_safe":
        if row.get("anchor_motion_weird"):
            return "clean_motion_but_anchor_motion_weird"
        if row.get("contact_support") in {"ambiguous_partner_contact", "possible_partner_contact"}:
            return "clean_motion_contact_uncertain"
        return "possible_clean_cowgirl_needs_confirmation"
    if bucket == "strict_cowgirl_clean_motion":
        return "strict_filtered_clean_cowgirl_candidate"
    if bucket == "strict_cowgirl_short_motion":
        return "strict_filtered_short_cowgirl_candidate"
    if bucket == "strict_cowgirl_lean_back_check":
        return "strict_filtered_lean_back_or_hands_behind_cowgirl_check"
    if bucket == "possible_cowgirl_lean_back_supported_review":
        return "possible_lean_back_supported_or_reverse_cowgirl_needs_human_orientation_check"
    if bucket == "cowgirl_clean_motion_low_confidence_short":
        return "short_window_may_be_valid_or_too_brief"
    if bucket == "cowgirl_pose_context_low_motion":
        return "pose_only_low_motion_may_be_misread_as_clean_motion"
    if bucket == "cowgirl_transition_setup":
        return "transition_or_intro_alignment_may_leak_into_clean_motion"
    if bucket == "not_cowgirl_standing_hand_head":
        return "standing_hand_head_negative_should_not_be_cowgirl"
    if bucket == "not_cowgirl_bj_oral":
        return "bj_oral_family_candidate_excluded_from_cowgirl"
    if bucket == "unknown_or_unusable_high_movement":
        return "high_movement_unknown_or_bad_pose"
    if bucket == "contact_support_ambiguous":
        return "contact_target_ambiguous_or_wrong_partner_context"
    if bucket in {"cowgirl_lean_back_supported_clean_motion", "cowgirl_lean_back_supported_pose_context"}:
        return "lean_back_supported_cowgirl_needs_front_vs_reverse_and_support_confirmation"
    if bucket == "cowgirl_hands_behind_support":
        return "hands_behind_support_needs_target_confirmation"
    if bucket == "cowgirl_hands_on_partner_legs_or_thighs":
        return "partner_leg_or_thigh_support_needs_confirmation"
    return "unknown_failure_mode"


def _possible_lean_back_review(row: dict[str, Any], context: dict[str, Any]) -> bool:
    if row.get("pose_subtype") == "cowgirl_lean_back_supported":
        return True
    if row.get("contact_support") in {"hands_behind_support", "hands_on_partner_legs_or_thighs", "ambiguous_behind_support"}:
        return True
    text = _row_context_text(row, context)
    if not any(token in text for token in ("lean back", "lean_back", "back support", "hands behind", "nach hinten", "hinten", "reverse cowgirl")):
        return False
    return row.get("semantic_family") == "cowgirl" or "cowgirl" in str(row.get("category") or "").lower()


def _row_context_text(row: dict[str, Any], context: dict[str, Any]) -> str:
    window = context["windows"].get(row.get("window_id"), {})
    sample = context["samples"].get(row.get("sample_id") or window.get("sample_id"), {})
    source = context["sources"].get(window.get("source_id") or sample.get("source_id"), {})
    values = [
        row.get("source_scene_file"),
        row.get("source_scene_path"),
        row.get("motion_subtype"),
        row.get("pose_subtype"),
        row.get("contact_support"),
        window.get("source_scene_file"),
        window.get("source_scene_path"),
        source.get("clip_name"),
        source.get("storable_id"),
        sample.get("clip_name"),
    ]
    return " ".join(str(v or "") for v in values).lower()


def _simple_review_label(idx: int, bucket: str) -> str:
    names = {
        "cowgirl_clean_motion_generation_safe": "cowgirl_clean",
        "strict_cowgirl_clean_motion": "cowgirl_strict",
        "strict_cowgirl_short_motion": "cowgirl_short_strict",
        "strict_cowgirl_lean_back_check": "cowgirl_lean_back_strict",
        "possible_cowgirl_lean_back_supported_review": "cowgirl_lean_back_check",
        "cowgirl_lean_back_supported_clean_motion": "cowgirl_lean_back_clean",
        "cowgirl_lean_back_supported_pose_context": "cowgirl_lean_back_pose",
        "cowgirl_hands_behind_support": "cowgirl_hands_behind",
        "cowgirl_hands_on_partner_legs_or_thighs": "cowgirl_hands_legs",
        "cowgirl_clean_motion_low_confidence_short": "cowgirl_short",
        "cowgirl_pose_context_low_motion": "cowgirl_pose_low_motion",
        "cowgirl_transition_setup": "cowgirl_transition",
        "not_cowgirl_standing_hand_head": "not_cowgirl_standing",
        "not_cowgirl_bj_oral": "not_cowgirl_bj_oral",
        "unknown_or_unusable_high_movement": "unknown_high_motion",
        "contact_support_ambiguous": "contact_ambiguous",
    }
    semantic = names.get(bucket, _safe_label(bucket))
    return f"{idx:03d}_{semantic}"


def _safe_label(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text).lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "review"


def _near_duplicate_group(row: dict[str, Any], window: dict[str, Any], bucket: str) -> str:
    start = round(_num(window.get("start_seconds") or row.get("start_seconds")) / 4.0) * 4
    return "|".join(
        [
            str(window.get("source_id") or row.get("source_id") or row.get("sample_id") or ""),
            str(row.get("technical_actor_id") or window.get("technical_atom_id") or ""),
            str(bucket),
            str(row.get("pose_subtype") or ""),
            str(row.get("motion_subtype") or ""),
            str(row.get("contact_support") or ""),
            str(start),
        ]
    )


def _write_answer_sheet(cards: list[dict[str, Any]], path: Path) -> None:
    data = {
        "allowed_review_labels": REVIEW_LABELS,
        "review_questions": REVIEW_QUESTIONS,
        "reviews": {
            card["review_id"]: {
                "semantic_family_correct": "unknown",
                "pose_correct": "unknown",
                "motion_correct": "unknown",
                "partner_relation_correct": "unknown",
                "contact_support_correct": "unknown",
                "generation_safe_correct": "unknown",
                "review_labels": [],
                "lean_back_supported_correct": "unknown",
                "front_vs_reverse_correct": "unknown",
                "hands_behind_support_correct": "unknown",
                "partner_legs_or_thighs_support_correct": "unknown",
                "usable_for_future_motion_primitive_extraction": "unknown",
                "notes": "",
            }
            for card in cards
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_manifest_csv(cards: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "review_id",
        "review_label",
        "why_selected",
        "source_scene_file",
        "source_id",
        "technical_atom_id",
        "clip_name",
        "start_seconds",
        "end_seconds",
        "semantic_family",
        "cowgirl_bucket",
        "pose_family",
        "pose_subtype",
        "motion_subtype",
        "hip_motion_strength",
        "pelvis_trajectory_strength",
        "vertical_movement",
        "forward_back_movement",
        "lateral_movement",
        "partner_context_confidence",
        "contact_support",
        "torso_lean_direction",
        "facing_context",
        "support_context",
        "hands_behind_support_score",
        "hands_on_partner_legs_score",
        "hands_on_partner_thighs_score",
        "likely_failure_mode",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for card in cards:
            writer.writerow({field: _csv_value(card.get(field)) for field in fields})


def _write_review_md(cards: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Focused New-Scenes Semantic Review",
        "",
        "This batch is for classifier calibration, not for training truth.",
        "",
        "## Reviewer Questions",
        "",
    ]
    lines.extend(f"- {q}" for q in REVIEW_QUESTIONS)
    for card in cards:
        lines.extend(
            [
                "",
                f"## {card['review_label']} - {card['why_selected']}",
                "",
                f"- Review ID: `{card.get('review_id')}`",
                f"- Scene: `{card.get('source_scene_file')}`",
                f"- Source ID: `{card.get('source_id')}`",
                f"- Actor: `{card.get('technical_atom_id')}`",
                f"- Timeline / clip: `{card.get('storable_id')}` / `{card.get('clip_name')}`",
                f"- Time: `{card.get('start_seconds')}` to `{card.get('end_seconds')}`",
                f"- Predicted family: `{card.get('semantic_family')}`",
                f"- Bucket: `{card.get('cowgirl_bucket')}`",
                f"- Pose: `{card.get('pose_family')}` / `{card.get('pose_subtype')}`",
                f"- Motion: `{card.get('motion_subtype')}` / `{card.get('phase')}`",
                f"- Hip motion: `{card.get('hip_motion_strength')}`",
                f"- Pelvis trajectory: `{card.get('pelvis_trajectory_strength')}`",
                f"- Axis breakdown: `{card.get('axis_breakdown')}`",
                f"- Partner context confidence: `{card.get('partner_context_confidence')}`",
                f"- Contact/support: `{card.get('contact_support')}`",
                f"- Likely failure mode: `{card.get('likely_failure_mode')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_html(cards: list[dict[str, Any]], path: Path) -> None:
    body = [
        "<!doctype html><meta charset='utf-8'><title>Focused New Scenes Review</title>",
        "<style>body{font-family:system-ui;margin:1.5rem;background:#f7f7f5}section{background:white;border:1px solid #ddd;border-radius:6px;padding:1rem;margin:1rem 0}code{background:#f0f0ea;padding:.1rem .25rem;border-radius:4px}</style>",
        "<h1>Focused New-Scenes Semantic Review</h1>",
        "<p>Audit-only calibration batch. Not training truth.</p>",
        "<h2>Reviewer Questions</h2><ul>",
    ]
    body.extend(f"<li>{html.escape(q)}</li>" for q in REVIEW_QUESTIONS)
    body.append("</ul>")
    for card in cards:
        body.append(
            f"<section><h2>{html.escape(str(card.get('review_label') or card['review_id']))}</h2>"
            f"<p><strong>Review ID:</strong> <code>{html.escape(card['review_id'])}</code> "
            f"<strong>Bucket:</strong> <code>{html.escape(str(card.get('why_selected')))}</code></p>"
            f"<p><strong>Scene:</strong> <code>{html.escape(str(card.get('source_scene_file')))}</code> "
            f"<strong>Actor:</strong> <code>{html.escape(str(card.get('technical_atom_id')))}</code> "
            f"<strong>Time:</strong> <code>{html.escape(str(card.get('start_seconds')))}-{html.escape(str(card.get('end_seconds')))}s</code></p>"
            f"<p><strong>Family:</strong> <code>{html.escape(str(card.get('semantic_family')))}</code> "
            f"<strong>Bucket:</strong> <code>{html.escape(str(card.get('cowgirl_bucket')))}</code></p>"
            f"<p><strong>Pose:</strong> <code>{html.escape(str(card.get('pose_family')))} / {html.escape(str(card.get('pose_subtype')))}</code> "
            f"<strong>Motion:</strong> <code>{html.escape(str(card.get('motion_subtype')))} / {html.escape(str(card.get('phase')))}</code></p>"
            f"<p><strong>Hip:</strong> <code>{html.escape(str(card.get('hip_motion_strength')))}</code> "
            f"<strong>Axes:</strong> <code>{html.escape(str(card.get('axis_breakdown')))}</code> "
            f"<strong>Contact:</strong> <code>{html.escape(str(card.get('contact_support')))}</code></p>"
            f"<p><strong>Likely failure:</strong> <code>{html.escape(str(card.get('likely_failure_mode')))}</code></p>"
            "</section>"
        )
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _write_selection_report(
    cards: list[dict[str, Any]],
    selection: dict[str, Any],
    path: Path,
    targets: dict[str, int] | None = None,
) -> None:
    targets = targets or TARGETS
    lines = [
        "# Focused New-Scenes Review Selection Report",
        "",
        f"- Items selected: {len(cards)}",
        "- Batch purpose: calibration coverage, not a showcase and not training truth.",
        "",
        "## Target Counts",
        "",
    ]
    for key, target in targets.items():
        lines.append(f"- `{key}`: target {target}, selected {selection['selected_counts'].get(key, 0)}")
    lines.extend(["", "## Duplicate Avoidance", ""])
    lines.append(f"- Rejected by rule: `{selection.get('rejected_by_rule')}`")
    lines.append(f"- Scene counts: `{selection.get('scene_counts')}`")
    lines.append(f"- Motion-shape counts: `{selection.get('motion_shape_counts')}`")
    lines.append(f"- Contact counts: `{selection.get('contact_counts')}`")
    lines.extend(["", "## Review Labels", ""])
    lines.extend(f"- `{label}`" for label in REVIEW_LABELS)
    lines.extend(["", "## Reviewer Questions", ""])
    lines.extend(f"- {question}" for question in REVIEW_QUESTIONS)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_codex_prompt_template(path: Path) -> None:
    lines = [
        "# Codex Summary Prompt After Review",
        "",
        "After exporting UI answers, paste a concise summary like this back into Codex:",
        "",
        "```text",
        "We reviewed the focused new-scenes batch.",
        "Findings:",
        "review_001: ...",
        "review_002: ...",
        "",
        "Common errors:",
        "- ...",
        "",
        "Needed fixes:",
        "- ...",
        "",
        "Do not treat these audit labels as manual training truth and do not modify manual_labels.yaml.",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Focused New-Scenes Review Summary",
        "",
        f"- Output: `{summary['out_dir']}`",
        f"- Selected cards: {summary['selected']}",
        "",
        "## Selected Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in summary["selected_counts"].items())
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Review package: `{(summary['vam_review_package'] or {}).get('out_dir')}`",
            f"- Static UI: `{(summary['static_review_ui'] or {}).get('index')}`",
            f"- Manifest: `{Path(summary['out_dir']) / 'focused_review_manifest.jsonl'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_inside(run: Path, out: Path) -> None:
    try:
        out.relative_to(run)
    except ValueError as exc:
        raise ValueError(f"focused review output must stay inside {run}") from exc
    if run.name == "clean_v3":
        raise ValueError("focused new-scene review must not write inside clean_v3")


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _duration(start: Any, end: Any) -> float | None:
    try:
        return float(end) - float(start)
    except Exception:
        return None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value != value:
            return default
        return float(value)
    except Exception:
        return default


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(x) for x in value)
    if isinstance(value, dict):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return value

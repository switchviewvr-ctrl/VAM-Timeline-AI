"""Focused clean_v3 rescan for Cowgirl lean-back supported pose semantics."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import html

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.features.partner_relative_features import extract_partner_relative_features_v0
from vam_timeline_ai.features.pose_features import extract_pose_features_v0
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.semantics.clean_v3_v16_calibration import _load_context_v2, _num, _write_csv
from vam_timeline_ai.semantics.interaction_classifier import classify_interactions_v0
from vam_timeline_ai.semantics.pose_classifier import classify_poses_v0
from vam_timeline_ai.ui.review_ui import build_static_review_ui


V8_CATEGORIES = [
    "cowgirl_clean_motion_generation_safe",
    "cowgirl_clean_motion_low_confidence_short",
    "cowgirl_pose_context_low_motion",
    "cowgirl_intro_alignment",
    "cowgirl_transition_setup",
    "cowgirl_no_clear_hip_motion",
    "cowgirl_lean_back_supported_clean_motion",
    "cowgirl_lean_back_supported_pose_context",
    "cowgirl_hands_on_partner_chest",
    "cowgirl_hands_on_partner_hips",
    "cowgirl_hands_on_partner_legs_or_thighs",
    "cowgirl_hands_behind_support",
    "cowgirl_ambiguous_partner_contact",
    "cowgirl_missing_partner_context",
    "not_cowgirl_standing_hand_head",
    "not_cowgirl_bj_oral",
    "not_cowgirl_receiver_response",
    "unknown_or_unusable",
]


def store_cowgirl_lean_back_observation(run_dir: str | Path) -> dict[str, Any]:
    out = Path(run_dir) / "audits" / "pose_observations"
    out.mkdir(parents=True, exist_ok=True)
    md = out / "cowgirl_lean_back_supported_note.md"
    yml = out / "cowgirl_lean_back_supported_note.yaml"
    md.write_text(
        "\n".join(
            [
                "# Cowgirl Lean-back Supported Pose Observation",
                "",
                "Audit-only observation; not a manual training label.",
                "",
                "- User provided a visual example of a frontal Cowgirl lean-back supported pose.",
                "- Woman/rider is leaning backward.",
                "- Hands are behind the body.",
                "- Hands support against the partner/man legs or thighs behind her back.",
                "- This is still frontal Cowgirl, not reverse Cowgirl.",
                "- Existing pose taxonomy must be extended.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    yml.write_text(
        yaml.safe_dump(
            {
                "audit_only": True,
                "is_human_ground_truth": False,
                "is_training_label": False,
                "pose_family": "cowgirl",
                "pose_subtype": "cowgirl_lean_back_supported",
                "facing": "front_cowgirl",
                "torso": "lean_back",
                "support_context": [
                    "hands_behind_support",
                    "hands_on_partner_legs_or_thighs",
                ],
                "not_labels": ["reverse_cowgirl", "hands_free", "unknown_support"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {"markdown": str(md), "yaml": str(yml)}


def run_clean_v3_pose_support_rescan(run_dir: str | Path, out_suffix: str = "lean_back_support_v1") -> dict[str, Any]:
    run = Path(run_dir)
    note = store_cowgirl_lean_back_observation(run)
    pose_features_path = run / "pose_semantics" / f"pose_features_{out_suffix}.jsonl"
    pose_semantics_path = run / "pose_semantics" / f"pose_semantics_{out_suffix}.jsonl"
    partner_features_path = run / "interaction_semantics" / f"partner_relative_features_{out_suffix}.jsonl"
    interaction_path = run / "interaction_semantics" / f"interaction_semantics_{out_suffix}.jsonl"

    pose_features = extract_pose_features_v0(
        run / "relative_motion" / "relative_motion_window_index.jsonl",
        run / "audits" / "body_motion_quality.jsonl",
        run / "audits" / "pose_anchor_completeness.jsonl",
        run / "audits" / "controller_validity.jsonl",
        pose_features_path,
        run / "pose_semantics" / f"pose_feature_report_{out_suffix}.md",
    )
    pose_rows = classify_poses_v0(
        pose_features_path,
        run / "relative_motion" / "relative_reference_matches.jsonl",
        run / "references" / "handmade_animations" / "handmade_relative_features.jsonl",
        pose_semantics_path,
        run / "pose_semantics" / f"pose_semantics_report_{out_suffix}.md",
    )
    partner_rows = extract_partner_relative_features_v0(
        run / "semantic" / "pair_windows_v1.jsonl",
        run / "features" / "cowgirl_pair_features_v0.jsonl",
        run / "relative_motion" / "relative_motion_window_index.jsonl",
        pose_semantics_path,
        partner_features_path,
        run / "interaction_semantics" / f"partner_relative_feature_report_{out_suffix}.md",
    )
    interaction_rows = classify_interactions_v0(
        partner_features_path,
        pose_semantics_path,
        None,
        interaction_path,
        run / "interaction_semantics" / f"interaction_semantics_report_{out_suffix}.md",
    )

    actions = _build_semantic_actions_v3(run, pose_rows, interaction_rows)
    actions_path = run / "semantic_actions" / "semantic_actions_v3.jsonl"
    write_jsonl(actions_path, actions)
    _write_action_report(actions, run / "semantic_actions" / "semantic_actions_v3_report.md")

    semantic_db = _build_semantic_candidate_db_v3(actions)
    write_jsonl(run / "datasets" / "semantic_candidate_db_v3.jsonl", semantic_db)
    _write_csv(semantic_db, run / "datasets" / "semantic_candidate_db_v3.csv", _semantic_csv_fields())
    _write_db_report(semantic_db, run / "datasets" / "semantic_candidate_db_v3_report.md", "Semantic Candidate DB V3")

    cowgirl_db = _build_cowgirl_db_v8(semantic_db)
    write_jsonl(run / "datasets" / "cowgirl_candidate_db_v8.jsonl", cowgirl_db)
    _write_csv(cowgirl_db, run / "datasets" / "cowgirl_candidate_db_v8.csv", _cowgirl_csv_fields())
    _write_db_report(cowgirl_db, run / "datasets" / "cowgirl_candidate_db_v8_report.md", "Cowgirl Candidate DB V8")

    review_dir = run / "audits" / "semantic_review_lean_back_support_020"
    review_summary = export_lean_back_support_review(run, cowgirl_db, review_dir)

    summary = {
        "status": "ok",
        "observation": note,
        "pose_feature_rows": len(pose_features),
        "pose_semantics_counts": dict(Counter(r.get("pose_subtype") for r in pose_rows)),
        "partner_feature_rows": len(partner_rows),
        "interaction_counts": dict(Counter(r.get("support_context") for r in interaction_rows)),
        "semantic_actions_v3": len(actions),
        "semantic_candidate_db_v3_counts": dict(Counter(r.get("semantic_family") for r in semantic_db)),
        "cowgirl_candidate_db_v8_counts": _category_counts(cowgirl_db),
        "lean_back_support_candidate_counts": _lean_back_counts(cowgirl_db),
        "focused_review": review_summary,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    _write_pipeline_summary(run / "reports" / "clean_v3_pose_support_rescan_summary.md", summary)
    dump_json(run / "reports" / "clean_v3_pose_support_rescan_summary.json", summary)
    return summary


def export_lean_back_support_review(run: Path, cowgirl_db: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context_v2(run)
    selected, shortage = _select_lean_back_review(cowgirl_db, context)
    rows = [_review_row(idx, row, context) for idx, row in enumerate(selected, start=1)]
    write_jsonl(out_dir / "semantic_review_010.jsonl", rows)
    _write_review_md(rows, shortage, out_dir / "semantic_review_010.md")
    _write_review_html(rows, shortage, out_dir / "semantic_review_010_index.html")
    _write_answer_sheet(rows, out_dir / "semantic_review_010_answer_sheet.yaml")
    package_summary = build_vam_review_package(out_dir / "semantic_review_010.jsonl", run, run.parent / "clean_v2", out_dir / "vam_review_package", attempt_timeline_segments=True)
    static_summary = build_static_review_ui(run, out_dir, out_dir / "review_ui_static")
    return {
        "review_dir": str(out_dir),
        "review_items": len(rows),
        "category_counts": dict(Counter(r.get("why_selected") for r in rows)),
        "shortage": shortage,
        "vam_package": package_summary,
        "static_review_ui": static_summary,
    }


def _build_semantic_actions_v3(run: Path, pose_rows: list[dict[str, Any]], interaction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pose = {r.get("window_id"): r for r in pose_rows if r.get("window_id")}
    interactions = defaultdict(list)
    for row in interaction_rows:
        interactions[row.get("window_id")].append(row)
    source = _load_first(run / "semantic_actions" / "semantic_actions_v2.jsonl", run / "semantic_actions" / "semantic_actions_v1.jsonl", run / "semantic_actions" / "semantic_actions_v0.jsonl")
    out = []
    for action in source:
        row = dict(action)
        wid = row.get("window_id")
        pose_row = pose.get(wid, {})
        interaction = _best_interaction(interactions.get(wid, []))
        if pose_row:
            row.update(
                {
                    "pose_family": pose_row.get("pose_family", row.get("pose_family")),
                    "pose_subtype": pose_row.get("pose_subtype", row.get("pose_subtype")),
                    "support_context": pose_row.get("support_context", row.get("support_context")),
                    "facing_context": pose_row.get("facing_context", row.get("facing_context")),
                    "torso_lean_direction": pose_row.get("torso_lean_direction", row.get("torso_lean_direction")),
                    "lean_back_pose_confidence": pose_row.get("lean_back_pose_confidence", 0.0),
                    "hands_behind_support_score": pose_row.get("hands_behind_support_confidence", 0.0),
                    "partner_leg_support_confidence": pose_row.get("partner_leg_support_confidence", 0.0),
                    "facing_confidence": pose_row.get("facing_confidence", 0.0),
                }
            )
        if interaction:
            support = interaction.get("support_context")
            if support and support != "unknown":
                row["contact_support"] = support
            row.update(
                {
                    "pair_window_id": row.get("pair_window_id") or interaction.get("pair_window_id"),
                    "partner_relation": interaction.get("partner_relation") or row.get("partner_relation") or ["unknown"],
                    "interaction_score": max(_num(row.get("interaction_score")), _num(interaction.get("interaction_confidence"))),
                    "contact_support_confidence": interaction.get("contact_support_confidence", row.get("contact_support_confidence")),
                    "contact_support_margin": interaction.get("contact_support_margin", row.get("contact_support_margin")),
                    "contact_support_ambiguous": interaction.get("contact_support_ambiguous", row.get("contact_support_ambiguous")),
                    "best_contact_target": interaction.get("best_contact_target", row.get("best_contact_target")),
                    "second_best_contact_target": interaction.get("second_best_contact_target", row.get("second_best_contact_target")),
                    "partner_context_confidence": interaction.get("partner_context_confidence", row.get("partner_context_confidence")),
                    "hands_on_partner_legs_score": interaction.get("hands_on_partner_legs_score", 0.0),
                    "hands_on_partner_thighs_score": interaction.get("hands_on_partner_thighs_score", 0.0),
                    "partner_leg_thigh_approximation_used": interaction.get("partner_leg_thigh_approximation_used", False),
                }
            )
        support_context = set(row.get("support_context") or [])
        if row.get("contact_support") in {None, "unknown", "unknown_contact"}:
            if "hands_on_partner_legs_or_thighs" in support_context:
                row["contact_support"] = "hands_on_partner_legs_or_thighs"
            elif "hands_behind_support" in support_context or "possible_hands_behind_support" in support_context:
                row["contact_support"] = "hands_behind_support"
        requirements = list(row.get("support_constraint_requirements") or [])
        if row.get("pose_subtype") == "cowgirl_lean_back_supported":
            requirements.extend(["keep_torso_lean_back", "keep_rider_pelvis_aligned_to_partner"])
            if row.get("contact_support") in {"hands_on_partner_legs_or_thighs", "hands_behind_support", "ambiguous_behind_support"}:
                requirements.append("keep_hands_behind_on_partner_legs_or_thighs")
            if row.get("facing_context") != "reverse_cowgirl":
                row["facing_context"] = "front_cowgirl"
        row["support_constraint_requirements"] = _dedupe(requirements)
        row["audit_calibration_source"] = "lean_back_support_v1"
        row["is_human_ground_truth"] = False
        row["is_training_label"] = False
        out.append(row)
    return out


def _build_semantic_candidate_db_v3(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(actions, start=1):
        rows.append(
            {
                "candidate_id": f"semantic_action_v3::{row.get('window_id') or idx}",
                "window_id": row.get("window_id"),
                "pair_window_id": row.get("pair_window_id"),
                "source_scene_file": row.get("source_scene_file"),
                "source_scene_path": row.get("source_scene_path"),
                "technical_actor_id": row.get("technical_atom_id") or row.get("technical_actor_id"),
                "sample_id": row.get("sample_id"),
                "semantic_family": row.get("semantic_family") or "unknown",
                "pose_family": row.get("pose_family"),
                "pose_subtype": row.get("pose_subtype"),
                "torso_lean_direction": row.get("torso_lean_direction", "unknown"),
                "facing_context": row.get("facing_context", "unknown"),
                "support_context": row.get("support_context") or [],
                "motion_subtype": row.get("motion_subtype"),
                "partner_relation": row.get("partner_relation") or ["unknown"],
                "contact_support": row.get("contact_support") or "unknown",
                "phase": row.get("phase") or "unknown",
                "generation_safe": bool(row.get("generation_safe")),
                "safe_for_learning": bool(row.get("generation_safe")) and not row.get("conflict_flags"),
                "invalidity_reason": ";".join(row.get("conflict_flags") or []),
                "semantic_score": row.get("semantic_score"),
                "pose_score": row.get("pose_score"),
                "motion_score": row.get("motion_score"),
                "interaction_score": row.get("interaction_score"),
                "clean_motion_gate": row.get("clean_motion_gate"),
                "clean_motion_gate_reason": row.get("clean_motion_gate_reason"),
                "hip_motion_strength": row.get("hip_motion_strength"),
                "pelvis_trajectory_strength": row.get("pelvis_trajectory_strength"),
                "hands_behind_support_score": row.get("hands_behind_support_score"),
                "hands_on_partner_legs_score": row.get("hands_on_partner_legs_score"),
                "hands_on_partner_thighs_score": row.get("hands_on_partner_thighs_score"),
                "facing_confidence": row.get("facing_confidence"),
                "partner_leg_support_confidence": row.get("partner_leg_support_confidence"),
                "contact_support_confidence": row.get("contact_support_confidence"),
                "support_constraint_requirements": row.get("support_constraint_requirements") or [],
                "warnings": row.get("warnings") or [],
                "preserve_for_future_dataset": row.get("semantic_family") in {"cowgirl", "bj_oral", "hand_gesture", "head_gesture", "receiver_response"},
                "audit_calibration_source": "lean_back_support_v1",
                "is_human_ground_truth": False,
                "is_training_label": False,
            }
        )
    rows.sort(key=lambda r: (_category_rank(_cowgirl_v8_category(r)), -_num(r.get("semantic_score"))))
    return rows


def _build_cowgirl_db_v8(semantic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in semantic_rows:
        category = _cowgirl_v8_category(row)
        if category == "skip":
            continue
        item = dict(row)
        item["candidate_id"] = f"cowgirl_v8::{row.get('window_id')}"
        item["category"] = category
        item["hand_contact_target"] = row.get("best_contact_target")
        item["generation_requires_partner_targets"] = row.get("contact_support") in {"hands_on_partner_chest", "hands_on_partner_hips", "hands_on_partner_legs_or_thighs"}
        item["generation_safe"] = bool(row.get("generation_safe")) and category in {"cowgirl_clean_motion_generation_safe", "cowgirl_lean_back_supported_clean_motion"}
        rows.append(item)
    rows.sort(key=lambda r: (_category_rank(r.get("category")), -_num(r.get("semantic_score"))))
    return rows


def _cowgirl_v8_category(row: dict[str, Any]) -> str:
    family = row.get("semantic_family")
    if family == "bj_oral":
        return "not_cowgirl_bj_oral"
    if family == "receiver_response":
        return "not_cowgirl_receiver_response"
    if family in {"hand_gesture", "head_gesture"} or row.get("pose_family") == "standing":
        return "not_cowgirl_standing_hand_head"
    if family == "unknown":
        return "unknown_or_unusable"
    if family != "cowgirl":
        return "skip"
    contact = row.get("contact_support")
    pose_subtype = row.get("pose_subtype")
    gate = row.get("clean_motion_gate")
    phase = row.get("phase")
    if pose_subtype == "cowgirl_lean_back_supported" and gate == "pass" and row.get("generation_safe"):
        return "cowgirl_lean_back_supported_clean_motion"
    if pose_subtype == "cowgirl_lean_back_supported":
        return "cowgirl_lean_back_supported_pose_context"
    if contact == "hands_on_partner_legs_or_thighs":
        return "cowgirl_hands_on_partner_legs_or_thighs"
    if contact == "hands_behind_support":
        return "cowgirl_hands_behind_support"
    if gate == "pass" and row.get("generation_safe"):
        return "cowgirl_clean_motion_generation_safe"
    if gate == "soft_pass_short":
        return "cowgirl_clean_motion_low_confidence_short"
    if phase == "transition_setup":
        return "cowgirl_transition_setup"
    if phase == "intro_alignment":
        return "cowgirl_intro_alignment"
    if gate == "fail_no_hip_motion":
        return "cowgirl_no_clear_hip_motion"
    if phase in {"low_motion_hold", "pose_context_only"} or gate == "fail_low_motion":
        return "cowgirl_pose_context_low_motion"
    if contact == "hands_on_partner_chest":
        return "cowgirl_hands_on_partner_chest"
    if contact == "hands_on_partner_hips":
        return "cowgirl_hands_on_partner_hips"
    if contact in {"ambiguous_partner_contact", "ambiguous_behind_support"}:
        return "cowgirl_ambiguous_partner_contact"
    if contact in {"unknown", "unknown_contact", None}:
        return "cowgirl_missing_partner_context"
    return "unknown_or_unusable"


def _select_lean_back_review(rows: list[dict[str, Any]], context: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = [
        ("cowgirl_lean_back_supported_clean_motion", 5),
        ("cowgirl_lean_back_supported_pose_context", 5),
        ("cowgirl_hands_behind_support", 5),
        ("cowgirl_hands_on_partner_legs_or_thighs", 5),
        ("cowgirl_ambiguous_partner_contact", 3),
        ("not_cowgirl_standing_hand_head", 1),
        ("unknown_or_unusable", 1),
    ]
    pools = defaultdict(list)
    for row in rows:
        pools[row.get("category")].append(row)
    selected: list[dict[str, Any]] = []
    seen_windows: set[str] = set()
    scene_counts: Counter[str] = Counter()
    shortage: dict[str, int] = {}
    for category, limit in targets:
        added = 0
        for row in pools.get(category, []):
            if added >= limit or len(selected) >= 20:
                break
            wid = str(row.get("window_id") or "")
            window = context["windows"].get(wid, {})
            scene = str(row.get("source_scene_file") or window.get("source_scene_file") or "unknown")
            if not wid or wid in seen_windows or scene_counts[scene] >= 2:
                continue
            selected.append(row)
            seen_windows.add(wid)
            scene_counts[scene] += 1
            added += 1
        if added < limit:
            shortage[category] = limit - added
    return selected[:20], shortage


def _review_row(idx: int, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    window = context["windows"].get(row.get("window_id"), {})
    return {
        "review_id": f"review_{idx:03d}",
        "window_id": row.get("window_id"),
        "pair_window_id": row.get("pair_window_id"),
        "semantic_family": row.get("semantic_family"),
        "pose_semantics": {"family": row.get("pose_family"), "subtype": row.get("pose_subtype")},
        "motion_semantics": {"subtype": row.get("motion_subtype"), "phase": row.get("phase")},
        "partner_relation": row.get("partner_relation") or ["unknown"],
        "contact_support": row.get("contact_support"),
        "interaction_family": "cowgirl" if row.get("semantic_family") == "cowgirl" else row.get("semantic_family"),
        "generation_safe": bool(row.get("generation_safe")),
        "why_selected": row.get("category"),
        "pose_subtype": row.get("pose_subtype"),
        "torso_lean_direction": row.get("torso_lean_direction"),
        "facing_context": row.get("facing_context"),
        "support_context": row.get("support_context"),
        "hands_behind_support_score": row.get("hands_behind_support_score"),
        "hands_on_partner_legs_score": row.get("hands_on_partner_legs_score"),
        "hands_on_partner_thighs_score": row.get("hands_on_partner_thighs_score"),
        "partner_leg_support_confidence": row.get("partner_leg_support_confidence"),
        "facing_confidence": row.get("facing_confidence"),
        "hip_motion_strength": row.get("hip_motion_strength"),
        "pelvis_trajectory_strength": row.get("pelvis_trajectory_strength"),
        "clean_motion_gate": row.get("clean_motion_gate"),
        "clean_motion_gate_reason": row.get("clean_motion_gate_reason"),
        "source_scene_file": row.get("source_scene_file") or window.get("source_scene_file"),
        "source_scene_path": row.get("source_scene_path") or window.get("source_scene_path"),
        "technical_atom_id": row.get("technical_actor_id") or window.get("technical_atom_id"),
        "sample_id": row.get("sample_id") or window.get("sample_id"),
        "start_seconds": window.get("start_seconds") or row.get("start_seconds"),
        "end_seconds": window.get("end_seconds") or row.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds") or row.get("duration_seconds"),
        "warnings": row.get("warnings") or [],
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _write_review_md(rows: list[dict[str, Any]], shortage: dict[str, int], path: Path) -> None:
    lines = ["# Lean-back Support Semantic Review", "", "Audit review only; not manual training labels.", "", "## Shortage", ""]
    lines.extend(f"- `{k}`: {v}" for k, v in shortage.items()) if shortage else lines.append("- None")
    for row in rows:
        lines.extend(
            [
                "",
                f"## {row['review_id']}",
                f"- Category: `{row.get('why_selected')}`",
                f"- Pose: `{(row.get('pose_semantics') or {}).get('subtype')}`",
                f"- Torso/facing: `{row.get('torso_lean_direction')}` / `{row.get('facing_context')}`",
                f"- Contact/support: `{row.get('contact_support')}`",
                f"- Hands behind score: `{row.get('hands_behind_support_score')}`",
                f"- Partner legs/thighs scores: `{row.get('hands_on_partner_legs_score')}` / `{row.get('hands_on_partner_thighs_score')}`",
                f"- Scene: `{row.get('source_scene_path') or row.get('source_scene_file')}`",
                f"- Time: `{row.get('start_seconds')}` - `{row.get('end_seconds')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_html(rows: list[dict[str, Any]], shortage: dict[str, int], path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"<section><h2>{html.escape(row['review_id'])}</h2>"
            f"<p><b>{html.escape(str(row.get('why_selected')))}</b> pose <code>{html.escape(str(row.get('pose_subtype')))}</code></p>"
            f"<p>torso <code>{html.escape(str(row.get('torso_lean_direction')))}</code>, facing <code>{html.escape(str(row.get('facing_context')))}</code>, support <code>{html.escape(str(row.get('contact_support')))}</code></p>"
            f"<p>hands behind <code>{html.escape(str(row.get('hands_behind_support_score')))}</code>, legs <code>{html.escape(str(row.get('hands_on_partner_legs_score')))}</code>, thighs <code>{html.escape(str(row.get('hands_on_partner_thighs_score')))}</code></p>"
            f"<p>{html.escape(str(row.get('source_scene_path') or row.get('source_scene_file')))} @ {html.escape(str(row.get('start_seconds')))}-{html.escape(str(row.get('end_seconds')))}s</p></section>"
        )
    body = "<!doctype html><meta charset='utf-8'><title>Lean-back Support Review</title><style>body{font-family:system-ui,Segoe UI,sans-serif;margin:24px;background:#f6f7f9}section{background:white;border:1px solid #ccd3dd;border-radius:8px;padding:16px;margin:14px 0}code{background:#eef2f7;padding:2px 5px;border-radius:4px}</style><h1>Lean-back Support Review</h1>"
    if shortage:
        body += "<h2>Shortage</h2><ul>" + "".join(f"<li><code>{html.escape(k)}</code>: {v}</li>" for k, v in shortage.items()) + "</ul>"
    body += "".join(cards)
    path.write_text(body, encoding="utf-8")


def _write_answer_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    data = {
        "reviews": {
            row["review_id"]: {
                "lean_back_supported_correct": "unknown",
                "facing_front_not_reverse_correct": "unknown",
                "hands_behind_support_correct": "unknown",
                "partner_legs_or_thighs_support_correct": "unknown",
                "cowgirl_family_correct": "unknown",
                "usable_for_future_generation": "unknown",
                "notes": "",
            }
            for row in rows
        }
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_action_report(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Semantic Actions V3 Report",
                "",
                "V3 adds Cowgirl lean-back supported pose/contact fields. These remain audit candidates, not training labels.",
                "",
                f"- Rows: {len(rows)}",
                "",
                "## Pose Subtypes",
                "",
                *_counter_lines(Counter(r.get("pose_subtype") for r in rows)),
                "",
                "## Contact Support",
                "",
                *_counter_lines(Counter(r.get("contact_support") for r in rows)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_db_report(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    lines = [f"# {title} Report", "", "Audit/candidate inventory only; not manual ground truth.", "", f"- Records: {len(rows)}", ""]
    if any("category" in r for r in rows):
        lines.extend(["## Categories", ""])
        counts = _category_counts(rows)
        lines.extend(f"- `{k}`: {v}" for k, v in counts.items())
    lines.extend(["", "## Pose Subtypes", ""])
    lines.extend(_counter_lines(Counter(r.get("pose_subtype") for r in rows)))
    lines.extend(["", "## Contact Support", ""])
    lines.extend(_counter_lines(Counter(r.get("contact_support") for r in rows)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pipeline_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# clean_v3 Pose Support Rescan Summary",
        "",
        "Focused audit layer for frontal Cowgirl lean-back supported pose. No ML training and no manual label merge were performed.",
        "",
        f"- Pose feature rows: {summary.get('pose_feature_rows')}",
        f"- Semantic actions v3: {summary.get('semantic_actions_v3')}",
        f"- Focused review: `{(summary.get('focused_review') or {}).get('review_dir')}`",
        "",
        "## Cowgirl DB V8 Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in (summary.get("cowgirl_candidate_db_v8_counts") or {}).items())
    lines.extend(["", "## Lean-back/Support Candidate Counts", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in (summary.get("lean_back_support_candidate_counts") or {}).items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _semantic_csv_fields() -> list[str]:
    return [
        "candidate_id",
        "window_id",
        "semantic_family",
        "pose_family",
        "pose_subtype",
        "torso_lean_direction",
        "facing_context",
        "motion_subtype",
        "phase",
        "clean_motion_gate",
        "contact_support",
        "generation_safe",
        "hands_behind_support_score",
        "hands_on_partner_legs_score",
        "hands_on_partner_thighs_score",
    ]


def _cowgirl_csv_fields() -> list[str]:
    return ["candidate_id", "window_id", "category"] + _semantic_csv_fields()[2:]


def _lean_back_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "cowgirl_lean_back_supported_pose_subtype": sum(1 for r in rows if r.get("pose_subtype") == "cowgirl_lean_back_supported"),
        "hands_behind_support": sum(1 for r in rows if r.get("contact_support") == "hands_behind_support"),
        "hands_on_partner_legs_or_thighs": sum(1 for r in rows if r.get("contact_support") == "hands_on_partner_legs_or_thighs"),
        "ambiguous_behind_support": sum(1 for r in rows if r.get("contact_support") == "ambiguous_behind_support"),
    }


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(r.get("category") for r in rows)
    counts = {category: int(counter.get(category, 0)) for category in V8_CATEGORIES}
    for category, value in counter.items():
        if category not in counts:
            counts[str(category)] = int(value)
    return counts


def _best_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(rows, key=lambda r: _num(r.get("interaction_confidence")), reverse=True)[0] if rows else {}


def _load_first(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def _category_rank(category: Any) -> int:
    try:
        return V8_CATEGORIES.index(str(category))
    except ValueError:
        return 50


def _counter_lines(counter: Counter[Any]) -> list[str]:
    return [f"- `{k}`: {v}" for k, v in counter.most_common()] if counter else ["- None"]


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out

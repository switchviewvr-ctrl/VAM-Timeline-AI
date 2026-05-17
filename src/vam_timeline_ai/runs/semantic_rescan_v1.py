"""One-command clean_v3 semantic rescan.

The rescan is intentionally conservative: it reuses clean_v2 technical artifacts
when schema-compatible, then rebuilds semantic layers around pose, motion,
partner relation, contact/support, and generation safety. It does not train ML.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import shutil

from vam_timeline_ai.datasets.cowgirl_candidate_database import build_cowgirl_candidate_db_v5
from vam_timeline_ai.datasets.semantic_candidate_database import build_semantic_candidate_db_from_actions_v0
from vam_timeline_ai.features.partner_relative_features import extract_partner_relative_features_v0
from vam_timeline_ai.features.pose_features import extract_pose_features_v0
from vam_timeline_ai.generation.baseline_pose import select_interaction_baseline_for_plan_v0
from vam_timeline_ai.generation.interaction_validation import validate_partner_relative_flow_v0
from vam_timeline_ai.generation.partner_relative_flow import synthesize_partner_relative_flow_v0
from vam_timeline_ai.generation.primitive_extractor import extract_cowgirl_motion_primitives_v1
from vam_timeline_ai.generation.primitive_groups import group_cowgirl_motion_primitives_v1
from vam_timeline_ai.generation.prompt_to_plan import draft_motion_plan_v1
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.semantics.interaction_classifier import classify_interactions_v0
from vam_timeline_ai.semantics.pose_classifier import classify_poses_v0
from vam_timeline_ai.semantics.semantic_action import build_semantic_actions_v0


RUN_DIRS = [
    "audits",
    "baked",
    "semantic",
    "features",
    "pose_semantics",
    "interaction_semantics",
    "semantic_actions",
    "references",
    "datasets",
    "generation",
    "labels",
    "reports",
    "relative_motion",
]


ARTIFACTS = [
    ("semantic/motion_source_index.jsonl", "semantic/motion_source_index.jsonl"),
    ("baked/motion_sample_index.jsonl", "baked/motion_sample_index.jsonl"),
    ("semantic/movement_windows.jsonl", "semantic/movement_windows.jsonl"),
    ("semantic/pair_windows_v1.jsonl", "semantic/pair_windows_v1.jsonl"),
    ("features/cowgirl_pair_features_v0.jsonl", "features/cowgirl_pair_features_v0.jsonl"),
    ("semantic/controller_bodypart_map.json", "semantic/controller_bodypart_map.json"),
    ("audits/body_motion_quality.jsonl", "audits/body_motion_quality.jsonl"),
    ("audits/pose_anchor_completeness.jsonl", "audits/pose_anchor_completeness.jsonl"),
    ("audits/controller_validity.jsonl", "audits/controller_validity.jsonl"),
    ("datasets/cowgirl_candidate_db_v3.jsonl", "datasets/cowgirl_candidate_db_v3.jsonl"),
    ("relative_motion/relative_motion_window_index.jsonl", "relative_motion/relative_motion_window_index.jsonl"),
    ("relative_motion/relative_motion_features.jsonl", "relative_motion/relative_motion_features.jsonl"),
    ("relative_motion/trajectory_shape_features.jsonl", "relative_motion/trajectory_shape_features.jsonl"),
    ("relative_motion/relative_reference_matches.jsonl", "relative_motion/relative_reference_matches.jsonl"),
]


def run_semantic_rescan_v1(source_run: str | Path, out_run: str | Path) -> dict[str, Any]:
    source = Path(source_run)
    out = Path(out_run)
    _create_structure(source, out)
    reuse = _reuse_technical_artifacts(source, out)
    _write_relative_reports(out)

    pose_features = out / "pose_semantics" / "pose_features_v0.jsonl"
    pose_feature_report = out / "pose_semantics" / "pose_feature_report_v0.md"
    extract_pose_features_v0(
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        out / "audits" / "body_motion_quality.jsonl",
        out / "audits" / "pose_anchor_completeness.jsonl",
        out / "audits" / "controller_validity.jsonl",
        pose_features,
        pose_feature_report,
    )
    pose_semantics = out / "pose_semantics" / "pose_semantics_v0.jsonl"
    classify_poses_v0(
        pose_features,
        out / "relative_motion" / "relative_reference_matches.jsonl",
        out / "references" / "handmade_animations" / "handmade_relative_features.jsonl",
        pose_semantics,
        out / "pose_semantics" / "pose_semantics_report_v0.md",
    )
    partner_features = out / "interaction_semantics" / "partner_relative_features_v0.jsonl"
    extract_partner_relative_features_v0(
        out / "semantic" / "pair_windows_v1.jsonl",
        out / "features" / "cowgirl_pair_features_v0.jsonl",
        out / "relative_motion" / "relative_motion_window_index.jsonl",
        pose_semantics,
        partner_features,
        out / "interaction_semantics" / "partner_relative_feature_report_v0.md",
    )
    interaction_semantics = out / "interaction_semantics" / "interaction_semantics_v0.jsonl"
    classify_interactions_v0(
        partner_features,
        pose_semantics,
        out / "semantic_actions" / "semantic_actions_v0.jsonl",
        interaction_semantics,
        out / "interaction_semantics" / "interaction_semantics_report_v0.md",
    )
    semantic_actions = out / "semantic_actions" / "semantic_actions_v0.jsonl"
    build_semantic_actions_v0(
        out / "datasets" / "cowgirl_candidate_db_v3.jsonl",
        pose_semantics,
        out / "relative_motion" / "relative_reference_matches.jsonl",
        interaction_semantics,
        semantic_actions,
        out / "semantic_actions" / "semantic_actions_report_v0.md",
    )
    semantic_db = out / "datasets" / "semantic_candidate_db_v0.jsonl"
    build_semantic_candidate_db_from_actions_v0(
        semantic_actions,
        semantic_db,
        out / "datasets" / "semantic_candidate_db_v0.csv",
        out / "datasets" / "semantic_candidate_db_v0_report.md",
    )
    cowgirl_v5 = out / "datasets" / "cowgirl_candidate_db_v5.jsonl"
    build_cowgirl_candidate_db_v5(
        semantic_db,
        cowgirl_v5,
        out / "datasets" / "cowgirl_candidate_db_v5.csv",
        out / "datasets" / "cowgirl_candidate_db_v5_report.md",
    )
    primitives = out / "generation" / "cowgirl_motion_primitives_v1.jsonl"
    extract_cowgirl_motion_primitives_v1(
        cowgirl_v5,
        out / "relative_motion" / "relative_motion_features.jsonl",
        out / "relative_motion" / "trajectory_shape_features.jsonl",
        pose_semantics,
        interaction_semantics,
        primitives,
        out / "generation" / "cowgirl_motion_primitives_v1_report.md",
    )
    primitive_groups = out / "generation" / "cowgirl_motion_primitive_groups_v1.json"
    group_cowgirl_motion_primitives_v1(primitives, primitive_groups, out / "generation" / "cowgirl_motion_primitive_groups_v1_report.md")

    plan_path = out / "generation" / "draft_motion_plan_v1.json"
    draft_motion_plan_v1("slow cowgirl grinding, leaning forward, hands on partner chest", plan_path)
    baseline_path = out / "generation" / "selected_interaction_baseline_v0.json"
    select_interaction_baseline_for_plan_v0(plan_path, baseline_path)
    flow_path = out / "generation" / "partner_relative_flow_v0.json"
    synthesize_partner_relative_flow_v0(plan_path, primitive_groups, baseline_path, flow_path, out / "generation" / "partner_relative_flow_v0_report.md")
    validation_report = out / "generation" / "partner_relative_flow_v0_validation.md"
    validation = validate_partner_relative_flow_v0(flow_path, validation_report)
    review_dir = out / "audits" / "semantic_review_010_v15"
    review = _export_review_v15(semantic_db, cowgirl_v5, review_dir)

    summary = _summary(out, reuse, validation, review)
    dump_json(out / "reports" / "semantic_rescan_v1_summary.json", summary)
    _write_summary_markdown(summary, out / "reports" / "semantic_rescan_v1_summary.md")
    return summary


def _create_structure(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in RUN_DIRS:
        (out / name).mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": out.name,
        "source_run": str(source),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "semantic_rescan_pose_motion_partner_interaction",
        "inherited_technical_artifacts": [dst for _, dst in ARTIFACTS],
        "regenerated_semantic_artifacts": [
            "pose_semantics/pose_features_v0.jsonl",
            "pose_semantics/pose_semantics_v0.jsonl",
            "interaction_semantics/partner_relative_features_v0.jsonl",
            "interaction_semantics/interaction_semantics_v0.jsonl",
            "semantic_actions/semantic_actions_v0.jsonl",
            "datasets/semantic_candidate_db_v0.jsonl",
            "datasets/cowgirl_candidate_db_v5.jsonl",
        ],
        "constraints": {
            "manual_labels_modified": False,
            "ml_training_performed": False,
            "source_world_coords_as_learning_targets": False,
            "person_root_tracks_allowed": False,
        },
    }
    dump_json(out / "run_manifest.json", manifest)


def _reuse_technical_artifacts(source: Path, out: Path) -> dict[str, Any]:
    copied: list[str] = []
    missing: list[str] = []
    for src_rel, dst_rel in ARTIFACTS:
        src = source / src_rel
        dst = out / dst_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst_rel)
        else:
            missing.append(src_rel)
    refs = source / "references"
    if refs.exists():
        dst_refs = out / "references"
        for item in refs.rglob("*"):
            if item.is_file():
                target = dst_refs / item.relative_to(refs)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    report = out / "reports" / "technical_artifact_reuse_report.md"
    lines = [
        "# Technical Artifact Reuse Report",
        "",
        "clean_v3 reuses schema-compatible clean_v2 technical artifacts and rebuilds semantic artifacts around them.",
        "",
        "## Copied/Referenced",
        "",
    ]
    lines.extend(f"- `{x}`" for x in copied) if copied else lines.append("- None")
    lines.extend(["", "## Missing", ""])
    lines.extend(f"- `{x}`" for x in missing) if missing else lines.append("- None")
    lines.extend(["", "## Safety", "", "- manual_labels.yaml was not read as training truth or modified.", "- Relative motion artifacts remain relative/local representations; source world tracks are not generation targets."])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"copied": copied, "missing": missing, "report": str(report)}


def _write_relative_reports(out: Path) -> None:
    rel_count = len(load_jsonl(out / "relative_motion" / "relative_motion_window_index.jsonl"))
    feat_count = len(load_jsonl(out / "relative_motion" / "relative_motion_features.jsonl"))
    traj_count = len(load_jsonl(out / "relative_motion" / "trajectory_shape_features.jsonl"))
    (out / "relative_motion" / "relative_motion_report.md").write_text(
        "# Relative Motion Report\n\n"
        f"- Windows: {rel_count}\n"
        "- Source: reused clean_v2 schema-compatible relative body motion artifacts.\n"
        "- Person/root/world tracks are not generation targets.\n",
        encoding="utf-8",
    )
    (out / "relative_motion" / "relative_motion_feature_report.md").write_text(
        "# Relative Motion Feature Report\n\n"
        f"- Feature rows: {feat_count}\n"
        "- Features are relative/local motion proxies.\n",
        encoding="utf-8",
    )
    (out / "relative_motion" / "trajectory_shape_report.md").write_text(
        "# Trajectory Shape Report\n\n"
        f"- Trajectory rows: {traj_count}\n"
        "- Trajectory shapes are reused for clean_v3 semantic action composition.\n",
        encoding="utf-8",
    )


def _export_review_v15(semantic_db: Path, cowgirl_v5: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    semantic = load_jsonl(semantic_db)
    cowgirl = load_jsonl(cowgirl_v5)
    selected: list[dict[str, Any]] = []
    selected.extend(_take(cowgirl, lambda r: r.get("category") == "cowgirl_clean_motion_generation_safe", 4))
    selected.extend(_take(cowgirl, lambda r: r.get("category") == "cowgirl_hands_on_partner_chest", 2))
    selected.extend(_take(semantic, lambda r: r.get("semantic_family") == "bj_oral", 1))
    selected.extend(_take(semantic, lambda r: r.get("semantic_family") == "receiver_response", 1))
    selected.extend(_take(semantic, lambda r: r.get("semantic_family") in {"hand_gesture", "head_gesture"}, 1))
    selected.extend(_take(semantic, lambda r: r.get("semantic_family") == "unknown", 1))
    seen: set[str] = set()
    review_rows: list[dict[str, Any]] = []
    for row in selected:
        wid = str(row.get("window_id"))
        if wid in seen:
            continue
        seen.add(wid)
        review_rows.append(_review_item(len(review_rows) + 1, row))
        if len(review_rows) >= 10:
            break
    for row in semantic:
        if len(review_rows) >= 10:
            break
        wid = str(row.get("window_id"))
        if wid not in seen:
            seen.add(wid)
            review_rows.append(_review_item(len(review_rows) + 1, row))
    write_jsonl(out_dir / "semantic_review_010.jsonl", review_rows)
    answer_lines = [
        "# Audit answer sheet for semantic_review_010_v15.",
        "# These are placeholders for human review notes, not training labels.",
        "",
    ]
    for row in review_rows:
        answer_lines.extend([
            f"{row['review_id']}:",
            "  user_verdict: null",
            "  actual_labels: []",
            "  notes: null",
        ])
    (out_dir / "semantic_review_010_answer_sheet.yaml").write_text("\n".join(answer_lines) + "\n", encoding="utf-8")
    (out_dir / "semantic_review_010_human_summary.md").write_text("# Semantic Review 010 V15 Human Summary\n\nPending manual VaM review.\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "count": len(review_rows), "items": review_rows}


def _take(rows: list[dict[str, Any]], pred: Any, count: int) -> list[dict[str, Any]]:
    return [r for r in rows if pred(r)][:count]


def _review_item(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": f"review_{index:03d}",
        "window_id": row.get("window_id"),
        "pair_window_id": row.get("pair_window_id"),
        "semantic_family": row.get("semantic_family"),
        "pose_semantics": {"family": row.get("pose_family"), "subtype": row.get("pose_subtype")},
        "motion_semantics": {"subtype": row.get("motion_subtype")},
        "partner_relation": row.get("partner_relation"),
        "contact_support": row.get("contact_support"),
        "generation_safe": row.get("generation_safe"),
        "why_selected": row.get("category") or row.get("semantic_family"),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _summary(out: Path, reuse: dict[str, Any], validation: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    pose_rows = load_jsonl(out / "pose_semantics" / "pose_semantics_v0.jsonl")
    inter_rows = load_jsonl(out / "interaction_semantics" / "interaction_semantics_v0.jsonl")
    action_rows = load_jsonl(out / "semantic_actions" / "semantic_actions_v0.jsonl")
    semantic_rows = load_jsonl(out / "datasets" / "semantic_candidate_db_v0.jsonl")
    cowgirl_rows = load_jsonl(out / "datasets" / "cowgirl_candidate_db_v5.jsonl")
    primitive_rows = load_jsonl(out / "generation" / "cowgirl_motion_primitives_v1.jsonl")
    return {
        "run_dir": str(out),
        "technical_reuse": reuse,
        "pose_semantic_counts": dict(Counter(r.get("pose_family") for r in pose_rows)),
        "interaction_semantic_counts": dict(Counter(r.get("interaction_family") for r in inter_rows)),
        "semantic_action_counts": dict(Counter(r.get("semantic_family") for r in action_rows)),
        "semantic_candidate_family_counts": dict(Counter(r.get("semantic_family") for r in semantic_rows)),
        "cowgirl_db_v5_counts": dict(Counter(r.get("category") for r in cowgirl_rows)),
        "motion_primitive_v1_count": len(primitive_rows),
        "partner_relative_flow_validation_passed": validation.get("passed"),
        "review_v15": {"out_dir": review.get("out_dir"), "count": review.get("count")},
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def _write_summary_markdown(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# Semantic Rescan V1 Summary",
        "",
        f"- Run dir: `{summary.get('run_dir')}`",
        f"- Motion primitive v1 count: {summary.get('motion_primitive_v1_count')}",
        f"- Partner-relative validation passed: {summary.get('partner_relative_flow_validation_passed')}",
        f"- Review v15: `{(summary.get('review_v15') or {}).get('out_dir')}` ({(summary.get('review_v15') or {}).get('count')} items)",
        "- manual_labels.yaml modified: false",
        "- ML training performed: false",
        "",
        "## Pose Semantic Counts",
        "",
    ]
    for k, v in (summary.get("pose_semantic_counts") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Interaction Semantic Counts", ""])
    for k, v in (summary.get("interaction_semantic_counts") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Semantic Candidate Families", ""])
    for k, v in (summary.get("semantic_candidate_family_counts") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## Cowgirl DB V5 Categories", ""])
    for k, v in (summary.get("cowgirl_db_v5_counts") or {}).items():
        lines.append(f"- `{k}`: {v}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

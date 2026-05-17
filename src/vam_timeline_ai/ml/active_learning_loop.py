"""Cowgirl ML active-learning loop v2."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from vam_timeline_ai.audits.ml_assisted_review_batch import export_ml_assisted_cowgirl_review_v1
from vam_timeline_ai.audits.review_answer_ingestion import ingest_review_ui_answers
from vam_timeline_ai.audits.review_deduplication import audit_review_duplicates, build_reviewed_window_index
from vam_timeline_ai.ml.cowgirl_baseline_model import train_cowgirl_ml_baseline_v1
from vam_timeline_ai.ml.cowgirl_model_scoring import score_clean_v3_with_cowgirl_model_v1
from vam_timeline_ai.ml.grouped_splits import split_cowgirl_ml_dataset_v1
from vam_timeline_ai.ml.human_label_dataset import build_human_reviewed_ml_labels_v1
from vam_timeline_ai.ml.ml_review_evaluation import evaluate_ml_assisted_review_v1
from vam_timeline_ai.ml.supervised_feature_table import build_cowgirl_ml_feature_table_v1


def run_cowgirl_ml_active_learning_v2(run_dir: str | Path, review_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    review = Path(review_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    answers = _find_answers(review)
    if answers is None:
        summary = _blocked_missing_answers(review, out)
        return summary

    canonical_answers = review / "human_review_ui_answers.jsonl"
    if answers.suffix.lower() == ".jsonl" and answers != canonical_answers:
        shutil.copy2(answers, canonical_answers)
        answers = canonical_answers

    ingestion = ingest_review_ui_answers(
        answers,
        review,
        run / "audits" / "human_review_ledger.jsonl",
        review / "review_ui_answer_ingestion_report.md",
        overwrite=False,
    )
    evaluation = evaluate_ml_assisted_review_v1(
        review,
        run / "ml" / "scores" / "cowgirl_model_scores_v1.jsonl",
        answers,
        run / "ml" / "reports" / "ml_assisted_cowgirl_review_v1_evaluation.md",
    )
    labels_v2 = build_human_reviewed_ml_labels_v1(
        run,
        run / "audits" / "human_review_ledger.jsonl",
        run / "ml" / "human_labels" / "human_reviewed_labels_v2.jsonl",
        run / "ml" / "human_labels" / "human_reviewed_labels_v2_report.md",
    )
    labels_v1 = _load_json_if_report(run / "ml" / "human_labels" / "human_reviewed_labels_v1_report.md")
    feature_table = build_cowgirl_ml_feature_table_v1(
        run,
        run / "ml" / "human_labels" / "human_reviewed_labels_v2.jsonl",
        run / "relative_motion" / "relative_motion_features.jsonl",
        run / "relative_motion" / "trajectory_shape_features.jsonl",
        run / "pose_semantics" / "pose_features_v0.jsonl",
        run / "pose_semantics" / "pose_semantics_v0.jsonl",
        run / "interaction_semantics" / "partner_relative_features_v0.jsonl",
        run / "interaction_semantics" / "interaction_semantics_v0.jsonl",
        _latest_existing(run / "semantic_actions", ["semantic_actions_v3.jsonl", "semantic_actions_v2.jsonl", "semantic_actions_v1.jsonl", "semantic_actions_v0.jsonl"]),
        _latest_existing(run / "datasets", ["cowgirl_candidate_db_v8.jsonl", "cowgirl_candidate_db_v7.jsonl", "cowgirl_candidate_db_v6.jsonl", "cowgirl_candidate_db_v5.jsonl"]),
        run / "ml" / "datasets" / "cowgirl_ml_feature_table_v2.npz",
        run / "ml" / "datasets" / "cowgirl_ml_feature_table_v2_metadata.jsonl",
        run / "ml" / "reports" / "cowgirl_ml_feature_table_v2_report.md",
    )
    split = split_cowgirl_ml_dataset_v1(
        run / "ml" / "datasets" / "cowgirl_ml_feature_table_v2.npz",
        run / "ml" / "datasets" / "cowgirl_ml_feature_table_v2_metadata.jsonl",
        run / "ml" / "splits" / "cowgirl_v2",
        "source_scene_file",
        42,
    )
    model = train_cowgirl_ml_baseline_v1(
        run / "ml" / "datasets" / "cowgirl_ml_feature_table_v2.npz",
        run / "ml" / "datasets" / "cowgirl_ml_feature_table_v2_metadata.jsonl",
        run / "ml" / "splits" / "cowgirl_v2",
        run / "ml" / "models" / "cowgirl_baseline_v2",
    )
    scores = score_clean_v3_with_cowgirl_model_v1(
        run,
        run / "ml" / "models" / "cowgirl_baseline_v2",
        "all_candidates",
        run / "ml" / "scores" / "cowgirl_model_scores_v2.jsonl",
        run / "ml" / "reports" / "cowgirl_model_scores_v2_report.md",
    )
    _augment_v2_scores_with_v1(run / "ml" / "scores" / "cowgirl_model_scores_v1.jsonl", run / "ml" / "scores" / "cowgirl_model_scores_v2.jsonl")
    reviewed_index = build_reviewed_window_index(
        run,
        [Path("data/runs/clean_v2"), run, Path("data/runs/clean_v3_new_scenes")],
        run / "audits" / "reviewed_window_index.jsonl",
        run / "audits" / "reviewed_window_index.csv",
        run / "audits" / "reviewed_window_index_report.md",
    )
    duplicate_audit = audit_review_duplicates(run / "audits" / "reviewed_window_index.jsonl", run / "audits" / "review_duplicate_audit_report.md")
    review_v2 = export_ml_assisted_cowgirl_review_v1(
        run,
        run / "ml" / "scores" / "cowgirl_model_scores_v2.jsonl",
        run / "audits" / "reviewed_window_index.jsonl",
        run / "audits" / "ml_assisted_cowgirl_review_v2",
        count=30,
        max_per_scene=2,
        max_per_sample=1,
        build_vam_package=True,
        build_static_ui=True,
    )
    summary = {
        "status": "ok",
        "answers_found": str(answers),
        "ingestion": ingestion,
        "evaluation": {k: v for k, v in evaluation.items() if k != "rows"},
        "labels_v2": labels_v2,
        "feature_table_v2": feature_table,
        "split_v2": split,
        "model_v2": {"trained": model.get("trained"), "model_type": model.get("model_type"), "metrics": model.get("metrics")},
        "scores_v2": scores,
        "reviewed_index": reviewed_index,
        "duplicate_audit": duplicate_audit,
        "review_v2": review_v2,
        "trust_verdict": "review ranking only; no auto-labels; no generation-safe automation",
        "manual_labels_modified": False,
        "generation_performed": False,
    }
    _write_summary(out / "active_learning_v2_summary.md", summary)
    (out / "active_learning_v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _find_answers(review: Path) -> Path | None:
    candidates = [
        review / "human_review_ui_answers.jsonl",
        review / "human_review_ui_answers.yaml",
        review / "human_review_ui_answers.yml",
    ]
    candidates.extend(sorted((review / "review_ui_static" / "downloads").glob("human_review_ui_answers*.jsonl")) if (review / "review_ui_static" / "downloads").exists() else [])
    candidates.extend(sorted((review / "review_ui_static").glob("human_review_ui_answers*.jsonl")))
    for path in candidates:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _blocked_missing_answers(review: Path, out: Path) -> dict[str, Any]:
    text = f"""# BLOCKED: Missing ML-assisted Review Answers

No human answer file was found for:

`{review}`

No training was performed.

## What to do

1. Open `{review / "review_ui_static" / "index.html"}`.
2. Review the 20 items.
3. Export answers as JSONL.
4. Save the file as `{review / "human_review_ui_answers.jsonl"}`.
5. Rerun `python -m vam_timeline_ai.cli run-cowgirl-ml-active-learning-v2 --run-dir data\\runs\\clean_v3 --review-dir data\\runs\\clean_v3\\audits\\ml_assisted_cowgirl_review_v1 --out-dir data\\runs\\clean_v3\\ml\\active_learning\\cowgirl_v2`.

This loop will not use model predictions as truth and will not modify `manual_labels.yaml`.
"""
    (out / "BLOCKED_MISSING_REVIEW_ANSWERS.md").write_text(text, encoding="utf-8")
    summary = {
        "status": "blocked_missing_review_answers",
        "answers_found": False,
        "training_performed": False,
        "generation_performed": False,
        "manual_labels_modified": False,
        "blocked_report": str(out / "BLOCKED_MISSING_REVIEW_ANSWERS.md"),
    }
    (out / "active_learning_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _augment_v2_scores_with_v1(v1_path: Path, v2_path: Path) -> None:
    if not v1_path.exists() or not v2_path.exists():
        return
    from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl

    v1 = {str(r.get("window_id")): r for r in load_jsonl(v1_path)}
    rows = []
    for row in load_jsonl(v2_path):
        old = v1.get(str(row.get("window_id") or "")) or {}
        for key in ["model_cowgirl_probability", "model_clean_motion_probability", "model_generation_safe_probability"]:
            old_key = "v1_" + key
            row[old_key] = old.get(key)
            if old.get(key) is not None and row.get(key) is not None:
                row[key + "_change_from_v1"] = float(row[key]) - float(old[key])
        rows.append(row)
    write_jsonl(v2_path, rows)


def _latest_existing(folder: Path, names: list[str]) -> Path:
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return folder / names[-1]


def _load_json_if_report(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Cowgirl ML Active Learning v2 Summary",
        "",
        f"- Answers found: `{summary.get('answers_found')}`",
        f"- Ingestion: `{summary.get('ingestion')}`",
        f"- ML v1 helped: `{(summary.get('evaluation') or {}).get('ml_v1_helped')}`",
        f"- Labels v2 Cowgirl counts: `{(summary.get('labels_v2') or {}).get('cowgirl_label_counts')}`",
        f"- Feature table v2 shape: `{(summary.get('feature_table_v2') or {}).get('shape')}`",
        f"- Model v2 trained: `{(summary.get('model_v2') or {}).get('trained')}`",
        f"- Model v2 type: `{(summary.get('model_v2') or {}).get('model_type')}`",
        f"- Scores v2: `{(summary.get('scores_v2') or {}).get('priority_counts')}`",
        f"- Review v2 path: `{Path('data/runs/clean_v3/audits/ml_assisted_cowgirl_review_v2')}`",
        "",
        "## Trust Verdict",
        "",
        summary.get("trust_verdict", "review ranking only"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

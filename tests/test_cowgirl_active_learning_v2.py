from __future__ import annotations

import json
from pathlib import Path

from vam_timeline_ai.audits.review_answer_ingestion import ingest_review_ui_answers
from vam_timeline_ai.ml.active_learning_loop import run_cowgirl_ml_active_learning_v2
from vam_timeline_ai.ml.ml_review_evaluation import evaluate_ml_assisted_review_v1


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_active_loop_blocks_when_answers_missing(tmp_path: Path) -> None:
    review = tmp_path / "audits" / "ml_assisted_cowgirl_review_v1"
    (review / "review_ui_static").mkdir(parents=True)
    summary = run_cowgirl_ml_active_learning_v2(tmp_path, review, tmp_path / "ml" / "active_learning" / "cowgirl_v2")
    assert summary["status"] == "blocked_missing_review_answers"
    assert (tmp_path / "ml" / "active_learning" / "cowgirl_v2" / "BLOCKED_MISSING_REVIEW_ANSWERS.md").exists()


def test_answer_ingestion_idempotent(tmp_path: Path) -> None:
    review = tmp_path / "review"
    _write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "source_scene_file": "s.json", "window_id": "w1"}])
    answers = tmp_path / "answers.jsonl"
    _write_jsonl(answers, [{"review_id": "review_001", "review_labels": ["correct_clean_cowgirl_motion"], "notes": "ok"}])
    ledger = tmp_path / "ledger.jsonl"
    first = ingest_review_ui_answers(answers, review, ledger, tmp_path / "report1.md")
    second = ingest_review_ui_answers(answers, review, ledger, tmp_path / "report2.md")
    assert first["new_ledger_records"] == 1
    assert second["new_ledger_records"] == 0
    assert second["duplicates_skipped"] == 1


def test_ml_review_evaluation_groups_by_bucket(tmp_path: Path) -> None:
    review = tmp_path / "review"
    _write_jsonl(
        review / "semantic_review_010.jsonl",
        [{"review_id": "review_001", "window_id": "w1", "recommended_review_priority": "high_confidence_cowgirl", "model_cowgirl_probability": 0.9, "category": "cowgirl_clean_motion_generation_safe"}],
    )
    _write_jsonl(tmp_path / "scores.jsonl", [{"window_id": "w1", "model_cowgirl_probability": 0.9}])
    _write_jsonl(tmp_path / "answers.jsonl", [{"review_id": "review_001", "review_labels": ["correct_clean_cowgirl_motion"], "notes": "cowgirl"}])
    summary = evaluate_ml_assisted_review_v1(review, tmp_path / "scores.jsonl", tmp_path / "answers.jsonl", tmp_path / "eval.md")
    assert summary["bucket_stats"]["high_confidence_cowgirl"]["item_count"] == 1
    assert summary["bucket_stats"]["high_confidence_cowgirl"]["model_correct_count"] == 1


def test_no_manual_labels_written(tmp_path: Path) -> None:
    review = tmp_path / "review"
    review.mkdir()
    run_cowgirl_ml_active_learning_v2(tmp_path, review, tmp_path / "out")
    assert not (tmp_path / "manual_labels.yaml").exists()

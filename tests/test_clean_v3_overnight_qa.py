from pathlib import Path

from vam_timeline_ai.audits.human_review_memory import build_human_review_ledger
from vam_timeline_ai.audits.review_batch_planner import plan_larger_review_batch_v1
from vam_timeline_ai.datasets.db_invariant_validator import validate_semantic_dbs
from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.reports.clean_v3_status import clean_v3_status
from vam_timeline_ai.reports.prompt_capability_matrix import write_prompt_capability_matrix
from vam_timeline_ai.reports.run_drift_report import compare_clean_v2_clean_v3
from vam_timeline_ai.reports.semantic_qa_dashboard import write_clean_v3_dashboard


def test_human_review_ledger_handles_missing_reviews(tmp_path):
    run = tmp_path / "data" / "runs" / "clean_v3"
    (run / "audits").mkdir(parents=True)

    summary = build_human_review_ledger(
        run,
        str(run),
        run / "audits" / "human_review_ledger.jsonl",
        run / "audits" / "human_review_ledger.csv",
        run / "audits" / "human_review_ledger_report.md",
    )

    assert summary["records"] == 0
    assert (run / "audits" / "human_review_ledger_report.md").exists()


def test_invariant_validator_catches_bad_generation_safe_records(tmp_path):
    run = tmp_path / "clean_v3"
    (run / "datasets").mkdir(parents=True)
    sem = run / "datasets" / "semantic_candidate_db_v1.jsonl"
    cow = run / "datasets" / "cowgirl_candidate_db_v6.jsonl"
    write_jsonl(sem, [{"candidate_id": "bad_unknown", "window_id": "w1", "semantic_family": "unknown", "generation_safe": True}])
    write_jsonl(cow, [{"candidate_id": "bad_cow", "window_id": "w2", "category": "cowgirl_clean_motion_generation_safe", "semantic_family": "bj_oral", "generation_safe": True}])

    summary = validate_semantic_dbs(run, sem, cow, run / "reports" / "semantic_db_invariant_report.md")

    assert summary["errors"] >= 2
    text = (run / "reports" / "semantic_db_invariant_report.md").read_text(encoding="utf-8")
    assert "unknown record marked generation_safe" in text
    assert "clean generation-safe category is not semantic_family cowgirl" in text


def test_dashboard_writes_report_with_missing_inputs(tmp_path):
    run = tmp_path / "clean_v3"
    run.mkdir()

    summary = write_clean_v3_dashboard(run, run / "reports" / "dashboard.md", run / "reports" / "dashboard.html")

    assert summary["semantic_records"] == 0
    assert (run / "reports" / "dashboard.md").exists()
    assert (run / "reports" / "dashboard.html").exists()


def test_drift_report_handles_missing_clean_v2_db(tmp_path):
    v2 = tmp_path / "clean_v2"
    v3 = tmp_path / "clean_v3"
    v2.mkdir()
    v3.mkdir()

    summary = compare_clean_v2_clean_v3(v2, v3, v3 / "reports" / "drift.md")

    assert summary["v2_cowgirl_records"] == 0
    assert "Missing or unavailable" in (v3 / "reports" / "drift.md").read_text(encoding="utf-8")


def test_review_planner_enforces_scene_and_sample_caps(tmp_path):
    run = tmp_path / "clean_v3"
    (run / "datasets").mkdir(parents=True)
    sem = run / "datasets" / "semantic_candidate_db_v1.jsonl"
    cow = run / "datasets" / "cowgirl_candidate_db_v6.jsonl"
    rows = [
        {
            "window_id": f"w{i}",
            "sample_id": f"s{i}",
            "source_scene_file": "one_scene.json",
            "category": "cowgirl_clean_motion_generation_safe",
            "semantic_family": "cowgirl",
            "semantic_score": 1.0 - i * 0.01,
        }
        for i in range(10)
    ]
    write_jsonl(cow, rows)
    write_jsonl(sem, rows)

    summary = plan_larger_review_batch_v1(run, sem, cow, run / "reports" / "larger_review_batch_plan_v1.md")

    assert summary["selected_counts"]["cowgirl_clean_motion_generation_safe"] == 5
    assert (run / "reports" / "larger_review_batch_plan_v1.md").exists()


def test_prompt_capability_matrix_does_not_overclaim_timeline_readiness(tmp_path):
    run = tmp_path / "clean_v3"
    run.mkdir()

    write_prompt_capability_matrix(run, run / "reports" / "prompt_capability_matrix.md")
    text = (run / "reports" / "prompt_capability_matrix.md").read_text(encoding="utf-8")

    assert "not final" in text
    assert "not ready" in text


def test_clean_v3_status_works_with_missing_optional_artifacts(tmp_path):
    run = tmp_path / "clean_v3"
    run.mkdir()

    summary = clean_v3_status(run)

    assert summary["run_exists"] is True
    assert (run / "reports" / "clean_v3_status.md").exists()
    assert not (run / "labels" / "manual_labels.yaml").exists()

from pathlib import Path

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.reports.candidate_lineage import write_candidate_lineage_report
from vam_timeline_ai.reports.reproducibility_audit import (
    run_clean_v3_reproducibility_audit,
    write_artifact_manifest,
    write_deprecated_artifacts_report,
    write_morning_checklist,
    write_real_generation_input_requirements,
    write_repo_snapshot_report,
    write_schema_registry,
)


def test_schema_registry_writes_required_schemas(tmp_path):
    run = tmp_path / "clean_v3"
    run.mkdir()

    summary = write_schema_registry(run)

    assert summary["schemas"] >= 10
    text = (run / "reports" / "schema_registry.md").read_text(encoding="utf-8")
    assert "pose_semantics_v0" in text
    assert "cowgirl_candidate_db_v6" in text
    assert "not manual ground truth" in text


def test_artifact_manifest_handles_missing_files(tmp_path):
    run = tmp_path / "clean_v3"
    run.mkdir()

    summary = write_artifact_manifest(run)
    rows = load_jsonl(run / "reports" / "artifact_manifest.jsonl")

    assert summary["artifacts"] == len(rows)
    assert any(row["status"] == "missing" for row in rows)
    assert (run / "reports" / "artifact_manifest.md").exists()


def test_lineage_report_detects_orphan_synthetic_records(tmp_path):
    run = tmp_path / "clean_v3"
    (run / "relative_motion").mkdir(parents=True)
    write_jsonl(run / "relative_motion" / "relative_motion_window_index.jsonl", [{"window_id": "orphan_window", "sample_id": "sample_1"}])

    summary = write_candidate_lineage_report(run, run / "reports" / "candidate_lineage_report.md")
    text = (run / "reports" / "candidate_lineage_report.md").read_text(encoding="utf-8")

    assert summary["orphan_windows"] == 1
    assert "orphan_window" in text


def test_deprecated_artifact_report_marks_review_only_timeline_schema(tmp_path):
    project = tmp_path / "project"
    run = project / "data" / "runs" / "clean_v3"
    (run / "reports").mkdir(parents=True)

    write_deprecated_artifacts_report(run)
    text = (run / "reports" / "deprecated_artifacts_report.md").read_text(encoding="utf-8")

    assert "review_only_timeline_v0 custom schema" in text
    assert "not native Timeline JSON" in text


def test_real_generation_input_spec_mentions_partner_references(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    run = project / "data" / "runs" / "clean_v3"
    run.mkdir(parents=True)

    summary = write_real_generation_input_requirements(run)
    text = Path(summary["report"]).read_text(encoding="utf-8")

    assert "Partner chest reference" in text
    assert "Partner pelvis reference" in text
    assert "synthetic review timelines" in text


def test_morning_checklist_exists(tmp_path):
    run = tmp_path / "clean_v3"
    run.mkdir()

    summary = write_morning_checklist(run)

    assert Path(summary["out"]).exists()
    assert "Do not train ML yet" in Path(summary["out"]).read_text(encoding="utf-8")


def test_repo_snapshot_handles_non_git_directory_gracefully(tmp_path):
    run = tmp_path / "clean_v3"
    run.mkdir()

    summary = write_repo_snapshot_report(run)

    assert Path(summary["out"]).exists()
    assert "Git available" in Path(summary["out"]).read_text(encoding="utf-8")


def test_reproducibility_audit_orchestrator_writes_summary(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    run = project / "data" / "runs" / "clean_v3"
    (run / "reports").mkdir(parents=True)

    summary = run_clean_v3_reproducibility_audit(run)

    assert Path(summary["summary"]).exists()
    assert (run / "reports" / "schema_registry.md").exists()
    assert not (run / "labels" / "manual_labels.yaml").exists()
    assert not (run / "ml").exists()

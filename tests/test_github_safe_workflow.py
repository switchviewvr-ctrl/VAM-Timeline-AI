import json
from pathlib import Path

import yaml

from vam_timeline_ai.audits.local_status import write_local_status
from vam_timeline_ai.audits import repo_safety
from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.semantics.edited_label_batch import inspect_edited_label_batch
from vam_timeline_ai.semantics.ingest_latest import ingest_latest_edited_batch
from vam_timeline_ai.semantics.labeling_next_step import write_labeling_next_step
from vam_timeline_ai.semantics.review_batch_discovery import find_latest_review_batch


def _make_batch(run_dir: Path, num: int, edited: dict | None = None, valid: bool = True) -> Path:
    batch = run_dir / "labels" / "batches" / f"batch_{num:03d}"
    batch.mkdir(parents=True, exist_ok=True)
    if valid:
        write_jsonl(batch / "review_batch.jsonl", [{"review_id": f"r{num}", "window_id": "win_a"}])
        (batch / "manual_labels.stub.yaml").write_text(yaml.safe_dump({"windows": {"win_a": {"labels": [], "confidence": 0.0}}}), encoding="utf-8")
    if edited is not None:
        (batch / "manual_labels.edited.yaml").write_text(yaml.safe_dump(edited), encoding="utf-8")
    previews = batch / "previews"
    previews.mkdir(exist_ok=True)
    (previews / "index.html").write_text("<html></html>", encoding="utf-8")
    write_jsonl(previews / "preview_manifest.jsonl", [{"review_id": f"r{num}"}])
    return batch


def _minimal_run(tmp_path: Path) -> Path:
    run = tmp_path / "data" / "runs" / "clean_v2"
    (run / "semantic").mkdir(parents=True, exist_ok=True)
    write_jsonl(run / "semantic" / "movement_windows.jsonl", [{"window_id": "win_a", "sample_id": "sample_a", "source_id": "src_a", "source_scene_file": "scene.json"}])
    write_jsonl(run / "semantic" / "pair_windows_v1.jsonl", [{"pair_window_id": "pair_a", "window_id_a": "win_a", "window_id_b": "win_b"}])
    return run


def test_latest_review_batch_prefers_highest_numeric_valid_batch(tmp_path):
    run = _minimal_run(tmp_path)
    _make_batch(run, 1)
    _make_batch(run, 2)
    _make_batch(run, 3)

    result = find_latest_review_batch(run, run / "labels" / "latest_review_batch_report.md")

    assert result["latest_batch"]["batch_name"] == "batch_003"
    assert result["status"] == "waiting_for_human_labels"
    assert (run / "labels" / "latest_review_batch.json").exists()


def test_latest_review_batch_does_not_hardcode_batch_002(tmp_path):
    run = _minimal_run(tmp_path)
    _make_batch(run, 7)

    result = find_latest_review_batch(run)

    assert result["latest_batch"]["batch_name"] == "batch_007"


def test_human_next_step_when_edited_missing(tmp_path):
    run = _minimal_run(tmp_path)
    _make_batch(run, 3)

    result = write_labeling_next_step(run, run / "labels" / "human_labeling_next_step.md")
    text = (run / "labels" / "human_labeling_next_step.md").read_text(encoding="utf-8")

    assert result["status"] == "waiting_for_human_labels"
    assert "manual_labels.edited.yaml" in text
    assert "weak_" in text


def test_ingest_latest_stops_safely_when_edited_missing(tmp_path):
    run = _minimal_run(tmp_path)
    _make_batch(run, 3)
    schema = tmp_path / "schema.yaml"
    schema.write_text("allowed_manual_labels: []", encoding="utf-8")

    result = ingest_latest_edited_batch(run, schema, stop_if_missing=True)

    assert result["status"] == "waiting_for_human_labels"
    assert not (run / "labels" / "manual_labels.yaml").exists()
    assert (run / "labels" / "human_labeling_next_step.md").exists()


def test_inspection_rejects_empty_stub_and_weak_labels(tmp_path):
    run = _minimal_run(tmp_path)
    batch = _make_batch(run, 3)
    edited = batch / "manual_labels.edited.yaml"
    edited.write_text((batch / "manual_labels.stub.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    same = inspect_edited_label_batch(batch / "manual_labels.stub.yaml", edited, run / "semantic" / "movement_windows.jsonl", run / "semantic" / "pair_windows_v1.jsonl", batch / "inspect_same.md")
    assert same["safe_to_merge"] is False

    edited.write_text(yaml.safe_dump({"windows": {"win_a": {"labels": ["weak_v2_fast_motion_candidate"], "confidence": 0.8}}}), encoding="utf-8")
    weak = inspect_edited_label_batch(batch / "manual_labels.stub.yaml", edited, run / "semantic" / "movement_windows.jsonl", run / "semantic" / "pair_windows_v1.jsonl", batch / "inspect_weak.md")
    assert weak["safe_to_merge"] is False
    assert any("weak labels" in error for error in weak["errors"])


def test_repo_safety_detects_tracked_generated_patterns(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("data/runs/\ndata/labels/manual_labels.yaml\n", encoding="utf-8")
    (root / ".gitattributes").write_text("*.npz binary\n", encoding="utf-8")
    old_git_available = repo_safety._git_available
    old_git_ok = repo_safety._git_ok
    old_git_text = repo_safety._git_text
    try:
        repo_safety._git_available = lambda _root: True
        repo_safety._git_ok = lambda _root, _args: True
        repo_safety._git_text = lambda _root, args: "\n".join([
            "main",
        ]) if args == ["branch", "--show-current"] else (
            "origin https://example.invalid/repo.git (fetch)" if args == ["remote", "-v"] else "\n".join([
                "data/runs/clean_v2/features/x.npz",
                "data/labels/manual_labels.yaml",
                "data/labels/batches/batch_001/manual_labels.edited.yaml",
                "data/labels/batches/batch_001/previews/x.png",
            ])
        )
        result = repo_safety.audit_repo_safety(root, root / "report.md")
    finally:
        repo_safety._git_available = old_git_available
        repo_safety._git_ok = old_git_ok
        repo_safety._git_text = old_git_text

    assert result["status"] == "ERROR"
    assert any(check["name"] == "data_runs_tracked" and check["status"] == "ERROR" for check in result["checks"])
    assert any(check["name"] == "manual_labels_yaml_tracked" and check["status"] == "ERROR" for check in result["checks"])


def test_gitignore_protection_patterns_exist():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for pattern in ["data/runs/", "data/**/samples/", "data/**/previews/", "data/**/*.npz", "data/labels/manual_labels.yaml", "data/labels/**/*.edited.yaml", "outputs/", "*.var"]:
        assert pattern in gitignore


def test_local_status_handles_missing_clean_run(tmp_path):
    missing_run = tmp_path / "data" / "runs" / "clean_v2"
    result = write_local_status(missing_run, tmp_path / "status.md")

    assert result["clean_v2_exists"] is False
    assert (tmp_path / "status.md").exists()


def test_docs_say_weak_labels_not_ground_truth():
    readme = Path("README.md").read_text(encoding="utf-8")
    safety = Path("references/GITHUB_REPO_DATA_SAFETY.md").read_text(encoding="utf-8")

    assert "Weak labels" in readme
    assert "not training targets" in readme or "not semantic truth" in readme
    assert "not semantic truth" in safety

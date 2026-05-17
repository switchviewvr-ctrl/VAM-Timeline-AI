from pathlib import Path

from vam_timeline_ai.audits.review_deduplication import (
    annotate_candidates_against_reviewed,
    annotate_duplicate_status,
    build_reviewed_window_index,
    export_strict_novel_review,
    write_review_quality_report,
)
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import build_static_review_ui


def test_exact_duplicate_detection_by_window_id():
    rows = [
        {"review_folder": "r1", "review_id": "a", "window_id": "w1"},
        {"review_folder": "r2", "review_id": "b", "window_id": "w1"},
    ]

    annotated = annotate_duplicate_status(rows)

    assert annotated[0]["duplicate_status"] == "unique"
    assert annotated[1]["duplicate_status"] == "exact_duplicate"
    assert annotated[1]["overlaps_with_review_ids"]


def test_near_duplicate_detection_by_scene_actor_overlap():
    rows = [
        {
            "review_folder": "r1",
            "review_id": "a",
            "source_scene_file": "scene.json",
            "technical_actor_id": "Person",
            "source_id": "src",
            "sample_id": "s1",
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "pose_subtype": "cowgirl_kneeling",
            "motion_subtype": "oval_grind",
            "category": "cowgirl_clean_motion_generation_safe",
        },
        {
            "review_folder": "r2",
            "review_id": "b",
            "source_scene_file": "scene.json",
            "technical_actor_id": "Person",
            "source_id": "src",
            "sample_id": "s2",
            "start_seconds": 1.0,
            "end_seconds": 5.0,
            "pose_subtype": "cowgirl_kneeling",
            "motion_subtype": "oval_grind",
            "category": "cowgirl_clean_motion_generation_safe",
        },
    ]

    annotated = annotate_duplicate_status(rows)

    assert annotated[1]["duplicate_status"] in {"near_duplicate", "previously_reviewed"}


def test_selector_excludes_reviewed_and_exports_fewer_if_only_duplicates(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "semantic_review_old"
    review.mkdir(parents=True)
    write_jsonl(
        review / "semantic_review_010.jsonl",
        [{"review_id": "review_001", "window_id": "w1", "sample_id": "s1", "source_id": "src1", "start_seconds": 0, "end_seconds": 4}],
    )
    db = run / "datasets" / "cowgirl_candidate_db_v7.jsonl"
    write_jsonl(
        db,
        [
            {
                "window_id": "w1",
                "sample_id": "s1",
                "source_id": "src1",
                "source_scene_file": "scene.json",
                "technical_actor_id": "Person",
                "start_seconds": 0,
                "end_seconds": 4,
                "category": "cowgirl_clean_motion_generation_safe",
                "semantic_family": "cowgirl",
                "generation_safe": True,
            }
        ],
    )
    idx = run / "audits" / "reviewed_window_index.jsonl"
    build_reviewed_window_index(run, [run], idx, run / "audits" / "idx.csv", run / "audits" / "idx.md")

    summary = export_strict_novel_review(run, db, idx, run / "audits" / "novel", count=1, build_vam_package=False, build_static_ui=False)

    assert summary["exported_count"] == 0
    assert summary["quality"]["trust_level"] == "medium"


def test_selector_enforces_max_per_sample():
    reviewed = []
    candidates = [
        {"window_id": "w1", "sample_id": "same", "source_scene_file": "a.json", "category": "cowgirl_clean_motion_generation_safe"},
        {"window_id": "w2", "sample_id": "same", "source_scene_file": "b.json", "category": "cowgirl_clean_motion_generation_safe"},
    ]
    annotated = annotate_candidates_against_reviewed(candidates, reviewed)
    from vam_timeline_ai.audits.review_deduplication import _select_novel_candidates

    selected, summary = _select_novel_candidates(
        annotated,
        count=2,
        max_per_scene=2,
        max_per_sample=1,
        allow_reviewed_overlap=False,
        allow_near_duplicates=False,
        diversity_mode="loose",
    )

    assert len(selected) == 1
    assert summary["rejected_by_rule"]["sample_cap"] == 1


def test_review_quality_report_marks_duplicate_batch_low_trust(tmp_path):
    summary = write_review_quality_report(
        [{"review_id": "r1", "duplicate_status": "near_duplicate", "previously_reviewed": True}],
        tmp_path / "quality.md",
        requested_count=1,
        max_per_scene=2,
        max_per_sample=1,
    )

    assert summary["trust_level"] == "low"
    assert (tmp_path / "quality.md").exists()


def test_ui_includes_duplicate_warning_fields(tmp_path):
    run = tmp_path / "run"
    review = run / "audits" / "semantic_review"
    review.mkdir(parents=True)
    write_jsonl(
        review / "semantic_review_010.jsonl",
        [
            {
                "review_id": "review_001",
                "window_id": "w1",
                "duplicate_status": "near_duplicate",
                "previously_reviewed": True,
                "duplicate_group_id": "dup_x",
                "review_trust_warning": "This item appears to overlap a previously reviewed sample/window.",
            }
        ],
    )

    build_static_review_ui(run, review, review / "review_ui_static")
    js = (review / "review_ui_static" / "review_data.js").read_text(encoding="utf-8")
    app = (review / "review_ui_static" / "app.js").read_text(encoding="utf-8")

    assert "duplicate_status" in js
    assert "previously_reviewed" in js
    assert "This item appears to overlap" in app

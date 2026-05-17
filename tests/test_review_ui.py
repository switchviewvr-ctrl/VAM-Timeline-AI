import json
from pathlib import Path

import yaml

from vam_timeline_ai.audits.review_answer_ingestion import ingest_review_ui_answers
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import (
    answer_schema,
    build_review_ui_data,
    build_static_review_ui,
    save_ui_answers,
    validate_answer,
)


def _make_review(tmp_path: Path):
    run = tmp_path / "data" / "runs" / "clean_v3"
    review = run / "audits" / "semantic_review_010_v16"
    package = review / "vam_review_package"
    (package / "items" / "review_001").mkdir(parents=True)
    (run / "datasets").mkdir(parents=True)
    row = {
        "review_id": "review_001",
        "window_id": "w1",
        "semantic_family": "cowgirl",
        "pose_semantics": {"family": "cowgirl", "subtype": "cowgirl_lean_forward_supported"},
        "motion_semantics": {"subtype": "grinding", "phase": "clean_motion"},
        "partner_relation": ["rider_above_partner"],
        "contact_support": "possible_partner_contact",
        "interaction_family": "cowgirl",
        "generation_safe": True,
        "why_selected": "cowgirl_clean_motion_generation_safe",
        "hands_on_partner_chest_score": 0.3,
    }
    manifest = dict(
        row,
        source_scene_file="scene.json",
        source_scene_path="local_scene_batch/scene.json",
        technical_atom_id="Person",
        start_seconds=1.0,
        end_seconds=5.0,
        timeline_export_path="timeline_segments/review_001/review_001.timeline.json",
        evidence_scores={"rider_above_partner_score": 1.0, "pelvis_alignment_score": 0.8},
    )
    write_jsonl(review / "semantic_review_010.jsonl", [row])
    write_jsonl(package / "vam_review_manifest.jsonl", [manifest])
    (package / "items" / "review_001" / "item_review.md").write_text("review", encoding="utf-8")
    write_jsonl(
        run / "datasets" / "cowgirl_candidate_db_v6.jsonl",
        [
            {
                "candidate_id": "c1",
                "window_id": "w1",
                "semantic_family": "cowgirl",
                "category": "cowgirl_clean_motion_generation_safe",
                "pose_subtype": "cowgirl_lean_forward_supported",
                "motion_subtype": "grinding",
                "phase": "clean_motion",
                "contact_support": "possible_partner_contact",
                "generation_safe": True,
            }
        ],
    )
    return run, review


def test_static_review_ui_builds_from_synthetic_review_data(tmp_path):
    run, review = _make_review(tmp_path)
    out = review / "review_ui_static"

    summary = build_static_review_ui(run, review, out)

    assert summary["review_items"] == 1
    assert (out / "index.html").exists()
    assert (out / "review_data.js").exists()
    assert "vam_timeline_ai_review_ui_v0" in (out / "review_data.js").read_text(encoding="utf-8")


def test_answer_schema_validates_and_normalizes():
    schema = answer_schema()
    answer = validate_answer({"review_id": "review_001", "semantic_family_correct": True, "error_tags": "low_motion_hold"})

    assert "semantic_family_correct" in schema["fields"]
    assert answer["semantic_family_correct"] == "true"
    assert answer["error_tags"] == ["low_motion_hold"]


def test_answer_ingestion_creates_ledger_records(tmp_path):
    run, review = _make_review(tmp_path)
    answers = review / "human_review_ui_answers.jsonl"
    write_jsonl(
        answers,
        [
            {
                "review_id": "review_001",
                "semantic_family_correct": "true",
                "pose_correct": "true",
                "motion_correct": "false",
                "actual_motion": "intro_alignment",
                "verdict": "partially_correct",
                "error_tags": ["intro_alignment"],
                "notes": "not clean motion",
            }
        ],
    )

    summary = ingest_review_ui_answers(answers, review, run / "audits" / "human_review_ledger.jsonl", run / "audits" / "ingest.md")
    rows = load_jsonl(run / "audits" / "human_review_ledger.jsonl")

    assert summary["new_ledger_records"] == 1
    assert rows[0]["is_training_label"] is False
    assert rows[0]["human_motion"] == "intro_alignment"


def test_ui_data_contains_review_fields(tmp_path):
    run, review = _make_review(tmp_path)

    data = build_review_ui_data(run, review)
    item = data["review_items"][0]

    assert item["semantic_family"] == "cowgirl"
    assert item["pose_family"] == "cowgirl"
    assert item["motion_subtype"] == "grinding"
    assert item["timeline_export_path"]
    assert data["candidates"][0]["category"] == "cowgirl_clean_motion_generation_safe"


def test_candidate_explorer_data_handles_missing_dbs(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "semantic_review_010_v16"
    review.mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "w1"}])

    data = build_review_ui_data(run, review)

    assert data["candidates"] == []
    assert data["candidate_summary"]["count"] == 0


def test_save_ui_answers_writes_jsonl_and_yaml(tmp_path):
    review = tmp_path / "review"
    review.mkdir()

    summary = save_ui_answers(review, [{"review_id": "review_001", "semantic_family_correct": "true"}])

    assert Path(summary["jsonl"]).exists()
    assert Path(summary["yaml"]).exists()
    data = yaml.safe_load(Path(summary["yaml"]).read_text(encoding="utf-8"))
    assert "review_001" in data["reviews"]


def test_no_manual_labels_or_ml_created_by_static_ui(tmp_path):
    run, review = _make_review(tmp_path)

    build_static_review_ui(run, review, review / "review_ui_static")

    assert not (run / "labels" / "manual_labels.yaml").exists()
    assert not (run / "ml").exists()

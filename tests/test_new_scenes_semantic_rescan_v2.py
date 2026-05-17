from pathlib import Path

from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.semantics.new_scenes_semantic_rescan_v2 import (
    _apply_v2_rules,
    _candidate_category,
    _select_review_items,
    build_new_scenes_ontology_candidate_db_v2,
)


def _base(family="cowgirl", primary="pelvis_hip", pose="cowgirl_kneeling"):
    return {
        "window_id": "w1",
        "sample_id": "s1",
        "source_scene_path": "local_scene_batch/szene.json",
        "source_scene_file": "szene.json",
        "resolved_semantic_family": family,
        "resolved_motion_subtype": family,
        "pose_family": "cowgirl" if "cowgirl" in pose else "unknown",
        "pose_subtype": pose,
        "primary_motion_center": primary,
        "target_region": "unknown",
        "partner_relation": ["pelvis_aligned"],
        "clean_motion_gate": "pass",
        "confidence": 0.7,
        "conflict_flags": [],
        "not_labels": [],
        "missing_requirements": [],
    }


def _ctx(old_family="cowgirl", phase="clean_motion"):
    return {
        "candidate": {"semantic_family": old_family, "phase": phase, "source_scene_file": "szene.json"},
        "pose": {},
        "relative": {},
        "interaction": {},
        "window": {},
        "sample": {},
    }


def test_cowgirl_like_head_driver_maps_to_bj_oral():
    row = _apply_v2_rules(_base(primary="head_neck"), _ctx())
    assert row["resolved_semantic_family"] == "bj_oral"
    assert "cowgirl_clean_motion" in row["not_labels"]


def test_hand_driver_maps_to_handjob():
    base = _base(primary="hands")
    base["target_region"] = "partner_pelvis_or_genital_area"
    row = _apply_v2_rules(base, _ctx())
    assert row["resolved_semantic_family"] == "handjob"
    assert "cowgirl_clean_motion" in row["not_labels"]


def test_doggy_requires_support_or_partner_behind():
    base = _base(family="doggy", primary="pelvis_hip", pose="cowgirl_kneeling")
    base["partner_relation"] = []
    row = _apply_v2_rules(base, _ctx(old_family="doggy"))
    assert "doggy" in row["not_labels"]
    assert any("requires_support" in flag or "partner_behind" in flag for flag in row["conflict_flags"])


def test_low_motion_pose_becomes_pose_context():
    row = _apply_v2_rules(_base(), _ctx(phase="low_motion_hold"))
    assert row["resolved_semantic_family"] == "pose_context_hold"
    assert row["clean_motion_gate"] == "fail_low_motion"


def test_candidate_db_uses_ontology_v2_fields(tmp_path: Path):
    resolved = tmp_path / "resolved.jsonl"
    out_jsonl = tmp_path / "candidates.jsonl"
    out_csv = tmp_path / "candidates.csv"
    report = tmp_path / "report.md"
    row = _base()
    row["schema"] = "new_scenes_semantic_rescan_v2"
    write_jsonl(resolved, [row])
    summary = build_new_scenes_ontology_candidate_db_v2(tmp_path, resolved, tmp_path / "motion_families_v2.yaml", tmp_path / "manual_gt.jsonl", out_jsonl, out_csv, report)
    assert summary["category_counts"]["cowgirl_clean_motion_candidate"] == 1
    text = out_jsonl.read_text(encoding="utf-8")
    assert '"manual_labels_modified": false' in text
    assert '"ml_training_performed": false' in text


def test_review_batch_excludes_duplicates_and_reviewed():
    rows = []
    for i in range(6):
        row = _base()
        row.update(
            {
                "window_id": f"w{i}",
                "sample_id": f"s{i // 2}",
                "source_scene_file": f"scene{i // 3}.json",
                "category": "cowgirl_clean_motion_candidate",
                "recommended_review_priority": "high",
            }
        )
        rows.append(row)
    selected = _select_review_items(rows, 4, {"w0"})
    assert all(r["window_id"] != "w0" for r in selected)
    assert len({r["sample_id"] for r in selected}) == len(selected)
    assert all("scene" in r["source_scene_file"] for r in selected)


def test_no_ml_training_or_timeline_generation_markers():
    row = _base()
    row["category"] = _candidate_category(row)
    text = str(row).lower()
    assert "manual_labels.yaml" not in text
    assert "timeline_generation_performed': true" not in text
    assert "ml_training_performed': true" not in text

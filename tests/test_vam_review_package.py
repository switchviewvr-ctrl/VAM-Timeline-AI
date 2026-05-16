from pathlib import Path

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def _make_review_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run = tmp_path / "data" / "runs" / "clean_v3"
    source_run = tmp_path / "data" / "runs" / "clean_v2"
    for rel in [
        "audits/semantic_review_010_v15",
        "semantic",
        "baked",
        "interaction_semantics",
    ]:
        (run / rel).mkdir(parents=True, exist_ok=True)
    (source_run / "baked").mkdir(parents=True, exist_ok=True)

    review = run / "audits" / "semantic_review_010_v15" / "semantic_review_010.jsonl"
    write_jsonl(
        review,
        [
            {
                "review_id": "review_001",
                "window_id": "win_001",
                "pair_window_id": "pair_001",
                "semantic_family": "cowgirl",
                "pose_semantics": {"family": "cowgirl", "subtype": "cowgirl_lean_forward_supported"},
                "motion_semantics": {"subtype": "grinding"},
                "partner_relation": ["rider_above_partner", "pelvis_aligned"],
                "contact_support": "hands_on_partner_chest",
                "generation_safe": True,
                "why_selected": "cowgirl_hands_on_partner_chest",
                "is_human_ground_truth": False,
                "is_training_label": False,
            },
            {
                "review_id": "review_002",
                "window_id": "win_002",
                "semantic_family": "unknown",
                "pose_semantics": {"family": "unknown", "subtype": "unknown"},
                "motion_semantics": {"subtype": "unknown"},
                "partner_relation": [],
                "contact_support": "unknown",
                "generation_safe": False,
                "why_selected": "unknown_or_unusable",
                "is_human_ground_truth": False,
                "is_training_label": False,
            },
        ],
    )
    write_jsonl(
        run / "semantic" / "movement_windows.jsonl",
        [
            {
                "window_id": "win_001",
                "sample_id": "sample_001",
                "source_id": "src_001",
                "source_scene_file": "scene_001.json",
                "source_scene_path": str(tmp_path / "scene_001.json"),
                "technical_atom_id": "Person",
                "start_seconds": 4.0,
                "end_seconds": 8.0,
                "duration_seconds": 4.0,
                "frame_start": 240,
                "frame_end": 480,
            },
            {
                "window_id": "win_002",
                "sample_id": "sample_002",
                "source_id": "src_002",
                "technical_atom_id": "Person",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "duration_seconds": 2.0,
            },
        ],
    )
    write_jsonl(
        run / "baked" / "motion_sample_index.jsonl",
        [
            {
                "sample_id": "sample_001",
                "source_id": "src_001",
                "source_type": "timeline_controller_motion",
                "source_scene_file": "scene_001.json",
                "source_scene_path": str(tmp_path / "scene_001.json"),
                "technical_atom_id": "Person",
                "storable_id": "plugin#0_VamTimeline.AtomPlugin",
                "clip_name": "Anim 1",
                "clip_index": 0,
                "fps": 60.0,
                "baked_npz_path": "missing.npz",
                "warnings": [],
            },
            {
                "sample_id": "sample_002",
                "source_id": "src_002",
                "source_type": "timeline_controller_motion",
                "technical_atom_id": "Person",
                "clip_name": "Anim 2",
                "clip_index": 1,
                "baked_npz_path": "",
                "warnings": [],
            },
        ],
    )
    write_jsonl(
        run / "semantic" / "motion_source_index.jsonl",
        [
            {
                "source_id": "src_001",
                "source_type": "timeline_controller_motion",
                "source_scene_file": "scene_001.json",
                "source_scene_path": str(tmp_path / "scene_001.json"),
                "technical_atom_id": "Person",
                "storable_id": "plugin#0_VamTimeline.AtomPlugin",
                "clip_name": "Anim 1",
                "clip_index": 0,
            }
        ],
    )
    write_jsonl(
        run / "semantic" / "pair_windows_v1.jsonl",
        [{"pair_window_id": "pair_001", "window_id_a": "win_001", "window_id_b": "partner_win", "technical_atom_id_b": "Partner"}],
    )
    write_jsonl(
        run / "interaction_semantics" / "partner_relative_features_v0.jsonl",
        [
            {
                "window_id": "win_001",
                "pair_window_id": "pair_001",
                "partner_actor_id": "Partner",
                "rider_above_partner_score": 0.9,
                "pelvis_alignment_score": 0.8,
                "hands_on_partner_chest_score": 0.7,
                "hands_on_partner_hips_score": 0.1,
                "partner_lying_score": 0.6,
            }
        ],
    )
    write_jsonl(
        run / "interaction_semantics" / "interaction_semantics_v0.jsonl",
        [
            {
                "window_id": "win_001",
                "pair_window_id": "pair_001",
                "partner_actor_id": "Partner",
                "interaction_family": "cowgirl",
                "partner_relation": ["rider_above_partner", "pelvis_aligned"],
                "support_context": "hands_on_partner_chest",
            }
        ],
    )
    out = run / "audits" / "semantic_review_010_v15" / "vam_review_package"
    return run, source_run, review, out


def test_vam_review_package_creates_manifest_scene_list_and_items(tmp_path):
    run, source_run, review, out = _make_review_fixture(tmp_path)

    summary = build_vam_review_package(review, run, source_run, out, attempt_timeline_segments=True)
    rows = load_jsonl(out / "vam_review_manifest.jsonl")

    assert summary["review_items"] == 2
    assert (out / "vam_review_manifest.csv").exists()
    assert (out / "vam_review_scene_list.md").exists()
    assert (out / "vam_review_index.html").exists()
    assert (out / "items" / "review_001" / "item_review.md").exists()
    assert (out / "items" / "review_001" / "item_metadata.json").exists()
    assert rows[0]["source_scene_path"]
    assert rows[0]["technical_atom_id"] == "Person"
    assert rows[0]["start_seconds"] == 4.0
    assert rows[0]["hands_on_partner_chest_score"] == 0.7


def test_vam_review_package_answer_sheet_includes_all_items(tmp_path):
    run, source_run, review, out = _make_review_fixture(tmp_path)

    build_vam_review_package(review, run, source_run, out, attempt_timeline_segments=True)
    data = yaml.safe_load((out / "vam_review_answer_sheet.yaml").read_text(encoding="utf-8"))

    assert set(data["reviews"]) == {"review_001", "review_002"}
    assert data["reviews"]["review_001"]["semantic_family_correct"] == "unknown"
    assert data["reviews"]["review_001"]["actual_contact_support"] == ""


def test_vam_review_package_handles_missing_source_and_does_not_fake_export(tmp_path):
    run, source_run, review, out = _make_review_fixture(tmp_path)

    summary = build_vam_review_package(review, run, source_run, out, attempt_timeline_segments=True)
    rows = load_jsonl(out / "vam_review_manifest.jsonl")

    assert summary["timeline_segments_successful"] == 0
    assert (out / "timeline_segments" / "review_001" / "export_unavailable.md").exists()
    assert (out / "timeline_segments" / "review_002" / "export_unavailable.md").exists()
    assert not (out / "timeline_segments" / "review_001" / "review_001.timeline.json").exists()
    assert rows[1]["can_review_in_original_scene"] is False
    assert rows[1]["review_method"] == "unavailable"
    assert rows[1]["timeline_export_status"] == "unavailable"


def test_vam_review_package_does_not_modify_manual_labels_or_train(tmp_path):
    run, source_run, review, out = _make_review_fixture(tmp_path)

    summary = build_vam_review_package(review, run, source_run, out, attempt_timeline_segments=True)

    assert summary["manual_labels_modified"] is False
    assert summary["ml_training_performed"] is False
    assert not (run / "labels" / "manual_labels.yaml").exists()

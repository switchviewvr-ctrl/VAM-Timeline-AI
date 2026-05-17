from pathlib import Path

from vam_timeline_ai.generation.semantic_motion_examples import build_semantic_motion_examples_v2_contact_aware
from vam_timeline_ai.generation.semantic_pose_library import build_semantic_stickman_pose_library_v1
from vam_timeline_ai.generation.stickman_to_vam_preview_mapper import map_stickman_to_vam_preview_v0
from vam_timeline_ai.generation.vam_semantic_preview_exporter import export_vam_semantic_preview_v0
from vam_timeline_ai.generation.vam_semantic_preview_validation import validate_vam_semantic_preview_v0
from vam_timeline_ai.io.json_utils import dump_json, load_json


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "data" / "ontology" / "motion_families_v2.yaml"


def _build_contact_examples(tmp_path: Path) -> Path:
    library = tmp_path / "pose_library.json"
    examples = tmp_path / "contact_examples.json"
    build_semantic_stickman_pose_library_v1(ONTOLOGY, library, tmp_path / "pose_report.md")
    build_semantic_motion_examples_v2_contact_aware(library, ONTOLOGY, examples, tmp_path / "contact_report.md")
    return examples


def _export_package(tmp_path: Path) -> Path:
    examples = _build_contact_examples(tmp_path)
    out_dir = tmp_path / "vam_preview"
    summary = export_vam_semantic_preview_v0(examples, out_dir, duration=2.0, fps=12)
    assert summary["exported_clips"] == 13
    return out_dir


def test_mapper_maps_pelvis_to_pelvis_control_and_keeps_review_metadata(tmp_path):
    examples = _build_contact_examples(tmp_path)
    out = tmp_path / "preview_data.json"
    summary = map_stickman_to_vam_preview_v0(examples, out, duration_seconds=2.0, fps=12)
    data = load_json(out)
    cowgirl = next(clip for clip in data["clips"] if clip["clip_id"] == "cowgirl_grinding")
    tracks = {track["source_bodypart"]: track["controller_name"] for track in cowgirl["controller_tracks"]}

    assert summary["clip_count"] == 13
    assert tracks["pelvis"] == "pelvisControl"
    assert cowgirl["review_only"] is True
    assert cowgirl["coordinate_space"] == "synthetic_review_local"
    assert cowgirl["target_points"]["partner_pelvis_target"]


def test_exporter_writes_timeline_json_without_person_root_world_tracks(tmp_path):
    out_dir = _export_package(tmp_path)
    manifest = (out_dir / "manifest.jsonl").read_text(encoding="utf-8")
    timeline = out_dir / "clips" / "cowgirl_grinding.timeline.json"
    payload = load_json(timeline)
    meta = payload["VAMTimelineAISemanticPreviewMetadata"]
    controllers = [row["Controller"] for row in payload["Clips"][0]["Controllers"]]

    assert "cowgirl_grinding" in manifest
    assert timeline.exists()
    assert meta["review_only"] is True
    assert meta["source_world_coords_used"] is False
    assert meta["person_root_tracks_included"] is False
    assert all("root" not in c.lower() and "world" not in c.lower() and "person" not in c.lower() for c in controllers)


def test_semantic_preview_validation_core_family_rules(tmp_path):
    out_dir = _export_package(tmp_path)
    validation = validate_vam_semantic_preview_v0(out_dir, out_dir / "reports" / "validation.md")
    data = load_json(out_dir / "preview_data" / "vam_semantic_preview_clips_v0.json")
    by_id = {clip["clip_id"]: clip for clip in data["clips"]}

    assert validation["status"] == "ok"
    assert by_id["cowgirl_grinding"]["target_points"]["partner_pelvis_target"]
    assert by_id["cowgirl_grinding"]["alignment_validation"]["valid"] is True
    assert by_id["bj_head_bob"]["labels"]["primary_driver"] == ["head_neck", "chest_abdomen"]
    assert by_id["doggy_forward_back"]["labels"]["anchors"]
    assert by_id["missionary_counter_thrust"]["pose_subtype"] == "missionary_supine"


def test_validation_catches_invalid_floating_cowgirl_preview(tmp_path):
    out_dir = _export_package(tmp_path)
    preview_path = out_dir / "preview_data" / "vam_semantic_preview_clips_v0.json"
    data = load_json(preview_path)
    for clip in data["clips"]:
        if clip["clip_id"] == "cowgirl_grinding":
            for track in clip["controller_tracks"]:
                if track["controller_name"] == "pelvisControl":
                    track["positions"] = [[p[0], p[1] + 2.0, p[2] + 2.0] for p in track["positions"]]
            break
    dump_json(preview_path, data)
    validation = validate_vam_semantic_preview_v0(out_dir, out_dir / "reports" / "bad_validation.md")

    assert validation["status"] == "failed"
    assert any("cowgirl_grinding" in msg for msg in validation["error_messages"])


def test_package_contains_import_and_partner_reference_instructions(tmp_path):
    out_dir = _export_package(tmp_path)

    assert (out_dir / "import_instructions.md").exists()
    assert (out_dir / "partner_reference" / "partner_reference_instructions.md").exists()
    assert (out_dir / "partner_reference" / "partner_reference_markers.json").exists()
    assert (out_dir / "index.html").exists()

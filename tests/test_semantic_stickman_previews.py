from pathlib import Path

from vam_timeline_ai.generation.semantic_motion_examples import build_semantic_motion_examples_v1
from vam_timeline_ai.generation.semantic_motion_examples import build_semantic_motion_examples_v2_contact_aware
from vam_timeline_ai.generation.semantic_pose_library import build_semantic_stickman_pose_library_v1
from vam_timeline_ai.generation.semantic_stickman_validation import validate_semantic_stickman_examples_v1, validate_semantic_stickman_examples_v2, validate_semantic_stickman_examples_v3
from vam_timeline_ai.io.json_utils import dump_json, load_json
from vam_timeline_ai.reports.semantic_stickman_gallery import build_semantic_stickman_gallery_v1, build_semantic_stickman_gallery_v2, build_semantic_stickman_gallery_v3
from vam_timeline_ai.visualization.semantic_stickman_renderer import render_semantic_stickman_previews_v1, render_semantic_stickman_previews_v2, render_semantic_stickman_previews_v3


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "data" / "ontology" / "motion_families_v2.yaml"


def _build_examples(tmp_path: Path) -> Path:
    library = tmp_path / "pose_library.json"
    examples = tmp_path / "examples.json"
    build_semantic_stickman_pose_library_v1(ONTOLOGY, library, tmp_path / "pose_report.md")
    build_semantic_motion_examples_v1(library, ONTOLOGY, examples, tmp_path / "examples_report.md")
    return examples


def _build_contact_examples(tmp_path: Path) -> Path:
    library = tmp_path / "pose_library.json"
    examples = tmp_path / "contact_examples.json"
    build_semantic_stickman_pose_library_v1(ONTOLOGY, library, tmp_path / "pose_report.md")
    build_semantic_motion_examples_v2_contact_aware(library, ONTOLOGY, examples, tmp_path / "contact_report.md")
    return examples


def test_pose_library_contains_required_canonical_poses(tmp_path):
    out = tmp_path / "library.json"
    summary = build_semantic_stickman_pose_library_v1(ONTOLOGY, out, tmp_path / "report.md")
    data = load_json(out)
    ids = {pose["concept_id"] for pose in data["poses"]}

    assert summary["pose_count"] >= 18
    assert "cowgirl_lean_back_supported" in ids
    assert "doggy_all_fours" in ids
    assert "bj_kneeling_forward" in ids
    assert "missionary_supine" in ids
    assert data["uses_person_root_or_world"] is False


def test_motion_examples_encode_core_driver_grammar(tmp_path):
    examples = _build_examples(tmp_path)
    data = load_json(examples)
    by_id = {ex["concept_id"]: ex for ex in data["examples"]}

    cowgirl = by_id["cowgirl_grinding"]
    bj = by_id["bj_head_bob"]
    doggy = by_id["doggy_forward_back"]
    missionary = by_id["missionary_counter_thrust"]
    lean_back = by_id["cowgirl_lean_back_supported"]

    assert "pelvis_hip" in cowgirl["labels"]["primary_driver"]
    assert cowgirl["driver_curves"]["pelvis_hip"] == "oval_or_figure8_xz_low_y"
    assert "head_neck" in bj["labels"]["primary_driver"]
    assert bj["follower_curves"]["pelvis"] == "static_isolator"
    assert "pelvis_hip" in doggy["labels"]["primary_driver"]
    assert "hands" in doggy["labels"]["anchors"]
    assert "knees" in doggy["labels"]["anchors"]
    assert "pelvis_counter_driver" in missionary["labels"]["primary_driver"]
    assert missionary["pose_subtype"] == "missionary_supine"
    assert "reverse_cowgirl" in lean_back["not_labels"]


def test_renderer_and_gallery_create_outputs_or_gracefully_skip_gif(tmp_path):
    examples = _build_examples(tmp_path)
    out_dir = tmp_path / "previews"
    summary = render_semantic_stickman_previews_v1(examples, out_dir, width=640, height=360, fps=8, make_gif=True, make_contact_sheet=True)
    gallery = build_semantic_stickman_gallery_v1(out_dir, out_dir / "index.html", tmp_path / "gallery.md")

    assert (out_dir / "semantic_stickman_preview_manifest_v1.json").exists()
    assert summary["concepts"] == 13
    if summary["rendered"]:
        assert summary["gif_count"] == 13
        assert summary["contact_sheet_count"] == 13
    assert gallery["items"] == 13
    assert (out_dir / "index.html").exists()


def test_validation_catches_wrong_driver(tmp_path):
    examples = _build_examples(tmp_path)
    data = load_json(examples)
    data["examples"][0]["labels"]["primary_driver"] = ["head_neck"]
    bad = tmp_path / "bad_examples.json"
    dump_json(bad, data)

    summary = validate_semantic_stickman_examples_v1(bad, ONTOLOGY, tmp_path / "validation.md")

    assert summary["status"] == "failed"
    assert summary["errors"] >= 1
    assert "Cowgirl example lacks pelvis_hip driver" in "\n".join(summary["error_messages"])


def test_validation_passes_and_no_generation_or_labels(tmp_path):
    examples = _build_examples(tmp_path)
    summary = validate_semantic_stickman_examples_v1(examples, ONTOLOGY, tmp_path / "validation.md")
    text = (tmp_path / "validation.md").read_text(encoding="utf-8")

    assert summary["status"] == "ok"
    assert "Timeline animation generated: false" in text
    assert "ML training performed: false" in text
    assert "manual_labels.yaml modified: false" in text


def test_v2_renderer_labels_partner_alignment_and_support_context(tmp_path):
    examples = _build_examples(tmp_path)
    out_dir = tmp_path / "previews_v2"
    summary = render_semantic_stickman_previews_v2(
        examples,
        out_dir,
        width=720,
        height=420,
        fps=8,
        make_gif=True,
        make_contact_sheet=True,
        show_labels=True,
        show_partner=True,
        show_alignment=True,
        show_support_targets=True,
    )
    manifest = load_json(out_dir / "semantic_stickman_preview_manifest_v2.json")
    by_id = {item["concept_id"]: item for item in manifest["items"]}

    assert summary["concepts"] == 13
    assert by_id["cowgirl_grinding"]["semantic_context"]["bodypart_labels_drawn"] is True
    assert by_id["cowgirl_grinding"]["semantic_context"]["has_partner_pelvis_target"] is True
    assert by_id["cowgirl_grinding"]["semantic_context"]["has_alignment_target"] is True
    assert by_id["cowgirl_grinding"]["semantic_context"]["appears_floating_warning"] is False
    assert by_id["cowgirl_lean_back_supported"]["semantic_context"]["has_support_targets"] is True
    assert any("lHand->partner.legs_or_thighs" in v for v in [by_id["cowgirl_lean_back_supported"]["semantic_context"]["support_vectors"]])
    assert by_id["doggy_forward_back"]["semantic_context"]["has_partner_reference"] is True
    assert "head_to_partner_pelvis" in by_id["bj_head_bob"]["semantic_context"]["target_vectors"]


def test_v2_gallery_includes_legends_and_validation_passes(tmp_path):
    examples = _build_examples(tmp_path)
    out_dir = tmp_path / "previews_v2"
    render_semantic_stickman_previews_v2(examples, out_dir, width=720, height=420, fps=8)
    gallery = build_semantic_stickman_gallery_v2(out_dir, out_dir / "index.html", tmp_path / "gallery_v2.md")
    validation = validate_semantic_stickman_examples_v2(examples, out_dir, ONTOLOGY, tmp_path / "validation_v2.md")
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    text = (tmp_path / "validation_v2.md").read_text(encoding="utf-8")

    assert gallery["items"] == 13
    assert "Legend" in html
    assert "alignment" in html
    assert validation["status"] == "ok"
    assert "Timeline animation generated: false" in text


def test_v3_contact_aware_examples_enforce_core_alignment(tmp_path):
    examples = _build_contact_examples(tmp_path)
    data = load_json(examples)
    by_id = {ex["concept_id"]: ex for ex in data["examples"]}

    cowgirl = by_id["cowgirl_grinding"]
    bounce = by_id["cowgirl_vertical_bounce"]
    lean_back = by_id["cowgirl_lean_back_supported"]
    reverse = by_id["reverse_cowgirl_standing_squat_bounce"]
    doggy = by_id["doggy_forward_back"]
    bj = by_id["bj_head_bob"]
    missionary = by_id["missionary_counter_thrust"]

    assert cowgirl["alignment_validation"]["valid"] is True
    assert cowgirl["alignment_validation"]["max_distance"] <= cowgirl["alignment_validation"]["target_distance_max"]
    assert "partner_pelvis_target" in cowgirl["target_points"]
    assert bounce["interaction_constraints"][0]["partner_target_bodypart"] == "partner_pelvis"
    assert lean_back["alignment_validation"]["valid"] is True
    assert lean_back["labels"]["facing_context"] == "front_cowgirl"
    assert "reverse_cowgirl" in lean_back["not_labels"]
    assert reverse["labels"]["facing_context"] == "back_to_partner"
    assert doggy["interaction_constraints"][0]["orientation_requirement"] == "partner_behind"
    assert "hands" in doggy["labels"]["anchors"]
    assert bj["alignment_validation"]["valid"] is True
    assert bj["interaction_constraints"][0]["actor_anchor_bodypart"] == "head"
    assert bj["follower_curves"]["pelvis"] == "static_isolator"
    assert missionary["alignment_validation"]["valid"] is True
    assert missionary["pose_subtype"] == "missionary_supine"


def test_v3_renderer_gallery_and_validation_show_contact_validity(tmp_path):
    examples = _build_contact_examples(tmp_path)
    out_dir = tmp_path / "previews_v3"
    summary = render_semantic_stickman_previews_v3(examples, out_dir, width=720, height=420, fps=8)
    gallery = build_semantic_stickman_gallery_v3(out_dir, out_dir / "index.html", tmp_path / "gallery_v3.md")
    validation = validate_semantic_stickman_examples_v3(examples, out_dir, ONTOLOGY, tmp_path / "validation_v3.md")
    html = (out_dir / "index.html").read_text(encoding="utf-8")

    assert summary["gif_count"] == 13
    assert gallery["items"] == 13
    assert "contact-valid" in html
    assert validation["status"] == "ok"
    assert validation["valid"] >= 12


def test_v3_validation_catches_floating_misaligned_cowgirl(tmp_path):
    examples = _build_contact_examples(tmp_path)
    data = load_json(examples)
    for ex in data["examples"]:
        if ex["concept_id"] == "cowgirl_grinding":
            for frame in ex["frames"]:
                point = frame["controller_points"]["pelvis"]
                frame["controller_points"]["pelvis"] = [point[0], point[1] + 1.5, point[2] + 1.5]
            break
    bad = tmp_path / "bad_contact_examples.json"
    dump_json(bad, data)
    out_dir = tmp_path / "bad_previews"
    render_semantic_stickman_previews_v3(bad, out_dir, width=720, height=420, fps=8)
    validation = validate_semantic_stickman_examples_v3(bad, out_dir, ONTOLOGY, tmp_path / "bad_validation.md")

    assert validation["status"] == "failed"
    assert any("cowgirl_grinding" in msg for msg in validation["error_messages"])

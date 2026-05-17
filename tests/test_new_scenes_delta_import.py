import json
from pathlib import Path

from vam_timeline_ai.io.json_utils import load_json, load_jsonl, write_jsonl
from vam_timeline_ai.motion.source_inventory import build_motion_source_index
from vam_timeline_ai.audits.new_scene_review_planner import (
    DIAGNOSTIC_FIELDS,
    REVIEW_LABELS,
    REVIEW_QUESTIONS,
    _review_card,
    _ensure_inside,
    select_focused_review_cards,
    select_strict_cowgirl_review_cards,
)
from vam_timeline_ai.audits.vam_timeline_segment_export import export_review_timeline_segments_to_vam
from vam_timeline_ai.reports.new_scene_delta_report import compare_new_scenes_to_clean_v3
from vam_timeline_ai.runs.new_scenes_delta_import import _create_structure, _select_review_items
from vam_timeline_ai.ui.review_ui import answer_schema


def _timeline_scene(path: Path, controller: str = "hipControl") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "atoms": [
                    {
                        "id": "Person",
                        "storables": [
                            {
                                "id": "Timeline",
                                "Animation": {
                                    "SerializeVersion": 283,
                                    "Clips": [
                                        {
                                            "AnimationName": "Anim 1",
                                            "AnimationLength": "2",
                                            "Controllers": [
                                                {"Controller": controller, "X": [], "Y": [], "Z": []}
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_new_scene_run_structure_created_and_manifest_records_source(tmp_path):
    raw = tmp_path / "scene_batch"
    raw.mkdir(parents=True)
    for idx in range(2):
        (raw / f"scene_{idx}.json").write_text('{"atoms":[]}', encoding="utf-8")
    base = tmp_path / "data" / "runs" / "clean_v3"
    base.mkdir(parents=True)
    out = tmp_path / "data" / "runs" / "clean_v3_new_scenes"

    _create_structure(raw, base, out)

    manifest = load_json(out / "run_manifest.json")
    assert manifest["source_folder"] == str(raw)
    assert manifest["parent_reference_run"] == "clean_v3"
    assert manifest["purpose"] == "new_scene_delta_import"
    assert manifest["scene_count_expected"] == 24
    for folder in ["audits", "baked", "semantic", "relative_motion", "pose_semantics", "interaction_semantics", "datasets", "reports"]:
        assert (out / folder).is_dir()


def test_raw_dir_filter_does_not_scan_parent_folder(tmp_path):
    parent = tmp_path / "scene_parent"
    raw = parent / "scene_batch"
    _timeline_scene(parent / "old_scene.json", controller="headControl")
    _timeline_scene(raw / "new_scene.json", controller="pelvisControl")

    rows = build_motion_source_index(raw, tmp_path / "out.jsonl", tmp_path / "report.md", recursive=True)

    assert len(rows) == 1
    assert rows[0]["source_scene_file"] == "new_scene.json"
    assert rows[0]["source_scene_path"] == str(raw / "new_scene.json")


def test_delta_report_handles_missing_base_db_gracefully(tmp_path):
    base = tmp_path / "clean_v3"
    new = tmp_path / "clean_v3_new_scenes"
    for rel in ["semantic", "baked", "relative_motion", "pose_semantics", "interaction_semantics", "datasets", "reports"]:
        (new / rel).mkdir(parents=True, exist_ok=True)
    (base / "datasets").mkdir(parents=True, exist_ok=True)
    (new / "run_manifest.json").write_text('{"run_name":"clean_v3_new_scenes","source_folder":"raw","parent_reference_run":"clean_v3"}', encoding="utf-8")
    write_jsonl(new / "semantic" / "motion_source_index.jsonl", [{"source_scene_file": "new.json", "source_scene_path": "raw/new.json"}])
    write_jsonl(new / "baked" / "motion_sample_index.jsonl", [{"sample_id": "s1", "bake_status": "ok"}])
    write_jsonl(new / "semantic" / "movement_windows.jsonl", [{"window_id": "w1"}])
    write_jsonl(new / "relative_motion" / "relative_motion_window_index.jsonl", [{"window_id": "w1", "safe_for_learning": True}])
    write_jsonl(new / "pose_semantics" / "pose_semantics_v0.jsonl", [{"window_id": "w1", "pose_family": "cowgirl"}])
    write_jsonl(new / "interaction_semantics" / "interaction_semantics_v0.jsonl", [{"window_id": "w1", "interaction_family": "cowgirl"}])
    write_jsonl(new / "datasets" / "semantic_candidate_db_v0.jsonl", [{"window_id": "w1", "semantic_family": "cowgirl", "generation_safe": True, "contact_support": "hands_free"}])
    write_jsonl(new / "datasets" / "cowgirl_candidate_db_v0.jsonl", [{"window_id": "w1", "category": "cowgirl_clean_motion_generation_safe", "source_scene_file": "new.json"}])

    summary = compare_new_scenes_to_clean_v3(base, new, new / "reports" / "delta.md")

    assert summary["base_semantic_records"] == 0
    assert summary["new_scenes"] == 1
    assert (new / "reports" / "delta.md").exists()


def test_review_selection_enforces_scene_and_sample_caps():
    context = {"windows": {}}
    cowgirl = []
    for idx in range(6):
        wid = f"w{idx}"
        sample = "same_sample" if idx < 2 else f"s{idx}"
        scene = "scene_a.json" if idx < 4 else f"scene_{idx}.json"
        context["windows"][wid] = {
            "window_id": wid,
            "sample_id": sample,
            "source_scene_file": scene,
            "technical_atom_id": "Person",
            "start_seconds": float(idx * 4),
        }
        cowgirl.append(
            {
                "window_id": wid,
                "sample_id": sample,
                "source_scene_file": scene,
                "category": "cowgirl_clean_motion_generation_safe",
                "semantic_family": "cowgirl",
                "semantic_score": 0.9 - idx * 0.01,
                "motion_score": 0.8,
            }
        )

    selected, summary = _select_review_items(cowgirl, [], context, count=6)

    scenes = [row["source_scene_file"] for row in selected]
    samples = [row["sample_id"] for row in selected]
    assert scenes.count("scene_a.json") <= 2
    assert len(samples) == len(set(samples))
    assert summary["rejected_by_rule"].get("scene_cap", 0) >= 1 or summary["rejected_by_rule"].get("sample_cap", 0) >= 1


def test_new_scene_helpers_do_not_touch_existing_clean_v3_or_manual_labels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    base = tmp_path / "data" / "runs" / "clean_v3"
    base.mkdir(parents=True)
    marker = base / "do_not_touch.txt"
    marker.write_text("keep", encoding="utf-8")
    out = tmp_path / "data" / "runs" / "clean_v3_new_scenes"

    _create_structure(raw, base, out)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (base / "labels" / "manual_labels.yaml").exists()
    assert not (out / "ml").exists()


def _focused_context():
    windows = {}
    samples = {}
    sources = {}
    relative = {}
    for idx in range(24):
        wid = f"w{idx}"
        sample = f"s{idx}"
        source = f"src{idx}"
        windows[wid] = {
            "window_id": wid,
            "sample_id": sample,
            "source_id": source,
            "source_scene_file": f"scene_{idx % 8}.json",
            "source_scene_path": f"local_scene_batch/scene_{idx % 8}.json",
            "technical_atom_id": "Person",
            "start_seconds": float(idx * 4),
            "end_seconds": float(idx * 4 + 4),
            "duration_seconds": 4.0,
        }
        samples[sample] = {"sample_id": sample, "source_id": source, "clip_name": f"clip_{idx}", "source_type": "timeline_controller_motion"}
        sources[source] = {"source_id": source, "clip_name": f"clip_{idx}", "clip_index": idx, "storable_id": "Timeline"}
        relative[wid] = {
            "window_id": wid,
            "feature_values": {
                "relative_pelvis_vertical_amplitude": 0.12,
                "relative_pelvis_forward_back_amplitude": 0.18,
                "relative_pelvis_lateral_amplitude": 0.08,
                "local_path_length": 0.8,
                "local_motion_energy": 0.05,
                "local_velocity_mean": 0.2,
            },
        }
    return {"windows": windows, "samples": samples, "sources": sources, "relative": relative}


def _focused_row(idx: int, category: str, family: str = "cowgirl") -> dict:
    return {
        "window_id": f"w{idx}",
        "sample_id": f"s{idx}",
        "source_scene_file": f"scene_{idx % 8}.json",
        "source_scene_path": f"local_scene_batch/scene_{idx % 8}.json",
        "technical_actor_id": "Person",
        "category": category,
        "semantic_family": family,
        "pose_family": "cowgirl" if family == "cowgirl" else "unknown",
        "pose_subtype": "cowgirl_kneeling",
        "motion_subtype": "vertical_bounce",
        "phase": "clean_motion",
        "contact_support": "ambiguous_partner_contact" if category == "contact" else "unknown_contact",
        "generation_safe": category == "cowgirl_clean_motion_generation_safe",
        "semantic_score": 0.9,
        "motion_score": 0.8,
        "hip_motion_strength": 0.7,
        "pelvis_trajectory_strength": 0.8,
        "partner_context_confidence": 0.4,
        "contact_support_confidence": 0.5,
        "contact_support_ambiguous": category == "contact",
    }


def test_focused_review_selection_respects_count_targets_and_caps():
    context = _focused_context()
    cowgirl = []
    cowgirl.extend(_focused_row(i, "cowgirl_clean_motion_generation_safe") for i in range(4))
    cowgirl.extend(_focused_row(i + 4, "not_cowgirl_bj_oral", family="bj_oral") for i in range(4))
    semantic = [_focused_row(i + 8, "contact", family="cowgirl") for i in range(4)]
    for row in semantic:
        row["contact_support"] = "ambiguous_partner_contact"
        row["contact_support_ambiguous"] = True
    targets = {"cowgirl_clean_motion_generation_safe": 3, "not_cowgirl_bj_oral": 2, "contact_support_ambiguous": 2}

    selected, summary = select_focused_review_cards(cowgirl, semantic, context, targets)

    counts = summary["selected_counts"]
    assert counts["cowgirl_clean_motion_generation_safe"] == 3
    assert counts["not_cowgirl_bj_oral"] == 2
    assert counts["contact_support_ambiguous"] == 2
    assert len({row["sample_id"] for row in selected}) == len(selected)


def test_focused_review_cards_include_required_diagnostics():
    context = _focused_context()
    row = _focused_row(0, "cowgirl_clean_motion_generation_safe")
    card = _review_card(1, row, context)

    assert card["review_label"] == "001_cowgirl_clean"
    for field in DIAGNOSTIC_FIELDS:
        assert field in card or field in card.get("diagnostic_fields_present", [])
    assert card["source_id"] == "src0"
    assert card["motion_metrics"]["local_path_length"] == 0.8
    assert card["axis_breakdown"]["forward_back"] == 0.18
    assert card["likely_failure_mode"]


def test_focused_review_labels_and_questions_are_in_ui_schema():
    schema = answer_schema()

    for label in REVIEW_LABELS:
        assert label in schema["review_labels"]
    for question in REVIEW_QUESTIONS:
        assert question in schema["review_questions"]
    assert "review_labels" in schema["fields"]


def test_strict_cowgirl_review_filters_setup_and_keeps_reverse_name():
    context = _focused_context()
    clean = _focused_row(0, "cowgirl_clean_motion_generation_safe")
    clean["hip_motion_strength"] = 1.0
    clean["pelvis_trajectory_strength"] = 1.0
    clean["clean_motion_gate"] = "pass"
    context["sources"]["src0"]["clip_name"] = "Reverse Cowgirl"
    intro = _focused_row(1, "cowgirl_clean_motion_generation_safe")
    intro["hip_motion_strength"] = 1.0
    intro["pelvis_trajectory_strength"] = 1.0
    intro["clean_motion_gate"] = "pass"
    context["sources"]["src1"]["clip_name"] = "Intro Mount Switch"
    standing = _focused_row(2, "cowgirl_clean_motion_generation_safe", family="hand_gesture")
    standing["hip_motion_strength"] = 1.0
    standing["pelvis_trajectory_strength"] = 1.0
    standing["clean_motion_gate"] = "pass"

    selected, summary = select_strict_cowgirl_review_cards([clean, intro, standing], context)

    assert [row["window_id"] for row in selected] == ["w0"]
    assert summary["rejected_by_rule"]["excluded_clip_token_switch"] == 1
    assert summary["rejected_by_rule"]["not_cowgirl_semantic_family"] == 1


def test_strict_cowgirl_review_uses_human_exclusions():
    context = _focused_context()
    row = _focused_row(0, "cowgirl_clean_motion_generation_safe")
    row["hip_motion_strength"] = 1.0
    row["pelvis_trajectory_strength"] = 1.0
    row["clean_motion_gate"] = "pass"

    selected, summary = select_strict_cowgirl_review_cards([row], context, human_exclusions={"w0"})

    assert selected == []
    assert summary["rejected_by_rule"]["rejected_by_human_notes_from_previous_batch"] == 1


def test_focused_review_output_paths_stay_inside_new_scene_run(tmp_path):
    run = (tmp_path / "data" / "runs" / "clean_v3_new_scenes").resolve()
    out = (run / "audits" / "focused").resolve()
    run.mkdir(parents=True)

    _ensure_inside(run, out)

    outside = (tmp_path / "data" / "runs" / "clean_v3" / "audits" / "focused").resolve()
    try:
        _ensure_inside(run, outside)
    except ValueError:
        pass
    else:
        raise AssertionError("outside clean_v3 path should be rejected")


def test_review_timeline_segments_copy_to_vam_animations_and_update_manifest(tmp_path):
    run = tmp_path / "data" / "runs" / "clean_v3_new_scenes"
    review = run / "audits" / "semantic_review_new_scenes_020_focused"
    package = review / "vam_review_package"
    segment_dir = package / "timeline_segments" / "review_001"
    segment_dir.mkdir(parents=True)
    segment = segment_dir / "review_001.timeline.json"
    segment.write_text('{"SerializeVersion":283}', encoding="utf-8")
    write_jsonl(
        package / "vam_review_manifest.jsonl",
        [
            {
                "review_id": "review_001",
                "timeline_export_path": str(segment),
                "source_scene_file": "scene.json",
                "technical_atom_id": "Person",
                "start_seconds": 0,
                "end_seconds": 2,
                "why_selected": "cowgirl_clean_motion_generation_safe",
                "semantic_family": "cowgirl",
            }
        ],
    )
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "w1"}])
    vam_dir = tmp_path / "VAM" / "Saves" / "PluginData" / "animations"

    summary = export_review_timeline_segments_to_vam(review, vam_dir, None, "VAMTimelineAI/test_review")
    manifest = load_jsonl(package / "vam_review_manifest.jsonl")
    review_rows = load_jsonl(review / "semantic_review_010.jsonl")

    assert summary["copied"] == 1
    assert Path(manifest[0]["vam_animation_path"]).exists()
    assert review_rows[0]["vam_animation_export_status"] == "copied"
    assert str(vam_dir) in manifest[0]["vam_animation_path"]


def test_review_timeline_segment_names_are_semantic_and_numbered(tmp_path):
    run = tmp_path / "data" / "runs" / "clean_v3_new_scenes"
    review = run / "audits" / "semantic_review_new_scenes_020_focused"
    package = review / "vam_review_package"
    segment_dir = package / "timeline_segments" / "review_001"
    segment_dir.mkdir(parents=True)
    segment = segment_dir / "review_001.timeline.json"
    segment.write_text('{"SerializeVersion":283}', encoding="utf-8")
    write_jsonl(
        package / "vam_review_manifest.jsonl",
        [
            {
                "review_id": "review_001",
                "review_label": "001_cowgirl_clean",
                "timeline_export_path": str(segment),
                "source_scene_file": "long_scene_name.json",
                "technical_atom_id": "Person",
                "why_selected": "cowgirl_clean_motion_generation_safe",
            }
        ],
    )
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "review_label": "001_cowgirl_clean"}])
    vam_dir = tmp_path / "VAM" / "Saves" / "PluginData" / "animations"

    summary = export_review_timeline_segments_to_vam(review, vam_dir, None, "short_names")
    copied = list((vam_dir / "short_names").glob("*.timeline.json"))

    assert summary["copied"] == 1
    assert copied[0].name == "001_cowgirl_clean.timeline.json"

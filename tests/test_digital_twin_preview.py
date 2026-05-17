from pathlib import Path

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import build_review_ui_data, build_static_review_ui
from vam_timeline_ai.visualization.digital_twin_preview import render_digital_twin_previews_v1, render_digital_twin_review_previews_v0
from vam_timeline_ai.visualization.visual_judge_schema import visual_judge_schema_v0
from vam_timeline_ai.visualization.visual_judge_requests import build_visual_judge_requests_v0


def _sample_npz(path: Path) -> None:
    names = [
        "pelvisControl",
        "chestControl",
        "headControl",
        "lElbowControl",
        "rElbowControl",
        "lHandControl",
        "rHandControl",
        "lKneeControl",
        "rKneeControl",
        "lFootControl",
        "rFootControl",
    ]
    times = np.linspace(0.0, 2.0, 121, dtype=np.float32)
    positions = np.zeros((len(times), len(names), 3), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}
    positions[:, idx["pelvisControl"], :] = np.stack(
        [0.05 * np.sin(times * 4), 0.75 + 0.03 * np.sin(times * 8), 0.04 * np.cos(times * 4)],
        axis=1,
    )
    offsets = {
        "chestControl": [0.0, 0.45, -0.05],
        "headControl": [0.0, 0.75, -0.08],
        "lElbowControl": [-0.25, 0.28, -0.04],
        "rElbowControl": [0.25, 0.28, -0.04],
        "lHandControl": [-0.35, 0.08, -0.05],
        "rHandControl": [0.35, 0.08, -0.05],
        "lKneeControl": [-0.22, -0.45, 0.04],
        "rKneeControl": [0.22, -0.45, 0.04],
        "lFootControl": [-0.26, -0.72, 0.18],
        "rFootControl": [0.26, -0.72, 0.18],
    }
    pelvis = positions[:, idx["pelvisControl"], :]
    for name, offset in offsets.items():
        positions[:, idx[name], :] = pelvis + np.asarray(offset, dtype=np.float32)
    np.savez_compressed(path, times=times, positions=positions, controller_names=np.asarray(names, dtype=object))


def test_digital_twin_preview_renders_or_reports_missing_matplotlib(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "ml_review"
    (run / "baked").mkdir(parents=True)
    review.mkdir(parents=True)
    npz = tmp_path / "sample.npz"
    _sample_npz(npz)
    write_jsonl(
        run / "baked" / "motion_sample_index.jsonl",
        [{"sample_id": "s1", "bake_status": "ok", "baked_npz_path": str(npz), "controller_names": ["pelvisControl"]}],
    )
    write_jsonl(
        review / "semantic_review_010.jsonl",
        [
            {
                "review_id": "review_001",
                "sample_id": "s1",
                "window_id": "w1",
                "start_seconds": 0,
                "end_seconds": 2,
                "semantic_family": "cowgirl",
                "pose_subtype": "cowgirl_lean_back_supported",
                "motion_subtype": "oval_grind",
                "phase": "clean_motion",
                "contact_support": "hands_on_partner_legs_or_thighs",
                "model_cowgirl_probability": 0.8,
            }
        ],
    )

    summary = render_digital_twin_review_previews_v0(run, review, review / "digital_twin_previews")
    manifest = load_jsonl(review / "digital_twin_previews" / "digital_twin_preview_manifest.jsonl")

    assert summary["review_items"] == 1
    assert summary["visual_judgments_are_ground_truth"] is False
    if manifest[0]["status"] == "rendered":
        assert (review / "digital_twin_previews" / "review_001" / "contact_sheet.png").exists()
    else:
        assert manifest[0]["warnings"]


def test_review_ui_data_includes_digital_twin_contact_sheet(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "ml_review"
    (review / "digital_twin_previews" / "review_001").mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "w1"}])
    write_jsonl(
        review / "digital_twin_previews" / "digital_twin_preview_manifest.jsonl",
        [{"review_id": "review_001", "status": "rendered", "warnings": ["partner reference markers are proxy/unavailable"]}],
    )

    data = build_review_ui_data(run, review)
    build_static_review_ui(run, review, review / "review_ui_static")
    js = (review / "review_ui_static" / "review_data.js").read_text(encoding="utf-8")
    app = (review / "review_ui_static" / "app.js").read_text(encoding="utf-8")

    assert data["review_items"][0]["digital_twin_contact_sheet"] == "../digital_twin_previews/review_001/contact_sheet.png"
    assert "digital_twin_contact_sheet" in js
    assert "digital twin preview" in app


def test_visual_judge_schema_is_audit_only():
    schema = visual_judge_schema_v0()

    assert schema["audit_only"] is True
    assert schema["may_be_used_as_ground_truth_without_human_confirmation"] is False
    assert "visual_contact_guess" in schema["fields"]


def test_digital_twin_v1_creates_frames_and_metadata(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "ml_review"
    (run / "baked").mkdir(parents=True)
    review.mkdir(parents=True)
    npz = tmp_path / "sample.npz"
    _sample_npz(npz)
    write_jsonl(run / "baked" / "motion_sample_index.jsonl", [{"sample_id": "s1", "bake_status": "ok", "baked_npz_path": str(npz)}])
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "sample_id": "s1", "start_seconds": 0, "end_seconds": 2, "semantic_family": "cowgirl"}])

    summary = render_digital_twin_previews_v1(run, review, review / "digital_twin_previews_v1", fps=8, width=480, height=360, frames=4, make_gif=True, make_mp4=False)
    manifest = load_jsonl(review / "digital_twin_previews_v1" / "digital_twin_preview_manifest_v1.jsonl")

    assert summary["previews_rendered"] == 1
    assert (review / "digital_twin_previews_v1" / "items" / "review_001" / "frames" / "frame_000.png").exists()
    assert (review / "digital_twin_previews_v1" / "items" / "review_001" / "contact_sheet_large.png").exists()
    assert (review / "digital_twin_previews_v1" / "items" / "review_001" / "metadata.json").exists()
    assert manifest[0]["primary_visual_type"] in {"gif", "contact_sheet"}
    assert manifest[0]["visual_judgments_are_ground_truth"] is False


def test_static_review_ui_prefers_v1_gif_over_static_plot(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "ml_review"
    (review / "digital_twin_previews_v1" / "items" / "review_001").mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "w1"}])
    write_jsonl(
        review / "digital_twin_previews_v1" / "digital_twin_preview_manifest_v1.jsonl",
        [{"review_id": "review_001", "status": "rendered", "gif_path": "preview.gif", "contact_sheet_large_path": "sheet.png", "primary_visual_type": "gif", "visual_quality": "high"}],
    )

    build_static_review_ui(run, review, review / "review_ui_static")
    js = (review / "review_ui_static" / "review_data.js").read_text(encoding="utf-8")
    app = (review / "review_ui_static" / "app.js").read_text(encoding="utf-8")

    assert "digital_twin_gif" in js
    assert "digital_twin_contact_sheet_large" in js
    assert "Only static technical plot available" in app


def test_visual_judge_request_prefers_gif_over_contact_sheet(tmp_path):
    review = tmp_path / "review"
    preview = review / "digital_twin_previews_v1"
    review.mkdir(parents=True)
    preview.mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "w1"}])
    write_jsonl(
        preview / "digital_twin_preview_manifest_v1.jsonl",
        [{"review_id": "review_001", "gif_path": "preview.gif", "contact_sheet_large_path": "sheet.png", "warnings": []}],
    )

    summary = build_visual_judge_requests_v0(review, preview, review / "visual_judge_requests.jsonl", mode="blind")
    rows = load_jsonl(review / "visual_judge_requests.jsonl")

    assert summary["requests"] == 1
    assert rows[0]["primary_visual_type"] == "gif"
    assert rows[0]["primary_visual_path"] == "preview.gif"
    assert rows[0]["visual_output_is_ground_truth"] is False

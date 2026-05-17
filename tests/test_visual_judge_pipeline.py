from pathlib import Path

from vam_timeline_ai.audits.multisignal_triage import build_multisignal_review_priorities_v0
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import build_review_ui_data
from vam_timeline_ai.vision.keyframe_sampling import (
    build_contact_sheet_metadata,
    choose_adaptive_frame_times,
    choose_frame_times,
)
from vam_timeline_ai.vision.lmstudio_vlm_judge import run_lmstudio_vlm_judge_v0
from vam_timeline_ai.vision.vam_capture_bridge_client import run_vam_reality_capture_v0
from vam_timeline_ai.vision.vam_capture_requests import build_vam_capture_requests_v0
from vam_timeline_ai.vision.visual_judge_calibration import build_visual_judge_calibration_set_v1
from vam_timeline_ai.vision.visual_judge_prompts import build_visual_judge_prompt
from vam_timeline_ai.vision.visual_judge_requests import build_visual_judge_requests_v1
from vam_timeline_ai.vision.visual_judge_schema import (
    parse_json_from_text_response,
    validate_visual_judge_result,
)
from vam_timeline_ai.vision.visual_judge_trust_gate import evaluate_vlm_visual_judge_v1


def test_visual_schema_normalizes_and_enforces_single_frame_rule():
    row = validate_visual_judge_result(
        {
            "visual_input_type": "single_frame",
            "partner_visible": False,
            "motion_visible": False,
            "suggested_family": "doggy",
            "family_confidence": 0.9,
            "body_pose_guess": "kneeling",
        }
    )

    assert row["suggested_family"] == "unknown"
    assert row["family_confidence"] <= 0.35
    assert row["body_pose_guess"] == "kneeling"
    assert row["warnings"]


def test_parse_json_from_text_response_handles_fenced_json():
    parsed, status = parse_json_from_text_response('text\n```json\n{"suggested_family":"cowgirl"}\n```')

    assert status == "ok"
    assert parsed["suggested_family"] == "cowgirl"


def test_prompt_builder_requires_json_and_technical_no_moralizing():
    prompt = build_visual_judge_prompt()

    assert "Return JSON only" in prompt
    assert "Do not give moral commentary" in prompt
    assert "do not infer doggy from kneeling alone" in prompt


def test_capture_request_builder_creates_item_requests(tmp_path):
    review = tmp_path / "review"
    (review / "vam_review_package").mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "source_scene_file": "scene.json", "technical_actor_id": "Person", "start_seconds": 1, "end_seconds": 3}])

    summary = build_vam_capture_requests_v0(review, review / "vam_capture_requests.jsonl", review / "vam_captures", frame_count=8, duration_seconds=2)
    rows = load_jsonl(review / "vam_capture_requests.jsonl")

    assert summary["requests"] == 1
    assert rows[0]["review_id"] == "review_001"
    assert rows[0]["captures_scene_automatically"] is False


def test_capture_client_handles_bridge_unavailable(tmp_path):
    requests = tmp_path / "requests.jsonl"
    out = tmp_path / "results.jsonl"
    write_jsonl(requests, [{"review_id": "review_001"}])

    summary = run_vam_reality_capture_v0(requests, "http://127.0.0.1:1", "status_only", out)

    assert summary["status"] == "blocked"
    assert Path(summary["blocked_report"]).exists()


def test_visual_request_builder_prefers_real_vam_capture_over_digital(tmp_path):
    review = tmp_path / "review"
    capture = review / "vam_capture_contact_sheets"
    digital = review / "digital_twin_previews_v1"
    capture.mkdir(parents=True)
    digital.mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "semantic_family": "cowgirl"}])
    write_jsonl(capture / "vam_capture_contact_sheet_manifest.jsonl", [{"review_id": "review_001", "contact_sheet_path": "real_sheet.png"}])
    write_jsonl(digital / "digital_twin_preview_manifest_v1.jsonl", [{"review_id": "review_001", "mp4_path": "digital.mp4"}])

    summary = build_visual_judge_requests_v1(review, capture, digital, review / "visual_judge_requests.jsonl")
    rows = load_jsonl(review / "visual_judge_requests.jsonl")

    assert summary["requests"] == 1
    assert rows[0]["primary_visual_path"] == "real_sheet.png"
    assert rows[0]["visual_quality"] == "high_real_vam_capture"


def test_visual_request_builder_prefers_digital_contact_sheet_over_mp4(tmp_path):
    review = tmp_path / "review"
    digital = review / "digital_twin_previews_v1"
    capture = review / "vam_capture_contact_sheets"
    digital.mkdir(parents=True)
    capture.mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "semantic_family": "cowgirl"}])
    write_jsonl(
        digital / "digital_twin_preview_manifest_v1.jsonl",
        [{"review_id": "review_001", "mp4_path": "digital.mp4", "gif_path": "digital.gif", "contact_sheet_large_path": "sheet.png"}],
    )

    build_visual_judge_requests_v1(review, capture, digital, review / "visual_judge_requests.jsonl")
    rows = load_jsonl(review / "visual_judge_requests.jsonl")

    assert rows[0]["primary_visual_path"] == "sheet.png"
    assert rows[0]["primary_visual_type"] == "contact_sheet"
    assert "digital.gif" in rows[0]["fallback_paths"]


def test_lmstudio_adapter_dry_run_writes_planned_payloads(tmp_path):
    requests = tmp_path / "requests.jsonl"
    out = tmp_path / "results.jsonl"
    raw = tmp_path / "raw"
    write_jsonl(requests, [{"review_id": "review_001", "primary_visual_path": "x.png", "primary_visual_type": "contact_sheet", "visual_quality": "medium_contact_sheet", "prompt_text": "Return JSON only"}])

    summary = run_lmstudio_vlm_judge_v0(requests, "http://localhost:1234/v1", "nsfwvision-v4-qwen3.5-9b", out, raw, dry_run=True)
    rows = load_jsonl(out)

    assert summary["status"] == "dry_run"
    assert rows[0]["parse_status"] == "dry_run"
    assert (raw / "review_001_planned_payload.json").exists()


def test_lmstudio_adapter_unavailable_writes_blocked_report(tmp_path):
    requests = tmp_path / "requests.jsonl"
    out = tmp_path / "results.jsonl"
    raw = tmp_path / "raw"
    write_jsonl(requests, [{"review_id": "review_001", "primary_visual_path": "x.png", "prompt_text": "Return JSON only"}])

    summary = run_lmstudio_vlm_judge_v0(requests, "http://127.0.0.1:1/v1", "nsfwvision-v4-qwen3.5-9b", out, raw, dry_run=False)

    assert summary["status"] == "blocked"
    assert Path(summary["blocked_report"]).exists()


def test_keyframe_sampling_prefers_motion_but_keeps_window_context():
    even = choose_frame_times(10, 14, 5)
    adaptive = choose_adaptive_frame_times(10, 14, 5, [{"time": 11.0, "delta": 0.2}, {"time": 12.5, "delta": 0.9}])
    meta = build_contact_sheet_metadata("review_001", 10, 14, adaptive)

    assert even == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert adaptive[0] == 10.0
    assert 12.5 in adaptive
    assert meta["recommended_for_vlm"] is True


def test_calibration_trust_gate_dry_run_is_disabled(tmp_path):
    run = tmp_path / "clean_v3"
    out = run / "vision" / "calib"
    calib = build_visual_judge_calibration_set_v1(run, out)
    trust = evaluate_vlm_visual_judge_v1(out / "calibration_items.jsonl", "http://localhost:1234/v1", "nsfwvision-v4-qwen3.5-9b", out / "eval", dry_run=True)

    assert calib["items"] >= 1
    assert trust["trust_gate"] == "disabled"


def test_multisignal_triage_ignores_dry_run_visual_family(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "review"
    review.mkdir(parents=True)
    scores = run / "ml" / "scores.jsonl"
    scores.parent.mkdir(parents=True)
    visual = review / "visual_judge_results.jsonl"
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001", "window_id": "w1", "semantic_family": "cowgirl"}])
    write_jsonl(scores, [{"window_id": "w1", "model_cowgirl_probability": 0.99}])
    write_jsonl(visual, [{"review_id": "review_001", "parse_status": "dry_run", "suggested_family": "cowgirl", "evidence_sufficient_for_family": True}])

    summary = build_multisignal_review_priorities_v0(run, review, scores, visual, review / "multi.jsonl", review / "multi.md")
    rows = load_jsonl(review / "multi.jsonl")

    assert summary["items"] == 1
    assert rows[0]["multisignal_priority"] == "must_review"


def test_review_ui_data_includes_visual_fields(tmp_path):
    run = tmp_path / "clean_v3"
    review = run / "audits" / "review"
    review.mkdir(parents=True)
    write_jsonl(review / "semantic_review_010.jsonl", [{"review_id": "review_001"}])
    write_jsonl(review / "visual_judge_results.jsonl", [{"review_id": "review_001", "suggested_family": "unknown", "parse_status": "dry_run", "body_pose_guess": "kneeling"}])
    write_jsonl(review / "multisignal_review_priorities.jsonl", [{"review_id": "review_001", "multisignal_priority": "must_review", "reason": "dry-run"}])

    data = build_review_ui_data(run, review)

    item = data["review_items"][0]
    assert item["visual_suggested_family"] == "unknown"
    assert item["multisignal_priority"] == "must_review"

from pathlib import Path
import json

from vam_timeline_ai.datasets.manual_pose_explanation_parser import parse_explanation_text
from vam_timeline_ai.datasets.manual_pose_ground_truth import build_manual_pose_ground_truth_v1
from vam_timeline_ai.datasets.manual_pose_measurements import compute_manual_pose_measurements
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl
from vam_timeline_ai.reports.manual_pose_ground_truth_report import (
    build_manual_pose_ground_truth_gallery_v1,
    report_manual_pose_ground_truth_v1,
)


def _ctrl(x, y, z):
    return {
        "exists": True,
        "world_position": [x, y, z],
        "world_rotation_quat": [0, 0, 0, 1],
        "local_position_to_atom": [x, y, z],
        "local_rotation_to_atom_quat": [0, 0, 0, 1],
        "active": True,
    }


def _capture():
    return {
        "schema_version": "pose_capture_v1",
        "created_at": "2026-05-16T20:41:10",
        "source": "VaM SkeletonPoseCaptureTool",
        "atoms": {
            "rider": {
                "atom_uid": "Quiet",
                "atom_name": "Quiet",
                "controllers": {
                    "pelvisControl": _ctrl(0, 1.0, 0),
                    "chestControl": _ctrl(0, 1.45, 0.1),
                    "headControl": _ctrl(0, 1.75, 0.1),
                    "lHandControl": _ctrl(-0.2, 1.2, 0.1),
                    "rHandControl": _ctrl(0.2, 1.2, 0.1),
                    "lKneeControl": _ctrl(-0.35, 0.55, 0.0),
                    "rKneeControl": _ctrl(0.35, 0.55, 0.0),
                    "lFootControl": _ctrl(-0.45, 0.05, 0.1),
                    "rFootControl": _ctrl(0.45, 0.05, 0.1),
                    "lThighControl": _ctrl(-0.25, 0.8, 0.0),
                    "rThighControl": _ctrl(0.25, 0.8, 0.0),
                },
                "missing_controllers": [],
            },
            "partner": {
                "atom_uid": "Person",
                "atom_name": "Person",
                "controllers": {
                    "pelvisControl": _ctrl(0, 0.85, 0),
                    "chestControl": _ctrl(0, 1.15, 0.1),
                    "headControl": _ctrl(0, 1.4, 0.2),
                    "lThighControl": _ctrl(-0.25, 0.7, -0.1),
                    "rThighControl": _ctrl(0.25, 0.7, -0.1),
                    "lKneeControl": _ctrl(-0.35, 0.45, -0.1),
                    "rKneeControl": _ctrl(0.35, 0.45, -0.1),
                },
                "missing_controllers": [],
            },
        },
        "derived": {
            "rider_pelvis_to_partner_pelvis": {
                "world_delta": [0, 0.15, 0],
                "distance": 0.15,
                "partner_local_delta": [0, 0.15, 0],
            }
        },
        "pose_quality_flags": {"warnings": []},
    }


def test_explanation_parser_maps_core_families_and_drivers():
    text = """
pose_capture_20260516_204110:
- klassische cowgirl pose, leicht nach vorn gelehnt
- man kann jeglich cowgirl motions nutzen wie bouncing, grinding, riding usw.
- fuesse meist komplett still

pose_capture_20260516_203203:
- BJ auf den knien
- die frau bewegt ihren kopf nach vorn und zurueck wie ueblich bei BJ

pose_capture_20260516_205956:
- HJ pose
- andere hand macht typische vor und rueckwaertsbewegung wie beim Handjob

pose_capture_20260516_204615:
- klassische doggy stellung
- frau passiv

pose_capture_20260516_210433:
- missionary pose mit legs up
- die fuesse bewegen sich passend zu den stoessen des mannes
"""
    labels = parse_explanation_text(text)
    by_id = {row["capture_id"]: row for row in labels}

    assert by_id["pose_capture_20260516_204110"]["family"] == "cowgirl"
    assert by_id["pose_capture_20260516_204110"]["primary_driver"] == "pelvis_hip"
    assert by_id["pose_capture_20260516_203203"]["family"] == "bj_oral"
    assert by_id["pose_capture_20260516_203203"]["primary_driver"] == "head_neck"
    assert by_id["pose_capture_20260516_205956"]["family"] == "handjob"
    assert by_id["pose_capture_20260516_205956"]["primary_driver"] == "hand"
    assert by_id["pose_capture_20260516_204615"]["role_active_passive"] == "female_passive_receiver"
    assert by_id["pose_capture_20260516_210433"]["family"] == "missionary"


def test_manual_pose_measurements_compute_pelvis_distance():
    metrics = compute_manual_pose_measurements(_capture(), {"primary_driver": "pelvis_hip", "foot_behavior": "mostly_static"})

    assert metrics["partner_relative"]["rider_pelvis_to_partner_pelvis_distance"] == 0.15
    assert metrics["controller_completeness"]["rider_pelvis"] is True
    assert metrics["anchor_expectations"]["pelvis_should_drive"] is True
    assert metrics["anchor_expectations"]["feet_should_be_static"] is True


def test_ground_truth_builder_matches_timestamp_and_writes_patch(tmp_path):
    capture_dir = tmp_path / "raw"
    capture_dir.mkdir()
    dump_json(capture_dir / "pose_capture_20260516_204110.json", _capture())
    (capture_dir / "pose_capture_20260516_204110.png").write_bytes(b"fake")
    labels = {
        "schema_version": "manual_pose_human_labels_v1",
        "labels": [
            {
                "capture_id": "pose_capture_20260516_204110",
                "family": "cowgirl",
                "pose_subtype": "cowgirl_classic_lean_forward_light",
                "motion_intent": "bouncing, grinding, riding",
                "primary_driver": "pelvis_hip",
                "anchors": ["feet_static"],
                "hand_support_options": ["hands_on_partner_chest"],
                "foot_behavior": "mostly_static",
                "knee_behavior": "may_phase_out_in",
                "generation_valid_motions": ["bouncing", "grinding", "riding"],
                "raw_notes": "klassische cowgirl pose",
            }
        ],
    }
    label_path = tmp_path / "labels.json"
    dump_json(label_path, labels)

    out_jsonl = tmp_path / "manual_pose_ground_truth_v1.jsonl"
    summary = build_manual_pose_ground_truth_v1(capture_dir, label_path, out_jsonl, tmp_path / "out.csv", tmp_path / "report.md")
    rows = load_jsonl(out_jsonl)
    patch = load_json if False else Path(summary["ontology_patch"]).read_text(encoding="utf-8")

    assert summary["captures"] == 1
    assert summary["matched_labels"] == 1
    assert rows[0]["capture_id"] == "pose_capture_20260516_204110"
    assert rows[0]["human_labels"]["family"] == "cowgirl"
    assert rows[0]["screenshot_path"].endswith(".png")
    assert "cowgirl_classic_lean_forward_light" in patch
    assert rows[0]["ml_training_run"] is False
    assert rows[0]["manual_labels_yaml_modified"] is False


def test_ground_truth_reports_and_gallery_work(tmp_path):
    capture_dir = tmp_path / "raw"
    capture_dir.mkdir()
    dump_json(capture_dir / "pose_capture_20260516_204110.json", _capture())
    (capture_dir / "pose_capture_20260516_204110.png").write_bytes(b"fake")
    label_path = tmp_path / "labels.json"
    dump_json(
        label_path,
        {
            "labels": [
                {
                    "capture_id": "pose_capture_20260516_204110",
                    "family": "cowgirl",
                    "pose_subtype": "cowgirl_classic",
                    "motion_intent": "grinding",
                    "primary_driver": "pelvis_hip",
                    "anchors": ["feet_static"],
                    "hand_support_options": [],
                    "generation_valid_motions": ["grinding"],
                    "raw_notes": "manual note",
                }
            ]
        },
    )
    out_jsonl = tmp_path / "gt.jsonl"
    build_manual_pose_ground_truth_v1(capture_dir, label_path, out_jsonl, tmp_path / "gt.csv", tmp_path / "build.md")

    report_summary = report_manual_pose_ground_truth_v1(out_jsonl, tmp_path / "reports")
    gallery_summary = build_manual_pose_ground_truth_gallery_v1(out_jsonl, tmp_path / "index.html")

    assert report_summary["family_counts"]["cowgirl"] == 1
    assert (tmp_path / "reports" / "overview.md").exists()
    assert (tmp_path / "reports" / "cowgirl_pose_ground_truth.md").exists()
    assert gallery_summary["captures"] == 1
    assert "Manual Pose Ground Truth V1" in (tmp_path / "index.html").read_text(encoding="utf-8")

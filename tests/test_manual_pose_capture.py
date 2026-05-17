from pathlib import Path
import re

from vam_timeline_ai.datasets.manual_pose_capture_importer import import_manual_pose_captures_v1
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl
from vam_timeline_ai.reports.manual_pose_capture_report import report_manual_pose_captures_v1


ROOT = Path(__file__).resolve().parents[1]
CS_SOURCE = ROOT / "vam_runtime" / "source" / "SkeletonPoseCaptureTool.cs"


def _synthetic_capture() -> dict:
    def ctrl(x, y, z):
        return {
            "exists": True,
            "world_position": [x, y, z],
            "world_rotation_quat": [0, 0, 0, 1],
            "local_position_to_atom": [x, y, z],
            "local_rotation_to_atom_quat": [0, 0, 0, 1],
            "active": True,
        }

    return {
        "schema_version": "pose_capture_v1",
        "created_at": "2026-05-16T10:00:00Z",
        "source": "VaM SkeletonPoseCaptureTool",
        "vam_version": "test",
        "scene_name": "unit",
        "human_labels": {
            "pose_family": "cowgirl",
            "pose_subtype": "cowgirl_lean_back_supported",
            "motion_intent": "pose hold",
            "human_notes": "manual test pose",
        },
        "atoms": {
            "rider": {
                "atom_uid": "Female",
                "atom_name": "Female",
                "controllers": {
                    "pelvisControl": ctrl(0, 1.05, 0),
                    "headControl": ctrl(0, 1.7, -0.2),
                    "chestControl": ctrl(0, 1.45, -0.1),
                },
                "missing_controllers": ["lToeControl"],
            },
            "partner": {
                "atom_uid": "Male",
                "atom_name": "Male",
                "controllers": {
                    "pelvisControl": ctrl(0, 0.9, 0),
                    "chestControl": ctrl(0, 1.2, 0.25),
                    "headControl": ctrl(0, 1.45, 0.45),
                },
                "missing_controllers": [],
            },
        },
        "derived": {
            "rider_pelvis_to_partner_pelvis": {
                "world_delta": [0, 0.15, 0],
                "distance": 0.15,
                "partner_local_delta": [0, 0.15, 0],
            },
            "orientation_hints": {
                "rider_facing_relative_to_partner": "front_to_partner",
                "pose_hint": "kneeling_or_squat",
            },
        },
        "pose_quality_flags": {
            "has_rider": True,
            "has_partner": True,
            "warnings": [],
        },
    }


def test_importer_validates_synthetic_capture_and_preserves_human_labels(tmp_path):
    input_dir = tmp_path / "captures"
    input_dir.mkdir()
    dump_json(input_dir / "pose_capture_test.json", _synthetic_capture())

    out_jsonl = tmp_path / "manual_pose_captures.jsonl"
    summary = import_manual_pose_captures_v1(input_dir, out_jsonl, tmp_path / "import_report.md")
    rows = load_jsonl(out_jsonl)

    assert summary["captures"] == 1
    assert rows[0]["human_labels"]["pose_family"] == "cowgirl"
    assert rows[0]["human_labels"]["pose_subtype"] == "cowgirl_lean_back_supported"
    assert rows[0]["metrics"]["rider_pelvis_to_partner_pelvis_distance"] == 0.15
    assert rows[0]["manual_labels_yaml_modified"] is False
    assert rows[0]["ml_training_run"] is False


def test_manual_pose_report_groups_by_family_and_reports_missing_controllers(tmp_path):
    input_dir = tmp_path / "captures"
    input_dir.mkdir()
    dump_json(input_dir / "pose_capture_test.json", _synthetic_capture())
    out_jsonl = tmp_path / "manual_pose_captures.jsonl"
    import_manual_pose_captures_v1(input_dir, out_jsonl, tmp_path / "import_report.md")

    summary = report_manual_pose_captures_v1(out_jsonl, tmp_path / "pose_report.md")
    text = (tmp_path / "pose_report.md").read_text(encoding="utf-8")

    assert summary["family_counts"]["cowgirl"] == 1
    assert summary["subtype_counts"]["cowgirl_lean_back_supported"] == 1
    assert summary["missing_controller_counts"]["rider:lToeControl"] == 1
    assert "cowgirl_lean_back_supported" in text


def test_importer_handles_empty_capture_folder(tmp_path):
    input_dir = tmp_path / "empty"
    summary = import_manual_pose_captures_v1(input_dir, tmp_path / "out.jsonl", tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert summary["captures"] == 0
    assert "No captures were imported yet" in text


def test_skeleton_pose_capture_tool_static_safety_checks():
    text = CS_SOURCE.read_text(encoding="utf-8")

    assert "Capture Pose Snapshot" in text
    assert "Enable Skeleton Overlay" in text
    assert "Saves/PluginData/VAMTimelineAI/pose_captures" in text
    assert "JSONClass root" in text
    assert "using System.IO;" not in text
    assert "Directory." not in text
    assert "File.WriteAllText" not in text
    assert "SaveScene" not in text
    assert "save or modify the scene" in text
    assert not re.search(r"\bfc\.transform\.position\s*=", text)
    assert not re.search(r"\bfc\.transform\.rotation\s*=", text)
    assert "Timeline" not in text or "does not create Timeline clips" in text

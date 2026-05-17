from __future__ import annotations

from pathlib import Path

from vam_timeline_ai.generation.prompt_to_plan import plan_from_prompt_v1
from vam_timeline_ai.semantics.interaction_classifier import _classify_interaction_row
from vam_timeline_ai.semantics.pose_classifier import _classify_pose_row
from vam_timeline_ai.semantics.pose_support_rescan import _build_cowgirl_db_v8, _review_row


def test_prompt_maps_lean_back_partner_legs_without_reverse() -> None:
    plan = plan_from_prompt_v1("slow cowgirl leaning back hands on partner legs")
    data = plan.to_dict()
    phase = data["sequence"][0]
    query = phase["primitive_query"]
    assert query["requested_pose_subtype"] == "cowgirl_lean_back_supported"
    assert query["support_context"] == "hands_on_partner_legs_or_thighs"
    assert query["torso_lean_direction"] == "backward"
    assert query["facing_context"] == "front_cowgirl"
    assert "keep_hands_behind_on_partner_legs_or_thighs" in phase["constraints"]


def test_reverse_requires_explicit_prompt() -> None:
    front = plan_from_prompt_v1("lean back cowgirl hands behind").to_dict()
    reverse = plan_from_prompt_v1("reverse cowgirl leaning back").to_dict()
    assert front["facing_context"] == "front_cowgirl"
    assert reverse["facing_context"] == "reverse_cowgirl"


def test_pose_classifier_handles_synthetic_lean_back_supported() -> None:
    row = _classify_pose_row(
        {
            "window_id": "w1",
            "sample_id": "s1",
            "source_scene_file": "scene.json",
            "technical_atom_id": "Person",
            "kneeling_score": 0.62,
            "squat_score": 0.2,
            "standing_score": 0.0,
            "lying_on_back_score": 0.0,
            "torso_lean_back_score": 0.78,
            "torso_lean_forward_score": 0.1,
            "hands_behind_support_score": 0.72,
            "hands_on_partner_legs_score": 0.64,
            "rider_front_facing_proxy": 0.7,
            "rider_reverse_facing_proxy": 0.0,
            "pose_anchor_completeness": 0.7,
        },
        {},
    ).to_dict()
    assert row["pose_family"] == "cowgirl"
    assert row["pose_subtype"] == "cowgirl_lean_back_supported"
    assert row["torso_lean_direction"] == "backward"
    assert row["facing_context"] == "front_cowgirl"
    assert "hands_behind_support" in row["support_context"]
    assert "hands_free" not in row["support_context"]


def test_interaction_classifier_legs_thighs_distinct_from_chest_hips() -> None:
    pose = {"w1": {"pose_family": "cowgirl", "support_context": ["hands_behind_support"]}}
    interaction = _classify_interaction_row(
        {
            "window_id": "w1",
            "pair_window_id": "p1",
            "partner_window_id": "w2",
            "rider_actor_id": "Person",
            "partner_actor_id": "Partner",
            "hands_on_partner_chest_score": 0.1,
            "hands_on_partner_hips_score": 0.22,
            "hands_on_partner_legs_score": 0.72,
            "hands_on_partner_thighs_score": 0.68,
            "hands_behind_partner_support_score": 0.74,
            "rider_above_partner_score": 0.65,
            "pelvis_alignment_score": 0.6,
            "partner_lying_score": 0.5,
            "partner_context_confidence": 0.8,
        },
        pose,
    ).to_dict()
    assert interaction["support_context"] == "hands_on_partner_legs_or_thighs"
    assert interaction["contact_targets"]["lHand"] == "partner.leg_or_thigh"
    assert interaction["support_context"] not in {"hands_on_partner_chest", "hands_on_partner_hips"}


def test_cowgirl_db_v8_categories_and_review_fields() -> None:
    rows = _build_cowgirl_db_v8(
        [
            {
                "window_id": "w1",
                "semantic_family": "cowgirl",
                "pose_family": "cowgirl",
                "pose_subtype": "cowgirl_lean_back_supported",
                "contact_support": "hands_on_partner_legs_or_thighs",
                "clean_motion_gate": "pass",
                "generation_safe": True,
                "semantic_score": 0.9,
                "torso_lean_direction": "backward",
                "facing_context": "front_cowgirl",
                "hands_behind_support_score": 0.8,
                "hands_on_partner_legs_score": 0.7,
                "hands_on_partner_thighs_score": 0.6,
            }
        ]
    )
    assert rows[0]["category"] == "cowgirl_lean_back_supported_clean_motion"
    review = _review_row(1, rows[0], {"windows": {"w1": {"source_scene_file": "scene.json", "start_seconds": 1, "end_seconds": 2}}})
    assert review["torso_lean_direction"] == "backward"
    assert review["facing_context"] == "front_cowgirl"
    assert "hands_on_partner_legs_score" in review


def test_no_manual_labels_path_written(tmp_path: Path) -> None:
    from vam_timeline_ai.semantics.pose_support_rescan import store_cowgirl_lean_back_observation

    run = tmp_path / "run"
    store_cowgirl_lean_back_observation(run)
    assert not (run / "labels" / "manual_labels.yaml").exists()

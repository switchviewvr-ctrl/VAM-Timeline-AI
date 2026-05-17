from pathlib import Path

from vam_timeline_ai.generation.motion_intent_translator import translate_motion_intent_v1
from vam_timeline_ai.generation.motion_parameter_calibration import calibrate_motion_parameters_v1
from vam_timeline_ai.io.json_utils import load_json, load_jsonl, write_jsonl
from vam_timeline_ai.semantics.ontology_alignment import align_candidates_to_motion_ontology_v1
from vam_timeline_ai.semantics.ontology_loader import load_motion_families
from vam_timeline_ai.semantics.pose_first_resolver import resolve_candidate, resolve_pose_first_semantics_v1


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "data" / "ontology" / "motion_families_v1.yaml"
PHRASES = ROOT / "data" / "ontology" / "motion_phrases_v1.yaml"


def test_ontology_yaml_loads_and_cowgirl_driver_is_pelvis():
    families = load_motion_families(ONTOLOGY)

    assert "cowgirl" in families
    assert "pelvis_hip" in families["cowgirl"]["primary_motion_centers"]
    assert "head_neck" in families["bj_oral"]["primary_motion_centers"]
    assert "cowgirl_lean_back_supported" in families["cowgirl"]["compatible_pose_subtypes"]


def test_cowgirl_rejects_head_and_hand_only_drivers():
    head = resolve_candidate(
        {"window_id": "w1", "semantic_family": "cowgirl", "pose_family": "cowgirl", "pose_subtype": "cowgirl_kneeling", "motion_score": 0.1},
        relative_features={"feature_values": {"head_relative_to_chest_motion": 0.9, "local_path_length": 0.05}},
        interaction={"partner_relation": ["rider_over_receiver", "pelvis_aligned"]},
    )
    hand = resolve_candidate(
        {"window_id": "w2", "semantic_family": "cowgirl", "pose_family": "cowgirl", "pose_subtype": "cowgirl_kneeling", "motion_score": 0.1},
        relative_features={"feature_values": {"hands_relative_to_chest_pelvis_head": 0.95, "local_path_length": 0.05}},
        interaction={"partner_relation": ["rider_over_receiver", "pelvis_aligned"]},
    )

    assert head["resolved_semantic_family"] == "bj_oral"
    assert hand["resolved_semantic_family"] == "handjob"
    assert "cowgirl_clean_motion" in head["not_labels"]
    assert "cowgirl_clean_motion" in hand["not_labels"]


def test_cowgirl_pose_without_hip_motion_becomes_context_hold():
    row = resolve_candidate(
        {"window_id": "w", "pose_family": "cowgirl", "pose_subtype": "cowgirl_kneeling", "motion_content_strength": 0.0, "clean_motion_gate": "fail_low_motion"},
        relative_features={"feature_values": {"local_path_length": 0.01}},
        interaction={"partner_relation": ["rider_over_receiver"]},
    )

    assert row["resolved_semantic_family"] == "pose_context_hold"
    assert row["clean_motion_gate"] == "fail_low_motion"


def test_doggy_not_inferred_from_kneeling_alone_and_reverse_requires_back_to_partner():
    kneeling = resolve_candidate({"window_id": "w", "pose_family": "kneeling_general", "pose_subtype": "kneeling_general"}, relative_features={"feature_values": {"local_path_length": 0.1}})
    reverse = resolve_candidate(
        {"window_id": "r", "pose_family": "cowgirl", "pose_subtype": "cowgirl_kneeling", "hip_motion_strength": 1.0},
        interaction={"partner_relation": ["rider_over_receiver", "pelvis_aligned", "back_to_partner"]},
    )

    assert kneeling["resolved_semantic_family"] != "doggy"
    assert reverse["resolved_semantic_family"] == "reverse_cowgirl"


def test_lean_back_prompt_is_front_cowgirl_not_reverse(tmp_path):
    out = tmp_path / "lean_back.json"
    summary = translate_motion_intent_v1("cowgirl zurückgelehnt, hände auf seinen oberschenkeln", ONTOLOGY, PHRASES, out)
    plan = load_json(out)

    assert summary["family"] == "cowgirl"
    assert plan["pose_subtype"] == "cowgirl_lean_back_supported"
    assert plan["facing_context"] == "front_cowgirl"
    assert plan["contact_support"] == "hands_on_partner_legs_or_thighs"
    assert "cowgirl_lean_back_not_mapped_to_reverse" in plan["invalid_mappings_prevented"]


def test_reverse_prompt_maps_to_reverse_cowgirl(tmp_path):
    out = tmp_path / "reverse.json"
    translate_motion_intent_v1("standing reverse cowgirl, hips bouncing up and down", ONTOLOGY, PHRASES, out)
    plan = load_json(out)

    assert plan["family"] == "reverse_cowgirl"
    assert plan["facing_context"] == "back_to_partner"
    assert plan["motion_driver"]["primary_bodyparts"] == ["pelvis_hip", "thighs"]


def test_intent_plan_includes_driver_followers_anchors(tmp_path):
    out = tmp_path / "plan.json"
    translate_motion_intent_v1("slow cowgirl grinding, leaning forward, hands on partner chest", ONTOLOGY, PHRASES, out)
    plan = load_json(out)

    assert plan["family"] == "cowgirl"
    assert plan["motion_subtype"] == "cowgirl_grinding"
    assert plan["motion_driver"]["primary_bodyparts"]
    assert plan["followers"]
    assert plan["anchors"]
    assert plan["contact_targets"]["lHand"] == "partner.chest"


def test_ontology_alignment_detects_contradiction(tmp_path):
    run = tmp_path / "run"
    (run / "datasets").mkdir(parents=True)
    (run / "semantic_actions").mkdir(parents=True)
    sem = run / "datasets" / "semantic_candidate_db_v2.jsonl"
    cow = run / "datasets" / "cowgirl_candidate_db_v7.jsonl"
    res = run / "semantic_actions" / "pose_first_semantic_resolved_v1.jsonl"
    write_jsonl(sem, [{"window_id": "w", "candidate_id": "c", "semantic_family": "cowgirl"}])
    write_jsonl(cow, [{"window_id": "w", "category": "cowgirl_clean_motion_generation_safe"}])
    write_jsonl(res, [{"window_id": "w", "resolved_semantic_family": "handjob", "primary_motion_center": "hands", "clean_motion_gate": "pass", "conflict_flags": ["cowgirl_pose_with_hands_driver"], "missing_requirements": []}])

    summary = align_candidates_to_motion_ontology_v1(run, ONTOLOGY, sem, cow, res, run / "datasets" / "aligned.jsonl", run / "reports" / "align.md")
    rows = load_jsonl(run / "datasets" / "aligned.jsonl")

    assert summary["records"] == 1
    assert rows[0]["ontology_match"] == "conflict"
    assert any("cowgirl" in flag for flag in rows[0]["ontology_conflict"])


def test_resolver_cli_function_writes_rows(tmp_path):
    run = tmp_path / "run"
    (run / "datasets").mkdir(parents=True)
    sem = run / "datasets" / "semantic_candidate_db_v2.jsonl"
    pose = run / "pose.jsonl"
    rel = run / "rel.jsonl"
    inter = run / "inter.jsonl"
    write_jsonl(sem, [{"window_id": "w", "pose_family": "cowgirl", "pose_subtype": "cowgirl_kneeling", "hip_motion_strength": 1.0, "semantic_family": "cowgirl"}])
    write_jsonl(pose, [{"window_id": "w", "pose_family": "cowgirl", "pose_subtype": "cowgirl_kneeling"}])
    write_jsonl(rel, [{"window_id": "w", "feature_values": {"local_path_length": 1.0}}])
    write_jsonl(inter, [{"window_id": "w", "partner_relation": ["rider_over_receiver", "pelvis_aligned"]}])

    summary = resolve_pose_first_semantics_v1(run, pose, rel, inter, sem, ROOT / "data" / "ontology" / "pose_first_motion_rules_v1.yaml", run / "out.jsonl", run / "report.md")
    rows = load_jsonl(run / "out.jsonl")

    assert summary["records"] == 1
    assert rows[0]["resolved_semantic_family"] == "cowgirl"


def test_parameter_calibration_reports_insufficient_honestly(tmp_path):
    run = tmp_path / "run"
    write_jsonl(run / "resolved.jsonl", [{"window_id": "w", "resolved_motion_subtype": "cowgirl_grinding", "clean_motion_gate": "pass", "conflict_flags": []}])
    write_jsonl(run / "rel.jsonl", [{"window_id": "w", "feature_values": {"local_path_length": 1.0, "local_grind_score": 0.8}}])
    write_jsonl(run / "traj.jsonl", [])
    write_jsonl(run / "ledger.jsonl", [])

    summary = calibrate_motion_parameters_v1(run, ONTOLOGY, run / "resolved.jsonl", run / "rel.jsonl", run / "traj.jsonl", run / "ledger.jsonl", run / "profiles.json", run / "report.md")
    data = load_json(run / "profiles.json")

    assert summary["profiles"] == 0
    assert data["insufficient_samples"]["cowgirl_grinding"] == 1

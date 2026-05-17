from pathlib import Path

from vam_timeline_ai.nlp.external_lexicon_import import build_nlp_lexicon_v1
from vam_timeline_ai.nlp.nlp_token_resolver import build_motion_intent_from_prompt_v1, resolve_nlp_tokens_v1
from vam_timeline_ai.research.web_context_ontology_patches import build_web_context_ontology_patches_v1
from vam_timeline_ai.research.web_context_research import collect_web_motion_context_v1
from vam_timeline_ai.semantics.ontology_loader import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_rig_anatomy_maps_hip_and_pelvis_roles():
    anatomy = load_yaml(ROOT / "data/ontology/rig_anatomy_v1.yaml")
    roles = load_yaml(ROOT / "data/ontology/rig_anatomy_roles_by_family_v1.yaml")
    lower = anatomy["regions"]["lower_body_core"]
    pelvis = anatomy["regions"]["pelvis"]
    assert "hipControl" in lower["primary_vam_controllers"]
    assert "pelvisControl" in pelvis["primary_vam_controllers"]
    assert roles["families"]["cowgirl"]["primary_driver_regions"] == ["lower_body_core"]
    assert roles["families"]["cowgirl"]["required_visible_driver_controllers"] == ["hipControl"]
    assert "pelvis" in roles["families"]["cowgirl"]["follower_regions"]


def test_manual_lexicon_maps_de_and_en_terms():
    lexicon = load_yaml(ROOT / "data/ontology/nlp_lexicon_manual_v1.yaml")
    entries = {entry["id"]: entry for entry in lexicon["entries"]}
    assert entries["anatomy_lower_body_core"]["maps_to"]["primary_controller"] == "hipControl"
    assert entries["anatomy_pelvis"]["maps_to"]["not_primary_for"] == ["cowgirl", "reverse_cowgirl"]
    assert entries["action_hold"]["maps_to"]["mode"] == "IK_locked_dynamic"
    assert entries["action_support"]["maps_to"]["mode"] == "support_anchor"
    assert entries["pose_lean_back"]["maps_to"]["not_reverse"] is True
    assert entries["family_reverse_cowgirl"]["maps_to"]["facing_context"] == "back_to_partner"


def test_prompt_resolver_understands_hold_target_and_speed_sequence(tmp_path):
    prompt = "Die Frau hält sich an den Schultern des Mannes fest und reitet erst langsam, dann schnell."
    out = tmp_path / "tokens.json"
    result = resolve_nlp_tokens_v1(
        prompt,
        ROOT / "data/ontology/nlp_lexicon_manual_v1.yaml",
        ROOT / "data/ontology/component_ontology_v1.yaml",
        out,
    )
    assert "cowgirl" in result["families"]
    assert "hold" in result["actions"]
    assert "partner.shoulders" in result["targets"]
    assert "slow_soft" in result["styles"]
    assert "fast_impact" in result["styles"]
    assert result["generated_timeline"] is False


def test_motion_intent_plan_keeps_sequence_order_and_no_timeline(tmp_path):
    prompt = "10 seconds fast cowgirl lean forward into 10 seconds slow teasing cowgirl upright"
    out = tmp_path / "intent.json"
    result = build_motion_intent_from_prompt_v1(
        prompt,
        ROOT / "data/ontology/nlp_lexicon_manual_v1.yaml",
        ROOT / "data/ontology/component_ontology_v1.yaml",
        out,
    )
    phases = result["intent_plan"]["phases"]
    assert len(phases) == 2
    assert phases[0]["duration_seconds"] == 10.0
    assert phases[1]["duration_seconds"] == 10.0
    assert phases[0]["motion_profile"]["tempo_profile"] == "fast_impact"
    assert phases[1]["motion_profile"]["tempo_profile"] == "slow_teasing"
    assert phases[0]["base_state"]["pose_subtype"] == "cowgirl_lean_forward_supported"
    assert phases[1]["base_state"]["pose_subtype"] == "cowgirl_upright"
    assert result["generated_timeline"] is False


def test_web_context_candidates_are_inactive_and_review_only(tmp_path):
    topics = tmp_path / "topics.yaml"
    topics.write_text(
        """
schema: web_context_sources_v1
categories:
  signal_processing_cycle_detection:
    allowed: true
    seed_urls:
      - https://example.invalid/not-real
""",
        encoding="utf-8",
    )
    research_dir = tmp_path / "research"
    summary = collect_web_motion_context_v1(topics, research_dir, allow_web=False)
    assert summary["timeline_generation_performed"] is False
    assert (research_dir / "BLOCKED_NO_WEB_ACCESS.md").exists()
    patch = tmp_path / "patch.yaml"
    report = tmp_path / "patch.md"
    patch_summary = build_web_context_ontology_patches_v1(
        research_dir,
        ROOT / "data/ontology/motion_families_v2.yaml",
        ROOT / "data/ontology/rig_anatomy_v1.yaml",
        patch,
        report,
    )
    data = load_yaml(patch)
    assert patch_summary["patch_candidates"] == 1
    assert all(candidate["accepted"] is False for candidate in data["patch_candidates"])
    assert all(candidate["active"] is False for candidate in data["lexicon_candidates"])


def test_build_nlp_lexicon_keeps_external_candidates_inactive(tmp_path):
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        """
schema: patch_candidates_v1
lexicon_candidates:
  - id: external_test_term
    type: action
    terms: [external motion term]
    maps_to: {action_id: unresolved_candidate}
""",
        encoding="utf-8",
    )
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        f"""
schema: lexicon_sources_v1
sources:
  - source_id: external_test
    type: external_candidate
    path: {candidates.as_posix()}
""",
        encoding="utf-8",
    )
    out = tmp_path / "lexicon.yaml"
    summary = build_nlp_lexicon_v1(
        ROOT / "data/ontology/nlp_lexicon_manual_v1.yaml",
        sources,
        out,
        tmp_path / "report.md",
        allow_web=True,
    )
    lexicon = load_yaml(out)
    external = [entry for entry in lexicon["entries"] if entry.get("id") == "external_test_term"][0]
    assert summary["candidate_entries"] == 1
    assert external["active"] is False
    assert external["accepted"] is False
    assert lexicon["manual_labels_modified"] is False
    assert lexicon["ml_training_performed"] is False

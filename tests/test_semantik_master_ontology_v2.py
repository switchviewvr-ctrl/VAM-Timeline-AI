from pathlib import Path
import zipfile

from vam_timeline_ai.generation.motion_intent_translator import translate_motion_intent_v1
from vam_timeline_ai.io.json_utils import load_json
from vam_timeline_ai.semantics.ontology_loader import load_motion_families, load_yaml
from vam_timeline_ai.semantics.ontology_sourcebook import ingest_semantik_sourcebook_v2
from vam_timeline_ai.semantics.pose_first_resolver import resolve_candidate


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_V2 = ROOT / "data" / "ontology" / "motion_families_v2.yaml"
PHRASES_V2 = ROOT / "data" / "ontology" / "motion_phrases_v2.yaml"
TRACE_V2 = ROOT / "data" / "ontology" / "sourcebook_trace_v2.yaml"


def test_semantik_master_v2_ontology_loads_and_marks_sourcebook():
    families = load_motion_families(ONTOLOGY_V2)
    trace = load_yaml(TRACE_V2)

    assert "cowgirl" in families
    assert "doggy" in families
    assert "bj_oral" in families
    assert "missionary" in families
    assert "pelvis_hip" in families["cowgirl"]["primary_motion_centers"]
    assert families["cowgirl"]["meaning_source"] if "meaning_source" in families["cowgirl"] else True
    assert "sourcebook" in trace
    assert "pelvis/hip/abdomen" in trace["trace"]["controller_legend"]["critical_mapping"]


def test_v2_root_mapping_never_person_root_world():
    data = load_yaml(ONTOLOGY_V2)
    mapping = data["core_architecture"]["root_mapping"]

    assert "pelvisControl" in mapping["sourcebook_root_node_means"]
    assert "VaM Person atom root" in mapping["never_means"]
    assert "world transform" in mapping["never_means"]


def test_v2_lean_back_prompt_from_sourcebook_is_front_cowgirl(tmp_path):
    out = tmp_path / "lean_back_v2.json"
    translate_motion_intent_v1("cowgirl zurückgelehnt, hände auf seinen oberschenkeln", ONTOLOGY_V2, PHRASES_V2, out)
    plan = load_json(out)

    assert plan["family"] == "cowgirl"
    assert plan["pose_subtype"] == "cowgirl_lean_back_supported"
    assert plan["facing_context"] == "front_cowgirl"
    assert plan["contact_support"] == "hands_on_partner_legs_or_thighs"
    assert "cowgirl_lean_back_not_mapped_to_reverse" in plan["invalid_mappings_prevented"]
    assert plan["sourcebook_trace"]["source"] == "Semantik_Master_Konsolidiert.docx"


def test_v2_intent_plan_carries_sourcebook_motion_grammar_fields(tmp_path):
    out = tmp_path / "spine_wave.json"
    translate_motion_intent_v1("slow cowgirl body roll hands on partner chest", ONTOLOGY_V2, PHRASES_V2, out)
    plan = load_json(out)

    assert plan["motion_driver"]["axis_priority"]
    assert "pelvisControl/hipControl" in plan["sourcebook_trace"]["root_mapping"]
    assert plan["micro_states"]
    assert plan["anomaly_guards"]
    assert "no_person_root_tracks" in plan["safety_rules"]


def test_v2_reverse_still_requires_back_to_partner_evidence():
    no_orientation = resolve_candidate(
        {"window_id": "r0", "pose_family": "reverse_cowgirl", "pose_subtype": "reverse_cowgirl_kneeling", "hip_motion_strength": 1.0},
        interaction={"partner_relation": ["rider_over_receiver", "pelvis_aligned"]},
    )
    with_orientation = resolve_candidate(
        {"window_id": "r1", "pose_family": "reverse_cowgirl", "pose_subtype": "reverse_cowgirl_kneeling", "hip_motion_strength": 1.0},
        interaction={"partner_relation": ["rider_over_receiver", "pelvis_aligned", "back_to_partner"]},
    )

    assert no_orientation["resolved_semantic_family"] != "reverse_cowgirl"
    assert "back_to_partner_or_facing_away" in no_orientation["missing_requirements"]
    assert with_orientation["resolved_semantic_family"] == "reverse_cowgirl"


def test_sourcebook_ingestion_extracts_docx_and_report(tmp_path):
    docx = tmp_path / "Semantik_Master_Konsolidiert.docx"
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Konsolidiertes Semantik-Dokument</w:t></w:r></w:p>
    <w:p><w:r><w:t>Root means pelvis driver here.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", xml)

    summary = ingest_semantik_sourcebook_v2(docx, tmp_path / "out", tmp_path / "report.md")

    assert summary["status"] == "ok"
    assert summary["paragraph_count"] == 2
    assert (tmp_path / "out" / "semantik_master_konsolidiert_manifest_v1.json").exists()
    assert "manual_labels.yaml modified: false" in (tmp_path / "report.md").read_text(encoding="utf-8")

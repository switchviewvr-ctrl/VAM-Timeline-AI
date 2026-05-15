from pathlib import Path

from vam_timeline_ai.cowgirl.features import FEATURE_GROUPS
from vam_timeline_ai.cowgirl.taxonomy import (
    COWGIRL_DEEP_SLOW,
    COWGIRL_HAND_SUPPORTED_ON_PARTNER_CHEST,
    COWGIRL_LABELS,
    COWGIRL_LEAN_FORWARD,
    MULTI_LABEL_EXAMPLE,
    semantic_labels,
    validate_label_set,
)
from vam_timeline_ai.semantics.schema import (
    ActorRoleGuess,
    CowgirlWindowRecord,
    ManualLabelOverride,
    MovementWindow,
    MovementWindowFeatures,
)


def test_taxonomy_labels_exist_and_are_multi_label_compatible():
    assert COWGIRL_LEAN_FORWARD in COWGIRL_LABELS
    assert COWGIRL_HAND_SUPPORTED_ON_PARTNER_CHEST in COWGIRL_LABELS
    assert COWGIRL_DEEP_SLOW in COWGIRL_LABELS
    assert len(MULTI_LABEL_EXAMPLE) > 1

    validate_label_set(list(MULTI_LABEL_EXAMPLE))
    labels = semantic_labels()
    assert labels[COWGIRL_LEAN_FORWARD].provisional is True


def test_feature_groups_describe_required_semantic_areas():
    assert set(FEATURE_GROUPS) == {"pelvis", "torso", "hands", "legs", "head_gaze", "rhythm_style"}
    assert "vertical_amplitude" in {field.name for field in FEATURE_GROUPS["pelvis"]}
    assert "hands_on_partner_chest_likelihood" in {field.name for field in FEATURE_GROUPS["hands"]}
    assert "head_down_score" in {field.name for field in FEATURE_GROUPS["head_gaze"]}


def test_schema_objects_can_be_constructed():
    window = MovementWindow(
        sample_id="206_man_plugin0_Anim_3",
        start_seconds=4.0,
        end_seconds=8.0,
        window_seconds=4.0,
        stride_seconds=2.0,
        source_scene_file="206 VR mode ver.json",
        technical_actor_atom_id="man",
    )
    role = ActorRoleGuess(
        technical_atom_id="man",
        semantic_role_guess="rider",
        rider_score=0.9,
        focus_actor_score=0.95,
        partner_context_atom="Person",
        confidence=0.8,
        needs_manual_review=True,
    )
    features = MovementWindowFeatures(
        pelvis={"vertical_amplitude": 0.12, "tempo_bpm_estimate": 72.0},
        torso={"lean_forward_score": 0.8},
        hands={"hands_on_partner_chest_likelihood": 0.7},
        head_gaze={"head_down_score": 0.6},
    )
    override = ManualLabelOverride(
        window_key=window.key,
        labels=[COWGIRL_LEAN_FORWARD, COWGIRL_HAND_SUPPORTED_ON_PARTNER_CHEST],
        confidence="manual",
    )
    record = CowgirlWindowRecord(
        window=window,
        actor_role=role,
        features=features,
        labels=[COWGIRL_LEAN_FORWARD, COWGIRL_LEAN_FORWARD, COWGIRL_DEEP_SLOW],
        source_scene_file="206 VR mode ver.json",
        partner_context_atom="Person",
        manual_overrides=[override],
    )

    assert window.key == "206_man_plugin0_Anim_3:4.000-8.000"
    assert record.actor_role.semantic_role_guess == "rider"
    assert record.labels == [COWGIRL_LEAN_FORWARD, COWGIRL_DEEP_SLOW]


def test_manual_label_template_exists():
    template = Path(__file__).resolve().parents[1] / "data" / "labels" / "manual_labels.template.yaml"

    assert template.exists()
    text = template.read_text(encoding="utf-8")
    assert "actors:" in text
    assert "samples:" in text
    assert "windows:" in text
    assert "cowgirl_hand_supported_on_partner_chest" in text

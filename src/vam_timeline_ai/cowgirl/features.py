"""Cowgirl/Riding semantic feature schema.

This module defines what the analyzer should measure later. It deliberately
does not pretend to compute semantic truth yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureField:
    name: str
    description: str
    value_type: str = "float"


PELVIS_FEATURES: tuple[FeatureField, ...] = (
    FeatureField("vertical_amplitude", "Vertical pelvis/hip displacement amplitude."),
    FeatureField("forward_back_amplitude", "Forward/back pelvis travel amplitude."),
    FeatureField("lateral_amplitude", "Left/right pelvis travel amplitude."),
    FeatureField("pelvis_pitch_range", "Pitch rotation range of pelvis/hip controller."),
    FeatureField("pelvis_roll_range", "Roll rotation range of pelvis/hip controller."),
    FeatureField("pelvis_yaw_range", "Yaw rotation range of pelvis/hip controller."),
    FeatureField("tempo_bpm_estimate", "Estimated rhythmic stroke tempo in BPM."),
    FeatureField("rhythm_regularity", "Regularity of the dominant pelvis rhythm."),
    FeatureField("depth_proxy", "Approximate depth signal from partner-relative or local travel."),
    FeatureField("stroke_depth_proxy", "Per-stroke depth variation proxy."),
    FeatureField("speed_change_count", "Number of meaningful speed changes in the window."),
    FeatureField("acceleration_peaks", "Count or magnitude of acceleration peaks."),
    FeatureField("movement_energy", "Aggregate pelvis/hip motion energy."),
    FeatureField("circularity_score", "How circular the pelvis path appears."),
    FeatureField("grind_score", "Likelihood that movement is grinding-like."),
    FeatureField("bounce_score", "Likelihood that movement is vertical bouncing."),
    FeatureField("rock_score", "Likelihood that movement is forward/back rocking."),
)

TORSO_FEATURES: tuple[FeatureField, ...] = (
    FeatureField("upright_score", "Likelihood that torso is mostly upright."),
    FeatureField("lean_forward_score", "Likelihood of forward torso lean."),
    FeatureField("lean_back_score", "Likelihood of backward torso lean."),
    FeatureField("torso_twist_range", "Range of torso yaw/twist."),
    FeatureField("torso_countermotion_score", "Torso counter-motion relative to pelvis."),
    FeatureField("chest_pelvis_relation", "Relationship between chest and pelvis movement.", "string_or_score"),
    FeatureField("torso_stability", "How stable the torso remains during the window."),
    FeatureField("torso_motion_energy", "Aggregate upper-body motion energy."),
    FeatureField("posture_change_count", "Count of posture changes."),
)

HANDS_FEATURES: tuple[FeatureField, ...] = (
    FeatureField("hands_on_partner_likelihood", "Likelihood hands support on partner."),
    FeatureField("hands_on_partner_chest_likelihood", "Likelihood hands support on partner chest."),
    FeatureField("hands_on_partner_shoulders_likelihood", "Likelihood hands support on partner shoulders."),
    FeatureField("hands_on_partner_hips_likelihood", "Likelihood hands contact partner hips."),
    FeatureField("hands_on_own_thighs_likelihood", "Likelihood hands rest on rider thighs."),
    FeatureField("hands_on_own_body_likelihood", "Likelihood hands contact rider body."),
    FeatureField("hands_on_floor_or_bed_likelihood", "Likelihood hands support on floor/bed."),
    FeatureField("hand_support_transition_count", "Count of support/contact target changes."),
    FeatureField("left_hand_contact_target_guess", "Best guess for left hand target.", "string"),
    FeatureField("right_hand_contact_target_guess", "Best guess for right hand target.", "string"),
    FeatureField("left_hand_motion_energy", "Left hand motion energy."),
    FeatureField("right_hand_motion_energy", "Right hand motion energy."),
    FeatureField("symmetric_hand_support_score", "Symmetric support likelihood."),
    FeatureField("asymmetric_hand_support_score", "Asymmetric support likelihood."),
)

LEGS_FEATURES: tuple[FeatureField, ...] = (
    FeatureField("kneeling_score", "Likelihood of kneeling support."),
    FeatureField("squat_score", "Likelihood of squat-like support."),
    FeatureField("feet_planted_score", "Likelihood feet are planted/stable."),
    FeatureField("knee_motion_energy", "Knee motion energy."),
    FeatureField("stance_width_proxy", "Approximate width between knees/feet."),
    FeatureField("thigh_angle_range", "Approximate thigh angle range."),
    FeatureField("leg_support_stability", "Stability of leg support."),
    FeatureField("weight_shift_left_right_score", "Left/right weight shift likelihood."),
)

HEAD_GAZE_FEATURES: tuple[FeatureField, ...] = (
    FeatureField("head_down_score", "Likelihood rider looks downward."),
    FeatureField("head_up_score", "Likelihood rider looks upward."),
    FeatureField("head_turn_range", "Head turn range."),
    FeatureField("head_motion_energy", "Head motion energy."),
    FeatureField("look_at_partner_likelihood", "Likelihood gaze attends to partner."),
    FeatureField("look_away_likelihood", "Likelihood gaze is away from partner."),
    FeatureField("gaze_direction_guess", "Coarse gaze direction guess.", "string"),
)

RHYTHM_STYLE_FEATURES: tuple[FeatureField, ...] = (
    FeatureField("slow_deep_score", "Likelihood of slow/deep riding."),
    FeatureField("fast_shallow_score", "Likelihood of fast/shallow riding."),
    FeatureField("steady_rhythm_score", "Likelihood of steady rhythm."),
    FeatureField("irregular_rhythm_score", "Likelihood of irregular human rhythm."),
    FeatureField("pause_hold_score", "Likelihood of pause/hold behavior."),
    FeatureField("adjustment_transition_score", "Likelihood of adjustment/transition."),
    FeatureField("escalation_score", "Intensity/tempo/depth increasing."),
    FeatureField("deescalation_score", "Intensity/tempo/depth decreasing."),
    FeatureField("intensity_score", "Overall movement intensity."),
)

FEATURE_GROUPS: dict[str, tuple[FeatureField, ...]] = {
    "pelvis": PELVIS_FEATURES,
    "torso": TORSO_FEATURES,
    "hands": HANDS_FEATURES,
    "legs": LEGS_FEATURES,
    "head_gaze": HEAD_GAZE_FEATURES,
    "rhythm_style": RHYTHM_STYLE_FEATURES,
}


@dataclass
class CowgirlFeatureSchema:
    groups: dict[str, tuple[FeatureField, ...]] = field(default_factory=lambda: FEATURE_GROUPS)

    def field_names(self, group: str) -> tuple[str, ...]:
        return tuple(field.name for field in self.groups[group])

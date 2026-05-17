"""VaM-specific controller mapping for semantic motion centers."""

from __future__ import annotations


SEMANTIC_TO_VAM_CONTROLLER_MAPPING = {
    "pelvis_hip": {
        "vam_primary_driver": "hipControl",
        "vam_secondary_driver": ["pelvisControl", "abdomenControl", "abdomen2Control"],
        "vam_optional_amplifiers": ["lThighControl", "rThighControl"],
    },
    "head_neck": {
        "vam_primary_driver": "headControl",
        "vam_secondary_driver": ["chestControl"],
        "vam_optional_amplifiers": [],
    },
    "hands": {
        "vam_primary_driver": "lHandControl/rHandControl",
        "vam_secondary_driver": ["lElbowControl", "rElbowControl"],
        "vam_optional_amplifiers": [],
    },
}

ACCEPTED_MANUAL_GT_TIMELINE_BASELINE = "data/runs/clean_v3/generation/manual_gt_timeline_examples_v4"
ACCEPTED_AMPLITUDE_PROFILE_DEFAULTS = "data/config/manual_gt_motion_amplitude_profiles_v1.yaml"

REQUIRED_PERSON_POSE_CONTROLLERS = [
    "hipControl",
    "pelvisControl",
    "abdomenControl",
    "chestControl",
    "headControl",
    "lHandControl",
    "rHandControl",
    "lKneeControl",
    "rKneeControl",
    "lFootControl",
    "rFootControl",
]

COWGIRL_HIP_PRIMARY_DRIVER = ["hipControl"]
COWGIRL_HIP_FOLLOWERS = ["pelvisControl", "abdomenControl", "chestControl", "headControl", "lThighControl", "rThighControl"]
COWGIRL_STATIC_ANCHORS = ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl", "lHandControl", "rHandControl"]
BJ_STATIC_HIPS = ["hipControl", "pelvisControl"]
HJ_STATIC_HIPS = ["hipControl", "pelvisControl"]

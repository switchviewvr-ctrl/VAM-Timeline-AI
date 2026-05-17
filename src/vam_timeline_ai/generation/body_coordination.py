"""Body coordination profiles for generated motion synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BodyCoordinationProfile:
    profile_id: str
    family: str
    subtype: str
    driver_controller: str
    follower_controllers: list[str]
    anchor_controllers: list[str]
    phase_offsets: dict[str, float]
    damping: dict[str, float]
    axis_weights: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_body_coordination_profile(profile_id: str) -> BodyCoordinationProfile:
    profiles = default_body_coordination_profiles()
    return profiles.get(profile_id, profiles["cowgirl_oval_grind_v1"])


def default_body_coordination_profiles() -> dict[str, BodyCoordinationProfile]:
    base_warnings = ["Review-only coordination profile. No source keyframes or world coordinates are copied."]
    return {
        "cowgirl_oval_grind_v1": BodyCoordinationProfile(
            profile_id="cowgirl_oval_grind_v1",
            family="cowgirl",
            subtype="oval_grind",
            driver_controller="pelvisControl",
            follower_controllers=["abdomenControl", "chestControl", "headControl"],
            anchor_controllers=["lKneeControl", "rKneeControl", "lFootControl", "rFootControl", "lHandControl", "rHandControl"],
            phase_offsets={"abdomenControl": 0.18, "chestControl": 0.34, "headControl": 0.46},
            damping={"abdomenControl": 0.48, "chestControl": 0.24, "headControl": 0.08},
            axis_weights={"forward_back": 1.0, "lateral": 0.70, "vertical": 1.25},
            warnings=base_warnings,
        ),
        "cowgirl_riding_forward_back_v1": BodyCoordinationProfile(
            profile_id="cowgirl_riding_forward_back_v1",
            family="cowgirl",
            subtype="forward_back_rock",
            driver_controller="pelvisControl",
            follower_controllers=["abdomenControl", "chestControl", "headControl"],
            anchor_controllers=["lKneeControl", "rKneeControl", "lFootControl", "rFootControl"],
            phase_offsets={"abdomenControl": 0.12, "chestControl": 0.28, "headControl": 0.40},
            damping={"abdomenControl": 0.42, "chestControl": 0.22, "headControl": 0.06},
            axis_weights={"forward_back": 1.20, "lateral": 0.35, "vertical": 0.85},
            warnings=base_warnings,
        ),
        "cowgirl_vertical_bounce_v1": BodyCoordinationProfile(
            profile_id="cowgirl_vertical_bounce_v1",
            family="cowgirl",
            subtype="vertical_bounce",
            driver_controller="pelvisControl",
            follower_controllers=["abdomenControl", "chestControl", "headControl"],
            anchor_controllers=["lKneeControl", "rKneeControl", "lFootControl", "rFootControl"],
            phase_offsets={"abdomenControl": 0.10, "chestControl": 0.22, "headControl": 0.36},
            damping={"abdomenControl": 0.40, "chestControl": 0.18, "headControl": 0.05},
            axis_weights={"forward_back": 0.55, "lateral": 0.25, "vertical": 1.35},
            warnings=base_warnings,
        ),
    }

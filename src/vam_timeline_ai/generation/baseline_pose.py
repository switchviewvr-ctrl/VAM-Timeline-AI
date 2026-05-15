"""Synthetic baseline pose schema for relative flow retargeting v0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import uuid

from vam_timeline_ai.generation.generated_motion import controller_bodypart
from vam_timeline_ai.io.json_utils import dump_json
from vam_timeline_ai.io.json_utils import load_json


@dataclass
class BaselineControllerPose:
    controller_name: str
    bodypart: str
    baseline_position: list[float]
    baseline_rotation: list[float] | None
    coordinate_space: str
    is_anchor: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselinePose:
    baseline_id: str
    source: str
    controller_poses: list[BaselineControllerPose]
    body_scale_estimate: float
    coordinate_space: str
    person_root_included: bool = False
    world_coords_allowed: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["controller_poses"] = [pose.to_dict() for pose in self.controller_poses]
        return data


def create_synthetic_baseline_pose_v0(out: str | Path) -> dict[str, Any]:
    """Create a body-shaped synthetic neutral pose in local review coordinates."""
    poses = [
        _pose("pelvisControl", [0.0, 1.00, 0.0], False),
        _pose("hipControl", [0.0, 0.92, 0.0], False),
        _pose("abdomenControl", [0.0, 1.18, 0.02], False),
        _pose("chestControl", [0.0, 1.42, 0.04], False),
        _pose("headControl", [0.0, 1.72, 0.06], False),
        _pose("lKneeControl", [-0.20, 0.56, 0.20], True),
        _pose("rKneeControl", [0.20, 0.56, 0.20], True),
        _pose("lFootControl", [-0.24, 0.08, 0.30], True),
        _pose("rFootControl", [0.24, 0.08, 0.30], True),
        _pose("lHandControl", [-0.36, 1.20, 0.16], False),
        _pose("rHandControl", [0.36, 1.20, 0.16], False),
    ]
    baseline = BaselinePose(
        baseline_id=f"synthetic_baseline_pose_v0::{uuid.uuid4().hex[:12]}",
        source="synthetic_neutral",
        controller_poses=poses,
        body_scale_estimate=1.0,
        coordinate_space="synthetic_neutral",
        person_root_included=False,
        world_coords_allowed=False,
        warnings=[
            "Synthetic review baseline only; not imported from VaM world coordinates.",
            "No Person/root/world transform is included.",
        ],
    )
    data = baseline.to_dict()
    dump_json(out, data)
    return data


def create_cowgirl_review_baseline_pose_v1(out: str | Path, style: str = "kneeling_forward") -> dict[str, Any]:
    """Create a Cowgirl-oriented kneeling/squat review baseline.

    The coordinates are synthetic local review coordinates. They are not copied
    from a source scene and do not include Person/root/world transforms.
    """
    if style != "kneeling_forward":
        style = "kneeling_forward"
    identity = [0.0, 0.0, 0.0, 1.0]
    slight_forward = [-0.087156, 0.0, 0.0, 0.996195]
    head_forward = [-0.043619, 0.0, 0.0, 0.999048]
    poses = [
        _pose("pelvisControl", [0.0, 0.92, 0.02], False, identity),
        _pose("hipControl", [0.0, 0.84, 0.00], False, identity),
        _pose("abdomenControl", [0.0, 1.08, -0.03], False, slight_forward),
        _pose("abdomen2Control", [0.0, 1.20, -0.08], False, slight_forward),
        _pose("chestControl", [0.0, 1.36, -0.16], False, slight_forward),
        _pose("headControl", [0.0, 1.62, -0.22], False, head_forward),
        _pose("lKneeControl", [-0.34, 0.46, 0.08], True, identity),
        _pose("rKneeControl", [0.34, 0.46, 0.08], True, identity),
        _pose("lFootControl", [-0.32, 0.10, 0.42], True, identity),
        _pose("rFootControl", [0.32, 0.10, 0.42], True, identity),
        _pose("lHandControl", [-0.34, 1.02, -0.34], True, slight_forward),
        _pose("rHandControl", [0.34, 1.02, -0.34], True, slight_forward),
    ]
    baseline = BaselinePose(
        baseline_id=f"cowgirl_review_baseline_pose_v1::{uuid.uuid4().hex[:12]}",
        source="synthetic_neutral",
        controller_poses=poses,
        body_scale_estimate=1.0,
        coordinate_space="synthetic_neutral",
        person_root_included=False,
        world_coords_allowed=False,
        warnings=[
            "Cowgirl review baseline only; not imported from source scene or VaM world coordinates.",
            "Use only for generated motion review, not production retargeting.",
        ],
    )
    data = baseline.to_dict()
    data.update({
        "style": style,
        "baseline_style": "cowgirl_kneeling_forward",
        "intended_family": "cowgirl",
        "anchor_profile": "kneeling_cowgirl",
        "generation_use": "review_baseline_only",
        "generated_baseline": True,
        "rotation_source": "synthetic_approximate",
    })
    dump_json(out, data)
    return data


def select_interaction_baseline_for_plan_v0(plan: str | Path | dict[str, Any], out: str | Path) -> dict[str, Any]:
    """Create a synthetic partner-relative baseline for an interaction plan."""
    data = load_json(plan) if not isinstance(plan, dict) else plan
    family = str(data.get("family") or "unknown")
    phase = (data.get("sequence") or [{}])[0]
    interaction = phase.get("interaction") or {}
    support = str(interaction.get("support_mode") or "hands_free")
    baseline = create_cowgirl_review_baseline_pose_v1(out, "kneeling_forward") if family == "cowgirl" else create_synthetic_baseline_pose_v0(out)
    partner_refs = {
        "partner_pelvis_reference": {"position": [0.0, 0.42, 0.0], "coordinate_space": "partner_pelvis_local"},
        "partner_chest_reference": {"position": [0.0, 0.92, -0.42], "coordinate_space": "partner_pelvis_local"},
        "partner_head_reference": {"position": [0.0, 1.12, -0.66], "coordinate_space": "partner_pelvis_local"},
    }
    if family == "cowgirl":
        baseline.update({
            "baseline_id": f"interaction_baseline_v0::{uuid.uuid4().hex[:12]}",
            "source": "synthetic_interaction_baseline",
            "generation_use": "review_baseline_with_partner_references",
            "requested_family": family,
            "support_mode": support,
            "partner_references": partner_refs,
            "partner_root_included": False,
            "partner_world_coords_allowed": False,
            "interaction_frame": interaction.get("coordinate_frame") or "partner_pelvis_local",
            "warnings": (baseline.get("warnings") or []) + [
                "Synthetic partner references are target points, not source-world coordinates.",
                "Hands-on-partner contact must be validated before export.",
            ],
        })
        if support == "hands_on_partner_chest":
            for pose in baseline.get("controller_poses", []):
                if pose.get("controller_name") == "lHandControl":
                    pose["baseline_position"] = [-0.22, 0.98, -0.40]
                    pose["is_anchor"] = True
                if pose.get("controller_name") == "rHandControl":
                    pose["baseline_position"] = [0.22, 0.98, -0.40]
                    pose["is_anchor"] = True
    dump_json(out, baseline)
    return baseline


def _pose(name: str, position: list[float], is_anchor: bool, rotation: list[float] | None = None) -> BaselineControllerPose:
    return BaselineControllerPose(
        controller_name=name,
        bodypart=controller_bodypart(name),
        baseline_position=[float(v) for v in position],
        baseline_rotation=[float(v) for v in rotation] if rotation is not None else None,
        coordinate_space="synthetic_neutral",
        is_anchor=is_anchor,
        warnings=["Synthetic baseline coordinate; review prototype only."],
    )

"""Build schematic semantic motion examples from stickman poses."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import math

from vam_timeline_ai.generation.semantic_stickman import SemanticStickmanMotion, as_point3, point_add
from vam_timeline_ai.generation.semantic_interaction_constraints import make_contact_aware_example
from vam_timeline_ai.io.json_utils import dump_json, load_json
from vam_timeline_ai.semantics.ontology_loader import load_motion_families


def build_semantic_motion_examples_v1(pose_library: str | Path, ontology: str | Path, out_json: str | Path, report: str | Path) -> dict[str, Any]:
    library = load_json(pose_library)
    poses = {p["concept_id"]: p for p in library.get("poses", [])}
    families = load_motion_families(ontology)
    examples = _examples(poses, families)
    data = {
        "schema_version": "semantic_motion_examples_v1",
        "source_pose_library": str(pose_library),
        "source_ontology": str(ontology),
        "production_vam_targets": False,
        "timeline_generation": False,
        "uses_person_root_or_world": False,
        "examples": [ex.to_dict() for ex in examples],
    }
    dump_json(out_json, data)
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.family] = counts.get(ex.family, 0) + 1
    lines = [
        "# Semantic Motion Examples V1",
        "",
        "Schematic driver/follower/anchor motions for ontology sanity checking.",
        "",
        f"- Examples: {len(examples)}",
        f"- Counts by family: {counts}",
        "- Person/root/world transforms used: false",
        "- Timeline animation generated: false",
        "- ML training performed: false",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "example_count": len(examples), "family_counts": counts, "out_json": str(out_json), "report": str(report)}


def build_semantic_motion_examples_v2_contact_aware(pose_library: str | Path, ontology: str | Path, out_json: str | Path, report: str | Path) -> dict[str, Any]:
    library = load_json(pose_library)
    poses = {p["concept_id"]: p for p in library.get("poses", [])}
    families = load_motion_families(ontology)
    base_examples = [ex.to_dict() for ex in _examples(poses, families)]
    examples = [make_contact_aware_example(ex) for ex in base_examples]
    data = {
        "schema_version": "semantic_motion_examples_v2_contact_aware",
        "source_pose_library": str(pose_library),
        "source_ontology": str(ontology),
        "production_vam_targets": False,
        "timeline_generation": False,
        "uses_person_root_or_world": False,
        "contact_aware": True,
        "examples": examples,
    }
    dump_json(out_json, data)
    counts: dict[str, int] = {}
    valid = 0
    invalid = 0
    for ex in examples:
        family = str(ex.get("family") or "unknown")
        counts[family] = counts.get(family, 0) + 1
        if (ex.get("alignment_validation") or {}).get("valid", True):
            valid += 1
        else:
            invalid += 1
    lines = [
        "# Semantic Motion Examples V2 Contact-Aware",
        "",
        "Schematic driver/follower/anchor motions rebuilt around partner-relative interaction constraints.",
        "",
        f"- Examples: {len(examples)}",
        f"- Counts by family: {counts}",
        f"- Alignment valid examples: {valid}",
        f"- Alignment invalid examples: {invalid}",
        "- Contact/alignment targets are constraints, not decorative markers.",
        "- Person/root/world transforms used: false",
        "- Timeline animation generated: false",
        "- ML training performed: false",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "example_count": len(examples), "family_counts": counts, "alignment_valid": valid, "alignment_invalid": invalid, "out_json": str(out_json), "report": str(report)}


def _examples(poses: dict[str, dict[str, Any]], families: dict[str, Any]) -> list[SemanticStickmanMotion]:
    specs = [
        ("cowgirl_grinding", "cowgirl_kneeling", "cowgirl", "cowgirl_grinding"),
        ("cowgirl_vertical_bounce", "cowgirl_kneeling", "cowgirl", "cowgirl_vertical_bounce"),
        ("cowgirl_lean_forward_supported", "cowgirl_lean_forward_supported", "cowgirl", "cowgirl_lean_forward_supported"),
        ("cowgirl_lean_back_supported", "cowgirl_lean_back_supported", "cowgirl", "cowgirl_lean_back_supported"),
        ("reverse_cowgirl_standing_squat_bounce", "reverse_cowgirl_standing_squat", "reverse_cowgirl", "reverse_cowgirl_vertical_bounce"),
        ("doggy_forward_back", "doggy_all_fours", "doggy", "doggy_forward_back"),
        ("doggy_elevated_support", "doggy_elevated_support", "doggy", "doggy_forward_back"),
        ("bj_head_bob", "bj_kneeling_forward", "bj_oral", "bj_head_bob"),
        ("bj_hand_assisted", "bj_hand_assisted", "bj_oral", "bj_hand_assisted"),
        ("missionary_counter_thrust", "missionary_supine", "missionary", "missionary_counter_thrust"),
        ("missionary_flat_passive", "missionary_flat_passive", "missionary", "missionary_passive_flat"),
        ("standing_hand_head_gesture", "standing_hand_head_gesture", "standing_hand_head", "standing_hand_head_gesture"),
        ("handjob_hand_driver", "bj_hand_assisted", "handjob", "hand_repetitive_up_down"),
    ]
    examples: list[SemanticStickmanMotion] = []
    for concept_id, pose_id, family, motion_subtype in specs:
        pose = poses[pose_id]
        examples.append(_make_motion(concept_id, pose, family, motion_subtype, families.get(family, {})))
    return examples


def _make_motion(concept_id: str, pose: dict[str, Any], family: str, motion_subtype: str, family_def: dict[str, Any]) -> SemanticStickmanMotion:
    fps = 12
    duration = 2.5
    count = int(fps * duration)
    base_points = {k: as_point3(v) for k, v in (pose.get("controller_points") or {}).items()}
    partner = {k: as_point3(v) for k, v in (pose.get("partner_reference_points") or {}).items()}
    frames: list[dict[str, Any]] = []
    trails: dict[str, list[tuple[float, float, float]]] = {"pelvis": [], "head": [], "lHand": [], "rHand": []}
    for i in range(count):
        phase = i / max(1, count - 1)
        pts = _frame_points(base_points, concept_id, motion_subtype, phase)
        for key in trails:
            if key in pts:
                trails[key].append(pts[key])
        frames.append({"time_seconds": round(i / fps, 3), "controller_points": pts, "partner_reference_points": partner})
    labels = {
        "family": family,
        "pose_subtype": pose.get("pose_subtype"),
        "motion_subtype": motion_subtype,
        "primary_driver": _driver_for(concept_id, family),
        "anchors": _anchors_for(concept_id, pose),
        "contact_support": _contact_for(concept_id, pose),
        "facing_context": pose.get("facing_context"),
        "exclusions": _not_labels_for(concept_id, pose),
        "sourcebook_semantics": True,
    }
    return SemanticStickmanMotion(
        concept_id=concept_id,
        family=family,
        pose_subtype=str(pose.get("pose_subtype") or ""),
        motion_subtype=motion_subtype,
        duration_seconds=duration,
        fps=fps,
        frames=frames,
        driver_curves=_driver_curve_for(concept_id, motion_subtype),
        follower_curves=_follower_curve_for(family, motion_subtype),
        anchor_policy={"anchors_remain_fixed": _anchors_for(concept_id, pose), "no_person_root_world": True},
        motion_trails=trails,
        labels=labels,
        contact_targets=dict(pose.get("contact_targets") or {}),
        not_labels=labels["exclusions"],
        warnings=_warnings_for(concept_id),
    )


def _frame_points(base: dict[str, tuple[float, float, float]], concept_id: str, motion_subtype: str, phase: float) -> dict[str, tuple[float, float, float]]:
    pts = copy.deepcopy(base)
    wave = math.sin(phase * math.tau)
    wave2 = math.cos(phase * math.tau)
    if concept_id in {"cowgirl_grinding", "cowgirl_lean_forward_supported", "cowgirl_lean_back_supported"}:
        pelvis_delta = (0.13 * wave2, 0.035 * math.sin(phase * math.tau * 2), 0.16 * wave)
        _move_chain(pts, ["pelvis", "abdomen"], pelvis_delta)
        _move_chain(pts, ["chest", "head"], (pelvis_delta[0] * 0.35, pelvis_delta[1] * 0.45, pelvis_delta[2] * -0.25))
    elif concept_id in {"cowgirl_vertical_bounce", "reverse_cowgirl_standing_squat_bounce"}:
        y = 0.20 * (0.5 - 0.5 * math.cos(phase * math.tau))
        _move_chain(pts, ["pelvis", "abdomen"], (0, y, 0))
        lag = 0.12 * (0.5 - 0.5 * math.cos((phase - 0.11) * math.tau))
        _move_chain(pts, ["chest", "head"], (0, lag, 0.03 * wave))
    elif concept_id.startswith("doggy"):
        dz = 0.25 * wave
        _move_chain(pts, ["pelvis", "abdomen"], (0, 0.02 * wave2, dz))
        _move_chain(pts, ["chest", "head"], (0, -0.02 * wave2, dz * 0.15))
    elif concept_id == "bj_head_bob":
        dz = 0.22 * wave
        dy = 0.06 * wave2
        _move_chain(pts, ["head", "chest"], (0, dy, dz))
    elif concept_id == "bj_hand_assisted":
        dz = 0.18 * wave
        _move_chain(pts, ["head", "chest"], (0, 0.04 * wave2, dz))
        _move_chain(pts, ["lHand", "rHand"], (0, 0.0, -dz * 0.75))
    elif concept_id == "missionary_counter_thrust":
        y = 0.15 * (0.5 - 0.5 * math.cos(phase * math.tau))
        _move_chain(pts, ["pelvis", "lThigh", "rThigh", "lKnee", "rKnee"], (0, y, -0.04 * wave))
    elif concept_id == "standing_hand_head_gesture":
        _move_chain(pts, ["rHand"], (0.12 * wave, 0.08 * wave2, 0))
        _move_chain(pts, ["head"], (0.04 * wave2, 0, 0.03 * wave))
    elif concept_id == "handjob_hand_driver":
        _move_chain(pts, ["lHand", "rHand"], (0, 0, 0.20 * wave))
    return pts


def _move_chain(points: dict[str, tuple[float, float, float]], keys: list[str], delta: tuple[float, float, float]) -> None:
    for key in keys:
        if key in points:
            points[key] = point_add(points[key], delta)


def _driver_for(concept_id: str, family: str) -> list[str]:
    if concept_id.startswith("bj"):
        return ["head_neck", "chest_abdomen"]
    if concept_id.startswith("handjob"):
        return ["hands"]
    if concept_id.startswith("standing"):
        return ["hands", "head_neck"]
    if family == "missionary":
        return ["pelvis_counter_driver"]
    return ["pelvis_hip"]


def _anchors_for(concept_id: str, pose: dict[str, Any]) -> list[str]:
    anchors = list(pose.get("anchors") or [])
    if concept_id.startswith("doggy") and "knees" not in anchors:
        anchors.append("knees")
    return anchors


def _contact_for(concept_id: str, pose: dict[str, Any]) -> str:
    if concept_id == "handjob_hand_driver":
        return "hands_near_partner_pelvis"
    return str(pose.get("support_context") or "unknown")


def _not_labels_for(concept_id: str, pose: dict[str, Any]) -> list[str]:
    labels = list(pose.get("not_labels") or [])
    if concept_id.startswith("bj"):
        labels.extend(["cowgirl"])
    if concept_id.startswith("handjob"):
        labels.extend(["cowgirl", "bj_oral_head_driver"])
    if concept_id.startswith("standing"):
        labels.extend(["cowgirl"])
    if concept_id == "cowgirl_lean_back_supported":
        labels.extend(["reverse_cowgirl"])
    return sorted(set(labels))


def _warnings_for(concept_id: str) -> list[str]:
    warnings = ["schematic semantic preview only", "not VaM production controller targets"]
    if concept_id == "cowgirl_lean_back_supported":
        warnings.append("lean-back is front Cowgirl unless back-to-partner evidence exists")
    return warnings


def _driver_curve_for(concept_id: str, motion_subtype: str) -> dict[str, Any]:
    if "grinding" in motion_subtype or "lean" in concept_id:
        return {"pelvis_hip": "oval_or_figure8_xz_low_y"}
    if "bounce" in motion_subtype:
        return {"pelvis_hip": "y_axis_bounce_bottom_impact_top_apex"}
    if concept_id.startswith("doggy"):
        return {"pelvis_hip": "z_axis_forward_back"}
    if concept_id.startswith("bj"):
        return {"head_neck": "target_vector_bob", "chest_abdomen": "support_driver"}
    if concept_id.startswith("handjob"):
        return {"hands": "partner_pelvis_target_stroke"}
    if concept_id.startswith("missionary"):
        return {"pelvis_counter_driver": "reactive_y_lift"}
    return {"unknown": "none"}


def _follower_curve_for(family: str, motion_subtype: str) -> dict[str, Any]:
    if family in {"cowgirl", "reverse_cowgirl"}:
        return {"chest": "delayed_damped_follow", "head": "stabilizing_lag_follow"}
    if family == "doggy":
        return {"chest": "support_damper", "head": "filtered_follow"}
    if family == "bj_oral":
        return {"pelvis": "static_isolator"}
    if family == "missionary":
        return {"chest": "grounded_or_impact_absorber", "head": "low_grounded"}
    return {}

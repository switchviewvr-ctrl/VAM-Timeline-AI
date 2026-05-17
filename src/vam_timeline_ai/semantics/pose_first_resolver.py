"""Top-down pose-first semantic resolver.

This module does not create ground truth labels. It aligns existing candidates
against the motion ontology and reports conflicts/missing requirements.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.contact_target_semantics import infer_target_region, normalize_contact_support
from vam_timeline_ai.semantics.motion_center_semantics import infer_motion_shape, infer_primary_motion_center
from vam_timeline_ai.semantics.ontology_loader import latest_existing


COWGIRL_POSES = {"cowgirl", "cowgirl_upright", "cowgirl_kneeling", "cowgirl_squat", "cowgirl_lean_forward_supported", "cowgirl_lean_back_supported", "cowgirl_sequential_spine_wave", "cowgirl_corkscrew_grind"}
REVERSE_COWGIRL_POSES = {"reverse_cowgirl_standing_squat", "reverse_cowgirl_hover", "reverse_cowgirl_kneeling", "reverse_cowgirl_lean_forward", "reverse_cowgirl_lean_back", "spider_reverse_cowgirl"}
DOGGY_POSES = {"doggy_all_fours", "doggy_bent_forward", "doggy_elevated_support", "doggy_drop_flat", "doggy_arched_upright", "doggy_vertical_kneeling"}
MISSIONARY_POSES = {"missionary_supine", "missionary_legs_up", "missionary_wrapped_legs", "missionary_flat_passive", "maximal_supine_flexion", "lying_on_back"}


def resolve_pose_first_semantics_v1(
    run_dir: str | Path,
    pose_semantics: str | Path,
    relative_features: str | Path,
    interaction_semantics: str | Path,
    candidate_db: str | Path,
    rules: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    candidate_path = Path(candidate_db)
    fallback = None
    if not candidate_path.exists():
        fallback = latest_existing(
            [
                run / "datasets" / "semantic_candidate_db_v3.jsonl",
                run / "datasets" / "semantic_candidate_db_v2.jsonl",
                run / "datasets" / "semantic_candidate_db_v1.jsonl",
                run / "datasets" / "semantic_candidate_db_v0.jsonl",
            ]
        )
        candidate_path = fallback or candidate_path
    candidates = load_jsonl(candidate_path)
    pose_by_window = {str(r.get("window_id")): r for r in load_jsonl(pose_semantics)}
    rel_by_window = {str(r.get("window_id")): r for r in load_jsonl(relative_features)}
    interaction_by_window = {str(r.get("window_id")): r for r in load_jsonl(interaction_semantics)}

    rows = []
    for candidate in candidates:
        wid = str(candidate.get("window_id") or "")
        pose = pose_by_window.get(wid) or {}
        rel = rel_by_window.get(wid) or {}
        inter = interaction_by_window.get(wid) or {}
        rows.append(resolve_candidate(candidate, pose, rel, inter))

    write_jsonl(out_jsonl, rows)
    counts = Counter(r["resolved_semantic_family"] for r in rows)
    gates = Counter(r["clean_motion_gate"] for r in rows)
    conflicts = sum(1 for r in rows if r.get("conflict_flags"))
    lines = [
        "# Pose-First Semantic Resolver V1",
        "",
        "Ontology-first audit layer. Existing labels are evidence, not truth.",
        "",
        f"- Candidate source: `{candidate_path}`",
        f"- Fallback used: {bool(fallback)}",
        f"- Records: {len(rows)}",
        f"- Family counts: {dict(counts)}",
        f"- Clean motion gate counts: {dict(gates)}",
        f"- Records with conflicts: {conflicts}",
        "- Person/root/world targets used: false",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "records": len(rows), "family_counts": dict(counts), "gate_counts": dict(gates), "conflicts": conflicts, "out_jsonl": str(out_jsonl), "report": str(report), "fallback_used": bool(fallback)}


def resolve_candidate(candidate: dict[str, Any], pose: dict[str, Any] | None = None, relative_features: dict[str, Any] | None = None, interaction: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = pose or {}
    interaction = interaction or {}
    pose_family = str(candidate.get("pose_family") or pose.get("pose_family") or "unknown")
    pose_subtype = str(candidate.get("pose_subtype") or pose.get("pose_subtype") or "unknown")
    partner_relation = _as_list(candidate.get("partner_relation") or interaction.get("partner_relation"))
    primary_center, center_flags = infer_primary_motion_center(candidate, relative_features)
    target_region, target_reason = infer_target_region(candidate, interaction)
    motion_shape = infer_motion_shape(candidate, relative_features)
    contact_support = normalize_contact_support(candidate.get("contact_support") or interaction.get("support_context"))

    resolved = "unknown"
    motion_subtype = str(candidate.get("motion_subtype") or "unknown")
    gate = "fail_no_driver"
    confidence = 0.25
    conflicts: list[str] = []
    not_labels: list[str] = []
    missing: list[str] = []

    if pose_family == "standing" or pose_subtype in {"standing_upright", "standing_gesture"}:
        if primary_center in {"hands", "head_neck", "unknown", "static_pose"}:
            resolved, gate, confidence = "standing_hand_head", "fail_wrong_pose", 0.75
            not_labels.extend(["cowgirl", "doggy"])

    if resolved == "unknown" and (pose_family == "cowgirl" or pose_subtype in COWGIRL_POSES) and primary_center == "head_neck":
        resolved, gate, confidence = "bj_oral", "fail_wrong_driver", 0.62
        conflicts.append("cowgirl_pose_with_head_neck_driver")
        not_labels.extend(["cowgirl_clean_motion", "doggy"])

    if resolved == "unknown" and (pose_family == "cowgirl" or pose_subtype in COWGIRL_POSES) and primary_center == "hands":
        resolved, gate, confidence = "handjob", "fail_wrong_driver", 0.6
        conflicts.append("cowgirl_pose_with_hands_driver")
        not_labels.append("cowgirl_clean_motion")

    if resolved == "unknown" and primary_center == "head_neck" and target_region in {"partner_pelvis_or_genital_area", "partner_hips", "partner_unknown"}:
        resolved, gate, confidence = "bj_oral", "pass", 0.72
        not_labels.extend(["cowgirl_clean_motion", "doggy"])

    if resolved == "unknown" and primary_center == "hands" and target_region in {"partner_pelvis_or_genital_area", "partner_hips", "partner_unknown"}:
        resolved, gate, confidence = "handjob", "pass", 0.7
        not_labels.append("cowgirl_clean_motion")

    if resolved == "unknown" and pose_subtype in REVERSE_COWGIRL_POSES:
        if "back_to_partner" in partner_relation or "facing_away_from_partner" in partner_relation or str(candidate.get("facing_context") or "") in {"back_to_partner", "reverse_cowgirl"}:
            resolved = "reverse_cowgirl"
            confidence = 0.72
            gate = "pass" if primary_center in {"pelvis_hip", "thighs"} else "fail_no_driver"
        else:
            gate = "fail_missing_partner_context"
            confidence = 0.35
            missing.append("back_to_partner_or_facing_away")
            conflicts.append("reverse_cowgirl_pose_without_orientation_evidence")

    if resolved == "unknown" and (pose_subtype in DOGGY_POSES or "partner_behind" in partner_relation or "behind_receiver" in partner_relation):
        if pose_subtype in DOGGY_POSES or "partner_behind" in partner_relation or "behind_receiver" in partner_relation:
            resolved, gate, confidence = "doggy", "pass" if primary_center in {"pelvis_hip", "receiver_pelvis_response"} else "fail_no_driver", 0.68
        if pose_subtype not in DOGGY_POSES and not ({"partner_behind", "behind_receiver"} & set(partner_relation)):
            conflicts.append("doggy_requires_all_fours_bent_forward_or_partner_behind")

    if resolved == "unknown" and pose_subtype in MISSIONARY_POSES:
        resolved, gate, confidence = "missionary", "pass" if primary_center in {"pelvis_hip", "pelvis_counter_driver", "static_pose"} else "fail_wrong_driver", 0.65

    if resolved == "unknown" and (pose_family == "cowgirl" or pose_subtype in COWGIRL_POSES):
        if "back_to_partner" in partner_relation or "facing_away_from_partner" in partner_relation:
            resolved = "reverse_cowgirl"
            confidence = 0.7
        else:
            resolved = "cowgirl"
            confidence = 0.72
        if primary_center in {"pelvis_hip", "thighs"}:
            if _low_motion(candidate):
                gate = "soft_pass_short" if _num(candidate.get("motion_duration_confidence"), 1.0) < 0.65 else "pass"
            else:
                gate = "pass"
        elif primary_center == "static_pose":
            gate = "fail_low_motion"
            resolved = "pose_context_hold"
            not_labels.append("cowgirl_clean_motion")
        elif primary_center in {"head_neck", "hands"}:
            gate = "fail_wrong_driver"
            conflicts.append(f"cowgirl_pose_with_{primary_center}_driver")
            not_labels.append("cowgirl_clean_motion")
        else:
            gate = "fail_no_driver"
            missing.append("pelvis_hip_driver")
        if pose_subtype == "cowgirl_lean_back_supported" and resolved == "reverse_cowgirl":
            conflicts.append("lean_back_cowgirl_requires_explicit_back_to_partner_for_reverse")

    if resolved == "unknown":
        if str(candidate.get("phase") or "") in {"transition_setup", "intro_alignment"}:
            resolved, gate, confidence = "transition_setup", "fail_no_driver", 0.55
        elif _low_motion(candidate):
            resolved, gate, confidence = "pose_context_hold", "fail_low_motion", 0.55

    if resolved == "cowgirl":
        if "rider_over_receiver" not in partner_relation and "pelvis_aligned" not in partner_relation:
            missing.append("rider_over_receiver_or_pelvis_aligned")
            if gate == "pass":
                gate = "fail_missing_partner_context"
        if primary_center not in {"pelvis_hip", "thighs"}:
            conflicts.append("cowgirl_clean_requires_pelvis_hip_or_thigh_driver")

    explanation = _explain(resolved, pose_subtype, primary_center, target_region, gate, conflicts, missing)
    return {
        "window_id": candidate.get("window_id"),
        "candidate_id": candidate.get("candidate_id"),
        "source_scene_file": candidate.get("source_scene_file"),
        "technical_actor_id": candidate.get("technical_actor_id"),
        "resolved_semantic_family": resolved,
        "resolved_motion_subtype": _resolve_motion_subtype(resolved, motion_subtype, motion_shape, gate),
        "pose_family": pose_family,
        "pose_subtype": pose_subtype,
        "primary_motion_center": primary_center,
        "secondary_motion_centers": _secondary_for(resolved),
        "target_region": target_region,
        "target_region_reason": target_reason,
        "contact_support": contact_support,
        "role_context": {"actor_role": candidate.get("actor_role") or interaction.get("actor_role"), "partner_role": candidate.get("partner_role") or interaction.get("partner_role")},
        "partner_relation": partner_relation,
        "motion_shape": motion_shape,
        "clean_motion_gate": gate,
        "confidence": round(confidence, 3),
        "conflict_flags": conflicts + center_flags,
        "missing_requirements": missing,
        "not_labels": sorted(set(not_labels)),
        "explanation": explanation,
        "source_semantic_family": candidate.get("semantic_family"),
        "is_human_ground_truth": False,
        "is_training_label": False,
        "person_root_world_targets_used": False,
    }


def _resolve_motion_subtype(family: str, old_subtype: str, shape: str, gate: str) -> str:
    if gate == "fail_low_motion":
        return f"{family}_pose_context_low_motion"
    if family == "cowgirl" and shape == "oval_or_grinding":
        return "cowgirl_grinding"
    if family == "cowgirl" and shape == "vertical_bounce":
        return "cowgirl_vertical_bounce"
    if family == "cowgirl" and shape == "forward_back_rock":
        return "cowgirl_rocking"
    if family == "bj_oral":
        return "bj_head_bob"
    if family == "handjob":
        return "hand_repetitive_up_down"
    if family == "doggy":
        return "doggy_forward_back"
    return old_subtype if old_subtype and old_subtype != "unknown" else family


def _secondary_for(family: str) -> list[str]:
    if family in {"cowgirl", "reverse_cowgirl"}:
        return ["abdomen_chest", "head_neck"]
    if family == "bj_oral":
        return ["chest_abdomen", "hands"]
    if family == "doggy":
        return ["chest_abdomen"]
    return []


def _explain(family: str, pose_subtype: str, primary: str, target: str, gate: str, conflicts: list[str], missing: list[str]) -> str:
    bits = [f"{family}: pose={pose_subtype}", f"driver={primary}", f"target={target}", f"gate={gate}"]
    if conflicts:
        bits.append("conflicts=" + ",".join(conflicts[:3]))
    if missing:
        bits.append("missing=" + ",".join(missing[:3]))
    return "; ".join(bits)


def _low_motion(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("phase") or "") in {"low_motion_hold", "pose_context_only"} or _num(candidate.get("motion_content_strength"), 1.0) < 0.25 or str(candidate.get("clean_motion_gate") or "") in {"fail_low_motion", "fail_no_hip_motion"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]

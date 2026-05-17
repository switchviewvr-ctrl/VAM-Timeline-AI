"""Semantic Action = pose + motion + partner relation + contact/support."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


@dataclass
class SemanticActionCandidate:
    window_id: str
    pair_window_id: str | None = None
    semantic_family: str = "unknown"
    actor_role: str = "unknown"
    partner_role: str = "unknown"
    pose_family: str = "unknown"
    pose_subtype: str = "unknown"
    motion_family: str = "unknown"
    motion_subtype: str = "unknown"
    partner_relation: list[str] = field(default_factory=list)
    contact_support: str = "unknown"
    facing_context: str = "unknown"
    torso_lean_direction: str = "unknown"
    phase: str = "unknown"
    generation_safe: bool = False
    semantic_score: float = 0.0
    pose_score: float = 0.0
    motion_score: float = 0.0
    interaction_score: float = 0.0
    consistency_score: float = 0.0
    conflict_flags: list[str] = field(default_factory=list)
    support_constraint_requirements: list[str] = field(default_factory=list)
    hands_behind_support_score: float = 0.0
    hands_on_partner_legs_score: float = 0.0
    hands_on_partner_thighs_score: float = 0.0
    facing_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    is_human_ground_truth: bool = False
    is_training_label: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_semantic_actions_v0(
    candidate_db: str | Path,
    pose_semantics: str | Path,
    relative_reference_matches: str | Path | None,
    interaction_semantics: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    poses = {r.get("window_id"): r for r in load_jsonl(pose_semantics) if r.get("window_id")}
    interactions_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(interaction_semantics):
        interactions_by_window.setdefault(str(row.get("window_id")), []).append(row)
    matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches or "") if r.get("window_id")}
    rows: list[dict[str, Any]] = []
    for candidate in load_jsonl(candidate_db):
        wid = candidate.get("window_id")
        if not wid:
            continue
        action = _action_from_candidate(candidate, poses.get(wid, {}), _best_interaction(interactions_by_window.get(str(wid), [])), matches.get(wid, {}))
        rows.append(action.to_dict())
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def _action_from_candidate(candidate: dict[str, Any], pose: dict[str, Any], interaction: dict[str, Any], match: dict[str, Any]) -> SemanticActionCandidate:
    category = str(candidate.get("category") or "")
    family_hint = str(candidate.get("semantic_family") or "").lower()
    motion_family = "bj_oral" if "bj" in family_hint or "not_cowgirl_bj_oral" in category else "cowgirl" if "cowgirl" in category or family_hint == "cowgirl" else "unknown"
    motion_subtype = str(candidate.get("cowgirl_subtype") or candidate.get("motion_subtype") or candidate.get("subtype") or "unknown")
    pose_family = str(pose.get("pose_family") or "unknown")
    pose_subtype = str(pose.get("pose_subtype") or "unknown")
    interaction_family = str(interaction.get("interaction_family") or "unknown")
    contact = str(interaction.get("support_context") or "unknown")
    facing_context = str(pose.get("facing_context") or "unknown")
    torso_lean_direction = str(pose.get("torso_lean_direction") or "unknown")
    phase = _phase_from_candidate(candidate, motion_family)
    conflicts: list[str] = []
    requirements: list[str] = []
    semantic_family = motion_family
    if motion_family == "cowgirl":
        if pose_family in {"standing", "lying_receiver"}:
            conflicts.append("cowgirl_motion_wrong_pose")
        if pose_family == "bj_oral":
            conflicts.append("cowgirl_motion_bj_oral_pose_context")
        if category == "cowgirl_context_intro_low_motion":
            phase = "intro"
        if contact == "hands_on_partner_chest" and not interaction.get("contact_targets"):
            conflicts.append("hands_on_partner_chest_missing_partner_target")
        if pose_subtype == "cowgirl_lean_back_supported":
            requirements.extend([
                "keep_torso_lean_back",
                "keep_rider_pelvis_aligned_to_partner",
            ])
            if contact in {"hands_on_partner_legs_or_thighs", "hands_behind_support", "ambiguous_behind_support"}:
                requirements.append("keep_hands_behind_on_partner_legs_or_thighs")
    if motion_family == "bj_oral":
        semantic_family = "bj_oral"
        conflicts.append("excluded_from_cowgirl_bj_oral")
    if "receiver_response" in category:
        semantic_family = "receiver_response"
        motion_family = "receiver_response"
    if "standing" in category:
        semantic_family = "hand_gesture" if pose_family == "standing" else "unknown"
    pose_score = float(pose.get("pose_confidence") or 0.0)
    motion_score = float(candidate.get("semantic_cowgirl_score") or candidate.get("family_confidence") or 0.0)
    interaction_score = float(interaction.get("interaction_confidence") or 0.0)
    semantic_score = max(motion_score, float(candidate.get("generation_candidate_score") or 0.0))
    consistency = _consistency_score(semantic_family, pose_family, interaction_family, conflicts)
    generation_safe = bool(candidate.get("generation_safe")) and not conflicts and semantic_family == "cowgirl" and pose_family in {"cowgirl", "kneeling_general"}
    warnings = []
    warnings.extend(str(x) for x in (candidate.get("warnings") or [])[:4])
    warnings.extend(str(x) for x in (pose.get("warnings") or [])[:3])
    warnings.extend(str(x) for x in (interaction.get("warnings") or [])[:3])
    if semantic_family == "bj_oral":
        warnings.append("BJ/oral candidate detected; excluded from Cowgirl generation-safe set, preserved for BJ/oral dataset.")
    if contact == "unknown" and motion_family == "cowgirl":
        warnings.append("Partner/contact context is unknown; do not invent hands-on-partner support.")
    return SemanticActionCandidate(
        window_id=str(candidate.get("window_id")),
        pair_window_id=interaction.get("pair_window_id"),
        semantic_family=semantic_family,
        actor_role=interaction.get("actor_role") or ("rider" if semantic_family == "cowgirl" else "unknown"),
        partner_role=interaction.get("partner_role") or "unknown",
        pose_family=pose_family,
        pose_subtype=pose_subtype,
        motion_family=motion_family,
        motion_subtype=motion_subtype,
        partner_relation=interaction.get("partner_relation") or ["unknown"],
        contact_support=contact,
        facing_context=facing_context,
        torso_lean_direction=torso_lean_direction,
        phase=phase,
        generation_safe=generation_safe,
        semantic_score=round(semantic_score, 6),
        pose_score=round(pose_score, 6),
        motion_score=round(motion_score, 6),
        interaction_score=round(interaction_score, 6),
        consistency_score=round(consistency, 6),
        conflict_flags=_dedupe(conflicts),
        support_constraint_requirements=_dedupe(requirements),
        hands_behind_support_score=round(float(pose.get("hands_behind_support_confidence") or interaction.get("hands_behind_partner_support_score") or 0.0), 6),
        hands_on_partner_legs_score=round(float(interaction.get("hands_on_partner_legs_score") or 0.0), 6),
        hands_on_partner_thighs_score=round(float(interaction.get("hands_on_partner_thighs_score") or 0.0), 6),
        facing_confidence=round(float(pose.get("facing_confidence") or 0.0), 6),
        warnings=_dedupe(warnings),
    )


def _best_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return sorted(rows, key=lambda r: float(r.get("interaction_confidence") or 0.0), reverse=True)[0]


def _phase_from_candidate(candidate: dict[str, Any], motion_family: str) -> str:
    category = str(candidate.get("category") or "")
    if "intro" in category or "context" in category:
        return "intro"
    if "receiver_response" in category:
        return "response"
    if "gesture" in category:
        return "gesture"
    return "clean_motion" if motion_family in {"cowgirl", "bj_oral"} else "unknown"


def _consistency_score(family: str, pose_family: str, interaction_family: str, conflicts: list[str]) -> float:
    score = 0.4
    if family == "cowgirl" and pose_family in {"cowgirl", "kneeling_general"}:
        score += 0.3
    if family == "bj_oral" and pose_family in {"bj_oral", "kneeling_general", "cowgirl"}:
        score += 0.2
    if family == interaction_family:
        score += 0.2
    if interaction_family == "unknown":
        score -= 0.05
    score -= min(0.5, 0.2 * len(conflicts))
    return max(0.0, min(1.0, score))


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    families = Counter(r.get("semantic_family") for r in rows)
    phases = Counter(r.get("phase") for r in rows)
    conflicts = Counter(flag for r in rows for flag in (r.get("conflict_flags") or []))
    lines = [
        "# Semantic Actions Report V0",
        "",
        "Semantic Actions combine pose, motion, partner relation, contact/support, phase, and generation safety.",
        "",
        f"- Rows: {len(rows)}",
        f"- Generation-safe actions: {sum(1 for r in rows if r.get('generation_safe'))}",
        "",
        "## Semantic Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common()) if families else lines.append("- None")
    lines.extend(["", "## Phases", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in phases.most_common()) if phases else lines.append("- None")
    lines.extend(["", "## Conflict Flags", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in conflicts.most_common()) if conflicts else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(str(item))
    return out

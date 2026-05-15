"""BJ/oral semantic-family candidate classifier.

BJ/oral motion is a valid animation family.  These rows are excluded from
Cowgirl generation-safe sets when appropriate, but preserved for future
BJ/oral datasets.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def classify_bj_oral_domain(
    run_dir: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    relative_reference_matches: str | Path,
    cowgirl_core_controllers: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    rel = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    traj = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches) if r.get("window_id")}
    core = {r.get("window_id"): r for r in load_jsonl(cowgirl_core_controllers) if r.get("window_id")}
    ids = sorted(set(rel) | set(traj) | set(matches) | set(core))
    rows = [bj_oral_domain_for_window(rel.get(wid, {}), traj.get(wid, {}), matches.get(wid, {}), core.get(wid, {})) for wid in ids]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def bj_oral_domain_for_window(
    relative_feature: dict[str, Any],
    trajectory: dict[str, Any],
    relative_match: dict[str, Any],
    core_controllers: dict[str, Any],
) -> dict[str, Any]:
    wid = relative_feature.get("window_id") or trajectory.get("window_id") or relative_match.get("window_id") or core_controllers.get("window_id")
    values = relative_feature.get("feature_values", {}) or {}
    bj_score = _f(relative_match.get("bj_relative_score"))
    head_score = _f(relative_match.get("head_relative_score"))
    hand_score = _f(relative_match.get("hand_relative_score"))
    cowgirl_score = _f(relative_match.get("cowgirl_relative_score"))
    cowgirl_grind = _f(relative_match.get("cowgirl_grind_trajectory_score"))
    pelvis_motion = max(
        _f(values.get("relative_pelvis_vertical_amplitude")),
        _f(values.get("relative_pelvis_forward_back_amplitude")),
        _f(values.get("relative_pelvis_lateral_amplitude")),
        _f(values.get("local_grind_score")),
        _f(values.get("local_bounce_score")),
    )
    head_motion = _f(values.get("head_relative_to_chest_motion"))
    limb_motion = _f(values.get("limb_motion_relative_energy"))
    core_status = str(core_controllers.get("core_gate_status") or core_controllers.get("cowgirl_core_controller_status") or "unknown")
    core_gate = core_controllers.get("generation_safe_core_controller_gate")
    core_weak = bool(core_status in {"hard_fail", "missing_core"} or core_gate is False)
    head_dominant = bool(head_motion > max(pelvis_motion, limb_motion, 1e-6) * 1.35 or head_score > max(cowgirl_score, 0.01) + 0.18)
    bj_high = bool(bj_score >= 0.45 or (bj_score > cowgirl_score + 0.12 and bj_score >= 0.25))
    kneeling_pose = bool(core_controllers.get("has_knee_controls") or core_controllers.get("has_foot_controls"))
    cowgirl_pose_but_bj = bool((cowgirl_score >= 0.30 or kneeling_pose) and (head_dominant or bj_high) and (core_weak or pelvis_motion < max(0.12, cowgirl_grind * 0.35)))
    bj_candidate = bool((bj_high or head_dominant) and (core_weak or pelvis_motion < 0.16 or cowgirl_pose_but_bj))
    confidence = min(1.0, max(bj_score, head_score) * 0.60 + (0.20 if core_weak else 0.0) + (0.15 if cowgirl_pose_but_bj else 0.0)) if bj_candidate else 0.0
    generation_candidate = bool(bj_candidate and confidence >= 0.45 and not core_controllers.get("has_hip_control"))
    subtype = "bj_oral"
    if head_dominant:
        subtype = "bj_head_dominant_motion"
    elif bj_score >= 0.60:
        subtype = "bj_deep_candidate"
    warnings = []
    if bj_candidate:
        warnings.append("BJ/oral candidate detected; excluded from Cowgirl generation-safe set, preserved for BJ/oral dataset.")
    return {
        "window_id": wid,
        "sample_id": relative_feature.get("sample_id") or relative_match.get("sample_id") or core_controllers.get("sample_id"),
        "source_id": relative_feature.get("source_id") or relative_match.get("source_id") or core_controllers.get("source_id"),
        "source_scene_file": relative_feature.get("source_scene_file") or relative_match.get("source_scene_file") or core_controllers.get("source_scene_file"),
        "technical_atom_id": relative_feature.get("technical_atom_id") or relative_match.get("technical_atom_id") or core_controllers.get("technical_atom_id"),
        "semantic_family": "bj_oral" if bj_candidate else "unknown",
        "bj_oral_motion_candidate": bj_candidate,
        "bj_oral_generation_candidate": generation_candidate,
        "bj_head_dominant_motion": head_dominant,
        "bj_reference_score_high": bj_high,
        "cowgirl_pose_but_bj_oral_motion": cowgirl_pose_but_bj,
        "kneeling_pose_bj_oral": bool(bj_candidate and kneeling_pose),
        "excluded_from_cowgirl": bj_candidate,
        "preserve_for_future_dataset": bj_candidate,
        "preserve_for_bj_dataset": bj_candidate,
        "not_cowgirl_bj_oral": bj_candidate,
        "bj_oral_confidence": round(float(confidence), 6),
        "bj_reference_score": round(float(bj_score), 6),
        "head_reference_score": round(float(head_score), 6),
        "hand_reference_score": round(float(hand_score), 6),
        "cowgirl_reference_score": round(float(cowgirl_score), 6),
        "pelvis_motion_proxy": round(float(pelvis_motion), 6),
        "head_motion_proxy": round(float(head_motion), 6),
        "core_gate_status": core_status,
        "core_controller_gate": core_gate,
        "missing_core_controllers": core_controllers.get("missing_core_controllers", []),
        "subtype": subtype,
        "warnings": warnings,
        "is_human_ground_truth": False,
        "is_training_label": False,
        # Compatibility fields for old reports/commands.
        "head_or_oral_domain_trap": bj_candidate,
        "cowgirl_pose_false_positive": cowgirl_pose_but_bj,
        "head_motion_dominant": head_dominant,
        "cowgirl_pose_but_not_cowgirl_motion": cowgirl_pose_but_bj,
        "likely_bj_or_oral_motion": bj_candidate,
        "bj_oral_trap_confidence": round(float(confidence), 6),
    }


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in rows if r.get("bj_oral_motion_candidate")]
    generation = [r for r in rows if r.get("bj_oral_generation_candidate")]
    kneeling = [r for r in rows if r.get("kneeling_pose_bj_oral")]
    family_counts = Counter(r.get("semantic_family") for r in rows)
    lines = [
        "# BJ/Oral Domain Classifier Report",
        "",
        "BJ/oral is a valid motion family. Candidates are excluded from Cowgirl generation-safe sets when appropriate and preserved for future BJ/oral datasets.",
        "",
        f"- Windows audited: {len(rows)}",
        f"- BJ/oral candidates: {len(candidates)}",
        f"- BJ/oral generation candidates: {len(generation)}",
        f"- Kneeling-pose BJ/oral candidates: {len(kneeling)}",
        f"- Excluded from Cowgirl: {sum(1 for r in rows if r.get('excluded_from_cowgirl'))}",
        f"- Preserved for BJ/oral dataset: {sum(1 for r in rows if r.get('preserve_for_future_dataset'))}",
        "",
        "## Semantic Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in family_counts.most_common()) if family_counts else lines.append("- None")
    lines.extend(["", "## Top BJ/Oral Candidates", ""])
    for row in sorted(candidates, key=lambda r: float(r.get("bj_oral_confidence") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` confidence={row.get('bj_oral_confidence')} "
            f"bj={row.get('bj_reference_score')} head={row.get('head_reference_score')} "
            f"excluded_from_cowgirl={row.get('excluded_from_cowgirl')} scene=`{row.get('source_scene_file')}`"
        )
    if not candidates:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

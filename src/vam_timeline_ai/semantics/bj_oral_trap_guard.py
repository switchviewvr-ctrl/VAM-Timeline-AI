"""Audit guard for head/BJ/oral-domain traps.

The guard detects review-trap cases where pose/context can look Cowgirl-like
but relative motion evidence is head/oral-domain or lacks core pelvis motion.
It emits audit signals only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def audit_bj_oral_trap_guard(
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
    rows = [bj_oral_trap_guard_for_window(rel.get(wid, {}), traj.get(wid, {}), matches.get(wid, {}), core.get(wid, {})) for wid in ids]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def bj_oral_trap_guard_for_window(
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
    pelvis_motion = max(
        _f(values.get("relative_pelvis_vertical_amplitude")),
        _f(values.get("relative_pelvis_forward_back_amplitude")),
        _f(values.get("relative_pelvis_lateral_amplitude")),
        _f(values.get("local_grind_score")),
        _f(values.get("local_bounce_score")),
    )
    head_motion = _f(values.get("head_relative_to_chest_motion"))
    limb_motion = _f(values.get("limb_motion_relative_energy"))
    core_gate = core_controllers.get("generation_safe_core_controller_gate")
    missing_core = core_controllers.get("cowgirl_core_controller_status") == "missing_core" or core_gate is False
    head_dominant = bool(head_motion > max(pelvis_motion, limb_motion, 1e-6) * 1.35 or head_score > max(cowgirl_score, 0.01) + 0.18)
    bj_high = bool(bj_score >= 0.45 or (bj_score > cowgirl_score + 0.12 and bj_score >= 0.25))
    cowgirl_pose_but_not_motion = bool(cowgirl_score >= 0.35 and (head_dominant or bj_high) and (missing_core or pelvis_motion < 0.12))
    trap = bool((bj_high or head_dominant) and (missing_core or pelvis_motion < 0.12 or cowgirl_pose_but_not_motion))
    confidence = 0.0
    if trap:
        confidence = min(1.0, 0.35 + max(bj_score, head_score) * 0.45 + (0.20 if missing_core else 0.0))
    warnings = []
    if trap:
        warnings.append("Head/BJ/oral-domain trap candidate; block generation-safe Cowgirl.")
    if cowgirl_pose_but_not_motion:
        warnings.append("Cowgirl-like pose/context but motion evidence is not Cowgirl-like.")
    return {
        "window_id": wid,
        "sample_id": relative_feature.get("sample_id") or relative_match.get("sample_id") or core_controllers.get("sample_id"),
        "source_id": relative_feature.get("source_id") or relative_match.get("source_id") or core_controllers.get("source_id"),
        "source_scene_file": relative_feature.get("source_scene_file") or relative_match.get("source_scene_file") or core_controllers.get("source_scene_file"),
        "technical_atom_id": relative_feature.get("technical_atom_id") or relative_match.get("technical_atom_id") or core_controllers.get("technical_atom_id"),
        "head_or_oral_domain_trap": trap,
        "cowgirl_pose_false_positive": cowgirl_pose_but_not_motion,
        "head_motion_dominant": head_dominant,
        "bj_reference_score_high": bj_high,
        "cowgirl_pose_but_not_cowgirl_motion": cowgirl_pose_but_not_motion,
        "likely_bj_or_oral_motion": bool(trap and bj_high),
        "bj_oral_trap_confidence": round(float(confidence), 6),
        "bj_reference_score": round(float(bj_score), 6),
        "head_reference_score": round(float(head_score), 6),
        "hand_reference_score": round(float(hand_score), 6),
        "cowgirl_reference_score": round(float(cowgirl_score), 6),
        "pelvis_motion_proxy": round(float(pelvis_motion), 6),
        "head_motion_proxy": round(float(head_motion), 6),
        "core_controller_gate": core_gate,
        "missing_core_controllers": core_controllers.get("missing_core_controllers", []),
        "warnings": warnings,
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    traps = [r for r in rows if r.get("head_or_oral_domain_trap")]
    pose_false = [r for r in rows if r.get("cowgirl_pose_false_positive")]
    counts = Counter(
        "trap" if r.get("head_or_oral_domain_trap") else "cowgirl_pose_false_positive" if r.get("cowgirl_pose_false_positive") else "clear"
        for r in rows
    )
    lines = [
        "# BJ/Oral Trap Guard Report",
        "",
        "This guard blocks head/BJ/oral-domain trap candidates from generation-safe Cowgirl. It does not create training labels.",
        "",
        f"- Windows audited: {len(rows)}",
        f"- Head/BJ/oral trap candidates: {len(traps)}",
        f"- Cowgirl-pose false-positive candidates: {len(pose_false)}",
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in counts.most_common()) if counts else lines.append("- None")
    lines.extend(["", "## Top Trap Candidates", ""])
    for row in sorted(traps, key=lambda r: float(r.get("bj_oral_trap_confidence") or 0.0), reverse=True)[:25]:
        lines.append(
            f"- `{row.get('window_id')}` confidence={row.get('bj_oral_trap_confidence')} "
            f"bj={row.get('bj_reference_score')} head={row.get('head_reference_score')} "
            f"cowgirl={row.get('cowgirl_reference_score')} scene=`{row.get('source_scene_file')}`"
        )
    if not traps:
        lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

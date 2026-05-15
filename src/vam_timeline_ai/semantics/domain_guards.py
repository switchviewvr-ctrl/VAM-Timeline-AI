"""Domain guardrails for Cowgirl/Riding semantic guesses."""

from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_domain_guards(feature_row: dict[str, Any], body_quality: dict[str, Any] | None = None) -> dict[str, Any]:
    values = feature_row.get("feature_values", {}) or {}
    body_quality = body_quality or {}
    warnings: list[str] = []
    audit_labels: list[str] = []
    confidence_multiplier = 1.0

    pelvis_energy = _f(values.get("pelvis_movement_energy"))
    head_energy = _f(values.get("head_motion_energy"))
    hand_energy = _f(values.get("left_hand_motion_energy")) + _f(values.get("right_hand_motion_energy"))
    leg_energy = _f(values.get("knee_motion_energy_left")) + _f(values.get("knee_motion_energy_right")) + _f(values.get("foot_motion_energy_left")) + _f(values.get("foot_motion_energy_right"))

    q = body_quality.get("body_motion_quality")
    if q in {"controller_only_whole_person_motion", "root_only_motion"}:
        audit_labels.append("root_only_motion_false_positive")
        if q == "controller_only_whole_person_motion":
            audit_labels.append("controller_only_whole_person_motion")
        warnings.append("Downgrade Cowgirl: motion appears root/whole-person dominant, not limb/body-controller animation.")
        confidence_multiplier = min(confidence_multiplier, 0.25)
    if q in {"static_or_empty", "static_or_micro_motion"} or body_quality.get("static_or_micro_motion"):
        audit_labels.append("static_or_micro_motion")
        warnings.append("Downgrade Cowgirl: static or micro-motion window.")
        confidence_multiplier = min(confidence_multiplier, 0.2)
    if body_quality.get("minimal_head_motion_only"):
        audit_labels.append("minimal_head_motion")
        warnings.append("Minimal head-only motion is a head gesture candidate, not Cowgirl.")
        confidence_multiplier = min(confidence_multiplier, 0.25)
    if body_quality.get("minimal_hand_jitter_only"):
        audit_labels.append("minimal_hand_jitter")
        warnings.append("Minimal hand jitter is isolated gesture/noise, not Cowgirl.")
        confidence_multiplier = min(confidence_multiplier, 0.25)

    if head_energy > max(pelvis_energy, hand_energy, leg_energy, 1e-9) * 2.0 and pelvis_energy < 0.05:
        audit_labels.append("possible_non_cowgirl_head_dominant_motion")
        warnings.append("Head-dominant rhythmic motion may be BJ/oral/head gesture rather than Cowgirl.")
        confidence_multiplier = min(confidence_multiplier, 0.45)

    if not audit_labels:
        audit_labels.append("domain_guard_clear")
    return {
        "domain_guard_warnings": warnings,
        "domain_guard_audit_labels": audit_labels,
        "cowgirl_confidence_multiplier": confidence_multiplier,
    }


def _f(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0

"""Audit-only schema for future visual review signals.

These fields are intentionally optional and never training truth unless a human
promotes them in a separate manual-label workflow.
"""

from __future__ import annotations

from typing import Any


VISUAL_JUDGE_SCHEMA_V0: dict[str, Any] = {
    "schema": "vam_timeline_ai_visual_judge_schema_v0",
    "purpose": "optional_future_visual_review_assistance",
    "audit_only": True,
    "human_verification_required": True,
    "may_be_used_as_ground_truth_without_human_confirmation": False,
    "fields": {
        "visual_family_guess": {"type": "string", "required": False},
        "visual_pose_guess": {"type": "string", "required": False},
        "visual_motion_guess": {"type": "string", "required": False},
        "visual_contact_guess": {"type": "string", "required": False},
        "visual_pose_broken_score": {"type": "number", "required": False, "range": [0.0, 1.0]},
        "visual_confidence": {"type": "number", "required": False, "range": [0.0, 1.0]},
        "visual_model_name": {"type": "string", "required": False},
        "visual_model_prompt": {"type": "string", "required": False},
        "human_verified": {"type": "boolean", "required": False},
    },
}


def visual_judge_schema_v0() -> dict[str, Any]:
    """Return a copy of the audit-only visual judge schema."""

    return {
        **VISUAL_JUDGE_SCHEMA_V0,
        "fields": dict(VISUAL_JUDGE_SCHEMA_V0["fields"]),
    }

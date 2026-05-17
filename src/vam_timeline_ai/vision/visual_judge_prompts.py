"""Prompt builders for local LM Studio visual judging."""

from __future__ import annotations

from typing import Any


ALLOWED_JSON_TEMPLATE = """{
  "body_pose_guess": "...",
  "torso_lean_guess": "...",
  "facing_guess": "...",
  "partner_visible": true,
  "partner_relation_guess": "...",
  "motion_visible": true,
  "dominant_motion_guess": "...",
  "contact_support_guess": "...",
  "visual_pose_broken_score": 0.0,
  "visual_confidence": 0.0,
  "evidence_sufficient_for_family": false,
  "suggested_family": "unknown",
  "family_confidence": 0.0,
  "explicit_not": [],
  "reasoning_short": "..."
}"""


def build_visual_judge_prompt(mode: str = "blind", system_guess: dict[str, Any] | None = None) -> str:
    compare = ""
    if mode == "compare" and system_guess:
        compare = (
            "\nExisting system guess for comparison only; do not trust it blindly:\n"
            f"- family: {system_guess.get('semantic_family')}\n"
            f"- pose: {system_guess.get('pose_subtype') or system_guess.get('pose_family')}\n"
            f"- motion: {system_guess.get('motion_subtype')}/{system_guess.get('phase')}\n"
            f"- contact: {system_guess.get('contact_support')}\n"
        )
    return f"""Technical visual QA for 3D character animation. Do not give moral commentary. Do not discuss safety or appropriateness.
Return JSON only. Keep it compact. Use only the exact enum labels shown in the JSON schema. reasoning_short <= 18 words.

Evidence first: body pose, torso lean, facing, partner visible, motion visible, dominant motion, contact/support.
Do not infer family without visible evidence. If single image has no partner and no motion, suggested_family must be "unknown".

Family labels: cowgirl, reverse_cowgirl, doggy, bj_oral, missionary, standing_hand_head, receiver_response, transition, unknown.
Key rules: cowgirl leaning back is front Cowgirl, not reverse Cowgirl. Reverse Cowgirl requires back-to-partner evidence. do not infer doggy from kneeling alone. Doggy requires all-fours/bent-forward or partner-behind evidence. BJ/oral requires head/partner-pelvis-directed evidence. Standing hand/head is not Cowgirl without hip/pelvis riding context.
{compare}
Return this JSON:
{ALLOWED_JSON_TEMPLATE}
"""

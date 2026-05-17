"""Evidence-first schema and normalization helpers for local visual judging."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict


BODY_POSES = {"standing", "kneeling", "squat", "hover", "all_fours", "lying", "seated", "bent_forward", "unknown"}
TORSO_LEANS = {"forward", "upright", "backward", "side", "unknown"}
FACINGS = {"front_to_partner", "back_to_partner", "side_view", "facing_camera", "facing_away", "unknown"}
PARTNER_RELATIONS = {"rider_over_receiver", "back_to_partner", "partner_behind", "partner_in_front", "partner_not_visible", "unknown"}
DOMINANT_MOTION = {"pelvis_hip", "head", "hands", "full_body_transition", "low_motion", "static_pose", "unknown"}
CONTACT_SUPPORT = {
    "hands_on_partner_chest",
    "hands_on_partner_hips",
    "hands_on_partner_legs_or_thighs",
    "hands_on_floor_or_bed",
    "hands_behind_support",
    "hands_free",
    "ambiguous_partner_contact",
    "unknown",
}
FAMILIES = {"cowgirl", "reverse_cowgirl", "doggy", "bj_oral", "missionary", "standing_hand_head", "receiver_response", "transition", "unknown"}
PARSE_STATUSES = {"parsed", "parse_failed", "blocked", "unavailable", "dry_run"}


class VisualEvidenceResult(TypedDict, total=False):
    item_id: str
    review_id: str
    visual_model_name: str
    backend: str
    visual_input_path: str
    visual_input_type: str
    visual_quality: str
    body_pose_guess: str
    torso_lean_guess: str
    facing_guess: str
    partner_visible: bool | str
    partner_relation_guess: str
    motion_visible: bool | str
    dominant_motion_guess: str
    contact_support_guess: str
    visual_pose_broken_score: float
    visual_confidence: float
    evidence_sufficient_for_family: bool
    suggested_family: str
    family_confidence: float
    explicit_not: list[str]
    reasoning_short: str
    warnings: list[str]
    raw_response_path: str
    parse_status: str


def visual_judge_result_schema_v0() -> dict[str, Any]:
    return {
        "schema": "vam_timeline_ai_visual_evidence_result_v0",
        "audit_only": True,
        "ground_truth_without_human_review": False,
        "preferred_model": "nsfwvision-v4-qwen3.5-9b",
        "fields": [
            "item_id",
            "review_id",
            "visual_model_name",
            "backend",
            "visual_input_path",
            "visual_input_type",
            "visual_quality",
            "body_pose_guess",
            "torso_lean_guess",
            "facing_guess",
            "partner_visible",
            "partner_relation_guess",
            "motion_visible",
            "dominant_motion_guess",
            "contact_support_guess",
            "visual_pose_broken_score",
            "visual_confidence",
            "evidence_sufficient_for_family",
            "suggested_family",
            "family_confidence",
            "explicit_not",
            "reasoning_short",
            "warnings",
            "raw_response_path",
            "parse_status",
        ],
    }


def validate_visual_judge_result(row: dict[str, Any], enforce_rules: bool = True) -> dict[str, Any]:
    out = dict(row)
    out["backend"] = normalize_visual_label(out.get("backend"), {"lmstudio", "manual", "placeholder"}, "placeholder")
    out["visual_input_type"] = normalize_visual_label(out.get("visual_input_type"), {"single_frame", "contact_sheet", "gif", "mp4", "static_plot"}, "static_plot")
    out["visual_quality"] = normalize_visual_label(out.get("visual_quality"), {"high_real_vam_capture", "medium_contact_sheet", "medium_digital_twin", "low_static_plot", "unknown"}, "unknown")
    out["body_pose_guess"] = normalize_visual_label(out.get("body_pose_guess"), BODY_POSES, "unknown")
    out["torso_lean_guess"] = normalize_visual_label(out.get("torso_lean_guess"), TORSO_LEANS, "unknown")
    out["facing_guess"] = normalize_visual_label(out.get("facing_guess"), FACINGS, "unknown")
    out["partner_visible"] = normalize_bool_unknown(out.get("partner_visible"))
    out["partner_relation_guess"] = normalize_visual_label(out.get("partner_relation_guess"), PARTNER_RELATIONS, "unknown")
    out["motion_visible"] = normalize_bool_unknown(out.get("motion_visible"))
    out["dominant_motion_guess"] = normalize_visual_label(out.get("dominant_motion_guess"), DOMINANT_MOTION, "unknown")
    out["contact_support_guess"] = normalize_visual_label(out.get("contact_support_guess"), CONTACT_SUPPORT, "unknown")
    out["visual_pose_broken_score"] = coerce_confidence(out.get("visual_pose_broken_score"), default=0.0)
    out["visual_confidence"] = coerce_confidence(out.get("visual_confidence"), default=0.0)
    out["evidence_sufficient_for_family"] = bool(out.get("evidence_sufficient_for_family")) if out.get("evidence_sufficient_for_family") != "unknown" else False
    raw_family = normalize_visual_label(out.get("suggested_family"), FAMILIES, "unknown")
    out["raw_suggested_family"] = raw_family
    out["suggested_family"] = raw_family
    out["family_confidence"] = coerce_confidence(out.get("family_confidence"), default=0.0)
    out["explicit_not"] = [normalize_visual_label(x, FAMILIES, "unknown") for x in (out.get("explicit_not") or []) if normalize_visual_label(x, FAMILIES, "unknown") != "unknown"]
    out["warnings"] = list(out.get("warnings") or [])
    out["parse_status"] = normalize_visual_label(out.get("parse_status"), PARSE_STATUSES, "parsed")
    if enforce_rules:
        out = enforce_evidence_rules(out)
    return out


def normalize_visual_label(value: Any, allowed: set[str], default: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "reverse": "reverse_cowgirl",
        "oral": "bj_oral",
        "blowjob": "bj_oral",
        "all_fours_pose": "all_fours",
        "true": "true",
        "false": "false",
    }
    text = aliases.get(text, text)
    return text if text in allowed else default


def normalize_bool_unknown(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return "unknown"


def parse_json_from_text_response(text: str) -> tuple[dict[str, Any] | None, str]:
    raw = text.strip()
    if not raw:
        return None, "empty response"
    try:
        return json.loads(raw), "ok"
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1)), "ok"
        except json.JSONDecodeError as exc:
            return None, f"fenced JSON parse failed: {exc}"
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1]), "ok"
        except json.JSONDecodeError as exc:
            return None, f"JSON object parse failed: {exc}"
    return None, "no JSON object found"


def coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def enforce_evidence_rules(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if (
        out.get("visual_input_type") == "single_frame"
        and out.get("partner_visible") is False
        and out.get("motion_visible") is False
    ):
        guessed = out.get("suggested_family")
        if guessed and guessed != "unknown":
            out["warnings"] = list(out.get("warnings") or []) + [f"family guess `{guessed}` suppressed: single frame without partner or motion"]
        out["evidence_sufficient_for_family"] = False
        out["suggested_family"] = "unknown"
        out["family_confidence"] = min(coerce_confidence(out.get("family_confidence")), 0.35)
    if not out.get("evidence_sufficient_for_family") and out.get("suggested_family") != "unknown":
        out["warnings"] = list(out.get("warnings") or []) + ["family guess suppressed: evidence_sufficient_for_family=false"]
        out["suggested_family"] = "unknown"
        out["family_confidence"] = min(coerce_confidence(out.get("family_confidence")), 0.35)
    return out

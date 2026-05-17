"""Error taxonomy summaries from the human review ledger."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl


TAXONOMY = [
    "root_person_world_false_positive",
    "receiver_as_rider",
    "bj_oral_as_cowgirl",
    "standing_hand_head_as_cowgirl",
    "low_motion_pose_context_as_clean_motion",
    "intro_alignment_as_clean_motion",
    "contact_support_wrong_target",
    "partner_context_missing",
    "duplicate_review_selection",
    "foot_anchor_weird",
    "missing_required_controllers",
    "controller_distance_outlier",
    "controller_orientation_invalid",
    "pose_broken",
    "generation_safe_false_positive",
    "semantic_correct_but_generation_unsafe",
]


def build_error_taxonomy_report(human_review_ledger: str | Path, out: str | Path) -> dict[str, Any]:
    rows = load_jsonl(human_review_ledger)
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {key: [] for key in TAXONOMY}
    for row in rows:
        for category in _categories(row):
            counts[category] += 1
            if len(examples[category]) < 5:
                examples[category].append(f"{Path(row.get('source_review_folder') or '').name}/{row.get('review_id')}")
    _write_report(rows, counts, examples, out)
    return {"status": "ok", "rows": len(rows), "counts": dict(counts), "top_items": counts.most_common(8)}


def _categories(row: dict[str, Any]) -> set[str]:
    tags = set(str(x) for x in (row.get("error_tags") or []))
    notes = str(row.get("notes") or "").lower()
    system_family = str(row.get("system_semantic_family") or "")
    human_family = str(row.get("human_semantic_family") or "")
    result: set[str] = set()
    if "root_only_motion_false_positive" in tags or "controller_only_whole_person_motion" in tags:
        result.add("root_person_world_false_positive")
    if "receiver_body_response" in tags or human_family == "receiver_response":
        result.add("receiver_as_rider")
    if human_family == "bj_oral" or "bj_oral_motion_candidate" in tags:
        if system_family == "cowgirl" or "not_cowgirl" in tags:
            result.add("bj_oral_as_cowgirl")
    if human_family == "standing_hand_head_gesture" or "standing_hand_head_gesture" in tags:
        result.add("standing_hand_head_as_cowgirl")
    if {"low_motion_hold", "low_motion", "cowgirl_pose_context"} & tags:
        result.add("low_motion_pose_context_as_clean_motion")
    if {"intro_alignment", "possible_insertion_setup"} & tags:
        result.add("intro_alignment_as_clean_motion")
    if "ambiguous_partner_contact" in tags or "wrong target" in notes:
        result.add("contact_support_wrong_target")
    if "missing_partner_context" in tags or "partner/contact context is unknown" in notes:
        result.add("partner_context_missing")
    if {"duplicate_like_low_motion_context", "repeated_duplicate_review_selection"} & tags:
        result.add("duplicate_review_selection")
    if {"foot_anchor_motion_weird", "lower_body_anchor_not_stable"} & tags:
        result.add("foot_anchor_weird")
    if "missing_required_controllers" in tags or "controller_data_insufficient" in tags:
        result.add("missing_required_controllers")
    if "controller_distance_outlier" in tags:
        result.add("controller_distance_outlier")
    if {"controller_orientation_invalid", "controller_rotation_invalid", "controller_twist_invalid"} & tags:
        result.add("controller_orientation_invalid")
    if "pose_broken" in tags:
        result.add("pose_broken")
    if str(row.get("system_generation_safe")).lower() == "true" and row.get("verdict") in {"wrong", "wrong_or_unclear"}:
        result.add("generation_safe_false_positive")
    if "cowgirl_true_segment" in tags and ({"foot_anchor_motion_weird", "lower_body_anchor_not_stable", "generation_pose_invalid"} & tags):
        result.add("semantic_correct_but_generation_unsafe")
    return result


def _write_report(rows: list[dict[str, Any]], counts: Counter[str], examples: dict[str, list[str]], out: str | Path) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fixed = {
        "bj_oral_as_cowgirl": "mitigated by preserving BJ/oral as its own semantic family",
        "low_motion_pose_context_as_clean_motion": "mitigated by clean_v3 calibration v1 phase split",
        "intro_alignment_as_clean_motion": "mitigated by intro/alignment phase split",
        "duplicate_review_selection": "mitigated by v16 sample/scene/low-motion caps",
        "foot_anchor_weird": "tracked as anchor warning; still needs visual review",
    }
    lines = [
        "# Error Taxonomy Report",
        "",
        f"- Ledger rows: {len(rows)}",
        "",
        "## Taxonomy Counts",
        "",
    ]
    for category in TAXONOMY:
        lines.append(f"- `{category}`: {counts.get(category, 0)}")
    lines.extend(["", "## Highest Priority", ""])
    for category, count in counts.most_common(8):
        lines.append(f"- `{category}`: {count} examples; status: {fixed.get(category, 'open / needs more review')}")
        if examples.get(category):
            lines.append(f"  Examples: {', '.join(examples[category])}")
    lines.extend(["", "## Suggested Next Calibration Target", ""])
    if counts.get("contact_support_wrong_target", 0) or counts.get("partner_context_missing", 0):
        lines.append("Contact/support and partner-context confidence remain the next highest-risk areas.")
    elif counts:
        lines.append(f"Focus next on `{counts.most_common(1)[0][0]}`.")
    else:
        lines.append("No reviewed error tags yet. Review v16 before further calibration.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

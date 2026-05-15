"""Rule-based pose classifier v0 for clean_v3."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.pose_semantics import PoseSemanticRecord, pose_subtype_from_context, support_context_from_features


def classify_poses_v0(
    pose_features: str | Path,
    relative_reference_matches: str | Path | None,
    handmade_features: str | Path | None,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    matches = {r.get("window_id"): r for r in load_jsonl(relative_reference_matches or "") if r.get("window_id")}
    rows: list[dict[str, Any]] = []
    for feat in load_jsonl(pose_features):
        rows.append(_classify_pose_row(feat, matches.get(feat.get("window_id"), {})).to_dict())
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def _classify_pose_row(feat: dict[str, Any], match: dict[str, Any]) -> PoseSemanticRecord:
    kneeling = _num(feat.get("kneeling_score"))
    squat = _num(feat.get("squat_score"))
    standing = _num(feat.get("standing_score"))
    lying_back = _num(feat.get("lying_on_back_score"))
    hands_support = _num(feat.get("hands_forward_support_score"))
    reference_text = str(match.get("best_family") or match.get("nearest_family") or match.get("matched_family") or match.get("top_reference_family") or "").lower()
    source_text = " ".join(str(feat.get(k) or "") for k in ("source_scene_file", "technical_atom_id")).lower()
    pose_family = "unknown"
    confidence = 0.15
    if standing >= 0.55:
        pose_family = "standing"
        confidence = standing
    elif lying_back >= 0.45:
        pose_family = "lying_receiver"
        confidence = lying_back
    elif "bj" in reference_text or "oral" in reference_text or "bj" in source_text or "oral" in source_text:
        pose_family = "bj_oral" if kneeling >= 0.25 else "unknown"
        confidence = max(kneeling, 0.45)
    elif kneeling >= 0.45 or squat >= 0.45 or "cowgirl" in reference_text or "riding" in source_text:
        pose_family = "cowgirl" if max(kneeling, squat) >= 0.35 or "cowgirl" in reference_text or "riding" in source_text else "kneeling_general"
        confidence = max(kneeling, squat, 0.45 if pose_family == "cowgirl" else 0.25)
    elif hands_support >= 0.45:
        pose_family = "hand_head_gesture"
        confidence = hands_support
    subtype = pose_subtype_from_context(pose_family, feat, reference_text + " " + source_text)
    support = support_context_from_features(feat)
    required = ["pelvis_or_hip", "chest_or_abdomen"]
    if pose_family in {"cowgirl", "kneeling_general", "bj_oral"}:
        required.extend(["knees", "feet"])
    optional = ["hands", "head"]
    warnings = list(feat.get("warnings") or [])
    if pose_family == "cowgirl":
        warnings.append("Pose is Cowgirl-compatible only; motion and partner relation must be checked separately.")
    if pose_family == "bj_oral":
        warnings.append("BJ/oral pose context is preserved as its own family and must not be folded into Cowgirl.")
    return PoseSemanticRecord(
        window_id=str(feat.get("window_id")),
        sample_id=feat.get("sample_id"),
        source_scene_file=feat.get("source_scene_file"),
        technical_atom_id=feat.get("technical_atom_id"),
        pose_family=pose_family,
        pose_subtype=subtype,
        support_context=support,
        anchor_requirements={"required_controllers": required, "optional_controllers": optional},
        pose_confidence=round(min(1.0, confidence), 6),
        pose_generation_safe=pose_family in {"cowgirl", "bj_oral", "kneeling_general"} and _num(feat.get("pose_anchor_completeness")) >= 0.35,
        warnings=_dedupe(warnings),
    )


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    families = Counter(r.get("pose_family") for r in rows)
    subtypes = Counter(r.get("pose_subtype") for r in rows)
    lines = [
        "# Pose Semantics Report V0",
        "",
        "Pose semantics are separate from motion semantics. They are not manual labels.",
        "",
        f"- Rows: {len(rows)}",
        "",
        "## Pose Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common()) if families else lines.append("- None")
    lines.extend(["", "## Pose Subtypes", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in subtypes.most_common()) if subtypes else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any) -> float:
    try:
        if value != value:
            return 0.0
        return float(value or 0.0)
    except Exception:
        return 0.0


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(str(item))
    return out

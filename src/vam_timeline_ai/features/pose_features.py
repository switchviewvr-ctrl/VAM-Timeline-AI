"""Approximate pose feature extraction for clean_v3.

These features are deliberately lightweight proxies. They use controller
coverage, relative-motion metadata, anchor audits, and candidate context; they
do not promote machine guesses into manual labels.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def extract_pose_features_v0(
    relative_index: str | Path,
    body_quality: str | Path,
    pose_anchor_completeness: str | Path,
    controller_validity: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    body = {r.get("window_id"): r for r in load_jsonl(body_quality) if r.get("window_id")}
    anchors = {r.get("window_id"): r for r in load_jsonl(pose_anchor_completeness) if r.get("window_id")}
    validity = {r.get("window_id"): r for r in load_jsonl(controller_validity) if r.get("window_id")}
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(relative_index):
        wid = row.get("window_id")
        if not wid:
            continue
        rows.append(_pose_feature_row(row, body.get(wid, {}), anchors.get(wid, {}), validity.get(wid, {})))
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report)
    return rows


def _pose_feature_row(index: dict[str, Any], body: dict[str, Any], anchors: dict[str, Any], validity: dict[str, Any]) -> dict[str, Any]:
    controllers = [str(c) for c in (index.get("controllers") or index.get("controller_names") or [])]
    bodyparts = [str(c) for c in (index.get("bodyparts") or [])]
    lower = {c.lower() for c in controllers + bodyparts}
    has = lambda token: any(token in c for c in lower)
    anchor_score = _num(anchors.get("pose_anchor_completeness_score"), 0.5 if (has("foot") and has("knee")) else 0.25)
    controller_coverage = len(controllers) / 12.0 if controllers else 0.0
    has_feet = has("foot")
    has_knees = has("knee")
    has_hands = has("hand")
    has_chest = has("chest") or has("abdomen")
    has_head = has("head")
    has_hip = has("hip") or has("pelvis")
    technical_atom = str(index.get("technical_atom_id") or "").lower()
    scene = str(index.get("source_scene_file") or "").lower()
    quality = str(body.get("body_motion_quality") or "")
    standing_hint = 0.5 if ("stand" in scene or "standing" in quality) else 0.0
    receiver_hint = 0.45 if ("receiver" in quality or "lying" in scene) else 0.0
    kneeling = 0.0
    if has_knees and has_feet and has_hip:
        kneeling += 0.42
    if has_hands and has_chest:
        kneeling += 0.12
    if anchor_score >= 0.65:
        kneeling += 0.16
    if "empty" in technical_atom:
        kneeling *= 0.4
        receiver_hint += 0.25
    squat = 0.35 if has_feet and has_hip and not has_knees else 0.18
    standing = max(standing_hint, 0.25 if has_feet and has_head and not has_knees else 0.0)
    lying_back = min(1.0, receiver_hint)
    lying_prone = 0.15 if "prone" in scene else 0.0
    hands_support = 0.5 if has_hands and has_chest else 0.15 if has_hands else 0.0
    feet_behind = 0.55 if has_feet and has_knees else 0.25 if has_feet else 0.0
    knees_under = 0.6 if has_knees and has_hip else 0.0
    fold = min(1.0, (feet_behind + knees_under + anchor_score) / 3.0)
    torso_forward = 0.45 if has_chest and has_hip else 0.0
    if has_hands and has_chest:
        torso_forward += 0.15
    warnings = []
    if "invalid" in str(validity.get("controller_validity_status") or ""):
        warnings.append("Controller validity is invalid; pose feature confidence reduced.")
    if not controllers:
        warnings.append("No controller list found; pose features are unknown proxies.")
    return {
        "window_id": index.get("window_id"),
        "sample_id": index.get("sample_id"),
        "source_scene_file": index.get("source_scene_file"),
        "technical_atom_id": index.get("technical_atom_id"),
        "pose_feature_version": "pose_features_v0",
        "pelvis_height_proxy": 0.55 if has_hip else 0.0,
        "chest_height_proxy": 0.72 if has_chest else 0.0,
        "head_height_proxy": 0.9 if has_head else 0.0,
        "knee_height_proxy": 0.28 if has_knees else 0.0,
        "foot_height_proxy": 0.08 if has_feet else 0.0,
        "hand_height_proxy": 0.58 if has_hands else 0.0,
        "torso_forward_lean_proxy": round(min(1.0, torso_forward), 6),
        "torso_upright_proxy": round(max(0.0, 1.0 - torso_forward), 6),
        "body_flatness_proxy": round(max(lying_back, lying_prone), 6),
        "kneeling_score": round(min(1.0, kneeling), 6),
        "squat_score": round(min(1.0, squat), 6),
        "standing_score": round(min(1.0, standing), 6),
        "lying_on_back_score": round(min(1.0, lying_back), 6),
        "lying_prone_score": round(min(1.0, lying_prone), 6),
        "hands_forward_support_score": round(min(1.0, hands_support), 6),
        "feet_behind_body_score": round(min(1.0, feet_behind), 6),
        "knees_under_body_score": round(min(1.0, knees_under), 6),
        "lower_body_fold_score": round(min(1.0, fold), 6),
        "pose_anchor_completeness": round(anchor_score, 6),
        "pose_controller_coverage": round(min(1.0, controller_coverage), 6),
        "controller_names": controllers,
        "warnings": warnings,
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    buckets = Counter()
    for row in rows:
        if _num(row.get("standing_score")) >= 0.5:
            buckets["standing_like"] += 1
        elif _num(row.get("kneeling_score")) >= 0.45:
            buckets["kneeling_like"] += 1
        elif _num(row.get("lying_on_back_score")) >= 0.45:
            buckets["lying_receiver_like"] += 1
        else:
            buckets["unknown_pose"] += 1
    lines = [
        "# Pose Feature Report V0",
        "",
        "Pose features are approximate geometry/controller proxies, not motion labels and not manual ground truth.",
        "",
        f"- Rows: {len(rows)}",
        "",
        "## Proxy Buckets",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in buckets.most_common()) if buckets else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value != value:
            return default
        return float(value)
    except Exception:
        return default

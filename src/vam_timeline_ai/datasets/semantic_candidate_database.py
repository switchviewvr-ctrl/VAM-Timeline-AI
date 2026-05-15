"""Global semantic candidate inventory.

This is review triage infrastructure, not ML training data.  It preserves
candidate semantic families such as Cowgirl and BJ/oral without promoting audit
labels into manual ground truth.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


FAMILIES = {
    "cowgirl",
    "bj_oral",
    "doggy",
    "hand_gesture",
    "head_gesture",
    "transition",
    "receiver_response",
    "unknown",
}


def build_semantic_candidate_db_v0(
    run_dir: str | Path,
    cowgirl_db: str | Path,
    bj_oral_domain: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    rel = {r.get("window_id"): r for r in load_jsonl(relative_features) if r.get("window_id")}
    traj = {r.get("window_id"): r for r in load_jsonl(trajectory_features) if r.get("window_id")}
    cowgirl_rows = [r for r in load_jsonl(cowgirl_db) if r.get("window_id")]
    bj_rows = {r.get("window_id"): r for r in load_jsonl(bj_oral_domain) if r.get("window_id")}

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in cowgirl_rows:
        wid = str(row.get("window_id"))
        rows.append(_from_cowgirl_record(row, bj_rows.get(wid, {}), rel.get(wid, {}), traj.get(wid, {})))
        seen.add(wid)

    for wid, bj in bj_rows.items():
        if wid in seen or not bj.get("bj_oral_motion_candidate"):
            continue
        rows.append(_from_bj_record(bj, rel.get(wid, {}), traj.get(wid, {})))

    rows.sort(key=lambda r: (r.get("semantic_family") != "cowgirl", r.get("semantic_family") != "bj_oral", -float(r.get("family_confidence") or 0.0)))
    write_jsonl(out_jsonl, rows)
    _write_csv(rows, out_csv)
    _write_report(rows, report)
    _write_larger_review_plan(run / "datasets" / "larger_review_batch_plan.md")
    return rows


def _from_cowgirl_record(row: dict[str, Any], bj: dict[str, Any], rel: dict[str, Any], traj: dict[str, Any]) -> dict[str, Any]:
    family = str(row.get("semantic_family") or "unknown")
    if family not in FAMILIES:
        family = "unknown"
    if bj.get("bj_oral_motion_candidate"):
        family = "bj_oral"
    confidence = float(row.get("semantic_cowgirl_score") or 0.0)
    if family == "bj_oral":
        confidence = float(row.get("bj_oral_confidence") or bj.get("bj_oral_confidence") or confidence)
    elif family in {"hand_gesture", "head_gesture", "receiver_response"}:
        confidence = max(confidence, 0.45)
    return {
        "candidate_id": f"semantic_v0::{row.get('window_id')}",
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_atom_id": row.get("technical_atom_id"),
        "semantic_family": family,
        "family_confidence": round(float(confidence), 6),
        "category": row.get("category"),
        "subtype": row.get("cowgirl_subtype") or bj.get("subtype") or "unknown",
        "generation_safe": bool(row.get("generation_safe")),
        "excluded_from_cowgirl": bool(row.get("excluded_from_cowgirl") or bj.get("excluded_from_cowgirl")),
        "preserve_for_future_dataset": bool(row.get("preserve_for_future_dataset") or bj.get("preserve_for_future_dataset")),
        "feature_refs": {
            "relative_features_window_id": rel.get("window_id"),
            "trajectory_features_window_id": traj.get("window_id"),
        },
        "warnings": row.get("warnings", []) or bj.get("warnings", []),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _from_bj_record(bj: dict[str, Any], rel: dict[str, Any], traj: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": f"semantic_v0::{bj.get('window_id')}",
        "window_id": bj.get("window_id"),
        "sample_id": bj.get("sample_id"),
        "source_scene_file": bj.get("source_scene_file"),
        "technical_atom_id": bj.get("technical_atom_id"),
        "semantic_family": "bj_oral",
        "family_confidence": bj.get("bj_oral_confidence"),
        "category": "bj_oral_motion_candidate",
        "subtype": bj.get("subtype") or "bj_oral",
        "generation_safe": bool(bj.get("bj_oral_generation_candidate")),
        "excluded_from_cowgirl": True,
        "preserve_for_future_dataset": True,
        "feature_refs": {
            "relative_features_window_id": rel.get("window_id"),
            "trajectory_features_window_id": traj.get("window_id"),
        },
        "warnings": bj.get("warnings", []),
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _write_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    target = Path(out_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "window_id",
        "sample_id",
        "source_scene_file",
        "technical_atom_id",
        "semantic_family",
        "family_confidence",
        "category",
        "subtype",
        "generation_safe",
        "excluded_from_cowgirl",
        "preserve_for_future_dataset",
    ]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    families = Counter(r.get("semantic_family") for r in rows)
    categories = Counter(r.get("category") for r in rows)
    conflicts = [r for r in rows if r.get("semantic_family") == "bj_oral" and str(r.get("category", "")).startswith("semantic_cowgirl")]
    lines = [
        "# Semantic Candidate DB V0 Report",
        "",
        "This is a global candidate inventory for semantic-family review. It is not ML training data and contains no human ground truth labels.",
        "",
        f"- Records: {len(rows)}",
        f"- Cowgirl candidates: {families.get('cowgirl', 0)}",
        f"- BJ/oral candidates: {families.get('bj_oral', 0)}",
        f"- Family conflicts: {len(conflicts)}",
        "",
        "## Semantic Families",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in families.most_common()) if families else lines.append("- None")
    lines.extend(["", "## Categories", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in categories.most_common()) if categories else lines.append("- None")
    lines.extend(["", "## Recommended Next Family-Specific DBs", "", "- BJ/oral candidate DB", "- Doggy candidate DB", "- Hand/head gesture candidate DB"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_larger_review_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Larger Review Batch Plan",
        "",
        "Prepared only as a proposal. No 60-item batch has been generated.",
        "",
        "Recommended mix:",
        "",
        "- 30 generation-safe Cowgirl candidates",
        "- 10 soft-fail/pose-invalid Cowgirl candidates",
        "- 10 BJ/oral candidates",
        "- 5 standing/hand/head candidates",
        "- 5 receiver-response negatives",
        "- 5 unknown/unusable candidates",
        "",
        "Run this only after the 10-item v13 review looks good.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

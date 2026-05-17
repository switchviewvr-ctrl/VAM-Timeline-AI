"""Cowgirl ML v3 labels from item-level human review.

This module builds supervised review-ranker labels. It does not auto-label
unreviewed windows and does not modify manual_labels.yaml.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ml.cowgirl_ml_label_dataset_v2 import FALSE, LABEL_KEYS, TRUE, UNKNOWN


LATEST_NEGATIVE_REVIEW_IDS = {
    "ml_cowgirl_v2_003": "no hip motion; gate failed",
    "ml_cowgirl_v2_004": "no cycle animation; gate failed",
    "ml_cowgirl_v2_007": "handjob/manual; cycle and hip-motion gates failed for Cowgirl",
    "ml_cowgirl_v2_008": "standing/walking animation; all Cowgirl gates failed",
    "ml_cowgirl_v2_009": "standing/walking animation; all Cowgirl gates failed",
}


def build_cowgirl_ml_labels_v3(
    new_run: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(new_run)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    positive_path = run / "audits" / "strict_gate_clean_cowgirl_v1_010" / "semantic_review_010.jsonl"
    for item in load_jsonl(positive_path):
        _append(rows, seen, _label(item, "strict_gate_clean_cowgirl_10_of_10_human_confirmed", TRUE, "User confirmed all 10 strict gate-clean Cowgirl clips were correct."))

    negative_path = run / "audits" / "ml_assisted_cowgirl_review_v2" / "semantic_review_010.jsonl"
    for item in load_jsonl(negative_path):
        rid = str(item.get("review_id") or "")
        if rid not in LATEST_NEGATIVE_REVIEW_IDS:
            continue
        _append(rows, seen, _label(item, "latest_item_level_negative_review", FALSE, LATEST_NEGATIVE_REVIEW_IDS[rid]))

    bj_path = run / "audits" / "bj_doggy_missionary_motion_cycles_v1" / "semantic_review_010.jsonl"
    for item in load_jsonl(bj_path):
        if len([r for r in rows if r.get("source_kind") == "bj_review_cowgirl_negative_5_of_5"]) >= 5:
            break
        _append(rows, seen, _label(item, "bj_review_cowgirl_negative_5_of_5", FALSE, "User confirmed BJ/oral extraction examples were true BJ and not Cowgirl."))

    write_jsonl(out_jsonl, rows)
    summary = {
        "schema": "cowgirl_ml_labels_v3",
        "rows": len(rows),
        "label_counts": {key: dict(Counter(row.get(key, UNKNOWN) for row in rows)) for key in LABEL_KEYS},
        "sources": {
            "positive_strict_gate_clean_review": str(positive_path),
            "negative_item_review": str(negative_path),
            "bj_negative_review": str(bj_path),
        },
        "target_policy": "human item-level review only; gates and ML scores are features, not label truth",
        "manual_labels_modified": False,
        "auto_labeling_performed": False,
        "timeline_generation_performed": False,
    }
    _write_report(report, summary)
    return summary


def _label(item: dict[str, Any], source_kind: str, is_cowgirl_clean: str, note: str) -> dict[str, Any]:
    row = {
        "schema": "cowgirl_ml_label_v3",
        "source_kind": source_kind,
        "review_id": item.get("review_id") or "",
        "window_id": item.get("window_id") or "",
        "sample_id": item.get("sample_id") or "",
        "source_id": item.get("source_id") or "",
        "source_scene_file": item.get("source_scene_file") or "",
        "technical_actor_id": item.get("technical_actor_id") or item.get("technical_atom_id") or "",
        "start_seconds": item.get("start_seconds"),
        "end_seconds": item.get("end_seconds"),
        "human_review_note": note,
        "is_human_reviewed_label": True,
        "manual_labels_yaml_modified": False,
    }
    row.update({key: UNKNOWN for key in LABEL_KEYS})
    row["label_cowgirl_semantic_family"] = is_cowgirl_clean
    row["label_cowgirl_clean_motion"] = is_cowgirl_clean
    row["label_cowgirl_pose_context"] = FALSE if is_cowgirl_clean == TRUE else UNKNOWN
    row["label_cowgirl_transition"] = FALSE if is_cowgirl_clean == TRUE else UNKNOWN
    row["label_not_cowgirl_bj_oral"] = TRUE if source_kind == "bj_review_cowgirl_negative_5_of_5" else FALSE if is_cowgirl_clean == TRUE else UNKNOWN
    row["label_not_cowgirl_handjob"] = TRUE if "handjob" in note.lower() or "manual" in note.lower() else FALSE if is_cowgirl_clean == TRUE else UNKNOWN
    row["label_not_cowgirl_standing_hand_head"] = TRUE if "standing" in note.lower() or "walking" in note.lower() else FALSE if is_cowgirl_clean == TRUE else UNKNOWN
    row["label_generation_safe_or_complete"] = TRUE if is_cowgirl_clean == TRUE else UNKNOWN
    row["label_pose_incomplete_missing_controllers"] = FALSE if is_cowgirl_clean == TRUE else UNKNOWN
    return row


def _append(rows: list[dict[str, Any]], seen: set[str], row: dict[str, Any]) -> None:
    wid = str(row.get("window_id") or "")
    if not wid or wid in seen:
        return
    rows.append(row)
    seen.add(wid)


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cowgirl ML Labels v3",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Target policy: {summary['target_policy']}",
        "- Auto-labeling performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Label Counts",
        "",
    ]
    for key, counts in summary["label_counts"].items():
        lines.append(f"- `{key}`: `{counts}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

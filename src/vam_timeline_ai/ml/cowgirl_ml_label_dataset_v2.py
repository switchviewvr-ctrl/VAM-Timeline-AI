"""Cowgirl ML v2 labels from human review findings only.

Gate and ontology outputs are useful features, but they are not targets here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


TRUE = "true"
FALSE = "false"
UNKNOWN = "unknown"


LABEL_KEYS = [
    "label_cowgirl_semantic_family",
    "label_cowgirl_clean_motion",
    "label_cowgirl_pose_context",
    "label_cowgirl_transition",
    "label_not_cowgirl_bj_oral",
    "label_not_cowgirl_handjob",
    "label_not_cowgirl_standing_hand_head",
    "label_generation_safe_or_complete",
    "label_pose_incomplete_missing_controllers",
]


def write_latest_semantic_review_findings(new_run: str | Path) -> dict[str, Any]:
    run = Path(new_run)
    out = run / "reports" / "latest_semantic_review_findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Latest Semantic Review Findings",
        "",
        "## Cowgirl Cycle Review",
        "",
        "- reviewed_candidates: 10",
        "- semantic_cowgirl_correct: 10",
        "- semantic_family_precision: 10/10",
        "- pose_controller_completeness_issues: 3",
        "- issue_type: missing hand controllers",
        "- interpretation: semantic family correct; generation safety/completeness partial",
        "",
        "## BJ Extraction Review",
        "",
        "- requested: BJ / Doggy / Missionary",
        "- bj_available_under_strict_gates: 17",
        "- bj_exported: 5",
        "- reviewed_bj_correct: 5/5",
        "- doggy_clean_candidates: 0",
        "- missionary_clean_candidates: 0",
        "- interpretation: strict gates did not fake or fill missing families",
        "",
        "## Tracking Distinction",
        "",
        "- semantic_correctness: family/motion meaning is correct",
        "- pose_completeness: required controllers/supports are present enough for analysis",
        "- generation_safety: pose/controller completeness is sufficient for future generation/export",
        "",
        "These findings may be used as supervised review-ranker labels. They do not modify `manual_labels.yaml`.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "report": str(out)}


def build_cowgirl_ml_labels_v2(
    base_run: str | Path,
    new_run: str | Path,
    human_ledger: str | Path,
    manual_gt: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    base = Path(base_run)
    run = Path(new_run)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    write_latest_semantic_review_findings(run)

    # Explicit latest human findings: 10/10 reviewed Cowgirl cycle candidates correct.
    cowgirl_review = _first_existing(
        [
            run / "audits" / "semantic_review_motion_cycles_v1_020" / "semantic_review_010.jsonl",
            run / "audits" / "cowgirl_motion_cycles_v1_010" / "semantic_review_010.jsonl",
        ]
    )
    for idx, item in enumerate(load_jsonl(cowgirl_review)[:10], start=1):
        label = _base_label(item, "latest_cowgirl_cycle_review_10_of_10")
        label.update(
            {
                "label_cowgirl_semantic_family": TRUE,
                "label_cowgirl_clean_motion": TRUE,
                "label_cowgirl_pose_context": FALSE,
                "label_cowgirl_transition": FALSE,
                "label_not_cowgirl_bj_oral": FALSE,
                "label_not_cowgirl_handjob": FALSE,
                "label_not_cowgirl_standing_hand_head": FALSE,
                "label_generation_safe_or_complete": UNKNOWN,
                "label_pose_incomplete_missing_controllers": UNKNOWN,
                "human_review_note": "Cowgirl semantic family correct. Aggregate review found 3/10 missing-hand controller completeness issues; this row was not individually marked.",
            }
        )
        _append_unique(rows, seen, label)

    # Explicit latest human finding: 5/5 exported BJ candidates were true BJ/oral and Cowgirl negatives.
    bj_review = run / "audits" / "bj_doggy_missionary_motion_cycles_v1" / "semantic_review_010.jsonl"
    for item in load_jsonl(bj_review)[:5]:
        label = _base_label(item, "latest_bj_extraction_review_5_of_5")
        label.update(
            {
                "label_cowgirl_semantic_family": FALSE,
                "label_cowgirl_clean_motion": FALSE,
                "label_cowgirl_pose_context": FALSE,
                "label_cowgirl_transition": FALSE,
                "label_not_cowgirl_bj_oral": TRUE,
                "label_not_cowgirl_handjob": FALSE,
                "label_not_cowgirl_standing_hand_head": FALSE,
                "label_generation_safe_or_complete": UNKNOWN,
                "label_pose_incomplete_missing_controllers": UNKNOWN,
                "human_review_note": "Reviewed as true BJ/oral animation; use as Cowgirl negative, not as all-family training truth.",
            }
        )
        _append_unique(rows, seen, label)

    # Existing human ledger remains truth where it can be matched later by window/time.
    for item in load_jsonl(human_ledger):
        label = _label_from_ledger(item)
        if any(label.get(k) in {TRUE, FALSE} for k in LABEL_KEYS):
            _append_unique(rows, seen, label)

    # Manual GT captures are trusted pose references. They are usually not window rows,
    # so the feature table records them as reference labels but does not force matches.
    for item in load_jsonl(manual_gt):
        label = _label_from_manual_gt(item)
        _append_unique(rows, seen, label)

    write_jsonl(out_jsonl, rows)
    summary = {
        "status": "ok",
        "schema": "cowgirl_ml_labels_v2",
        "rows": len(rows),
        "label_counts": {key: dict(Counter(row.get(key, UNKNOWN) for row in rows)) for key in LABEL_KEYS},
        "sources": {
            "base_run": str(base),
            "new_run": str(run),
            "human_ledger": str(human_ledger),
            "manual_gt": str(manual_gt),
            "latest_cowgirl_review": str(cowgirl_review),
            "latest_bj_review": str(bj_review),
        },
        "target_policy": "human review and manual GT only; gates are features, not labels",
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    _write_report(report, summary)
    return summary


def _base_label(item: dict[str, Any], source_kind: str) -> dict[str, Any]:
    return {
        "schema": "cowgirl_ml_label_v2",
        "source_kind": source_kind,
        "review_id": item.get("review_id") or "",
        "capture_id": item.get("capture_id") or "",
        "window_id": item.get("window_id") or "",
        "sample_id": item.get("sample_id") or "",
        "source_id": item.get("source_id") or "",
        "source_scene_file": item.get("source_scene_file") or "",
        "technical_actor_id": item.get("technical_actor_id") or item.get("technical_atom_id") or "",
        "start_seconds": item.get("start_seconds"),
        "end_seconds": item.get("end_seconds"),
        "human_semantic_family": item.get("resolved_motion_family") or item.get("resolved_semantic_family") or item.get("semantic_family") or "",
        "human_motion": item.get("motion_subtype") or item.get("resolved_motion_subtype") or "",
        "is_human_reviewed_label": True,
        "is_training_truth_source": "human_review_or_manual_gt",
        "manual_labels_yaml_modified": False,
    }


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _label_from_ledger(item: dict[str, Any]) -> dict[str, Any]:
    label = _base_label(item, "human_review_ledger")
    text = " ".join(str(item.get(k) or "") for k in ["human_semantic_family", "human_motion", "verdict", "notes"]).lower()
    tags = {str(t).lower() for t in (item.get("error_tags") or [])}
    family = str(item.get("human_semantic_family") or "").lower()

    if family == "cowgirl" or ("cowgirl" in text and "not_cowgirl" not in tags and "nicht cowgirl" not in text):
        cowgirl = TRUE
    elif family in {"bj_oral", "handjob", "standing_hand_head", "receiver_response", "unknown"} or "not_cowgirl" in tags or "bj" in text:
        cowgirl = FALSE
    else:
        cowgirl = UNKNOWN
    label.update({key: UNKNOWN for key in LABEL_KEYS})
    label["label_cowgirl_semantic_family"] = cowgirl
    label["label_cowgirl_clean_motion"] = TRUE if cowgirl == TRUE and ("clean" in text or "grind" in text or "bounce" in text or "riding" in text) else UNKNOWN
    if "transition" in text or "pose_context" in text or "low_motion" in tags:
        label["label_cowgirl_clean_motion"] = FALSE
        label["label_cowgirl_transition"] = TRUE if "transition" in text else UNKNOWN
        label["label_cowgirl_pose_context"] = TRUE if "pose" in text or "low_motion" in tags else UNKNOWN
    label["label_not_cowgirl_bj_oral"] = TRUE if ("bj" in text or family == "bj_oral") and cowgirl == FALSE else FALSE if cowgirl == TRUE else UNKNOWN
    label["label_not_cowgirl_handjob"] = TRUE if ("handjob" in text or family == "handjob") and cowgirl == FALSE else FALSE if cowgirl == TRUE else UNKNOWN
    label["label_not_cowgirl_standing_hand_head"] = TRUE if ("standing" in text or family == "standing_hand_head") and cowgirl == FALSE else FALSE if cowgirl == TRUE else UNKNOWN
    if "missing" in text or "controller" in text or "broken_pose" in tags:
        label["label_generation_safe_or_complete"] = FALSE
        label["label_pose_incomplete_missing_controllers"] = TRUE
    return label


def _label_from_manual_gt(item: dict[str, Any]) -> dict[str, Any]:
    human = item.get("human_labels") if isinstance(item.get("human_labels"), dict) else {}
    family = str(human.get("family") or "").lower()
    label = _base_label(
        {
            "capture_id": item.get("capture_id"),
            "source_scene_file": "manual_pose_ground_truth_v1",
            "technical_actor_id": item.get("rider_atom") or "",
        },
        "manual_pose_ground_truth_reference",
    )
    label.update({key: UNKNOWN for key in LABEL_KEYS})
    label["human_semantic_family"] = family
    label["human_motion"] = human.get("motion_intent") or ""
    label["label_cowgirl_semantic_family"] = TRUE if family in {"cowgirl", "reverse_cowgirl"} else FALSE if family in {"bj_oral", "handjob", "doggy", "missionary"} else UNKNOWN
    label["label_not_cowgirl_bj_oral"] = TRUE if family == "bj_oral" else FALSE if label["label_cowgirl_semantic_family"] == TRUE else UNKNOWN
    label["label_not_cowgirl_handjob"] = TRUE if family == "handjob" else FALSE if label["label_cowgirl_semantic_family"] == TRUE else UNKNOWN
    label["is_window_matchable"] = False
    return label


def _append_unique(rows: list[dict[str, Any]], seen: set[tuple[str, str, str]], row: dict[str, Any]) -> None:
    key = (str(row.get("window_id") or row.get("capture_id") or ""), str(row.get("review_id") or ""), str(row.get("source_kind") or ""))
    if key not in seen:
        rows.append(row)
        seen.add(key)


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cowgirl ML Labels v2",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Target policy: {summary['target_policy']}",
        "- VLM labels used as truth: `false`",
        "- Machine/gate labels used as truth: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Label Counts",
        "",
    ]
    for key, counts in summary["label_counts"].items():
        lines.append(f"- `{key}`: `{counts}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

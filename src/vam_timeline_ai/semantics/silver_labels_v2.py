"""Build deduplicated silver v2 labels from aggregated machine scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


ROLE_LABELS = {"rider_active", "receiver_passive", "partner_context_static"}
EXCLUDED_DEFAULT_LABELS = ROLE_LABELS | {"contact_unknown"}


def build_silver_labels_v2(
    window_scores: str | Path,
    pair_scores: str | Path,
    out_window_jsonl: str | Path,
    out_pair_jsonl: str | Path,
    out_yaml: str | Path,
    report: str | Path,
    min_score: float = 0.78,
) -> dict[str, Any]:
    window_rows = _build_records(load_jsonl(window_scores), "window", min_score)
    pair_rows = _build_records(load_jsonl(pair_scores), "pair", min_score)
    write_jsonl(out_window_jsonl, window_rows)
    write_jsonl(out_pair_jsonl, pair_rows)
    _write_yaml(window_rows, pair_rows, out_yaml, min_score)
    summary = _summary(window_rows, pair_rows, Path(window_scores).parent, min_score)
    _write_report(summary, report)
    return summary


def _build_records(score_rows: list[dict[str, Any]], scope: str, min_score: float) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for score in score_rows:
        label = str(score.get("label") or "")
        status = str(score.get("recommended_status") or "")
        final_score = _float(score.get("final_score"))
        if final_score < min_score and status not in {"reject_conflict", "review_only"}:
            continue
        if label == "contact_unknown":
            if final_score >= min_score:
                _add_review_only(grouped, score, scope, "contact_unknown is not a positive training label")
            continue
        if status == "silver_positive_candidate":
            _add_label(grouped, score, scope, "positive_labels", final_score)
        elif status == "silver_negative_candidate":
            _add_label(grouped, score, scope, "negative_labels", final_score)
        elif status in {"review_only", "reject_conflict"} and final_score >= min_score:
            reason = "conflict" if status == "reject_conflict" else "review_only"
            _add_review_only(grouped, score, scope, reason)
    rows: list[dict[str, Any]] = []
    for key, entry in sorted(grouped.items()):
        rows.append(_finalize(entry))
    return rows


def _add_label(grouped: dict[str, dict[str, Any]], score: dict[str, Any], scope: str, field: str, final_score: float) -> None:
    entry = _entry(grouped, score, scope)
    label = str(score.get("label") or "")
    entry[field].add(label)
    entry["scores_by_label"][label] = max(float(entry["scores_by_label"].get(label, 0.0)), final_score)
    entry["confidence_by_label"][label] = max(float(entry["confidence_by_label"].get(label, 0.0)), _float(score.get("max_confidence")))
    entry["rule_ids"].update(score.get("rule_ids", []) or [])
    entry["evidence_summary"][label] = {
        "evidence_count": score.get("evidence_count"),
        "distinct_rule_count": score.get("distinct_rule_count"),
        "supporting_pair_window_count": score.get("supporting_pair_window_count"),
        "recommended_status": score.get("recommended_status"),
        "high_risk_proxy_label": score.get("high_risk_proxy_label", False),
    }
    if label in EXCLUDED_DEFAULT_LABELS or scope == "pair":
        entry["excluded_from_default_training"][label] = "high-risk role/contact or pair-specific label"
    else:
        entry["default_trainable_labels"].add(label)


def _add_review_only(grouped: dict[str, dict[str, Any]], score: dict[str, Any], scope: str, reason: str) -> None:
    entry = _entry(grouped, score, scope)
    label = str(score.get("label") or "")
    entry["review_only_labels"].add(label)
    entry["scores_by_label"][label] = max(float(entry["scores_by_label"].get(label, 0.0)), _float(score.get("final_score")))
    entry["confidence_by_label"][label] = max(float(entry["confidence_by_label"].get(label, 0.0)), _float(score.get("max_confidence")))
    entry["excluded_from_default_training"][label] = reason
    entry["rule_ids"].update(score.get("rule_ids", []) or [])


def _entry(grouped: dict[str, dict[str, Any]], score: dict[str, Any], scope: str) -> dict[str, Any]:
    key = str(score.get("window_id") if scope == "window" else score.get("pair_window_id"))
    entry = grouped.setdefault(
        key,
        {
            "scope": scope,
            "window_id": score.get("window_id") if scope == "window" else None,
            "pair_window_id": score.get("pair_window_id") if scope == "pair" else None,
            "window_ids": set(score.get("window_ids", []) or ([score.get("window_id")] if score.get("window_id") else [])),
            "sample_ids": set(score.get("sample_ids", []) or []),
            "source_scene_files": set(score.get("source_scene_files", []) or []),
            "technical_atom_ids": set(score.get("technical_atom_ids", []) or []),
            "positive_labels": set(),
            "negative_labels": set(),
            "review_only_labels": set(),
            "default_trainable_labels": set(),
            "excluded_from_default_training": {},
            "scores_by_label": {},
            "confidence_by_label": {},
            "rule_ids": set(),
            "evidence_summary": {},
            "label_source": "silver_machine_v2",
            "is_human_ground_truth": False,
        },
    )
    entry["window_ids"].update(score.get("window_ids", []) or [])
    entry["sample_ids"].update(score.get("sample_ids", []) or [])
    entry["source_scene_files"].update(score.get("source_scene_files", []) or [])
    entry["technical_atom_ids"].update(score.get("technical_atom_ids", []) or [])
    return entry


def _finalize(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": entry["scope"],
        "window_id": entry.get("window_id"),
        "pair_window_id": entry.get("pair_window_id"),
        "window_ids": sorted(v for v in entry["window_ids"] if v),
        "sample_ids": sorted(v for v in entry["sample_ids"] if v),
        "source_scene_files": sorted(v for v in entry["source_scene_files"] if v),
        "technical_atom_ids": sorted(v for v in entry["technical_atom_ids"] if v),
        "positive_labels": sorted(entry["positive_labels"]),
        "negative_labels": sorted(entry["negative_labels"]),
        "review_only_labels": sorted(entry["review_only_labels"]),
        "default_trainable_labels": sorted(entry["default_trainable_labels"]),
        "excluded_from_default_training": dict(sorted(entry["excluded_from_default_training"].items())),
        "scores_by_label": {k: round(float(v), 5) for k, v in sorted(entry["scores_by_label"].items())},
        "confidence_by_label": {k: round(float(v), 5) for k, v in sorted(entry["confidence_by_label"].items())},
        "rule_ids": sorted(entry["rule_ids"]),
        "evidence_summary": entry["evidence_summary"],
        "label_source": "silver_machine_v2",
        "is_human_ground_truth": False,
    }


def _summary(window_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], folder: Path, min_score: float) -> dict[str, Any]:
    v1_path = folder / "silver_labels_v1.jsonl"
    prop_path = folder / "machine_label_proposals_v1.jsonl"
    v1_count = sum(1 for _ in v1_path.open("r", encoding="utf-8-sig")) if v1_path.exists() else None
    prop_count = sum(1 for _ in prop_path.open("r", encoding="utf-8-sig")) if prop_path.exists() else None
    window_counts = Counter()
    pair_counts = Counter()
    excluded = Counter()
    for row in window_rows:
        window_counts.update(row.get("positive_labels", []) or [])
        window_counts.update(row.get("negative_labels", []) or [])
        excluded.update(row.get("excluded_from_default_training", {}).keys())
    for row in pair_rows:
        pair_counts.update(row.get("positive_labels", []) or [])
        pair_counts.update(row.get("negative_labels", []) or [])
        excluded.update(row.get("excluded_from_default_training", {}).keys())
    return {
        "min_score": min_score,
        "raw_proposal_count": prop_count,
        "v1_silver_record_count": v1_count,
        "v2_silver_window_records": len(window_rows),
        "v2_silver_pair_records": len(pair_rows),
        "v2_window_label_counts": dict(window_counts.most_common()),
        "v2_pair_label_counts": dict(pair_counts.most_common()),
        "classes_excluded_from_default_training": dict(excluded.most_common()),
        "is_human_ground_truth": False,
    }


def _write_yaml(window_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], out: str | Path, min_score: float) -> None:
    data = {
        "metadata": {
            "label_source": "silver_machine_v2",
            "is_human_ground_truth": False,
            "min_score": min_score,
            "warning": "Silver v2 labels are aggregated machine labels, not human semantic ground truth.",
        },
        "windows": {row["window_id"]: row for row in window_rows if row.get("window_id")},
        "pair_windows": {row["pair_window_id"]: row for row in pair_rows if row.get("pair_window_id")},
    }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Silver Labels v2",
        "",
        "Silver v2 is built from aggregated scores, not raw proposals. It remains machine-generated and not human ground truth.",
        "",
        f"- Raw proposal count: {summary['raw_proposal_count']}",
        f"- Silver v1 record count: {summary['v1_silver_record_count']}",
        f"- Silver v2 window records: {summary['v2_silver_window_records']}",
        f"- Silver v2 pair records: {summary['v2_silver_pair_records']}",
        f"- Minimum score: {summary['min_score']}",
        "",
        "## Window Label Counts",
        "",
    ]
    lines.extend(f"- `{key}`: {count}" for key, count in summary["v2_window_label_counts"].items() or [("None", 0)])
    lines.extend(["", "## Pair Label Counts", ""])
    lines.extend(f"- `{key}`: {count}" for key, count in summary["v2_pair_label_counts"].items() or [("None", 0)])
    lines.extend(["", "## Excluded From Default Training", ""])
    lines.extend(f"- `{key}`: {count}" for key, count in summary["classes_excluded_from_default_training"].items() or [("None", 0)])
    lines.extend(
        [
            "",
            "Role labels and pair-specific labels are high-risk proxies. `contact_unknown` is never a positive training label by default.",
        ]
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0

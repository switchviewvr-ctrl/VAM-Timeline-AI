"""Aggregate raw machine proposals into canonical v2 label scores."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.identity import stable_hash
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


ROLE_LABELS = {"rider_active", "receiver_passive", "partner_context_static"}
CONTACT_LABELS = {"cowgirl_hand_supported_on_partner", "cowgirl_hand_supported_on_partner_chest", "cowgirl_hand_supported_on_partner_hips", "contact_unknown"}
HIGH_RISK_LABELS = ROLE_LABELS | {"contact_unknown"}
FAST_SLOW = {"cowgirl_fast_shallow", "cowgirl_deep_slow"}


def aggregate_machine_labels_v2(
    proposals: str | Path,
    out_window_jsonl: str | Path,
    out_pair_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    proposal_rows = load_jsonl(proposals)
    labels_by_window: dict[str, set[str]] = defaultdict(set)
    for row in proposal_rows:
        wid = str(row.get("window_id") or "")
        label = str(row.get("label") or "")
        if wid and label:
            labels_by_window[wid].add(label)

    window_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in proposal_rows:
        wid = str(row.get("window_id") or "")
        pid = str(row.get("pair_window_id") or "")
        label = str(row.get("label") or "")
        if wid and label:
            window_groups[(wid, label)].append(row)
        if pid and label:
            pair_groups[(pid, label)].append(row)

    window_rows = [_score_group("window", key[0], key[1], rows, labels_by_window.get(key[0], set())) for key, rows in window_groups.items()]
    pair_rows = [_score_group("pair", key[0], key[1], rows, labels_by_window.get(str(rows[0].get("window_id") or ""), set())) for key, rows in pair_groups.items()]
    window_rows.sort(key=lambda row: (row["window_id"], row["label"]))
    pair_rows.sort(key=lambda row: (row["pair_window_id"], row["label"]))
    write_jsonl(out_window_jsonl, window_rows)
    write_jsonl(out_pair_jsonl, pair_rows)
    summary = _summary(window_rows, pair_rows, proposal_rows)
    _write_report(summary, report)
    return summary


def _score_group(scope: str, group_id: str, label: str, rows: list[dict[str, Any]], window_labels: set[str]) -> dict[str, Any]:
    confidences = [_float(row.get("confidence")) for row in rows]
    rule_ids = {str(row.get("rule_id") or "") for row in rows if row.get("rule_id")}
    pair_ids = {str(row.get("pair_window_id") or "") for row in rows if row.get("pair_window_id")}
    sources = {str(row.get("source") or "") for row in rows if row.get("source")}
    ptypes = {str(row.get("proposal_type") or "") for row in rows if row.get("proposal_type")}
    labels = {str(row.get("label") or "") for row in rows}
    sample_ids = {str(row.get("sample_id") or "") for row in rows if row.get("sample_id")}
    window_ids = {str(row.get("window_id") or "") for row in rows if row.get("window_id")}
    scene_files = {str(row.get("source_scene_file") or "") for row in rows if row.get("source_scene_file")}
    atom_ids = {str(row.get("technical_atom_id") or "") for row in rows if row.get("technical_atom_id")}
    conflicts = _conflicts(label, window_labels)
    max_conf = max(confidences) if confidences else 0.0
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    pair_cap = min(1.0, math.log1p(len(pair_ids)) / math.log(8.0)) if pair_ids else 0.0
    rule_bonus = min(0.06, 0.02 * max(0, len(rule_ids) - 1))
    evidence_bonus = min(0.05, 0.01 * math.log1p(len(rows)))
    pair_bonus = 0.04 * pair_cap
    conflict_penalty = 0.18 if conflicts else 0.0
    broad_penalty = 0.12 if label == "contact_unknown" else 0.0
    final_score = max(0.0, min(1.0, max_conf * 0.78 + mean_conf * 0.14 + rule_bonus + evidence_bonus + pair_bonus - conflict_penalty - broad_penalty))
    status = _status(scope, label, ptypes, final_score, conflicts, len(rows), len(rule_ids))
    return {
        "score_id": f"score_{scope}_{stable_hash([scope, group_id, label], 14)}",
        "scope": scope,
        "window_id": group_id if scope == "window" else None,
        "pair_window_id": group_id if scope == "pair" else None,
        "window_ids": sorted(window_ids),
        "sample_ids": sorted(sample_ids),
        "source_scene_files": sorted(scene_files),
        "technical_atom_ids": sorted(atom_ids),
        "label": label,
        "proposal_types": sorted(ptypes),
        "max_confidence": round(float(max_conf), 5),
        "mean_confidence": round(float(mean_conf), 5),
        "evidence_count": len(rows),
        "distinct_rule_count": len(rule_ids),
        "distinct_pair_context_count": len(pair_ids),
        "supporting_pair_window_count": len(pair_ids),
        "source_types": sorted(sources),
        "rule_ids": sorted(rule_ids),
        "conflict_flags": conflicts,
        "high_risk_proxy_label": label in HIGH_RISK_LABELS or label in CONTACT_LABELS,
        "final_score": round(float(final_score), 5),
        "recommended_status": status,
        "is_human_ground_truth": False,
        "evidence_summary": {
            "pair_context_evidence_capped": round(pair_cap, 5),
            "labels_seen_in_same_window": sorted(window_labels),
            "raw_repetition_not_counted_linearly": True,
        },
    }


def _status(scope: str, label: str, ptypes: set[str], score: float, conflicts: list[str], evidence_count: int, rule_count: int) -> str:
    if conflicts:
        return "reject_conflict"
    if label == "contact_unknown":
        return "review_only"
    if label in ROLE_LABELS and scope == "window":
        return "review_only"
    if "uncertain" in ptypes:
        return "review_only"
    if evidence_count < 1 or rule_count < 1:
        return "reject_insufficient_evidence"
    if label in CONTACT_LABELS and scope == "window":
        return "review_only"
    if "negative" in ptypes:
        return "silver_negative_candidate" if score >= 0.78 else "review_only"
    if score >= 0.78:
        return "silver_positive_candidate"
    if score >= 0.62:
        return "review_only"
    return "reject_insufficient_evidence"


def _conflicts(label: str, labels: set[str]) -> list[str]:
    flags: list[str] = []
    if label in FAST_SLOW and FAST_SLOW.issubset(labels):
        flags.append("fast_and_slow")
    if label in {"cowgirl_pause_hold", "cowgirl_fast_shallow"} and {"cowgirl_pause_hold", "cowgirl_fast_shallow"}.issubset(labels):
        flags.append("pause_hold_and_fast_shallow")
    if label in ROLE_LABELS and {"rider_active", "partner_context_static"}.issubset(labels):
        flags.append("rider_active_and_partner_context_static")
    if label == "contact_unknown" and labels & (CONTACT_LABELS - {"contact_unknown"}):
        flags.append("contact_unknown_with_specific_contact")
    if label in ROLE_LABELS and len(labels & ROLE_LABELS) > 1:
        flags.append("multiple_role_candidates")
    return sorted(set(flags))


def _summary(window_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "raw_proposals": len(proposals),
        "window_score_rows": len(window_rows),
        "pair_score_rows": len(pair_rows),
        "window_status_counts": dict(Counter(row["recommended_status"] for row in window_rows).most_common()),
        "pair_status_counts": dict(Counter(row["recommended_status"] for row in pair_rows).most_common()),
        "window_label_counts": dict(Counter(row["label"] for row in window_rows).most_common()),
        "pair_label_counts": dict(Counter(row["label"] for row in pair_rows).most_common()),
        "conflicted_window_scores": sum(1 for row in window_rows if row["conflict_flags"]),
        "conflicted_pair_scores": sum(1 for row in pair_rows if row["conflict_flags"]),
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Machine Label Aggregation v2",
        "",
        "Raw proposals are collapsed into one score per window-label or pair-window-label. Pair-context evidence is capped/log-scaled.",
        "",
        f"- Raw proposals: {summary['raw_proposals']}",
        f"- Window score rows: {summary['window_score_rows']}",
        f"- Pair score rows: {summary['pair_score_rows']}",
        f"- Conflicted window scores: {summary['conflicted_window_scores']}",
        f"- Conflicted pair scores: {summary['conflicted_pair_scores']}",
        "",
        "## Window Status Counts",
        "",
    ]
    lines.extend(f"- `{key}`: {count}" for key, count in summary["window_status_counts"].items())
    lines.extend(["", "## Pair Status Counts", ""])
    lines.extend(f"- `{key}`: {count}" for key, count in summary["pair_status_counts"].items())
    lines.extend(["", "## Window Labels", ""])
    lines.extend(f"- `{key}`: {count}" for key, count in list(summary["window_label_counts"].items())[:30])
    lines.extend(["", "## Pair Labels", ""])
    lines.extend(f"- `{key}`: {count}" for key, count in list(summary["pair_label_counts"].items())[:30])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")

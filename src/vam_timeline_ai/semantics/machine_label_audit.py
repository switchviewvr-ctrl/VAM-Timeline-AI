"""Audit raw machine label proposals and silver v1 duplication/conflicts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl


FAST_LABELS = {"cowgirl_fast_shallow"}
SLOW_LABELS = {"cowgirl_deep_slow"}
SPECIFIC_CONTACT_LABELS = {
    "cowgirl_hand_supported_on_partner",
    "cowgirl_hand_supported_on_partner_chest",
    "cowgirl_hand_supported_on_partner_hips",
}
ROLE_LABELS = {"rider_active", "receiver_passive", "partner_context_static"}


def audit_machine_labels_v1(
    run_dir: str | Path,
    proposals: str | Path,
    silver_labels: str | Path,
    windows: str | Path,
    pair_windows: str | Path,
    out: str | Path,
    out_json: str | Path,
) -> dict[str, Any]:
    proposal_rows = load_jsonl(proposals)
    silver_rows = load_jsonl(silver_labels) if Path(silver_labels).exists() else []
    window_rows = load_jsonl(windows)
    pair_window_rows = load_jsonl(pair_windows) if Path(pair_windows).exists() else []

    prop_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prop_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    silver_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    silver_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposal_key_counts = Counter()
    label_counts = Counter()
    silver_counts = Counter()
    type_counts = Counter()
    conf_by_label: dict[str, list[float]] = defaultdict(list)
    window_label_pair_counts: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in proposal_rows:
        wid = str(row.get("window_id") or "")
        pid = str(row.get("pair_window_id") or "")
        label = str(row.get("label") or "")
        ptype = str(row.get("proposal_type") or "")
        if wid:
            prop_by_window[wid].append(row)
        if pid:
            prop_by_pair[pid].append(row)
            window_label_pair_counts[(wid, label)].add(pid)
        proposal_key_counts[(wid, pid, label, ptype)] += 1
        label_counts[label] += 1
        type_counts[ptype] += 1
        conf_by_label[label].append(_float(row.get("confidence")))

    for row in silver_rows:
        wid = str(row.get("window_id") or "")
        pid = str(row.get("pair_window_id") or "")
        labels = _silver_labels(row)
        if wid:
            silver_by_window[wid].append(row)
        if pid:
            silver_by_pair[pid].append(row)
        silver_counts.update(labels)

    conflicts = _find_conflicts(prop_by_window)
    duplicate_keys = { "|".join(key): count for key, count in proposal_key_counts.items() if count > 1 }
    labels_many_pair_contexts = [
        {"window_id": wid, "label": label, "pair_context_count": len(pairs)}
        for (wid, label), pairs in window_label_pair_counts.items()
        if len(pairs) > 5
    ]
    labels_many_pair_contexts.sort(key=lambda item: item["pair_context_count"], reverse=True)

    summary = {
        "run_dir": str(run_dir),
        "total_windows": len(window_rows),
        "total_pair_windows": len(pair_window_rows),
        "total_proposals": len(proposal_rows),
        "total_silver_records": len(silver_rows),
        "unique_window_ids_with_proposals": len(prop_by_window),
        "unique_window_ids_with_silver": len(silver_by_window),
        "duplicate_proposal_key_count": len(duplicate_keys),
        "proposal_count_by_label": dict(label_counts.most_common()),
        "silver_count_by_label": dict(silver_counts.most_common()),
        "proposal_type_counts": dict(type_counts.most_common()),
        "proposals_per_window_distribution": _dist([len(v) for v in prop_by_window.values()]),
        "proposals_per_pair_window_distribution": _dist([len(v) for v in prop_by_pair.values()]),
        "silver_records_per_window_distribution": _dist([len(v) for v in silver_by_window.values()]),
        "silver_records_per_pair_window_distribution": _dist([len(v) for v in silver_by_pair.values()]),
        "confidence_distribution_by_label": {label: _dist(vals) for label, vals in conf_by_label.items()},
        "duplicate_proposal_keys_top": sorted(duplicate_keys.items(), key=lambda item: item[1], reverse=True)[:50],
        "labels_many_times_same_window_through_pair_contexts_top": labels_many_pair_contexts[:50],
        "top_windows_by_proposal_count": [{"window_id": wid, "proposal_count": len(rows)} for wid, rows in sorted(prop_by_window.items(), key=lambda item: len(item[1]), reverse=True)[:20]],
        "top_pair_windows_by_proposal_count": [{"pair_window_id": pid, "proposal_count": len(rows)} for pid, rows in sorted(prop_by_pair.items(), key=lambda item: len(item[1]), reverse=True)[:20]],
        "conflict_counts": dict(Counter(item["conflict_type"] for item in conflicts)),
        "conflict_examples": conflicts[:100],
        "warnings": _warnings(label_counts, labels_many_pair_contexts, conflicts),
    }
    dump_json(out_json, summary)
    _write_report(summary, out)
    return summary


def _find_conflicts(prop_by_window: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for wid, rows in prop_by_window.items():
        labels = {str(row.get("label") or "") for row in rows}
        high_specific_contact = {
            str(row.get("label"))
            for row in rows
            if row.get("label") in SPECIFIC_CONTACT_LABELS and _float(row.get("confidence")) >= 0.75
        }
        if labels & FAST_LABELS and labels & SLOW_LABELS:
            conflicts.append({"window_id": wid, "conflict_type": "fast_and_slow", "labels": sorted((labels & FAST_LABELS) | (labels & SLOW_LABELS))})
        if "cowgirl_pause_hold" in labels and "cowgirl_fast_shallow" in labels:
            conflicts.append({"window_id": wid, "conflict_type": "pause_hold_and_fast_shallow", "labels": ["cowgirl_pause_hold", "cowgirl_fast_shallow"]})
        if "rider_active" in labels and "partner_context_static" in labels:
            conflicts.append({"window_id": wid, "conflict_type": "rider_active_and_partner_context_static", "labels": ["rider_active", "partner_context_static"]})
        if "contact_unknown" in labels and high_specific_contact:
            conflicts.append({"window_id": wid, "conflict_type": "contact_unknown_with_specific_contact", "labels": ["contact_unknown", *sorted(high_specific_contact)]})
        role_hits = sorted(labels & ROLE_LABELS)
        if len(role_hits) > 1:
            conflicts.append({"window_id": wid, "conflict_type": "multiple_role_candidates", "labels": role_hits})
    return conflicts


def _warnings(label_counts: Counter[str], inflated: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    total = sum(label_counts.values()) or 1
    for label, count in label_counts.most_common():
        if count / total > 0.2:
            warnings.append(f"`{label}` is very broad ({count} proposals, >20% of all proposals).")
    for label in ["rider_active", "partner_context_static", "contact_unknown"]:
        if label_counts.get(label, 0) > 10000:
            warnings.append(f"`{label}` is likely dominated by pair-window multiplication and should be review-only or excluded from default training.")
    if inflated:
        warnings.append("Some window-label pairs are supported by many pair contexts; aggregation must cap this evidence.")
    if conflicts:
        warnings.append("Contradictory window labels were found; conflicted labels should not become default silver positives.")
    return warnings


def _silver_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ["positive_labels", "negative_labels", "uncertain_labels", "role_candidates", "contact_candidates"]:
        labels.extend(row.get(key, []) or [])
    return labels


def _dist(values: list[float | int]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "min": None, "mean": None, "max": None, "p50": None, "p90": None, "p99": None}
    vals_sorted = sorted(vals)
    return {
        "count": len(vals_sorted),
        "min": vals_sorted[0],
        "mean": round(mean(vals_sorted), 5),
        "max": vals_sorted[-1],
        "p50": _pct(vals_sorted, 50),
        "p90": _pct(vals_sorted, 90),
        "p99": _pct(vals_sorted, 99),
    }


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return round(float(values[idx]), 5)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Machine Label Audit v1",
        "",
        "This report audits raw machine proposals and silver v1 labels. It does not modify labels.",
        "",
        f"- Total windows: {summary['total_windows']}",
        f"- Total pair windows: {summary['total_pair_windows']}",
        f"- Total proposals: {summary['total_proposals']}",
        f"- Total silver records: {summary['total_silver_records']}",
        f"- Unique windows with proposals: {summary['unique_window_ids_with_proposals']}",
        f"- Unique windows with silver labels: {summary['unique_window_ids_with_silver']}",
        f"- Duplicate proposal keys: {summary['duplicate_proposal_key_count']}",
        "",
        "## Distribution",
        "",
        f"- Proposals/window: {summary['proposals_per_window_distribution']}",
        f"- Proposals/pair window: {summary['proposals_per_pair_window_distribution']}",
        f"- Silver records/window: {summary['silver_records_per_window_distribution']}",
        f"- Silver records/pair window: {summary['silver_records_per_pair_window_distribution']}",
        "",
        "## Proposal Counts By Label",
        "",
    ]
    lines.extend(f"- `{label}`: {count}" for label, count in list(summary["proposal_count_by_label"].items())[:40])
    lines.extend(["", "## Silver Counts By Label", ""])
    lines.extend(f"- `{label}`: {count}" for label, count in list(summary["silver_count_by_label"].items())[:40])
    lines.extend(["", "## Proposal Type Counts", ""])
    lines.extend(f"- `{label}`: {count}" for label, count in summary["proposal_type_counts"].items())
    lines.extend(["", "## Conflict Counts", ""])
    if summary["conflict_counts"]:
        lines.extend(f"- `{key}`: {count}" for key, count in summary["conflict_counts"].items())
    else:
        lines.append("- None")
    lines.extend(["", "## Top Pair-Inflated Window Labels", ""])
    for item in summary["labels_many_times_same_window_through_pair_contexts_top"][:20]:
        lines.append(f"- `{item['window_id']}` `{item['label']}`: {item['pair_context_count']} pair contexts")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in summary["warnings"] or ["None"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")

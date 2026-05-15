"""Build silver labels from high-confidence machine proposals.

Silver labels are machine-generated convenience labels. They are not human
ground truth and must never be merged into manual_labels.yaml.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.machine_label_schema import SilverLabelRecord


def build_silver_labels_v1(
    proposals: str | Path,
    out_jsonl: str | Path,
    out_yaml: str | Path,
    report: str | Path,
    min_confidence: float = 0.75,
) -> list[dict[str, Any]]:
    proposal_rows = load_jsonl(proposals)
    grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
    for prop in proposal_rows:
        if prop.get("is_human_ground_truth") is True:
            continue
        if str(prop.get("label", "")).startswith("weak_"):
            continue
        confidence = _float(prop.get("confidence"))
        if confidence < min_confidence:
            continue
        if not prop.get("is_silver_candidate", confidence >= min_confidence):
            continue
        if prop.get("proposal_type") == "uncertain":
            continue
        key = (str(prop.get("window_id") or ""), prop.get("pair_window_id"))
        if not key[0]:
            continue
        entry = grouped.setdefault(
            key,
            {
                "positive_labels": set(),
                "negative_labels": set(),
                "uncertain_labels": set(),
                "role_candidates": set(),
                "contact_candidates": set(),
                "confidence_by_label": {},
                "rule_ids": set(),
                "evidence_summary": defaultdict(list),
            },
        )
        label = str(prop.get("label") or "")
        ptype = str(prop.get("proposal_type") or "")
        group = str(prop.get("label_group") or "")
        if ptype == "negative":
            entry["negative_labels"].add(label)
        elif ptype == "role_candidate" or group == "role_candidate":
            entry["role_candidates"].add(label)
        elif ptype == "contact_candidate" or group == "contact_candidate":
            entry["contact_candidates"].add(label)
        else:
            entry["positive_labels"].add(label)
        entry["confidence_by_label"][label] = max(float(entry["confidence_by_label"].get(label, 0.0)), confidence)
        entry["rule_ids"].add(str(prop.get("rule_id") or "unknown_rule"))
        for feature in prop.get("evidence_features", []) or []:
            entry["evidence_summary"][label].append(feature)

    rows: list[dict[str, Any]] = []
    for (window_id, pair_window_id), entry in sorted(grouped.items(), key=lambda item: (item[0][0], "" if item[0][1] is None else str(item[0][1]))):
        record = SilverLabelRecord(
            window_id=window_id,
            pair_window_id=pair_window_id,
            positive_labels=sorted(entry["positive_labels"]),
            negative_labels=sorted(entry["negative_labels"]),
            uncertain_labels=sorted(entry["uncertain_labels"]),
            role_candidates=sorted(entry["role_candidates"]),
            contact_candidates=sorted(entry["contact_candidates"]),
            confidence_by_label={k: round(float(v), 5) for k, v in sorted(entry["confidence_by_label"].items())},
            rule_ids=sorted(entry["rule_ids"]),
            evidence_summary={k: sorted(set(v)) for k, v in sorted(entry["evidence_summary"].items())},
            is_human_ground_truth=False,
        ).to_dict()
        rows.append(record)
    write_jsonl(out_jsonl, rows)
    _write_yaml(rows, out_yaml, min_confidence)
    _write_report(rows, proposal_rows, report, min_confidence)
    return rows


def _write_yaml(rows: list[dict[str, Any]], out: str | Path, min_confidence: float) -> None:
    data: dict[str, Any] = {
        "metadata": {
            "label_source": "silver_machine_v1",
            "is_human_ground_truth": False,
            "min_confidence": min_confidence,
            "warning": "These labels are generated from numeric rules/proxies and are not human semantic ground truth.",
        },
        "windows": {},
    }
    for row in rows:
        data["windows"][row["window_id"]] = {
            "pair_window_id": row.get("pair_window_id"),
            "positive_labels": row.get("positive_labels", []),
            "negative_labels": row.get("negative_labels", []),
            "uncertain_labels": row.get("uncertain_labels", []),
            "role_candidates": row.get("role_candidates", []),
            "contact_candidates": row.get("contact_candidates", []),
            "confidence_by_label": row.get("confidence_by_label", {}),
            "rule_ids": row.get("rule_ids", []),
            "label_source": "silver_machine_v1",
            "is_human_ground_truth": False,
        }
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_report(rows: list[dict[str, Any]], proposals: list[dict[str, Any]], out: str | Path, min_confidence: float) -> None:
    label_counts = Counter()
    for row in rows:
        for key in ["positive_labels", "negative_labels", "role_candidates", "contact_candidates"]:
            label_counts.update(row.get(key, []) or [])
    confidence_bins = Counter()
    for row in rows:
        for value in (row.get("confidence_by_label", {}) or {}).values():
            confidence_bins[_bin(_float(value))] += 1
    lines = [
        "# Silver Labels v1",
        "",
        "Silver labels are high-confidence machine labels. They are not human ground truth and are not written to manual_labels.yaml.",
        "",
        f"- Machine proposals read: {len(proposals)}",
        f"- Minimum confidence: {min_confidence}",
        f"- Silver window records: {len(rows)}",
        f"- Silver label assignments: {sum(label_counts.values())}",
        "",
        "## Counts By Class",
        "",
    ]
    if label_counts:
        lines.extend(f"- `{label}`: {count}" for label, count in label_counts.most_common())
    else:
        lines.append("- No silver labels passed the threshold.")
    lines.extend(["", "## Confidence Distribution", ""])
    if confidence_bins:
        lines.extend(f"- `{key}`: {count}" for key, count in sorted(confidence_bins.items()))
    else:
        lines.append("- No confidence values.")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- `label_source`: `silver_machine_v1`",
            "- `is_human_ground_truth`: `false` for every record",
            "- Uncertain proposals are not promoted to positive silver labels.",
            "- Weak labels remain separate and are not copied into silver labels.",
        ]
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _bin(value: float) -> str:
    if value >= 0.9:
        return "0.90-1.00"
    if value >= 0.8:
        return "0.80-0.90"
    if value >= 0.75:
        return "0.75-0.80"
    return "below-threshold"

"""Review batch for aggregated machine/silver v2 labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.identity import make_review_id
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


ROLE_LABELS = {"rider_active", "receiver_passive", "partner_context_static"}


def build_machine_proposal_review_batch_v2(
    run_dir: str | Path,
    window_scores: str | Path,
    pair_scores: str | Path,
    silver_window_labels: str | Path,
    silver_pair_labels: str | Path,
    out_dir: str | Path,
    batch_size: int = 120,
    max_per_scene: int = 15,
    max_per_sample: int = 3,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    features = {r.get("window_id"): r for r in load_jsonl(run / "features" / "cowgirl_window_features_v1.jsonl") if r.get("window_id")}
    weak = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "weak_labels_v2.jsonl") if r.get("window_id")}
    win_scores = load_jsonl(window_scores)
    pair_score_rows = load_jsonl(pair_scores)
    silver_win = {r.get("window_id"): r for r in load_jsonl(silver_window_labels) if r.get("window_id")}
    silver_pair = {r.get("pair_window_id"): r for r in load_jsonl(silver_pair_labels) if r.get("pair_window_id")}

    candidates = _candidate_scores(win_scores, pair_score_rows, silver_win, silver_pair)
    selected: list[dict[str, Any]] = []
    per_scene = Counter()
    per_sample = Counter()
    per_label = Counter()
    role_rows = 0
    for score, wid, pid, reasons, labels, score_rows in sorted(candidates, key=lambda item: item[0], reverse=True):
        if not wid or wid not in windows:
            continue
        wrow = windows.get(wid, {})
        frow = features.get(wid, {})
        scene = str(frow.get("source_scene_file") or wrow.get("source_scene_file") or "")
        sample = str(frow.get("sample_id") or wrow.get("sample_id") or "")
        if per_scene[scene] >= max_per_scene or per_sample[sample] >= max_per_sample:
            continue
        if labels & ROLE_LABELS:
            if role_rows >= max(10, batch_size // 6):
                continue
            role_rows += 1
        record = _record(out.name, wid, pid, score, reasons, labels, score_rows, wrow, frow, weak.get(wid, {}), silver_win.get(wid), silver_pair.get(pid))
        selected.append(record)
        per_scene[scene] += 1
        per_sample[sample] += 1
        per_label.update(labels)
        if len(selected) >= batch_size:
            break
    write_jsonl(out / "review_batch.jsonl", selected)
    _write_markdown(selected, out / "review_batch.md")
    _write_machine_yaml(selected, out / "machine_label_review_v2.yaml")
    _write_stub(selected, out / "manual_labels.stub.yaml")
    _write_summary(selected, per_scene, per_sample, per_label, out / "batch_summary.md")
    return selected


def _candidate_scores(win_scores: list[dict[str, Any]], pair_scores: list[dict[str, Any]], silver_win: dict[str, dict[str, Any]], silver_pair: dict[str, dict[str, Any]]) -> list[tuple[float, str, str | None, list[str], set[str], list[dict[str, Any]]]]:
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in win_scores:
        if row.get("window_id"):
            by_window[str(row["window_id"])].append(row)
    candidates = []
    for wid, rows in by_window.items():
        labels = {str(row.get("label")) for row in rows if row.get("label")}
        max_score = max(float(row.get("final_score") or 0.0) for row in rows)
        reasons = []
        if silver_win.get(wid):
            reasons.append("silver v2 window example")
            max_score += 1.0
        if any(row.get("recommended_status") == "reject_conflict" for row in rows):
            reasons.append("rejected/conflicted example")
            max_score += 0.8
        if any(0.72 <= float(row.get("final_score") or 0.0) <= 0.82 for row in rows):
            reasons.append("borderline threshold example")
            max_score += 0.5
        if labels & ROLE_LABELS:
            reasons.append("role high-risk example")
            max_score += 0.3
        candidates.append((max_score, wid, None, sorted(set(reasons or ["aggregated machine score example"])), labels, rows))
    for row in pair_scores:
        pid = str(row.get("pair_window_id") or "")
        label = str(row.get("label") or "")
        for wid in row.get("window_ids", []) or []:
            score = float(row.get("final_score") or 0.0) + (1.1 if silver_pair.get(pid) else 0.0)
            reasons = ["pair/contact candidate"]
            if row.get("recommended_status") == "reject_conflict":
                reasons.append("rejected/conflicted pair example")
            if label in ROLE_LABELS:
                reasons.append("role high-risk example")
            candidates.append((score, str(wid), pid, sorted(set(reasons)), {label}, [row]))
    return candidates


def _record(batch_name: str, wid: str, pid: str | None, score: float, reasons: list[str], labels: set[str], score_rows: list[dict[str, Any]], wrow: dict[str, Any], frow: dict[str, Any], weak: dict[str, Any], silver_win: dict[str, Any] | None, silver_pair: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "review_id": make_review_id(wid + (pid or ""), batch_name=batch_name),
        "window_id": wid,
        "pair_window_id": pid,
        "sample_id": frow.get("sample_id") or wrow.get("sample_id"),
        "source_scene_file": frow.get("source_scene_file") or wrow.get("source_scene_file"),
        "technical_atom_id": frow.get("technical_atom_id") or wrow.get("technical_atom_id"),
        "start_seconds": wrow.get("start_seconds"),
        "end_seconds": wrow.get("end_seconds"),
        "duration_seconds": wrow.get("duration_seconds"),
        "top_features": _top_features(frow.get("feature_values", {})),
        "weak_labels_v2": weak.get("weak_labels", []),
        "machine_scores_v2": [_score_hint(row) for row in sorted(score_rows, key=lambda r: float(r.get("final_score") or 0.0), reverse=True)[:10]],
        "silver_v2_window_labels": _silver_hint(silver_win),
        "silver_v2_pair_labels": _silver_hint(silver_pair),
        "machine_proposals": [_score_hint(row) for row in score_rows[:10]],
        "silver_labels": _silver_hint(silver_win) or _silver_hint(silver_pair),
        "why_selected": reasons,
        "selection_score": round(float(score), 5),
        "machine_label_warning": "Aggregated machine/silver v2 labels are review hints only, not human truth.",
        "suggested_labels_empty": [],
        "suggested_role_empty": "unknown",
        "notes_empty": "",
    }


def _score_hint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("label"),
        "final_score": row.get("final_score"),
        "recommended_status": row.get("recommended_status"),
        "conflict_flags": row.get("conflict_flags", []),
        "high_risk_proxy_label": row.get("high_risk_proxy_label", False),
        "rule_ids": row.get("rule_ids", []),
        "is_human_ground_truth": False,
    }


def _silver_hint(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "positive_labels": row.get("positive_labels", []),
        "negative_labels": row.get("negative_labels", []),
        "review_only_labels": row.get("review_only_labels", []),
        "default_trainable_labels": row.get("default_trainable_labels", []),
        "excluded_from_default_training": row.get("excluded_from_default_training", {}),
        "scores_by_label": row.get("scores_by_label", {}),
        "is_human_ground_truth": False,
    }


def _top_features(values: dict[str, Any], n: int = 8) -> dict[str, Any]:
    pairs = []
    for key, value in values.items():
        try:
            val = float(value)
        except Exception:
            continue
        if val == val:
            pairs.append((key, abs(val), val))
    return {key: round(float(val), 5) for key, _, val in sorted(pairs, key=lambda item: item[1], reverse=True)[:n]}


def _write_markdown(rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Machine Proposal Review Batch v2", "", "Manual label fields remain empty. Machine scores are hints only.", "", f"- Review items: {len(rows)}", ""]
    for row in rows[:120]:
        labels = ", ".join(h.get("label", "") for h in row.get("machine_scores_v2", [])[:6])
        lines.extend([f"## `{row['review_id']}`", "", f"- Window: `{row['window_id']}`", f"- Pair window: `{row.get('pair_window_id')}`", f"- Scene: `{row.get('source_scene_file')}`", f"- Machine labels: {labels}", f"- Why selected: {', '.join(row.get('why_selected', []))}", ""])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_machine_yaml(rows: list[dict[str, Any]], out: Path) -> None:
    data = {
        "metadata": {"label_source": "machine_label_review_hints_v2", "is_human_ground_truth": False, "warning": "Suggestions only. Do not merge as manual labels."},
        "windows": {row["window_id"]: {"machine_scores_v2": row.get("machine_scores_v2", []), "silver_v2_window_labels": row.get("silver_v2_window_labels", {}), "silver_v2_pair_labels": row.get("silver_v2_pair_labels", {})} for row in rows},
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_stub(rows: list[dict[str, Any]], out: Path) -> None:
    data: dict[str, Any] = {"windows": {}, "pair_windows": {}}
    for row in rows:
        data["windows"][row["window_id"]] = {"labels": [], "negative_labels": [], "uncertain_labels": [], "semantic_role": "unknown", "focus_actor": "unknown", "movement_quality": "questionable", "include_for_ml": False, "confidence": 0.0, "notes": ""}
        if row.get("pair_window_id"):
            data["pair_windows"][row["pair_window_id"]] = {"rider_window_id": "", "receiver_window_id": "", "pair_labels": [], "contact_labels": [], "confidence": 0.0, "notes": ""}
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_summary(rows: list[dict[str, Any]], per_scene: Counter[str], per_sample: Counter[str], per_label: Counter[str], out: Path) -> None:
    lines = ["# Machine Proposal Review Batch v2 Summary", "", f"- Items: {len(rows)}", "", "## Labels", ""]
    lines.extend(f"- `{label}`: {count}" for label, count in per_label.most_common())
    lines.extend(["", "## By Scene", ""])
    lines.extend(f"- `{scene}`: {count}" for scene, count in per_scene.most_common())
    lines.extend(["", "## By Sample", ""])
    lines.extend(f"- `{sample}`: {count}" for sample, count in per_sample.most_common(40))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

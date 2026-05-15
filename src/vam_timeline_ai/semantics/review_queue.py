"""Build diverse manual review queues."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import write_jsonl


def build_review_queue_v1(features_path: str | Path, weak_labels_path: str | Path, clusters_path: str | Path, windows_path: str | Path, out: str | Path, markdown: str | Path, max_per_sample: int = 3, max_records: int = 300) -> list[dict[str, Any]]:
    features = {r["window_id"]: r for r in _load_jsonl(features_path)}
    weak = {r["window_id"]: r for r in _load_jsonl(weak_labels_path)}
    clusters = {r["window_id"]: r.get("cluster_id") for r in _load_jsonl(clusters_path)} if Path(clusters_path).exists() else {}
    windows = {r["window_id"]: r for r in _load_jsonl(windows_path)}
    candidates = []
    for wid, frow in features.items():
        wrow = windows.get(wid, {})
        score, reasons = _review_score(frow, weak.get(wid, {}), clusters.get(wid), wrow)
        if score > 0:
            candidates.append((score, wid, reasons))
    selected = []
    per_sample = Counter()
    per_cluster = Counter()
    for _, wid, reasons in sorted(candidates, key=lambda item: item[0], reverse=True):
        frow = features[wid]
        sample_id = frow.get("sample_id")
        cluster_id = clusters.get(wid)
        if per_sample[sample_id] >= max_per_sample:
            continue
        if cluster_id is not None and per_cluster[cluster_id] >= 20:
            continue
        selected.append(_review_record(wid, frow, windows.get(wid, {}), weak.get(wid, {}), cluster_id, reasons))
        per_sample[sample_id] += 1
        if cluster_id is not None:
            per_cluster[cluster_id] += 1
        if len(selected) >= max_records:
            break
    write_jsonl(out, selected)
    _write_markdown(selected, markdown)
    return selected


def _review_score(frow: dict[str, Any], weak: dict[str, Any], cluster_id: int | None, window: dict[str, Any]) -> tuple[float, list[str]]:
    v = frow.get("feature_values", {})
    reasons = []
    score = 0.0
    for name, label in [
        ("pelvis_movement_energy", "high pelvis energy"),
        ("pause_hold_score_proxy", "pause/hold candidate"),
        ("irregular_rhythm_score_proxy", "irregular motion"),
        ("torso_motion_energy", "torso active"),
        ("left_hand_motion_energy", "hand motion"),
        ("right_hand_motion_energy", "hand motion"),
    ]:
        val = _f(v, name)
        if np.isfinite(val) and val > 0.2:
            score += float(min(val, 2.0))
            reasons.append(label)
    weak_labels = [item["label"] for item in weak.get("weak_labels", [])]
    if weak_labels:
        score += min(len(weak_labels) * 0.25, 2.0)
        reasons.append("has weak labels")
    scene = str(frow.get("source_scene_file", "")).lower()
    if any(token in scene for token in ["cow", "ride", "riding"]):
        score += 0.5
        reasons.append("filename hint for review priority only")
    else:
        score += 0.1
        reasons.append("negative/control scene candidate")
    if cluster_id is not None:
        score += 0.2
        reasons.append("cluster representative candidate")
    return score, sorted(set(reasons))


def _review_record(wid: str, frow: dict[str, Any], window: dict[str, Any], weak: dict[str, Any], cluster_id: int | None, reasons: list[str]) -> dict[str, Any]:
    top = _top_numeric(frow.get("feature_values", {}))
    return {
        "review_id": f"review_{wid}",
        "window_id": wid,
        "sample_id": frow.get("sample_id"),
        "source_scene_file": frow.get("source_scene_file"),
        "technical_atom_id": frow.get("technical_atom_id"),
        "start_seconds": window.get("start_seconds"),
        "end_seconds": window.get("end_seconds"),
        "duration_seconds": window.get("duration_seconds"),
        "cluster_id": cluster_id,
        "top_numeric_features": top,
        "weak_labels": weak.get("weak_labels", []),
        "why_selected": reasons,
        "suggested_manual_label_slots": ["semantic_role", "labels", "confidence", "needs_manual_review", "notes"],
        "notes_empty": "",
    }


def _top_numeric(values: dict[str, Any], n: int = 8) -> dict[str, float]:
    pairs = []
    for key, value in values.items():
        val = _f(values, key)
        if np.isfinite(val):
            pairs.append((key, abs(val), val))
    return {key: round(float(val), 5) for key, _, val in sorted(pairs, key=lambda item: item[1], reverse=True)[:n]}


def _write_markdown(rows: list[dict[str, Any]], path: str | Path) -> None:
    lines = ["# Manual Review Queue v1", "", f"- Review windows: {len(rows)}", "", "## Examples For manual_labels.yaml", ""]
    for row in rows[:60]:
        labels = ", ".join(item["label"] for item in row.get("weak_labels", [])[:5])
        lines.extend(
            [
                f"### `{row['window_id']}`",
                "",
                f"- Scene: `{row.get('source_scene_file')}`",
                f"- Technical atom: `{row.get('technical_atom_id')}`",
                f"- Time: {row.get('start_seconds')} - {row.get('end_seconds')}",
                f"- Weak labels: {labels}",
                f"- Why selected: {', '.join(row.get('why_selected', []))}",
                "",
                "```yaml",
                "windows:",
                f"  \"{row['window_id']}\":",
                "    labels:",
                "      - unknown_needs_manual_review",
                "    confidence: manual",
                "    needs_manual_review: true",
                "    notes: \"\"",
                "```",
                "",
            ]
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _f(values: dict[str, Any], name: str) -> float:
    try:
        return float(values.get(name, np.nan))
    except Exception:
        return np.nan


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

"""Build active-labeling review batches from coverage gaps."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.review_batch import _pair_by_window, _pair_summary, _record, _score_candidate, _write_markdown, _write_stub, _write_summary


def build_active_review_batch_v3(
    windows: str | Path,
    features: str | Path,
    weak_labels: str | Path,
    pair_windows: str | Path,
    pair_features: str | Path,
    manual_labels: str | Path,
    supervised_readiness: str | Path,
    out_dir: str | Path,
    batch_size: int = 120,
    max_per_scene: int = 15,
    max_per_sample: int = 3,
    prefer_coverage_gaps: bool = True,
) -> list[dict[str, Any]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    windows_by_id = {r.get("window_id"): r for r in load_jsonl(windows)}
    features_by_id = {r.get("window_id"): r for r in load_jsonl(features)}
    weak_by_id = {r.get("window_id"): r for r in load_jsonl(weak_labels)}
    pair_by_window = _pair_by_window(load_jsonl(pair_windows))
    pair_feature_by_id = {r.get("pair_window_id"): r for r in load_jsonl(pair_features)}
    manual = _load_yaml(Path(manual_labels)) if Path(manual_labels).exists() and "template" not in Path(manual_labels).name.lower() else {}
    labeled_windows = set((manual.get("windows", {}) or {}).keys()) if isinstance(manual.get("windows", {}), dict) else set()
    sparse = _sparse_classes(manual)

    candidates = []
    for wid, frow in features_by_id.items():
        if wid in labeled_windows:
            continue
        score, reasons = _score_candidate(frow, weak_by_id.get(wid, {}), pair_by_window.get(wid), None, True)
        labels = [item.get("label") for item in weak_by_id.get(wid, {}).get("weak_labels", [])]
        if prefer_coverage_gaps:
            if not sparse:
                score += 0.5
                reasons.append("manual label coverage is empty; seed broad coverage")
            if any(_weak_supports_sparse(label, sparse) for label in labels):
                score += 1.0
                reasons.append("weak hint may support sparse manual class")
            if pair_by_window.get(wid):
                score += 0.8
                reasons.append("pair/contact context for sparse labeling")
        candidates.append((score, wid, sorted(set(reasons))))

    selected = []
    per_scene = Counter()
    per_sample = Counter()
    per_label = Counter()
    for score, wid, reasons in sorted(candidates, key=lambda item: item[0], reverse=True):
        wrow = windows_by_id.get(wid, {})
        frow = features_by_id.get(wid, {})
        scene = str(frow.get("source_scene_file") or wrow.get("source_scene_file") or "")
        sample_id = str(frow.get("sample_id") or wrow.get("sample_id") or "")
        if per_scene[scene] >= max_per_scene or per_sample[sample_id] >= max_per_sample:
            continue
        pair = pair_by_window.get(wid)
        record = _record(wid, score, reasons, wrow, frow, weak_by_id.get(wid, {}), pair, _pair_summary(pair, pair_feature_by_id), None, out_path.name)
        selected.append(record)
        per_scene[scene] += 1
        per_sample[sample_id] += 1
        for item in record.get("weak_labels_v2", []):
            per_label[item["label"]] += 1
        if len(selected) >= batch_size:
            break
    write_jsonl(out_path / "review_batch.jsonl", selected)
    _write_markdown(selected, out_path / "review_batch.md")
    _write_stub(selected, out_path / "manual_labels.stub.yaml")
    _write_summary(selected, per_scene, per_sample, per_label, out_path / "batch_summary.md")
    _write_active_notes(selected, sparse, labeled_windows, out_path / "active_selection_notes.md")
    return selected


def _sparse_classes(manual: dict[str, Any]) -> set[str]:
    counts = Counter()
    for entry in (manual.get("windows", {}) or {}).values() if isinstance(manual.get("windows", {}), dict) else []:
        for label in entry.get("labels", []) or []:
            counts[label] += 1
    return {label for label, count in counts.items() if count < 20}


def _weak_supports_sparse(weak_label: str, sparse: set[str]) -> bool:
    if not sparse:
        return False
    mapping = {
        "vertical": "cowgirl_vertical_bounce",
        "forward_back": "cowgirl_forward_back_rock",
        "lateral": "cowgirl_lateral_sway",
        "circular": "cowgirl_circular_grind",
        "pause": "cowgirl_pause_hold",
        "fast": "cowgirl_fast_shallow",
        "slow": "cowgirl_deep_slow",
    }
    return any(target in sparse and token in weak_label for token, target in mapping.items())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_active_notes(rows: list[dict[str, Any]], sparse: set[str], labeled_windows: set[str], out: Path) -> None:
    lines = [
        "# Active Review Batch Notes",
        "",
        f"- Selected rows: {len(rows)}",
        f"- Already labeled windows avoided: {len(labeled_windows)}",
        f"- Sparse manual classes targeted: {sorted(sparse) if sparse else 'No manual classes yet; broad seed batch'}",
        "",
        "Weak labels are shown only as hints. Manual label fields remain empty.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Build balanced manual review batches."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.identity import make_review_id
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def build_review_batch_v2(
    windows: str | Path,
    features: str | Path,
    weak_labels: str | Path,
    pair_windows: str | Path,
    pair_features: str | Path,
    clusters: str | Path,
    out_dir: str | Path,
    batch_size: int = 120,
    max_per_scene: int = 15,
    max_per_sample: int = 3,
    prefer_pair_context: bool = True,
) -> list[dict[str, Any]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    windows_by_id = {r.get("window_id"): r for r in load_jsonl(windows)}
    features_by_id = {r.get("window_id"): r for r in load_jsonl(features)}
    weak_by_id = {r.get("window_id"): r for r in load_jsonl(weak_labels)}
    pair_by_window = _pair_by_window(load_jsonl(pair_windows))
    pair_feature_by_id = {r.get("pair_window_id"): r for r in load_jsonl(pair_features)}
    cluster_by_id = {r.get("window_id"): r.get("cluster_id") for r in load_jsonl(clusters)} if Path(clusters).exists() else {}

    candidates = []
    for wid, frow in features_by_id.items():
        score, reasons = _score_candidate(frow, weak_by_id.get(wid, {}), pair_by_window.get(wid), cluster_by_id.get(wid), prefer_pair_context)
        if score > 0:
            candidates.append((score, wid, reasons))
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
        pair_summary = _pair_summary(pair, pair_feature_by_id)
        record = _record(wid, score, reasons, wrow, frow, weak_by_id.get(wid, {}), pair, pair_summary, cluster_by_id.get(wid), out_path.name)
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
    return selected


def _pair_by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ["window_id_a", "window_id_b"]:
            wid = row.get(key)
            if wid and wid not in out:
                out[wid] = row
    return out


def _score_candidate(frow: dict[str, Any], weak: dict[str, Any], pair: dict[str, Any] | None, cluster_id: Any, prefer_pair_context: bool) -> tuple[float, list[str]]:
    v = frow.get("feature_values", {})
    reasons: list[str] = []
    score = 0.0
    for key, reason in [
        ("pelvis_movement_energy", "high motion-energy example"),
        ("pause_hold_score_proxy", "pause/hold candidate"),
        ("irregular_rhythm_score_proxy", "irregular/transition candidate"),
        ("torso_motion_energy", "torso activity candidate"),
        ("head_motion_energy", "head activity candidate"),
        ("left_hand_motion_energy", "hand motion candidate"),
        ("right_hand_motion_energy", "hand motion candidate"),
    ]:
        val = _f(v, key)
        if val > 0.2:
            score += min(val, 2.0)
            reasons.append(reason)
    labels = [item.get("label") for item in weak.get("weak_labels", [])]
    if labels:
        score += min(len(labels) * 0.3, 2.0)
        reasons.append("weak_v2 label coverage")
    if pair and prefer_pair_context:
        score += 1.2
        reasons.append("pair-context candidate")
    if cluster_id is not None:
        score += 0.2
        reasons.append("cluster representative candidate")
    scene = str(frow.get("source_scene_file", "")).lower()
    if any(token in scene for token in ["cow", "ride", "riding"]):
        score += 0.4
        reasons.append("filename hint for review triage only")
    else:
        score += 0.15
        reasons.append("negative/control scene candidate")
    return score, sorted(set(reasons))


def _record(wid: str, score: float, reasons: list[str], wrow: dict[str, Any], frow: dict[str, Any], weak: dict[str, Any], pair: dict[str, Any] | None, pair_summary: dict[str, Any], cluster_id: Any, batch_name: str) -> dict[str, Any]:
    return {
        "review_id": make_review_id(wid, batch_name=batch_name),
        "window_id": wid,
        "pair_window_id": pair.get("pair_window_id") if pair else None,
        "sample_id": frow.get("sample_id") or wrow.get("sample_id"),
        "source_scene_file": frow.get("source_scene_file") or wrow.get("source_scene_file"),
        "technical_atom_id": frow.get("technical_atom_id") or wrow.get("technical_atom_id"),
        "start_seconds": wrow.get("start_seconds"),
        "end_seconds": wrow.get("end_seconds"),
        "duration_seconds": wrow.get("duration_seconds"),
        "cluster_id": cluster_id,
        "top_features": _top_features(frow.get("feature_values", {})),
        "weak_labels_v2": weak.get("weak_labels", []),
        "pair_context_summary": pair_summary,
        "why_selected": reasons,
        "suggested_labels_empty": [],
        "suggested_role_empty": "unknown",
        "notes_empty": "",
        "selection_score": round(float(score), 4),
    }


def _pair_summary(pair: dict[str, Any] | None, pair_features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not pair:
        return {}
    feat = pair_features.get(pair.get("pair_window_id"), {})
    q = feat.get("feature_quality", {})
    return {
        "pair_window_id": pair.get("pair_window_id"),
        "sample_id_a": pair.get("sample_id_a"),
        "sample_id_b": pair.get("sample_id_b"),
        "technical_atom_id_a": pair.get("technical_atom_id_a"),
        "technical_atom_id_b": pair.get("technical_atom_id_b"),
        "window_id_a": pair.get("window_id_a"),
        "window_id_b": pair.get("window_id_b"),
        "frame_start_a": pair.get("frame_start_a"),
        "frame_end_a": pair.get("frame_end_a"),
        "frame_start_b": pair.get("frame_start_b"),
        "frame_end_b": pair.get("frame_end_b"),
        "other_window_id": pair.get("window_id_b"),
        "pair_confidence": pair.get("pair_confidence"),
        "active_actor_candidate_motion_only": q.get("active_actor_candidate"),
        "active_actor_confidence_motion_only": q.get("active_actor_confidence"),
        "has_hand_to_partner_features": q.get("has_hand_to_partner_features"),
    }


def _top_features(values: dict[str, Any], n: int = 8) -> dict[str, float]:
    pairs = []
    for key, value in values.items():
        val = _f(values, key)
        if val == val:
            pairs.append((key, abs(val), val))
    return {key: round(float(val), 5) for key, _, val in sorted(pairs, key=lambda item: item[1], reverse=True)[:n]}


def _write_markdown(rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Review Batch v2", "", f"- Review items: {len(rows)}", "", "Weak labels are hints only. Manual label fields start empty.", ""]
    for row in rows[:80]:
        weak = ", ".join(item["label"] for item in row.get("weak_labels_v2", [])[:5])
        lines.extend([
            f"## `{row['review_id']}`",
            "",
            f"- Window: `{row['window_id']}`",
            f"- Scene: `{row.get('source_scene_file')}`",
            f"- Technical atom: `{row.get('technical_atom_id')}`",
            f"- Time: {row.get('start_seconds')} - {row.get('end_seconds')}",
            f"- Weak hints: {weak}",
            f"- Why selected: {', '.join(row.get('why_selected', []))}",
            "",
        ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stub(rows: list[dict[str, Any]], out: Path) -> None:
    data: dict[str, Any] = {"windows": {}, "pair_windows": {}}
    for row in rows:
        data["windows"][row["window_id"]] = {
            "labels": [],
            "negative_labels": [],
            "uncertain_labels": [],
            "semantic_role": "unknown",
            "focus_actor": "unknown",
            "movement_quality": "questionable",
            "include_for_ml": False,
            "confidence": 0.0,
            "notes": "",
        }
        pid = row.get("pair_window_id")
        if pid:
            data["pair_windows"].setdefault(pid, {
                "rider_window_id": "",
                "receiver_window_id": "",
                "pair_labels": [],
                "contact_labels": [],
                "confidence": 0.0,
                "notes": "",
            })
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_summary(rows: list[dict[str, Any]], per_scene: Counter[str], per_sample: Counter[str], per_label: Counter[str], out: Path) -> None:
    lines = ["# Review Batch Summary", "", f"- Items: {len(rows)}", "", "## By Scene", ""]
    for key, count in per_scene.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## By Sample", ""])
    for key, count in per_sample.most_common(30):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Weak Hint Coverage", ""])
    for key, count in per_label.most_common():
        lines.append(f"- `{key}`: {count}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _f(values: dict[str, Any], key: str) -> float:
    try:
        return float(values.get(key, 0.0))
    except Exception:
        return 0.0

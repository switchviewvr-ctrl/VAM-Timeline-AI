"""Build review batches focused on checking machine label proposals."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.identity import make_review_id
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def build_machine_proposal_review_batch(
    run_dir: str | Path,
    proposals: str | Path,
    silver_labels: str | Path,
    out_dir: str | Path,
    batch_size: int = 120,
    max_per_scene: int = 15,
    max_per_sample: int = 3,
) -> list[dict[str, Any]]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proposal_rows = load_jsonl(proposals)
    silver_rows = {r.get("window_id"): r for r in load_jsonl(silver_labels) if r.get("window_id")}
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    features = {r.get("window_id"): r for r in load_jsonl(run / "features" / "cowgirl_window_features_v1.jsonl") if r.get("window_id")}
    weak = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "weak_labels_v2.jsonl") if r.get("window_id")}
    pair_features = {r.get("pair_window_id"): r for r in load_jsonl(run / "features" / "cowgirl_pair_features_v0.jsonl") if r.get("pair_window_id")}
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prop in proposal_rows:
        by_window[str(prop.get("window_id") or "")].append(prop)

    candidates = []
    for wid, props in by_window.items():
        if not wid:
            continue
        score, reasons = _score_props(props, bool(silver_rows.get(wid)))
        candidates.append((score, wid, reasons, props))
    selected: list[dict[str, Any]] = []
    per_scene = Counter()
    per_sample = Counter()
    per_label = Counter()
    for score, wid, reasons, props in sorted(candidates, key=lambda item: item[0], reverse=True):
        wrow = windows.get(wid, {})
        frow = features.get(wid, {})
        scene = str(frow.get("source_scene_file") or wrow.get("source_scene_file") or props[0].get("source_scene_file") or "")
        sample = str(frow.get("sample_id") or wrow.get("sample_id") or props[0].get("sample_id") or "")
        if per_scene[scene] >= max_per_scene or per_sample[sample] >= max_per_sample:
            continue
        record = _record(out.name, wid, score, reasons, props, silver_rows.get(wid), wrow, frow, weak.get(wid, {}), pair_features)
        selected.append(record)
        per_scene[scene] += 1
        per_sample[sample] += 1
        per_label.update(prop.get("label") for prop in props if prop.get("label"))
        if len(selected) >= batch_size:
            break
    write_jsonl(out / "review_batch.jsonl", selected)
    _write_markdown(selected, out / "review_batch.md")
    _write_machine_yaml(selected, out / "machine_label_review.yaml")
    _write_stub(selected, out / "manual_labels.stub.yaml")
    _write_summary(selected, per_scene, per_sample, per_label, out / "batch_summary.md")
    return selected


def _score_props(props: list[dict[str, Any]], has_silver: bool) -> tuple[float, list[str]]:
    confs = [float(p.get("confidence") or 0.0) for p in props]
    labels = {p.get("label") for p in props}
    ptypes = {p.get("proposal_type") for p in props}
    score = max(confs) * 3.0 + len(labels) * 0.2
    reasons = []
    if has_silver:
        score += 1.5
        reasons.append("high-confidence silver candidate")
    if any(0.68 <= c <= 0.78 for c in confs):
        score += 0.8
        reasons.append("borderline proposal near silver threshold")
    if "contact_candidate" in ptypes:
        score += 1.0
        reasons.append("pair/contact proposal")
    if "role_candidate" in ptypes:
        score += 0.8
        reasons.append("role candidate")
    if "negative" in ptypes:
        score += 0.7
        reasons.append("negative/control candidate")
    if len(ptypes) > 1:
        score += 0.4
        reasons.append("mixed proposal types")
    return score, sorted(set(reasons or ["machine proposal review candidate"]))


def _record(
    batch_name: str,
    wid: str,
    score: float,
    reasons: list[str],
    props: list[dict[str, Any]],
    silver: dict[str, Any] | None,
    wrow: dict[str, Any],
    frow: dict[str, Any],
    weak: dict[str, Any],
    pair_features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    props_sorted = sorted(props, key=lambda p: (-(float(p.get("confidence") or 0.0)), str(p.get("label"))))
    first = props_sorted[0]
    pair_window_id = first.get("pair_window_id")
    return {
        "review_id": make_review_id(wid, batch_name=batch_name),
        "window_id": wid,
        "pair_window_id": pair_window_id,
        "sample_id": frow.get("sample_id") or wrow.get("sample_id") or first.get("sample_id"),
        "source_scene_file": frow.get("source_scene_file") or wrow.get("source_scene_file") or first.get("source_scene_file"),
        "technical_atom_id": frow.get("technical_atom_id") or wrow.get("technical_atom_id") or first.get("technical_atom_id"),
        "start_seconds": wrow.get("start_seconds"),
        "end_seconds": wrow.get("end_seconds"),
        "duration_seconds": wrow.get("duration_seconds"),
        "top_features": _top_evidence(props_sorted, frow.get("feature_values", {})),
        "weak_labels_v2": weak.get("weak_labels", []),
        "machine_proposals": [_proposal_hint(p) for p in props_sorted[:12]],
        "silver_labels": _silver_hint(silver),
        "pair_context_summary": _pair_summary(pair_window_id, pair_features),
        "why_selected": reasons,
        "selection_score": round(float(score), 4),
        "machine_label_warning": "Machine proposals are hints only. They are not human ground truth and must not be copied blindly.",
        "suggested_labels_empty": [],
        "suggested_role_empty": "unknown",
        "notes_empty": "",
    }


def _proposal_hint(prop: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": prop.get("label"),
        "proposal_type": prop.get("proposal_type"),
        "label_group": prop.get("label_group"),
        "confidence": prop.get("confidence"),
        "rule_id": prop.get("rule_id"),
        "evidence_features": prop.get("evidence_features", []),
        "is_silver_candidate": prop.get("is_silver_candidate", False),
        "is_human_ground_truth": False,
    }


def _silver_hint(silver: dict[str, Any] | None) -> dict[str, Any]:
    if not silver:
        return {}
    return {
        "positive_labels": silver.get("positive_labels", []),
        "negative_labels": silver.get("negative_labels", []),
        "role_candidates": silver.get("role_candidates", []),
        "contact_candidates": silver.get("contact_candidates", []),
        "confidence_by_label": silver.get("confidence_by_label", {}),
        "is_human_ground_truth": False,
    }


def _pair_summary(pair_window_id: str | None, pair_features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not pair_window_id:
        return {}
    row = pair_features.get(pair_window_id, {})
    q = row.get("feature_quality", {})
    return {
        "pair_window_id": pair_window_id,
        "sample_id_a": row.get("sample_id_a"),
        "sample_id_b": row.get("sample_id_b"),
        "window_id_a": row.get("window_id_a"),
        "window_id_b": row.get("window_id_b"),
        "active_actor_candidate_motion_only": q.get("active_actor_candidate"),
        "active_actor_confidence_motion_only": q.get("active_actor_confidence"),
        "has_hand_to_partner_features": q.get("has_hand_to_partner_features"),
    }


def _top_evidence(props: list[dict[str, Any]], values: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    keys: list[str] = []
    for prop in props:
        keys.extend(prop.get("evidence_features", []) or [])
    out: dict[str, Any] = {}
    for key in keys:
        if key in values and key not in out:
            out[key] = values[key]
        if len(out) >= limit:
            break
    return out


def _write_markdown(rows: list[dict[str, Any]], out: Path) -> None:
    lines = [
        "# Machine Proposal Review Batch",
        "",
        "This batch is for checking Codex-generated machine proposals. Manual label fields remain empty.",
        "",
        f"- Review items: {len(rows)}",
        "",
    ]
    for row in rows[:120]:
        props = ", ".join(f"{p['label']}({p['confidence']})" for p in row.get("machine_proposals", [])[:6])
        lines.extend(
            [
                f"## `{row['review_id']}`",
                "",
                f"- Window: `{row['window_id']}`",
                f"- Scene: `{row.get('source_scene_file')}`",
                f"- Technical atom: `{row.get('technical_atom_id')}`",
                f"- Time: {row.get('start_seconds')} - {row.get('end_seconds')}",
                f"- Machine proposals: {props}",
                f"- Why selected: {', '.join(row.get('why_selected', []))}",
                "",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_machine_yaml(rows: list[dict[str, Any]], out: Path) -> None:
    data = {
        "metadata": {
            "label_source": "machine_label_review_hints_v1",
            "is_human_ground_truth": False,
            "warning": "These suggestions are machine-generated hints for human review, not manual labels.",
        },
        "windows": {
            row["window_id"]: {
                "machine_proposals": row.get("machine_proposals", []),
                "silver_labels": row.get("silver_labels", {}),
                "notes": "Use as review hints only. Do not paste weak_/machine hints directly as manual truth.",
            }
            for row in rows
        },
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


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
        if row.get("pair_window_id"):
            data["pair_windows"][row["pair_window_id"]] = {
                "rider_window_id": "",
                "receiver_window_id": "",
                "pair_labels": [],
                "contact_labels": [],
                "confidence": 0.0,
                "notes": "",
            }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_summary(rows: list[dict[str, Any]], per_scene: Counter[str], per_sample: Counter[str], per_label: Counter[str], out: Path) -> None:
    lines = ["# Machine Proposal Batch Summary", "", f"- Items: {len(rows)}", "", "## Proposal Labels", ""]
    lines.extend(f"- `{label}`: {count}" for label, count in per_label.most_common())
    lines.extend(["", "## By Scene", ""])
    lines.extend(f"- `{scene}`: {count}" for scene, count in per_scene.most_common())
    lines.extend(["", "## By Sample", ""])
    lines.extend(f"- `{sample}`: {count}" for sample, count in per_sample.most_common(40))
    lines.extend(["", "Manual stubs are intentionally empty. Machine suggestions are review hints only."])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

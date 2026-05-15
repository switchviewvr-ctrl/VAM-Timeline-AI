"""Reality-audit exports for checking data interpretation before ML work."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import html
import math
import random

import numpy as np
import yaml

from vam_timeline_ai.io.identity import stable_hash
from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl, safe_id_for_path, write_jsonl


MOVEMENT_LABELS = [
    "cowgirl_vertical_bounce",
    "cowgirl_forward_back_rock",
    "cowgirl_lateral_sway",
    "cowgirl_circular_grind",
    "cowgirl_fast_shallow",
    "cowgirl_deep_slow",
    "cowgirl_pause_hold",
    "cowgirl_adjustment_transition",
]

CONTACT_LABEL_TOKENS = ["hand_supported_on_partner", "partner_chest", "partner_hips", "rider_active", "partner_context_static"]
CORE_PARTS = ["pelvis", "hip", "root", "abdomen", "chest", "head", "left_hand", "right_hand", "left_foot", "right_foot"]
CORE_FEATURES = [
    "pelvis_vertical_amplitude",
    "pelvis_forward_back_amplitude",
    "pelvis_lateral_amplitude",
    "pelvis_movement_energy",
    "pelvis_mean_speed",
    "pelvis_max_speed",
    "pelvis_pause_ratio",
    "pelvis_circularity_score_proxy",
    "pelvis_grind_score_proxy",
    "slow_motion_score_proxy",
    "fast_motion_score_proxy",
    "adjustment_transition_score_proxy",
    "irregular_rhythm_score_proxy",
    "left_hand_motion_energy",
    "right_hand_motion_energy",
    "torso_motion_energy",
    "head_motion_energy",
]
PAIR_FEATURES = [
    "activity_ratio_a_over_b",
    "activity_ratio_b_over_a",
    "a_pelvis_motion_energy",
    "b_pelvis_motion_energy",
    "pelvis_to_pelvis_distance_mean",
    "pelvis_vertical_offset_a_minus_b_mean",
    "a_hands_near_b_chest_proxy",
    "a_hands_near_b_pelvis_proxy",
    "b_hands_near_a_chest_proxy",
    "b_hands_near_a_pelvis_proxy",
    "receiver_static_context_proxy_a_active",
    "receiver_static_context_proxy_b_active",
]
QUESTIONS_TO_ANSWER = [
    "Is real motion visible?",
    "Is the correct actor being shown?",
    "Does pelvis/hip/root mapping look plausible?",
    "Does vertical movement match the preview?",
    "Does forward/back movement match the preview?",
    "Does lateral movement match the preview?",
    "Does the window timing match the relevant motion?",
    "If pair item: is the pair context plausible?",
    "If pair item: is the active actor candidate plausible?",
    "If hand/contact item: does the hand/contact proxy look plausible?",
    "Are machine labels plausible?",
    "Should this item be trusted for ML?",
]
ALLOWED_AUDIT_VALUES = ["true", "false", "unknown", "not_applicable"]
FAILURE_REASONS = [
    "wrong_controller_mapping",
    "wrong_axis_assumption",
    "wrong_pairing",
    "wrong_timing",
    "static_or_empty_motion",
    "feature_mismatch",
    "machine_label_wrong",
    "preview_insufficient",
    "not_cowgirl",
    "wrong_actor",
    "other",
]


def export_reality_audit_100(run_dir: str | Path, out_dir: str | Path, count: int = 100, render_previews: bool = True) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = _load_run_data(run)
    selected = _select_audit_items(data, count=count)
    rows = [_make_audit_item(idx, selected_item, data) for idx, selected_item in enumerate(selected, start=1)]
    write_jsonl(out / "reality_audit_batch.jsonl", rows)
    _write_batch_markdown(rows, out / "reality_audit_batch.md")
    _write_annotation_schema(out / "reality_audit_annotation.schema.yaml")
    _write_annotation_stub(rows, out / "reality_audit_annotation.stub.yaml")
    _write_summary(rows, out / "reality_audit_summary.md")
    preview_summary = _render_reality_previews(rows, data, out / "previews") if render_previews else _write_preview_without_render(rows, out / "previews", "rendering disabled")
    result = summarize_reality_audit(out / "reality_audit_annotation.edited.yaml", out / "reality_audit_batch.jsonl", out / "reality_audit_result.md")
    return {
        "status": "ok",
        "audit_items": len(rows),
        "category_distribution": dict(Counter(row["category"] for row in rows)),
        "scene_distribution": dict(Counter(row.get("source_scene_file") for row in rows)),
        "pair_window_examples": sum(1 for row in rows if row.get("pair_window_id")),
        "preview_items": preview_summary.get("items", 0),
        "preview_images": preview_summary.get("image_count", 0),
        "annotation_status": result.get("status"),
        "manual_labels_modified": False,
    }


def summarize_reality_audit(annotations: str | Path, audit_batch: str | Path, out: str | Path) -> dict[str, Any]:
    annotation_path = Path(annotations)
    out_path = Path(out)
    batch_rows = load_jsonl(audit_batch)
    if not annotation_path.exists():
        lines = [
            "# Reality Audit Result",
            "",
            "Review is not completed yet.",
            "",
            f"- Expected edited annotation file: `{annotation_path}`",
            f"- Audit items: {len(batch_rows)}",
            "",
            "No ML, labels, or training decisions should be made from this audit until the edited file exists.",
        ]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"status": "not_completed", "audit_items": len(batch_rows)}

    data = yaml.safe_load(annotation_path.read_text(encoding="utf-8")) or {}
    items = data.get("audit_items", {}) or {}
    counts = {
        "data_read_correctly": Counter(),
        "real_motion_visible": Counter(),
        "correct_actor_visible": Counter(),
        "window_timing_ok": Counter(),
        "pair_context_ok": Counter(),
        "active_actor_candidate_ok": Counter(),
        "hand_contact_proxy_ok": Counter(),
        "trust_for_ml": Counter(),
    }
    nested_counts = {
        "controller_mapping_ok": defaultdict(Counter),
        "axis_interpretation_ok": defaultdict(Counter),
    }
    failure_counts: Counter[str] = Counter()
    true_labels: Counter[str] = Counter()
    false_labels: Counter[str] = Counter()
    uncertain_labels: Counter[str] = Counter()
    investigate: list[str] = []
    for audit_id, item in items.items():
        if not isinstance(item, dict):
            continue
        for field in counts:
            counts[field][str(item.get(field, "unknown"))] += 1
        for group in nested_counts:
            for key, value in (item.get(group, {}) or {}).items():
                nested_counts[group][str(key)][str(value)] += 1
        for reason in item.get("failure_reason", []) or []:
            failure_counts[str(reason)] += 1
        ml = item.get("machine_labels_plausible", {}) or {}
        true_labels.update(str(v) for v in ml.get("true_labels", []) or [])
        false_labels.update(str(v) for v in ml.get("false_labels", []) or [])
        uncertain_labels.update(str(v) for v in ml.get("uncertain_labels", []) or [])
        if item.get("trust_for_ml") == "false" or item.get("data_read_correctly") == "false":
            investigate.append(str(audit_id))

    verdict = {
        "phase1_data_extraction_trusted": _verdict(counts["data_read_correctly"]),
        "feature_extraction_trusted": _verdict(counts["real_motion_visible"], counts["window_timing_ok"]),
        "pair_context_trusted": _verdict(counts["pair_context_ok"]),
        "machine_labels_trusted_for_proxy_ml": _verdict(counts["trust_for_ml"]),
    }
    _write_result_report(
        out_path,
        len(batch_rows),
        len(items),
        counts,
        nested_counts,
        failure_counts,
        true_labels,
        false_labels,
        uncertain_labels,
        investigate,
        verdict,
    )
    return {
        "status": "completed",
        "audit_items": len(batch_rows),
        "annotated_items": len(items),
        "failure_reasons": dict(failure_counts),
        "verdict": verdict,
    }


def _load_run_data(run: Path) -> dict[str, Any]:
    windows = load_jsonl(run / "semantic" / "movement_windows.jsonl")
    features = load_jsonl(run / "features" / "cowgirl_window_features_v1.jsonl")
    samples = load_jsonl(run / "baked" / "motion_sample_index.jsonl")
    weak = load_jsonl(run / "semantic" / "weak_labels_v2.jsonl")
    pair_windows = load_jsonl(run / "semantic" / "pair_windows_v1.jsonl")
    pair_features = load_jsonl(run / "features" / "cowgirl_pair_features_v0.jsonl")
    window_scores = load_jsonl(run / "labels" / "machine_proposals" / "machine_window_label_scores_v2.jsonl")
    pair_scores = load_jsonl(run / "labels" / "machine_proposals" / "machine_pair_label_scores_v2.jsonl")
    silver_windows = load_jsonl(run / "labels" / "machine_proposals" / "silver_window_labels_v2.jsonl")
    silver_pairs = load_jsonl(run / "labels" / "machine_proposals" / "silver_pair_labels_v2.jsonl")
    baked_audit = load_jsonl(run / "audits" / "baked_sample_audit.jsonl")
    controller_map_path = run / "semantic" / "controller_bodypart_map.json"
    controller_map = load_json(controller_map_path) if controller_map_path.exists() else {"controller_mappings": {}}

    by_window_pair = _pair_by_window(pair_windows)
    pair_feature_by_id = {r.get("pair_window_id"): r for r in pair_features if r.get("pair_window_id")}
    return {
        "run_dir": run,
        "windows": {r.get("window_id"): r for r in windows if r.get("window_id")},
        "features": {r.get("window_id"): r for r in features if r.get("window_id")},
        "samples": {r.get("sample_id"): r for r in samples if r.get("sample_id")},
        "weak": {r.get("window_id"): r for r in weak if r.get("window_id")},
        "pair_windows": {r.get("pair_window_id"): r for r in pair_windows if r.get("pair_window_id")},
        "pair_by_window": by_window_pair,
        "pair_features": pair_feature_by_id,
        "window_scores": _group_by(window_scores, "window_id"),
        "pair_scores": _group_pair_scores(pair_scores),
        "silver_windows": {r.get("window_id"): r for r in silver_windows if r.get("window_id")},
        "silver_pairs": {r.get("pair_window_id"): r for r in silver_pairs if r.get("pair_window_id")},
        "baked_audit": {r.get("sample_id"): r for r in baked_audit if r.get("sample_id")},
        "controller_map": controller_map.get("controller_mappings", {}),
    }


def _select_audit_items(data: dict[str, Any], count: int) -> list[dict[str, Any]]:
    targets = _targets(count)
    all_candidates: dict[str, list[dict[str, Any]]] = {
        "random_baseline": _random_candidates(data),
        "high_confidence_movement": _movement_candidates(data),
        "pair_contact": _pair_candidates(data),
        "suspicious_problem": _suspicious_candidates(data),
        "negative_control": _negative_candidates(data),
    }
    for rows in all_candidates.values():
        _enrich_candidates(rows, data)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    per_scene: Counter[str] = Counter()
    per_sample: Counter[str] = Counter()
    ranges_by_sample: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for category, target in targets.items():
        _take_candidates(all_candidates[category], target, selected, seen, per_scene, per_sample, ranges_by_sample, strict_overlap=True, category_quota=True)
    if len(selected) < count:
        leftovers = [c for rows in all_candidates.values() for c in rows]
        _take_candidates(leftovers, count - len(selected), selected, seen, per_scene, per_sample, ranges_by_sample, strict_overlap=False, category_quota=False)
    if len(selected) < count:
        fallback = _fallback_candidates(data)
        _enrich_candidates(fallback, data)
        _take_candidates(fallback, count - len(selected), selected, seen, per_scene, per_sample, ranges_by_sample, strict_overlap=False, category_quota=False)
    if len(selected) != count:
        raise ValueError(f"Could not select exactly {count} reality-audit items; selected {len(selected)}")
    return selected


def _targets(count: int) -> dict[str, int]:
    if count == 100:
        return {
            "random_baseline": 15,
            "high_confidence_movement": 30,
            "pair_contact": 20,
            "suspicious_problem": 20,
            "negative_control": 15,
        }
    ratios = {
        "random_baseline": 0.15,
        "high_confidence_movement": 0.30,
        "pair_contact": 0.20,
        "suspicious_problem": 0.20,
        "negative_control": 0.15,
    }
    out = {key: int(math.floor(count * value)) for key, value in ratios.items()}
    for key in ratios:
        if sum(out.values()) >= count:
            break
        out[key] += 1
    return out


def _random_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(20260515)
    rows = []
    for wid, row in data["windows"].items():
        rows.append(_candidate("random_baseline", wid, None, rng.random(), ["deterministic random baseline example"]))
    rng.shuffle(rows)
    return rows


def _movement_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    by_label: dict[str, list[tuple[float, str, dict[str, Any]]]] = defaultdict(list)
    for wid, rows in data["window_scores"].items():
        for row in rows:
            label = str(row.get("label") or "")
            if label in MOVEMENT_LABELS:
                by_label[label].append((float(row.get("final_score") or row.get("max_confidence") or 0.0), wid, row))
    for label in MOVEMENT_LABELS:
        for score, wid, row in sorted(by_label.get(label, []), key=lambda item: item[0], reverse=True)[:80]:
            candidates.append(_candidate("high_confidence_movement", wid, None, score, [f"high-confidence machine movement proposal: {label}"], [label], [row]))
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _pair_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for pid, rows in data["pair_scores"].items():
        pair = data["pair_windows"].get(pid, {})
        labels = {str(row.get("label") or "") for row in rows}
        if not _has_contact_or_role_label(labels):
            continue
        score = max(float(row.get("final_score") or 0.0) for row in rows)
        wid = pair.get("window_id_a") or (rows[0].get("window_ids") or [None])[0]
        candidates.append(_candidate("pair_contact", wid, pid, score, ["pair/contact or active-passive machine proposal"], sorted(labels), rows))
    for pid, pair_feature in data["pair_features"].items():
        q = pair_feature.get("feature_quality", {}) or {}
        if q.get("has_hand_to_partner_features") or q.get("active_actor_confidence", 0) not in {None, 0}:
            wid = pair_feature.get("window_id_a")
            score = float(q.get("active_actor_confidence") or 0.2) + (0.4 if q.get("has_hand_to_partner_features") else 0.0)
            candidates.append(_candidate("pair_contact", wid, pid, score, ["pair/context feature proxy example"], [], []))
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _suspicious_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    pair_context_count = Counter()
    for pid, rows in data["pair_scores"].items():
        for row in rows:
            for wid in row.get("window_ids", []) or []:
                pair_context_count[str(wid)] += 1
    for wid, rows in data["window_scores"].items():
        labels = {str(r.get("label") or "") for r in rows}
        reasons = []
        if any(r.get("recommended_status") == "reject_conflict" or r.get("conflict_flags") for r in rows):
            reasons.append("contradictory or conflicted machine label score")
        if len(rows) >= 10:
            reasons.append("window has unusually many aggregated machine scores")
        if pair_context_count[wid] >= 20:
            reasons.append("window appears in many pair contexts")
        if "cowgirl_fast_shallow" in labels and "cowgirl_deep_slow" in labels:
            reasons.append("fast and slow labels both present")
        if reasons:
            candidates.append(_candidate("suspicious_problem", wid, None, float(len(rows) + pair_context_count[wid]), reasons, sorted(labels), rows))
    for wid, frow in data["features"].items():
        q = frow.get("feature_quality", {}) or {}
        missing = frow.get("missing_controller_groups", []) or []
        warnings = frow.get("warnings", []) or []
        reasons = []
        if missing:
            reasons.append("missing controller groups")
        if q.get("root_mapping_confidence") not in {"high", None}:
            reasons.append("low or uncertain root/pelvis mapping confidence")
        if warnings:
            reasons.append("feature extraction warning")
        if reasons:
            candidates.append(_candidate("suspicious_problem", wid, None, 5 + len(missing) + len(warnings), reasons))
    for sample_id, audit in data["baked_audit"].items():
        if audit.get("suspiciously_static") or audit.get("suspiciously_huge_motion"):
            for wid, row in data["windows"].items():
                if row.get("sample_id") == sample_id:
                    reason = "suspicious huge-motion sample" if audit.get("suspiciously_huge_motion") else "suspicious static sample"
                    candidates.append(_candidate("suspicious_problem", wid, None, 10.0, [reason]))
                    break
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _negative_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for wid, frow in data["features"].items():
        v = frow.get("feature_values", {}) or {}
        mean_speed = _num(v.get("pelvis_mean_speed"), 999.0)
        energy = _num(v.get("pelvis_movement_energy"), 999.0)
        pause = _num(v.get("pause_hold_score_proxy"), 0.0)
        scene = str(frow.get("source_scene_file") or "").lower()
        score = max(0.0, 1.0 - min(mean_speed * 10.0, 1.0)) + max(0.0, 1.0 - min(energy * 20.0, 1.0)) + pause
        reasons = ["low pelvis/root motion control example"]
        if not any(token in scene for token in ["cow", "ride", "riding"]):
            score += 0.25
            reasons.append("non-riding filename hint for review triage only")
        sample_audit = data["baked_audit"].get(frow.get("sample_id"), {})
        if sample_audit.get("suspiciously_static"):
            score += 1.0
            reasons.append("static/passive context sample candidate")
        candidates.append(_candidate("negative_control", wid, None, score, reasons))
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _fallback_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [_candidate("random_baseline", wid, None, 0.0, ["fallback fill example"]) for wid in sorted(data["windows"])]


def _take_candidates(
    candidates: list[dict[str, Any]],
    needed: int,
    selected: list[dict[str, Any]],
    seen: set[tuple[str, str | None]],
    per_scene: Counter[str],
    per_sample: Counter[str],
    ranges_by_sample: dict[str, list[tuple[float, float]]],
    strict_overlap: bool,
    category_quota: bool,
) -> None:
    start_len = len(selected)
    for item in candidates:
        if category_quota and len([r for r in selected if r["category"] == item["category"]]) >= needed and needed > 0:
            break
        if not category_quota and len(selected) - start_len >= needed:
            break
        key = (str(item.get("window_id")), item.get("pair_window_id"))
        if key in seen:
            continue
        scene = str(item.get("source_scene_file") or "")
        sample = str(item.get("sample_id") or "")
        if per_scene[scene] >= 10 or per_sample[sample] >= 3:
            continue
        if strict_overlap and _overlaps_existing(sample, float(item.get("start_seconds") or 0.0), float(item.get("end_seconds") or 0.0), ranges_by_sample):
            continue
        selected.append(item)
        seen.add(key)
        per_scene[scene] += 1
        per_sample[sample] += 1
        ranges_by_sample[sample].append((float(item.get("start_seconds") or 0.0), float(item.get("end_seconds") or 0.0)))


def _candidate(category: str, wid: Any, pid: Any, score: float, reasons: list[str], labels: list[str] | None = None, score_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "category": category,
        "window_id": str(wid) if wid is not None else "",
        "pair_window_id": str(pid) if pid else None,
        "score": float(score),
        "why_selected": sorted(set(reasons)),
        "candidate_labels": labels or [],
        "score_rows": score_rows or [],
    }


def _enrich_candidates(candidates: list[dict[str, Any]], data: dict[str, Any]) -> None:
    for item in candidates:
        wid = item.get("window_id")
        wrow = data["windows"].get(wid, {})
        frow = data["features"].get(wid, {})
        item["sample_id"] = wrow.get("sample_id") or frow.get("sample_id")
        item["source_scene_file"] = wrow.get("source_scene_file") or frow.get("source_scene_file")
        item["start_seconds"] = wrow.get("start_seconds")
        item["end_seconds"] = wrow.get("end_seconds")


def _make_audit_item(index: int, selected: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    wid = selected["window_id"]
    pid = selected.get("pair_window_id")
    wrow = data["windows"].get(wid, {})
    frow = data["features"].get(wid, {})
    sample = data["samples"].get(wrow.get("sample_id") or frow.get("sample_id"), {})
    pair = data["pair_windows"].get(pid, {}) if pid else data["pair_by_window"].get(wid, {})
    pair_feature = data["pair_features"].get(pair.get("pair_window_id"), {}) if pair else {}
    controller_names = list(sample.get("controller_names", []))
    mapped_body_parts = _mapped_body_parts(controller_names, data["controller_map"])
    window_scores = data["window_scores"].get(wid, [])
    pair_scores = data["pair_scores"].get(pair.get("pair_window_id"), []) if pair else []
    silver_window = data["silver_windows"].get(wid, {})
    silver_pair = data["silver_pairs"].get(pair.get("pair_window_id"), {}) if pair else {}
    category = selected["category"]
    audit_id = f"reality_{index:03d}_{category}_{stable_hash([wid, pid or '', category], 8)}"
    row = {
        "audit_id": audit_id,
        "category": category,
        "window_id": wid,
        "pair_window_id": pair.get("pair_window_id") if pair else None,
        "sample_id": wrow.get("sample_id") or frow.get("sample_id"),
        "source_id": wrow.get("source_id") or frow.get("source_id"),
        "source_scene_file": wrow.get("source_scene_file") or frow.get("source_scene_file"),
        "technical_atom_id": wrow.get("technical_atom_id") or frow.get("technical_atom_id"),
        "start_seconds": wrow.get("start_seconds"),
        "end_seconds": wrow.get("end_seconds"),
        "duration_seconds": wrow.get("duration_seconds"),
        "frame_start": wrow.get("frame_start"),
        "frame_end": wrow.get("frame_end"),
        "controller_names": controller_names,
        "mapped_body_parts": mapped_body_parts,
        "top_feature_values": _top_features(frow.get("feature_values", {})),
        "feature_quality": frow.get("feature_quality", {}),
        "controllers_used": frow.get("controllers_used", {}),
        "missing_controller_groups": frow.get("missing_controller_groups", []),
        "weak_v2_labels": _weak_hints(data["weak"].get(wid, {})),
        "machine_proposals": _score_hints(window_scores + selected.get("score_rows", []), limit=12),
        "silver_labels": _silver_hint(silver_window),
        "pair_feature_summary": _pair_summary(pair, pair_feature, pair_scores, silver_pair),
        "why_selected": selected["why_selected"],
        "questions_to_answer": QUESTIONS_TO_ANSWER,
        "machine_label_warning": "Weak, machine, and silver labels are audit hints only. They are not human truth.",
        "is_human_ground_truth": False,
    }
    if pid and not row["pair_feature_summary"]:
        row["why_selected"].append("pair window exists but pair feature summary is missing")
    return row


def _render_reality_previews(rows: list[dict[str, Any]], data: dict[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        matplotlib_available = True
    except Exception as exc:  # noqa: BLE001
        plt = None
        matplotlib_available = False
        warnings.append(f"matplotlib unavailable: {exc}")

    manifest = []
    image_count = 0
    for row in rows:
        item_dir = out / safe_id_for_path(row["audit_id"])
        item_dir.mkdir(parents=True, exist_ok=True)
        item_warnings: list[str] = []
        images: list[str] = []
        if matplotlib_available and plt is not None:
            try:
                images.extend(_render_single_item(row, data, item_dir, plt))
                images.extend(_render_pair_item(row, data, item_dir, plt))
            except Exception as exc:  # noqa: BLE001
                item_warnings.append(f"render failed: {exc}")
        else:
            item_warnings.extend(warnings)
        dump_json(item_dir / "metadata.json", {"audit_item": row, "warnings": item_warnings})
        image_count += len(images)
        manifest.append({
            "audit_id": row["audit_id"],
            "category": row["category"],
            "window_id": row["window_id"],
            "pair_window_id": row.get("pair_window_id"),
            "preview_dir": str(item_dir),
            "image_files": images,
            "warnings": item_warnings,
        })
    write_jsonl(out / "preview_manifest.jsonl", manifest)
    _write_preview_index(rows, manifest, out)
    _write_preview_report(rows, manifest, warnings, out / "preview_report.md")
    return {"items": len(rows), "image_count": image_count, "matplotlib_available": matplotlib_available, "warnings": warnings}


def _write_preview_without_render(rows: list[dict[str, Any]], out: Path, reason: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest = [{"audit_id": row["audit_id"], "category": row["category"], "window_id": row["window_id"], "pair_window_id": row.get("pair_window_id"), "preview_dir": "", "image_files": [], "warnings": [reason]} for row in rows]
    write_jsonl(out / "preview_manifest.jsonl", manifest)
    _write_preview_index(rows, manifest, out)
    _write_preview_report(rows, manifest, [reason], out / "preview_report.md")
    return {"items": len(rows), "image_count": 0, "matplotlib_available": False, "warnings": [reason]}


def _render_single_item(row: dict[str, Any], data: dict[str, Any], item_dir: Path, plt: Any) -> list[str]:
    sample = data["samples"].get(row.get("sample_id"))
    if not sample:
        return []
    loaded = _load_sample_window(sample, row.get("frame_start"), row.get("frame_end"), data["run_dir"])
    if loaded is None:
        return []
    pos, times, names = loaded["pos"], loaded["times"], loaded["names"]
    indices = _core_indices(names, data["controller_map"])
    root_idx = _root_index(names, data["controller_map"])
    images = []
    _plot_trajectory(pos, names, indices, 0, 2, "X lateral (axis under audit)", "Z forward/back (axis under audit)", item_dir / "trajectory_top.png", plt)
    images.append("trajectory_top.png")
    _plot_trajectory(pos, names, indices, 2, 1, "Z forward/back (axis under audit)", "Y vertical (axis under audit)", item_dir / "trajectory_side.png", plt)
    images.append("trajectory_side.png")
    _plot_trajectory(pos, names, indices, 0, 1, "X lateral (axis under audit)", "Y vertical (axis under audit)", item_dir / "trajectory_front.png", plt)
    images.append("trajectory_front.png")
    if root_idx is not None:
        _plot_pelvis_xyz(pos, times, names[root_idx], root_idx, item_dir / "pelvis_xyz_timeseries.png", plt)
        images.append("pelvis_xyz_timeseries.png")
    else:
        _plot_placeholder("No pelvis/hip/root controller mapped for XYZ timeseries", item_dir / "pelvis_xyz_timeseries.png", plt)
        images.append("pelvis_xyz_timeseries.png")
    _plot_speed(pos, times, names, [root_idx] if root_idx is not None else indices[:1], item_dir / "pelvis_speed.png", plt, "Pelvis/root speed proxy")
    images.append("pelvis_speed.png")
    _plot_speed(pos, times, names, indices, item_dir / "key_controller_motion.png", plt, "Key controller speed proxies")
    images.append("key_controller_motion.png")
    return images


def _render_pair_item(row: dict[str, Any], data: dict[str, Any], item_dir: Path, plt: Any) -> list[str]:
    pair_id = row.get("pair_window_id")
    if not pair_id:
        return []
    pair = data["pair_windows"].get(pair_id)
    if not pair:
        return []
    sample_a = data["samples"].get(pair.get("sample_id_a"))
    sample_b = data["samples"].get(pair.get("sample_id_b"))
    if not sample_a or not sample_b:
        return []
    a = _load_sample_window(sample_a, pair.get("frame_start_a"), pair.get("frame_end_a"), data["run_dir"])
    b = _load_sample_window(sample_b, pair.get("frame_start_b"), pair.get("frame_end_b"), data["run_dir"])
    if a is None or b is None:
        return []
    ia = _root_index(a["names"], data["controller_map"])
    ib = _root_index(b["names"], data["controller_map"])
    if ia is None or ib is None:
        return []
    n = min(len(a["pos"]), len(b["pos"]))
    if n < 2:
        return []
    images = []
    _plot_pair_trajectory(a, b, ia, ib, 0, 2, "X lateral (axis under audit)", "Z forward/back (axis under audit)", item_dir / "pair_trajectory_top.png", plt)
    images.append("pair_trajectory_top.png")
    _plot_pair_trajectory(a, b, ia, ib, 2, 1, "Z forward/back (axis under audit)", "Y vertical (axis under audit)", item_dir / "pair_trajectory_side.png", plt)
    images.append("pair_trajectory_side.png")
    _plot_pair_root_distance(a, b, ia, ib, item_dir / "pair_pelvis_distance.png", plt)
    images.append("pair_pelvis_distance.png")
    _plot_pair_hand_distances(a, b, data["controller_map"], item_dir / "pair_hand_distances.png", plt)
    images.append("pair_hand_distances.png")
    _plot_pair_activity(a, b, data["controller_map"], item_dir / "pair_activity_contrast.png", plt)
    images.append("pair_activity_contrast.png")
    return images


def _load_sample_window(sample: dict[str, Any], start_frame: Any, end_frame: Any, run_dir: Path) -> dict[str, Any] | None:
    path = Path(str(sample.get("baked_npz_path") or ""))
    if not path.is_absolute():
        project_root = run_dir.parents[2] if len(run_dir.parents) > 2 else Path.cwd()
        path = project_root / path if str(path).startswith("data") else run_dir / path
    if not path.exists():
        path = Path(str(sample.get("baked_npz_path") or ""))
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["positions"], dtype=np.float32)
        times = np.asarray(data["times"], dtype=np.float32)
        names = [str(x) for x in data["controller_names"].tolist()]
    start = max(0, min(int(start_frame or 0), len(times) - 1))
    end = len(times) if end_frame is None else int(end_frame)
    end = max(start + 1, min(end, len(times)))
    return {"pos": pos[start:end], "times": times[start:end] - times[start], "names": names}


def _plot_trajectory(pos: np.ndarray, names: list[str], indices: list[int], ax0: int, ax1: int, xlabel: str, ylabel: str, path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx in indices:
        ax.plot(pos[:, idx, ax0], pos[:, idx, ax1], label=names[idx])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Controller trajectory preview")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pelvis_xyz(pos: np.ndarray, times: np.ndarray, name: str, idx: int, path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for axis, label in enumerate(["X lateral?", "Y vertical?", "Z forward/back?"]):
        ax.plot(times, pos[:, idx, axis], label=f"{name} {label}")
    ax.set_xlabel("Window time seconds")
    ax.set_ylabel("Position value; axis interpretation under audit")
    ax.set_title("Pelvis/hip/root XYZ over time")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_speed(pos: np.ndarray, times: np.ndarray, names: list[str], indices: list[int], path: Path, plt: Any, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx in [i for i in indices if i is not None]:
        if len(pos) < 2:
            continue
        dt = np.diff(times.astype(np.float64))
        dt = np.where(dt <= 0, 1.0 / 60.0, dt)
        speed = np.linalg.norm(np.diff(pos[:, idx, :].astype(np.float64), axis=0) / dt[:, None], axis=1)
        ax.plot(times[1:], speed, label=names[idx])
    ax.set_xlabel("Window time seconds")
    ax.set_ylabel("Speed proxy")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pair_trajectory(a: dict[str, Any], b: dict[str, Any], ia: int, ib: int, ax0: int, ax1: int, xlabel: str, ylabel: str, path: Path, plt: Any) -> None:
    n = min(len(a["pos"]), len(b["pos"]))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(a["pos"][:n, ia, ax0], a["pos"][:n, ia, ax1], label=f"A {a['names'][ia]}")
    ax.plot(b["pos"][:n, ib, ax0], b["pos"][:n, ib, ax1], label=f"B {b['names'][ib]}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Pair root trajectory preview")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pair_root_distance(a: dict[str, Any], b: dict[str, Any], ia: int, ib: int, path: Path, plt: Any) -> None:
    n = min(len(a["pos"]), len(b["pos"]))
    t = a["times"][:n]
    dist = np.linalg.norm(a["pos"][:n, ia, :] - b["pos"][:n, ib, :], axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, dist, label="root/pelvis distance proxy")
    ax.set_xlabel("Window time seconds")
    ax.set_ylabel("Distance proxy")
    ax.set_title("Pair pelvis/root distance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pair_hand_distances(a: dict[str, Any], b: dict[str, Any], mappings: dict[str, Any], path: Path, plt: Any) -> None:
    n = min(len(a["pos"]), len(b["pos"]))
    hand_indices_a = _indices_for_parts(a["names"], mappings, {"left_hand", "right_hand"})
    hand_indices_b = _indices_for_parts(b["names"], mappings, {"left_hand", "right_hand"})
    root_a = _root_index(a["names"], mappings)
    root_b = _root_index(b["names"], mappings)
    chest_a = _first_index_for_parts(a["names"], mappings, {"chest"})
    chest_b = _first_index_for_parts(b["names"], mappings, {"chest"})
    targets = [(b, chest_b, "A hand to B chest"), (b, root_b, "A hand to B pelvis/root"), (a, chest_a, "B hand to A chest"), (a, root_a, "B hand to A pelvis/root")]
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for target, idx, label in targets[:2]:
        if idx is None:
            continue
        for hand in hand_indices_a:
            ax.plot(a["times"][:n], np.linalg.norm(a["pos"][:n, hand, :] - target["pos"][:n, idx, :], axis=1), label=f"{label} {a['names'][hand]}")
            plotted = True
    for target, idx, label in targets[2:]:
        if idx is None:
            continue
        for hand in hand_indices_b:
            ax.plot(b["times"][:n], np.linalg.norm(b["pos"][:n, hand, :] - target["pos"][:n, idx, :], axis=1), label=f"{label} {b['names'][hand]}")
            plotted = True
    if not plotted:
        plt.close(fig)
        _plot_placeholder("No mapped hand/body controllers available for pair hand-distance plot", path, plt)
        return
    ax.set_xlabel("Window time seconds")
    ax.set_ylabel("Distance proxy")
    ax.set_title("Pair hand/body distance proxies")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pair_activity(a: dict[str, Any], b: dict[str, Any], mappings: dict[str, Any], path: Path, plt: Any) -> None:
    ia = _root_index(a["names"], mappings)
    ib = _root_index(b["names"], mappings)
    fig, ax = plt.subplots(figsize=(7, 4))
    for loaded, idx, label in [(a, ia, "A root speed"), (b, ib, "B root speed")]:
        if idx is None or len(loaded["pos"]) < 2:
            continue
        dt = np.diff(loaded["times"].astype(np.float64))
        dt = np.where(dt <= 0, 1.0 / 60.0, dt)
        speed = np.linalg.norm(np.diff(loaded["pos"][:, idx, :].astype(np.float64), axis=0) / dt[:, None], axis=1)
        ax.plot(loaded["times"][1:], speed, label=label)
    ax.set_xlabel("Window time seconds")
    ax.set_ylabel("Speed proxy")
    ax.set_title("Pair activity contrast")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_placeholder(message: str, path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    ax.set_title("Preview unavailable")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_preview_index(rows: list[dict[str, Any]], manifest: list[dict[str, Any]], out: Path) -> None:
    manifest_by_id = {m["audit_id"]: m for m in manifest}
    sections = []
    for row in rows:
        m = manifest_by_id.get(row["audit_id"], {})
        image_tags = []
        safe = safe_id_for_path(row["audit_id"])
        for name in m.get("image_files", []):
            image_tags.append(f'<figure><img src="{safe}/{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>')
        yaml_block = html.escape(_annotation_block(row))
        hints = _format_hints(row)
        top_features = html.escape(yaml.safe_dump(row.get("top_feature_values", {}), sort_keys=False, allow_unicode=True))
        pair_summary = html.escape(yaml.safe_dump(row.get("pair_feature_summary", {}), sort_keys=False, allow_unicode=True))
        sections.append(
            f"<section>"
            f"<h2>{html.escape(row['audit_id'])}</h2>"
            f"<p><b>Category:</b> {html.escape(row['category'])}<br>"
            f"<b>Scene:</b> {html.escape(str(row.get('source_scene_file')))}<br>"
            f"<b>Time:</b> {row.get('start_seconds')} - {row.get('end_seconds')} seconds<br>"
            f"<b>Technical atom id only:</b> {html.escape(str(row.get('technical_atom_id')))}<br>"
            f"<b>Window:</b> {html.escape(str(row.get('window_id')))}<br>"
            f"<b>Pair window:</b> {html.escape(str(row.get('pair_window_id')))}</p>"
            f"<p><b>Why selected:</b> {html.escape('; '.join(row.get('why_selected', [])))}</p>"
            f"<p><b>Hints only, not truth:</b> {hints}</p>"
            f"<details open><summary>Top feature values</summary><pre>{top_features}</pre></details>"
            f"<details><summary>Pair/context summary</summary><pre>{pair_summary}</pre></details>"
            + "".join(image_tags)
            + f"<details open><summary>Copy/edit annotation YAML</summary><pre>{yaml_block}</pre></details>"
            f"</section>"
        )
    html_text = """<!doctype html>
<meta charset="utf-8">
<title>Reality Audit 001</title>
<style>
body{font-family:system-ui,Segoe UI,sans-serif;line-height:1.35;margin:1.5rem;background:#fafafa;color:#202020}
section{background:white;border:1px solid #ddd;border-radius:6px;padding:1rem;margin:1rem 0}
figure{display:inline-block;margin:.35rem;vertical-align:top}
img{max-width:440px;border:1px solid #ccc;background:white}
figcaption{font-size:.8rem;color:#555}
pre{white-space:pre-wrap;background:#f2f2f2;padding:.6rem;border-radius:4px}
.warn{padding:.75rem;background:#fff4cc;border:1px solid #e2c96d}
</style>
<h1>Reality Audit 001</h1>
<p class="warn">This is a technical reality audit. Weak, machine, and silver labels are hints only, not semantic truth. Axis interpretation is under audit.</p>
""" + "\n".join(sections)
    (out / "index.html").write_text(html_text, encoding="utf-8")


def _write_preview_report(rows: list[dict[str, Any]], manifest: list[dict[str, Any]], warnings: list[str], out: Path) -> None:
    item_warnings = sum(1 for item in manifest if item.get("warnings"))
    image_count = sum(len(item.get("image_files", [])) for item in manifest)
    lines = [
        "# Reality Audit Preview Report",
        "",
        "These are offline technical previews, not VaM playback and not semantic proof.",
        "",
        f"- Audit items: {len(rows)}",
        f"- Preview manifest rows: {len(manifest)}",
        f"- Preview images: {image_count}",
        f"- Items with warnings: {item_warnings}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {w}" for w in warnings) if warnings else lines.append("- None")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_batch_markdown(rows: list[dict[str, Any]], out: Path) -> None:
    lines = ["# Reality Audit Batch 001", "", f"- Items: {len(rows)}", "", "Machine/silver/weak labels are hints only. This batch is for checking whether extraction, mapping, features, pairing, and machine proposals match reality.", ""]
    for row in rows:
        hints = ", ".join(_hint_labels(row)[:8])
        lines.extend([
            f"## `{row['audit_id']}`",
            "",
            f"- Category: `{row['category']}`",
            f"- Window: `{row['window_id']}`",
            f"- Pair window: `{row.get('pair_window_id')}`",
            f"- Scene: `{row.get('source_scene_file')}`",
            f"- Technical atom id: `{row.get('technical_atom_id')}`",
            f"- Time: {row.get('start_seconds')} - {row.get('end_seconds')}",
            f"- Hint labels: {hints}",
            f"- Why selected: {'; '.join(row.get('why_selected', []))}",
            "",
        ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(rows: list[dict[str, Any]], out: Path) -> None:
    base = out.parent
    lines = [
        "# Reality Audit 001",
        "",
        "This audit is for verifying whether the clean_v2 data interpretation is actually correct before any further ML work.",
        "",
        "It asks the human reviewer to check the actual previewed motion against controller mappings, feature values, pair context, and machine/silver suggestions.",
        "",
        "This is not training labels yet. Do not copy these answers into `manual_labels.yaml`.",
        "",
        "## Open",
        "",
        f"`{base / 'previews' / 'index.html'}`",
        "",
        "## Copy",
        "",
        f"`{base / 'reality_audit_annotation.stub.yaml'}`",
        "",
        "## Save As",
        "",
        f"`{base / 'reality_audit_annotation.edited.yaml'}`",
        "",
        "## How To Review",
        "",
        "- Fill each field with `true`, `false`, `unknown`, or `not_applicable` where appropriate.",
        "- Do not modify audit IDs, window IDs, or pair window IDs.",
        "- Mark uncertainty instead of guessing.",
        "- Mark machine labels as plausible/false/uncertain only after checking previews.",
        "- Use failure reasons when the preview exposes a likely extraction, mapping, timing, feature, or pairing problem.",
        "- `trust_for_ml=true` should mean this item looks reliable enough for later ML work, not that it is semantically labeled.",
        "",
        "## What Counts As Failure",
        "",
        "- no real motion where motion is expected",
        "- wrong actor or wrong pair context",
        "- pelvis/hip/root mapping visibly wrong",
        "- axis interpretation does not match previewed motion",
        "- window timing misses the relevant movement",
        "- hand/contact proxy contradicts pair preview",
        "- machine labels are visually implausible",
        "- preview is insufficient to answer",
        "",
        "After finishing all 100 examples, run `summarize-reality-audit` to decide whether extraction, features, pairing, and machine labels can be trusted.",
        "",
        "## Batch Counts",
        "",
    ]
    lines.extend(f"- `{cat}`: {count}" for cat, count in Counter(row["category"] for row in rows).items())
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_annotation_stub(rows: list[dict[str, Any]], out: Path) -> None:
    data = {
        "metadata": {
            "audit_name": "reality_audit_001",
            "purpose": "Verify data extraction, controller mapping, feature calculations, pair windows, and machine label plausibility before ML.",
            "is_training_label_file": False,
            "do_not_merge_into_manual_labels_yaml": True,
        },
        "audit_items": {row["audit_id"]: _empty_annotation(row) for row in rows},
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_annotation_schema(out: Path) -> None:
    data = {
        "metadata": {
            "schema_name": "reality_audit_annotation_schema_v1",
            "is_training_label_schema": False,
            "allowed_values": ALLOWED_AUDIT_VALUES,
            "allowed_failure_reasons": FAILURE_REASONS,
        },
        "audit_item_fields": {
            "data_read_correctly": ALLOWED_AUDIT_VALUES,
            "real_motion_visible": ALLOWED_AUDIT_VALUES,
            "correct_actor_visible": ALLOWED_AUDIT_VALUES,
            "controller_mapping_ok": ["pelvis_or_hip", "chest", "hands", "legs", "head"],
            "axis_interpretation_ok": ["vertical", "forward_back", "lateral"],
            "window_timing_ok": ALLOWED_AUDIT_VALUES,
            "pair_context_ok": ALLOWED_AUDIT_VALUES,
            "active_actor_candidate_ok": ALLOWED_AUDIT_VALUES,
            "hand_contact_proxy_ok": ALLOWED_AUDIT_VALUES,
            "machine_labels_plausible": ["true_labels", "false_labels", "uncertain_labels"],
            "trust_for_ml": ALLOWED_AUDIT_VALUES,
            "failure_reason": FAILURE_REASONS,
            "notes": "free text",
        },
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _empty_annotation(row: dict[str, Any]) -> dict[str, Any]:
    pair_default = "unknown" if row.get("pair_window_id") else "not_applicable"
    return {
        "data_read_correctly": "unknown",
        "real_motion_visible": "unknown",
        "correct_actor_visible": "unknown",
        "controller_mapping_ok": {
            "pelvis_or_hip": "unknown",
            "chest": "unknown",
            "hands": "unknown",
            "legs": "unknown",
            "head": "unknown",
        },
        "axis_interpretation_ok": {
            "vertical": "unknown",
            "forward_back": "unknown",
            "lateral": "unknown",
        },
        "window_timing_ok": "unknown",
        "pair_context_ok": pair_default,
        "active_actor_candidate_ok": pair_default,
        "hand_contact_proxy_ok": pair_default,
        "machine_labels_plausible": {
            "true_labels": [],
            "false_labels": [],
            "uncertain_labels": [],
        },
        "trust_for_ml": "unknown",
        "failure_reason": [],
        "notes": "",
    }


def _annotation_block(row: dict[str, Any]) -> str:
    return yaml.safe_dump({"audit_items": {row["audit_id"]: _empty_annotation(row)}}, sort_keys=False, allow_unicode=True)


def _write_result_report(
    out: Path,
    batch_count: int,
    annotated_count: int,
    counts: dict[str, Counter[str]],
    nested_counts: dict[str, dict[str, Counter[str]]],
    failure_counts: Counter[str],
    true_labels: Counter[str],
    false_labels: Counter[str],
    uncertain_labels: Counter[str],
    investigate: list[str],
    verdict: dict[str, str],
) -> None:
    lines = ["# Reality Audit Result", "", f"- Audit items: {batch_count}", f"- Annotated items: {annotated_count}", "", "## Field Counts", ""]
    for field, counter in counts.items():
        lines.append(f"### {field}")
        lines.extend(f"- `{key}`: {value}" for key, value in counter.most_common())
        lines.append("")
    lines.append("## Controller Mapping OK Rates")
    for group, sub in nested_counts["controller_mapping_ok"].items():
        lines.append(f"- `{group}`: {dict(sub)}")
    lines.extend(["", "## Axis Interpretation OK Rates"])
    for group, sub in nested_counts["axis_interpretation_ok"].items():
        lines.append(f"- `{group}`: {dict(sub)}")
    lines.extend(["", "## Machine Label Plausibility", ""])
    lines.append("### Plausible")
    lines.extend(f"- `{label}`: {count}" for label, count in true_labels.most_common())
    lines.append("### False")
    lines.extend(f"- `{label}`: {count}" for label, count in false_labels.most_common())
    lines.append("### Uncertain")
    lines.extend(f"- `{label}`: {count}" for label, count in uncertain_labels.most_common())
    lines.extend(["", "## Top Failure Reasons", ""])
    lines.extend(f"- `{reason}`: {count}" for reason, count in failure_counts.most_common()) if failure_counts else lines.append("- None")
    lines.extend(["", "## Examples To Investigate", ""])
    lines.extend(f"- `{item}`" for item in investigate[:50]) if investigate else lines.append("- None flagged yet")
    lines.extend(["", "## Final Verdict", ""])
    lines.extend(f"- `{key}`: `{value}`" for key, value in verdict.items())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verdict(*counters: Counter[str]) -> str:
    true_count = sum(counter.get("true", 0) for counter in counters)
    false_count = sum(counter.get("false", 0) for counter in counters)
    unknown_count = sum(counter.get("unknown", 0) for counter in counters)
    if false_count > max(2, true_count * 0.2):
        return "false"
    if true_count >= max(5, false_count * 3) and true_count > unknown_count:
        return "true"
    return "uncertain"


def _pair_by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ["window_id_a", "window_id_b"]:
            wid = row.get(key)
            if wid and wid not in out:
                out[wid] = row
    return out


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key):
            out[str(row[key])].append(row)
    return out


def _group_pair_scores(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pair_window_id"):
            out[str(row["pair_window_id"])].append(row)
    return out


def _has_contact_or_role_label(labels: set[str]) -> bool:
    joined = " ".join(labels)
    return any(token in joined for token in CONTACT_LABEL_TOKENS)


def _overlaps_existing(sample: str, start: float, end: float, ranges_by_sample: dict[str, list[tuple[float, float]]]) -> bool:
    for old_start, old_end in ranges_by_sample.get(sample, []):
        overlap = max(0.0, min(end, old_end) - max(start, old_start))
        if overlap >= 0.75 * max(0.001, min(end - start, old_end - old_start)):
            return True
    return False


def _mapped_body_parts(names: list[str], mappings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "body_part": mappings.get(name, {}).get("body_part", "unknown"),
            "mapping_confidence": mappings.get(name, {}).get("mapping_confidence", "unknown"),
            "matched_pattern": mappings.get(name, {}).get("matched_pattern"),
        }
        for name in names
    }


def _top_features(values: dict[str, Any], limit: int = 20) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in CORE_FEATURES:
        if key in values:
            out[key] = _clean_num(values.get(key))
    extras = []
    for key, value in values.items():
        val = _num(value)
        if val == val:
            extras.append((key, abs(val), val))
    for key, _, val in sorted(extras, key=lambda item: item[1], reverse=True):
        if key not in out:
            out[key] = round(float(val), 6)
        if len(out) >= limit:
            break
    return out


def _pair_summary(pair: dict[str, Any], feature: dict[str, Any], score_rows: list[dict[str, Any]], silver_pair: dict[str, Any]) -> dict[str, Any]:
    if not pair:
        return {}
    values = feature.get("feature_values", {}) or {}
    return {
        "pair_window_id": pair.get("pair_window_id"),
        "sample_id_a": pair.get("sample_id_a"),
        "sample_id_b": pair.get("sample_id_b"),
        "technical_atom_id_a": pair.get("technical_atom_id_a"),
        "technical_atom_id_b": pair.get("technical_atom_id_b"),
        "window_id_a": pair.get("window_id_a"),
        "window_id_b": pair.get("window_id_b"),
        "pair_confidence": pair.get("pair_confidence"),
        "pairing_reasons": pair.get("pairing_reasons", []),
        "feature_quality": feature.get("feature_quality", {}),
        "top_pair_feature_values": {key: _clean_num(values.get(key)) for key in PAIR_FEATURES if key in values},
        "machine_pair_scores": _score_hints(score_rows, limit=10),
        "silver_pair_labels": _silver_hint(silver_pair),
        "warning": "Pair roles are motion/context candidates only and not semantic truth.",
    }


def _weak_hints(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"label": item.get("label"), "confidence": item.get("confidence"), "proxy_score": item.get("proxy_score"), "is_human_ground_truth": False}
        for item in row.get("weak_labels", [])[:20]
    ]


def _score_hints(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    seen = set()
    hints = []
    for row in sorted(rows, key=lambda item: float(item.get("final_score") or item.get("max_confidence") or 0.0), reverse=True):
        label = row.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        hints.append({
            "label": label,
            "final_score": row.get("final_score"),
            "max_confidence": row.get("max_confidence"),
            "recommended_status": row.get("recommended_status"),
            "proposal_types": row.get("proposal_types", []),
            "conflict_flags": row.get("conflict_flags", []),
            "high_risk_proxy_label": row.get("high_risk_proxy_label", False),
            "rule_ids": row.get("rule_ids", []),
            "is_human_ground_truth": False,
        })
        if len(hints) >= limit:
            break
    return hints


def _silver_hint(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "positive_labels": row.get("positive_labels", []),
        "negative_labels": row.get("negative_labels", []),
        "review_only_labels": row.get("review_only_labels", []),
        "default_trainable_labels": row.get("default_trainable_labels", []),
        "excluded_from_default_training": row.get("excluded_from_default_training", {}),
        "scores_by_label": row.get("scores_by_label", {}),
        "label_source": row.get("label_source", "silver_machine_v2"),
        "is_human_ground_truth": False,
    }


def _hint_labels(row: dict[str, Any]) -> list[str]:
    labels = []
    labels.extend(item.get("label", "") for item in row.get("weak_v2_labels", []))
    labels.extend(item.get("label", "") for item in row.get("machine_proposals", []))
    labels.extend(row.get("silver_labels", {}).get("positive_labels", []) or [])
    labels.extend(row.get("pair_feature_summary", {}).get("silver_pair_labels", {}).get("positive_labels", []) or [])
    return [label for label in labels if label]


def _format_hints(row: dict[str, Any]) -> str:
    labels = _hint_labels(row)[:12]
    return html.escape(", ".join(labels) if labels else "None")


def _core_indices(names: list[str], mappings: dict[str, Any]) -> list[int]:
    indices = []
    for idx, name in enumerate(names):
        if mappings.get(name, {}).get("body_part") in CORE_PARTS:
            indices.append(idx)
    return indices[:8] or list(range(min(len(names), 4)))


def _root_index(names: list[str], mappings: dict[str, Any]) -> int | None:
    for wanted in ["pelvis", "hip", "root", "abdomen"]:
        idx = _first_index_for_parts(names, mappings, {wanted})
        if idx is not None:
            return idx
    return None


def _first_index_for_parts(names: list[str], mappings: dict[str, Any], parts: set[str]) -> int | None:
    for idx, name in enumerate(names):
        if mappings.get(name, {}).get("body_part") in parts:
            return idx
    return None


def _indices_for_parts(names: list[str], mappings: dict[str, Any], parts: set[str]) -> list[int]:
    return [idx for idx, name in enumerate(names) if mappings.get(name, {}).get("body_part") in parts]


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    return val


def _clean_num(value: Any) -> float | None:
    val = _num(value)
    if val != val or math.isinf(val):
        return None
    return round(float(val), 6)

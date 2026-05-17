"""ML-assisted Cowgirl review batch export."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.audits.review_deduplication import annotate_candidates_against_reviewed, write_review_quality_report
from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import build_static_review_ui


BUCKET_TARGETS = [
    ("high_confidence_cowgirl", 6),
    ("high_confidence_negative", 4),
    ("model_heuristic_disagreement", 4),
    ("uncertain_boundary", 4),
    ("generation_safe_candidate_check", 2),
]


def export_ml_assisted_cowgirl_review_v1(
    run_dir: str | Path,
    model_scores: str | Path,
    reviewed_index: str | Path,
    out_dir: str | Path,
    count: int = 20,
    max_per_scene: int = 2,
    max_per_sample: int = 1,
    build_vam_package: bool = True,
    build_static_ui: bool = True,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = load_jsonl(model_scores)
    reviewed = load_jsonl(reviewed_index)
    annotated = annotate_candidates_against_reviewed(scores, reviewed)
    selected, rejected = _select(annotated, count, max_per_scene, max_per_sample)
    rows = [_review_row(i, row) for i, row in enumerate(selected, 1)]
    write_jsonl(out / "semantic_review_010.jsonl", rows)
    _write_answer_sheet(out / "semantic_review_010_answer_sheet.yaml", rows)
    _write_report(out / "ml_assisted_review_report.md", rows, rejected, count)
    quality = write_review_quality_report(
        rows,
        out / "review_quality_report.md",
        requested_count=count,
        max_per_scene=max_per_scene,
        max_per_sample=max_per_sample,
        shortage_reasons=rejected,
    )
    package = None
    static = None
    if build_vam_package:
        package = build_vam_review_package(out / "semantic_review_010.jsonl", run_dir, run_dir, out / "vam_review_package", attempt_timeline_segments=True)
    if build_static_ui:
        static = build_static_review_ui(run_dir, out, out / "review_ui_static")
    summary = {
        "status": "ok",
        "requested_count": count,
        "exported_count": len(rows),
        "bucket_counts": dict(Counter(r.get("recommended_review_priority") for r in selected)),
        "rejected_by_rule": rejected,
        "quality": quality,
        "vam_review_package": package,
        "static_review_ui": static,
        "review_assist_only": True,
    }
    dump_json(out / "ml_assisted_review_summary.json", summary)
    return summary


def _select(rows: list[dict[str, Any]], count: int, max_per_scene: int, max_per_sample: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    chosen_windows: set[str] = set()

    buckets = {name: target for name, target in BUCKET_TARGETS}
    for bucket, target in buckets.items():
        candidates = [r for r in rows if r.get("recommended_review_priority") == bucket]
        candidates.sort(key=_sort_key)
        for row in candidates:
            if len([r for r in selected if r.get("recommended_review_priority") == bucket]) >= target:
                break
            if len(selected) >= count:
                break
            if not _can_select(row, scene_counts, sample_counts, chosen_windows, max_per_scene, max_per_sample, rejected):
                continue
            selected.append(row)
            _mark(row, scene_counts, sample_counts, chosen_windows)
    if len(selected) < count:
        for row in sorted(rows, key=_sort_key):
            if len(selected) >= count:
                break
            if row in selected:
                continue
            if not _can_select(row, scene_counts, sample_counts, chosen_windows, max_per_scene, max_per_sample, rejected):
                continue
            selected.append(row)
            _mark(row, scene_counts, sample_counts, chosen_windows)
    return selected, dict(rejected)


def _can_select(row: dict[str, Any], scene_counts: Counter[str], sample_counts: Counter[str], chosen_windows: set[str], max_per_scene: int, max_per_sample: int, rejected: Counter[str]) -> bool:
    if row.get("previously_reviewed") or row.get("duplicate_status") in {"exact_duplicate", "near_duplicate", "previously_reviewed"}:
        rejected["previously_reviewed_or_duplicate"] += 1
        return False
    wid = str(row.get("window_id") or "")
    if wid and wid in chosen_windows:
        rejected["duplicate_window"] += 1
        return False
    scene = str(row.get("source_scene_file") or "")
    sample = str(row.get("sample_id") or "")
    if scene and scene_counts[scene] >= max_per_scene:
        rejected["scene_cap"] += 1
        return False
    if sample and sample_counts[sample] >= max_per_sample:
        rejected["sample_cap"] += 1
        return False
    return True


def _mark(row: dict[str, Any], scene_counts: Counter[str], sample_counts: Counter[str], chosen_windows: set[str]) -> None:
    if row.get("source_scene_file"):
        scene_counts[str(row["source_scene_file"])] += 1
    if row.get("sample_id"):
        sample_counts[str(row["sample_id"])] += 1
    if row.get("window_id"):
        chosen_windows.add(str(row["window_id"]))


def _sort_key(row: dict[str, Any]) -> tuple[float, float]:
    priority = str(row.get("recommended_review_priority") or "")
    if priority == "high_confidence_cowgirl":
        return (-(row.get("model_cowgirl_probability") or 0.0), -(row.get("model_clean_motion_probability") or 0.0))
    if priority == "high_confidence_negative":
        return ((row.get("model_cowgirl_probability") or 0.0), 0.0)
    return (-(row.get("uncertainty_score") or 0.0), -(row.get("model_cowgirl_probability") or 0.0))


def _review_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    rid = f"review_{index:03d}"
    return {
        "review_id": rid,
        "title": f"{rid}_ml_assisted_{row.get('recommended_review_priority')}",
        "semantic_family": row.get("heuristic_semantic_family") or "",
        "category": row.get("heuristic_category") or "",
        "source_scene_file": row.get("source_scene_file") or "",
        "technical_actor_id": row.get("technical_actor_id") or "",
        "source_id": row.get("source_id") or "",
        "sample_id": row.get("sample_id") or "",
        "window_id": row.get("window_id") or "",
        "start_seconds": row.get("start_seconds"),
        "end_seconds": row.get("end_seconds"),
        "pose_subtype": row.get("pose_subtype") or "",
        "motion_subtype": row.get("motion_subtype") or "",
        "phase": row.get("phase") or "",
        "contact_support": row.get("contact_support") or "",
        "model_cowgirl_probability": row.get("model_cowgirl_probability"),
        "model_clean_motion_probability": row.get("model_clean_motion_probability"),
        "model_generation_safe_probability": row.get("model_generation_safe_probability"),
        "recommended_review_priority": row.get("recommended_review_priority"),
        "disagreement_flags": row.get("disagreement_flags") or [],
        "why_selected": f"ML review priority: {row.get('recommended_review_priority')}",
        "duplicate_status": row.get("duplicate_status", "unique"),
        "previously_reviewed": bool(row.get("previously_reviewed")),
        "review_questions": ["Was ist sichtbar?", "Ist das Cowgirl?", "Ist es clean/cyclic motion oder nur setup/transition?", "Ist es für Primitive später nutzbar?"],
        "review_assist_only": True,
    }


def _write_answer_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    data = {"metadata": {"review_batch": "ml_assisted_cowgirl_review_v1", "is_training_label_file": False}, "reviews": {}}
    for row in rows:
        data["reviews"][row["review_id"]] = {"verdict": "", "notes": "", "review_labels": []}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_report(path: Path, rows: list[dict[str, Any]], rejected: dict[str, int], requested: int) -> None:
    lines = [
        "# ML-assisted Cowgirl Review v1",
        "",
        f"- Requested: {requested}",
        f"- Exported: {len(rows)}",
        f"- Priority counts: `{dict(Counter(r.get('recommended_review_priority') for r in rows))}`",
        f"- Rejected by rule: `{rejected}`",
        "",
        "This batch is for human review only. Model scores are suggestions, not labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

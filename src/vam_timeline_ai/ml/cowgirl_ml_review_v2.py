"""Export ML-assisted Cowgirl review v2 batches."""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import load_jsonl, safe_id_for_path, write_jsonl
from vam_timeline_ai.ui.review_ui import build_static_review_ui


BUCKET_QUOTAS = [
    ("high_confidence_clean_cowgirl", 10),
    ("cowgirl_pose_context", 5),
    ("likely_bj_or_hj_negative", 5),
    ("model_gate_disagreement", 5),
    ("uncertain_boundary", 5),
    ("incomplete_pose_but_semantic_cowgirl", 5),
]


def export_ml_assisted_cowgirl_review_v2(
    new_run: str | Path,
    scores: str | Path,
    candidates: str | Path,
    out_dir: str | Path,
    count: int,
    build_static_ui: bool = True,
    build_vam_package: bool = True,
) -> dict[str, Any]:
    run = Path(new_run)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_rows = load_jsonl(scores)
    candidate_by_window = {str(r.get("window_id")): r for r in load_jsonl(candidates) if r.get("window_id")}
    rows = _select(score_rows, count)
    review_rows = []
    for idx, row in enumerate(rows, start=1):
        merged = dict(candidate_by_window.get(str(row.get("window_id") or ""), {}))
        merged.update(row)
        rid = f"ml_cowgirl_v2_{idx:03d}"
        merged.update(
            {
                "review_id": rid,
                "review_index": idx,
                "review_label": f"{rid}_{safe_id_for_path(str(row.get('recommended_review_bucket') or 'candidate'))}",
                "semantic_family": merged.get("resolved_motion_family") or merged.get("resolved_semantic_family") or "unknown",
                "motion_subtype": merged.get("resolved_motion_subtype") or "",
                "why_selected": row.get("recommended_review_bucket"),
                "selection_reason": _selection_reason(row),
                "review_question": "Ist das Cowgirl semantisch korrekt, clean motion oder nur Pose/Transition/Negativfall?",
                "review_only": True,
                "not_training_truth": True,
                "manual_labels_modified": False,
                "ml_training_performed": False,
                "timeline_generation_performed": False,
            }
        )
        review_rows.append(merged)
    write_jsonl(out / "semantic_review_010.jsonl", review_rows)
    _write_index(out / "semantic_review_010_index.html", review_rows)
    _write_quality(out / "review_quality_report.md", score_rows, review_rows, count)
    if build_vam_package:
        build_vam_review_package(out / "semantic_review_010.jsonl", run, run, out / "vam_review_package", attempt_timeline_segments=True)
    if build_static_ui:
        build_static_review_ui(run, out, out / "review_ui_static")
    return {
        "status": "ok",
        "out_dir": str(out),
        "selected": len(review_rows),
        "bucket_counts": dict(Counter(r.get("recommended_review_bucket") for r in review_rows)),
        "static_ui": str(out / "review_ui_static" / "index.html") if build_static_ui else "",
        "vam_package": str(out / "vam_review_package") if build_vam_package else "",
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }


def _select(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_windows: set[str] = set()
    used_samples: set[str] = set()
    scene_counts: defaultdict[str, int] = defaultdict(int)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = str(row.get("recommended_review_bucket") or "uncertain_boundary")
        if bucket == "high_confidence_clean_cowgirl" and not _gate_clean_cowgirl(row):
            row = dict(row)
            row["recommended_review_bucket"] = "model_gate_disagreement"
            row["disagreement_with_gates"] = list(row.get("disagreement_with_gates") or []) + ["ml_clean_but_gate_not_clean_cowgirl"]
            bucket = "model_gate_disagreement"
        by_bucket[bucket].append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=_sort_key)
    for bucket, quota in BUCKET_QUOTAS:
        for row in by_bucket.get(bucket, []):
            if len([r for r in selected if r.get("recommended_review_bucket") == bucket]) >= quota:
                break
            if len(selected) >= count:
                break
            if _eligible(row, used_windows, used_samples, scene_counts):
                selected.append(row)
                _mark(row, used_windows, used_samples, scene_counts)
    for row in sorted(rows, key=_sort_key):
        if len(selected) >= count:
            break
        if _eligible(row, used_windows, used_samples, scene_counts):
            selected.append(row)
            _mark(row, used_windows, used_samples, scene_counts)
    return selected


def _eligible(row: dict[str, Any], used_windows: set[str], used_samples: set[str], scene_counts: dict[str, int]) -> bool:
    wid = str(row.get("window_id") or "")
    sample = str(row.get("sample_id") or "")
    scene = str(row.get("source_scene_file") or "")
    if wid and wid in used_windows:
        return False
    if sample and sample in used_samples:
        return False
    if scene and scene_counts[scene] >= 2:
        return False
    return True


def _mark(row: dict[str, Any], used_windows: set[str], used_samples: set[str], scene_counts: dict[str, int]) -> None:
    if row.get("window_id"):
        used_windows.add(str(row.get("window_id")))
    if row.get("sample_id"):
        used_samples.add(str(row.get("sample_id")))
    if row.get("source_scene_file"):
        scene_counts[str(row.get("source_scene_file"))] += 1


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    bucket = str(row.get("recommended_review_bucket") or "")
    if bucket == "high_confidence_clean_cowgirl":
        return (-(row.get("model_cowgirl_clean_motion_probability") or 0), -(row.get("cyclicity_score") or 0), row.get("transition_score") or 9)
    if bucket == "likely_bj_or_hj_negative":
        return (-(row.get("model_bj_oral_negative_probability") or 0), -(row.get("cyclicity_score") or 0), 0)
    return (-(row.get("uncertainty_score") or 0), -(row.get("model_cowgirl_family_probability") or 0), 0)


def _gate_clean_cowgirl(row: dict[str, Any]) -> bool:
    return (
        row.get("category") == "cowgirl_clean_cyclic_motion"
        and row.get("final_clean_motion_gate") == "pass"
        and row.get("primary_driver_controller") == "hipControl"
    )


def _selection_reason(row: dict[str, Any]) -> str:
    return (
        f"bucket={row.get('recommended_review_bucket')}; "
        f"p_family={row.get('model_cowgirl_family_probability')}; "
        f"p_clean={row.get('model_cowgirl_clean_motion_probability')}; "
        f"p_bj_negative={row.get('model_bj_oral_negative_probability')}; "
        f"disagreement={row.get('disagreement_with_gates')}"
    )


def _write_quality(path: Path, all_rows: list[dict[str, Any]], selected: list[dict[str, Any]], requested: int) -> None:
    lines = [
        "# ML-Assisted Cowgirl Review v2 Quality",
        "",
        f"- Requested: `{requested}`",
        f"- Selected: `{len(selected)}`",
        f"- Source score rows: `{len(all_rows)}`",
        f"- Selected buckets: `{dict(Counter(r.get('recommended_review_bucket') for r in selected))}`",
        "- Strict dedup: max 2 per scene, max 1 per sample/window",
        "- Auto-labeling performed: `false`",
        "- Timeline generation performed: `false`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<section>"
            f"<h2>{html.escape(str(row.get('review_id')))}</h2>"
            f"<p><b>{html.escape(str(row.get('recommended_review_bucket')))}</b> {html.escape(str(row.get('source_scene_file')))} {row.get('start_seconds')}-{row.get('end_seconds')}s</p>"
            f"<p>p_family={row.get('model_cowgirl_family_probability')} p_clean={row.get('model_cowgirl_clean_motion_probability')} p_bj={row.get('model_bj_oral_negative_probability')}</p>"
            f"<p>category={html.escape(str(row.get('category')))} driver={html.escape(str(row.get('primary_driver_controller')))} gate={html.escape(str(row.get('final_clean_motion_gate')))}</p>"
            f"<p>{html.escape(str(row.get('selection_reason')))}</p>"
            "</section>"
        )
    path.write_text("<!doctype html><meta charset='utf-8'><style>body{font-family:Arial;margin:24px auto;max-width:1100px}section{border:1px solid #ccc;margin:12px;padding:12px}</style><h1>ML-Assisted Cowgirl Review v2</h1>" + "\n".join(cards), encoding="utf-8")

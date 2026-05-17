"""Review deduplication and novelty QA utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import hashlib
import html

import yaml

from vam_timeline_ai.audits.vam_review_package import build_vam_review_package
from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import build_static_review_ui


REVIEW_FILE_NAMES = [
    "semantic_review_010.jsonl",
    "focused_review_manifest.jsonl",
    "strict_cowgirl_review_manifest.jsonl",
]


def build_reviewed_window_index(
    run_dir: str | Path,
    include_runs: str | list[str | Path],
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    include = _parse_include_runs(include_runs)
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for root in include:
        if not root.exists():
            warnings.append(f"include run missing: {root}")
            continue
        records.extend(_scan_run_reviews(root, warnings))
    records = annotate_duplicate_status(records)
    write_jsonl(out_jsonl, records)
    _write_csv(Path(out_csv), records)
    summary = _duplicate_summary(records)
    summary.update(
        {
            "status": "ok",
            "run_dir": str(run),
            "include_runs": [str(p) for p in include],
            "records": len(records),
            "warnings": warnings,
            "out_jsonl": str(out_jsonl),
            "out_csv": str(out_csv),
        }
    )
    _write_index_report(Path(report), summary)
    return summary


def audit_review_duplicates(reviewed_index: str | Path, out: str | Path) -> dict[str, Any]:
    rows = load_jsonl(reviewed_index)
    summary = _duplicate_summary(rows)
    by_folder = Counter(str(r.get("review_folder") or "unknown") for r in rows)
    dup_by_folder = Counter(
        str(r.get("review_folder") or "unknown")
        for r in rows
        if r.get("duplicate_status") in {"exact_duplicate", "near_duplicate", "previously_reviewed"}
    )
    summary["folder_counts"] = dict(by_folder)
    summary["duplicate_folder_counts"] = dict(dup_by_folder)
    _write_duplicate_audit(Path(out), rows, summary)
    return summary


def export_strict_novel_review(
    run_dir: str | Path,
    candidate_db: str | Path,
    reviewed_index: str | Path,
    out_dir: str | Path,
    count: int = 10,
    max_per_scene: int = 2,
    max_per_sample: int = 1,
    allow_reviewed_overlap: bool = False,
    allow_near_duplicates: bool = False,
    build_vam_package: bool = True,
    build_static_ui: bool = True,
    diversity_mode: str = "strict",
) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = _enrich_candidates_from_run(load_jsonl(candidate_db), run)
    reviewed = load_jsonl(reviewed_index)
    annotated = annotate_candidates_against_reviewed(candidates, reviewed)
    selected, selection = _select_novel_candidates(
        annotated,
        count=count,
        max_per_scene=max_per_scene,
        max_per_sample=max_per_sample,
        allow_reviewed_overlap=allow_reviewed_overlap,
        allow_near_duplicates=allow_near_duplicates,
        diversity_mode=diversity_mode,
    )
    rows = [_review_row(i, row) for i, row in enumerate(selected, start=1)]
    write_jsonl(out / "semantic_review_010.jsonl", rows)
    _write_answer_sheet(out / "semantic_review_010_answer_sheet.yaml", rows)
    _write_review_md(out / "semantic_review_010.md", rows)
    _write_review_html(out / "semantic_review_010_index.html", rows)
    quality = write_review_quality_report(
        rows,
        out / "review_quality_report.md",
        requested_count=count,
        max_per_scene=max_per_scene,
        max_per_sample=max_per_sample,
        shortage_reasons=selection["rejected_by_rule"],
    )
    package = None
    static = None
    if build_vam_package:
        package = build_vam_review_package(out / "semantic_review_010.jsonl", run, run.parent / "clean_v2", out / "vam_review_package", attempt_timeline_segments=True)
    if build_static_ui:
        static = build_static_review_ui(run, out, out / "review_ui_static")
    summary = {
        "status": "ok",
        "out_dir": str(out),
        "requested_count": count,
        "exported_count": len(rows),
        "selection": selection,
        "quality": quality,
        "vam_review_package": package,
        "static_review_ui": static,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    dump_json(out / "strict_novel_review_summary.json", summary)
    return summary


def annotate_duplicate_status(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_window: dict[str, str] = {}
    seen_pair: dict[str, str] = {}
    seen_sample_time: dict[str, str] = {}
    seen_source_time: dict[str, str] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        rec = dict(row)
        rid = _global_review_id(rec, index)
        duplicate_status = "unique"
        reason = ""
        overlaps: list[str] = []
        exact_key, exact_seen = _exact_duplicate_key(rec, seen_window, seen_pair, seen_sample_time, seen_source_time)
        if exact_key and exact_seen:
            duplicate_status = "exact_duplicate"
            reason = exact_key
            overlaps.append(exact_seen)
        elif _previously_reviewed_by_source(rec, out):
            duplicate_status = "previously_reviewed"
            reason = "same source/sample/time already reviewed"
            overlaps.extend(_overlap_review_ids(rec, out))
        else:
            near = _near_duplicate_ids(rec, out)
            if near:
                duplicate_status = "near_duplicate"
                reason = "same scene/actor/source with overlapping or nearby time"
                overlaps.extend(near)
        group_id = _duplicate_group_id(rec, duplicate_status, reason, overlaps)
        rec["duplicate_status"] = duplicate_status
        rec["duplicate_group_id"] = group_id
        rec["duplicate_reason"] = reason
        rec["overlaps_with_review_ids"] = _dedupe(overlaps)
        rec["previously_reviewed"] = duplicate_status != "unique"
        rec["review_trust_warning"] = (
            "This item appears to overlap a previously reviewed sample/window."
            if duplicate_status != "unique"
            else ""
        )
        out.append(rec)
        _mark_seen(rec, rid, seen_window, seen_pair, seen_sample_time, seen_source_time)
        groups[group_id].append(rid)
    return out


def annotate_candidates_against_reviewed(candidates: list[dict[str, Any]], reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reviewed_norm = [dict(r) for r in reviewed]
    out: list[dict[str, Any]] = []
    for row in candidates:
        rec = _normalize_candidate(row)
        exact = _candidate_exact_overlap(rec, reviewed_norm)
        near = [] if exact else _near_duplicate_ids(rec, reviewed_norm)
        if exact:
            status = "previously_reviewed"
            reason = "candidate exact key exists in reviewed index"
            overlaps = exact
        elif near:
            status = "near_duplicate"
            reason = "candidate overlaps reviewed scene/actor/source/time"
            overlaps = near
        else:
            status = "unique"
            reason = ""
            overlaps = []
        rec["duplicate_status"] = status
        rec["duplicate_group_id"] = _duplicate_group_id(rec, status, reason, overlaps)
        rec["duplicate_reason"] = reason
        rec["overlaps_with_review_ids"] = overlaps
        rec["previously_reviewed"] = status != "unique"
        rec["review_trust_warning"] = (
            "This item appears to overlap a previously reviewed sample/window."
            if status != "unique"
            else ""
        )
        out.append(rec)
    return out


def write_review_quality_report(
    rows: list[dict[str, Any]],
    out: str | Path,
    requested_count: int,
    max_per_scene: int,
    max_per_sample: int,
    shortage_reasons: Counter[str] | dict[str, int] | None = None,
) -> dict[str, Any]:
    exact = sum(1 for r in rows if r.get("duplicate_status") == "exact_duplicate")
    near = sum(1 for r in rows if r.get("duplicate_status") == "near_duplicate")
    prev = sum(1 for r in rows if r.get("previously_reviewed"))
    trust = "high"
    if exact or near or prev:
        trust = "low"
    elif len(rows) < requested_count:
        trust = "medium"
    summary = {
        "requested_count": requested_count,
        "exported_count": len(rows),
        "unique_scene_count": len({r.get("source_scene_file") for r in rows if r.get("source_scene_file")}),
        "unique_sample_count": len({r.get("sample_id") for r in rows if r.get("sample_id")}),
        "unique_source_count": len({r.get("source_id") for r in rows if r.get("source_id")}),
        "exact_duplicates": exact,
        "near_duplicates": near,
        "previously_reviewed_overlaps": prev,
        "max_per_scene": max_per_scene,
        "max_per_sample": max_per_sample,
        "category_distribution": dict(Counter(str(r.get("category") or r.get("why_selected") or "unknown") for r in rows)),
        "shortage_reasons": dict(shortage_reasons or {}),
        "trust_level": trust,
    }
    _write_quality_report(Path(out), summary)
    return summary


def _scan_run_reviews(run: Path, warnings: list[str]) -> list[dict[str, Any]]:
    audits = run / "audits"
    if not audits.exists():
        warnings.append(f"audits folder missing: {audits}")
        return []
    roots: set[Path] = set()
    for name in REVIEW_FILE_NAMES:
        for path in audits.rglob(name):
            roots.add(path.parent)
    for path in audits.rglob("vam_review_manifest.jsonl"):
        if path.parent.name == "vam_review_package":
            roots.add(path.parent.parent)
    rows: list[dict[str, Any]] = []
    for root in sorted(roots):
        rows.extend(_load_review_root(run, root, warnings))
    return rows


def _load_review_root(run: Path, root: Path, warnings: list[str]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for name in REVIEW_FILE_NAMES:
        for row in load_jsonl(root / name):
            rid = str(row.get("review_id") or row.get("id") or f"review_{len(merged)+1:03d}")
            merged.setdefault(rid, {}).update(row)
    for row in load_jsonl(root / "vam_review_package" / "vam_review_manifest.jsonl"):
        rid = str(row.get("review_id") or row.get("id") or f"review_{len(merged)+1:03d}")
        base = merged.setdefault(rid, {})
        base.update({k: v for k, v in row.items() if _has_value(v)})
    answers = _load_ui_answers(root)
    human_notes = _load_human_notes(root)
    records = []
    for rid, row in merged.items():
        rec = _review_index_record(run, root, rid, row)
        answer = answers.get(rid) or {}
        note = human_notes.get(rid) or {}
        if answer:
            rec["human_verdict"] = answer.get("verdict") or ""
            rec["human_labels"] = answer.get("review_labels") or answer.get("error_tags") or []
            rec["human_notes"] = answer.get("notes") or ""
        if note:
            rec["human_verdict"] = rec.get("human_verdict") or note.get("user_verdict") or ""
            rec["human_labels"] = rec.get("human_labels") or note.get("actual_labels") or []
            rec["human_notes"] = rec.get("human_notes") or note.get("notes") or ""
        if not rec.get("window_id") and not rec.get("sample_id") and not rec.get("source_id"):
            warnings.append(f"review item has weak lineage: {root} {rid}")
        records.append(rec)
    return records


def _review_index_record(run: Path, root: Path, rid: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_folder": str(root),
        "review_id": rid,
        "run_id": run.name,
        "source_scene_file": row.get("source_scene_file") or "",
        "source_scene_path": row.get("source_scene_path") or "",
        "technical_actor_id": row.get("technical_actor_id") or row.get("technical_atom_id") or "",
        "source_id": row.get("source_id") or "",
        "sample_id": row.get("sample_id") or "",
        "window_id": row.get("window_id") or "",
        "pair_window_id": row.get("pair_window_id") or "",
        "start_seconds": _num_or_none(row.get("start_seconds")),
        "end_seconds": _num_or_none(row.get("end_seconds")),
        "duration": _first(row.get("duration_seconds"), _duration(row.get("start_seconds"), row.get("end_seconds"))),
        "semantic_family_guess": row.get("semantic_family") or row.get("system_semantic_family") or "",
        "pose_subtype": row.get("pose_subtype") or ((row.get("pose_semantics") or {}).get("subtype") if isinstance(row.get("pose_semantics"), dict) else ""),
        "motion_subtype": row.get("motion_subtype") or ((row.get("motion_semantics") or {}).get("subtype") if isinstance(row.get("motion_semantics"), dict) else ""),
        "category": row.get("category") or row.get("cowgirl_bucket") or row.get("why_selected") or "",
        "clip_name": row.get("clip_name") or "",
        "human_verdict": "",
        "human_labels": [],
    }


def _load_ui_answers(root: Path) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("human_review_ui_answers*.jsonl")):
        for row in load_jsonl(path):
            if row.get("review_id"):
                answers[str(row["review_id"])] = row
    return answers


def _load_human_notes(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "semantic_review_010_human_notes.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if isinstance(data.get("reviews"), dict):
        data = data["reviews"]
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def _normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "technical_actor_id": row.get("technical_actor_id") or row.get("technical_atom_id") or "",
        "semantic_family_guess": row.get("semantic_family") or "",
        "start_seconds": _num_or_none(row.get("start_seconds")),
        "end_seconds": _num_or_none(row.get("end_seconds")),
    }


def _enrich_candidates_from_run(candidates: list[dict[str, Any]], run: Path) -> list[dict[str, Any]]:
    windows = {r.get("window_id"): r for r in load_jsonl(run / "semantic" / "movement_windows.jsonl") if r.get("window_id")}
    samples = {r.get("sample_id"): r for r in load_jsonl(run / "baked" / "motion_sample_index.jsonl") if r.get("sample_id")}
    sources = {r.get("source_id"): r for r in load_jsonl(run / "semantic" / "motion_source_index.jsonl") if r.get("source_id")}
    out = []
    for row in candidates:
        rec = dict(row)
        window = windows.get(rec.get("window_id"), {})
        sample = samples.get(rec.get("sample_id") or window.get("sample_id"), {})
        source = sources.get(window.get("source_id") or sample.get("source_id") or rec.get("source_id"), {})
        for key, value in {
            "source_id": window.get("source_id") or sample.get("source_id") or source.get("source_id"),
            "sample_id": window.get("sample_id") or sample.get("sample_id"),
            "source_scene_file": window.get("source_scene_file") or sample.get("source_scene_file") or source.get("source_scene_file"),
            "source_scene_path": window.get("source_scene_path") or sample.get("source_scene_path") or source.get("source_scene_path"),
            "technical_actor_id": window.get("technical_atom_id") or sample.get("technical_atom_id") or source.get("technical_atom_id"),
            "start_seconds": window.get("start_seconds"),
            "end_seconds": window.get("end_seconds"),
            "duration_seconds": window.get("duration_seconds"),
            "clip_name": sample.get("clip_name") or source.get("clip_name"),
        }.items():
            if not _has_value(rec.get(key)) and _has_value(value):
                rec[key] = value
        out.append(rec)
    return out


def _select_novel_candidates(
    candidates: list[dict[str, Any]],
    count: int,
    max_per_scene: int,
    max_per_sample: int,
    allow_reviewed_overlap: bool,
    allow_near_duplicates: bool,
    diversity_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools = _candidate_pools(candidates)
    order = [
        "cowgirl_clean_motion_generation_safe",
        "cowgirl_clean_motion_low_confidence_short",
        "cowgirl_pose_context_low_motion",
        "cowgirl_transition_setup",
        "not_cowgirl_bj_oral",
        "not_cowgirl_standing_hand_head",
        "not_cowgirl_receiver_response",
        "unknown_or_unusable",
    ]
    selected: list[dict[str, Any]] = []
    state = {"scene": Counter(), "sample": Counter(), "near": set(), "rejected": Counter(), "categories": Counter()}
    while len(selected) < count:
        progressed = False
        for category in order:
            if len(selected) >= count:
                break
            row = _pop_next_eligible(pools.get(category, []), state, max_per_scene, max_per_sample, allow_reviewed_overlap, allow_near_duplicates, diversity_mode)
            if row:
                selected.append(row)
                state["categories"][category] += 1
                progressed = True
        if not progressed:
            break
    return selected, {
        "requested": count,
        "selected": len(selected),
        "category_counts": dict(state["categories"]),
        "rejected_by_rule": state["rejected"],
        "allow_reviewed_overlap": allow_reviewed_overlap,
        "allow_near_duplicates": allow_near_duplicates,
        "diversity_mode": diversity_mode,
    }


def _candidate_pools(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        category = str(row.get("category") or row.get("semantic_family") or "unknown_or_unusable")
        pools[category].append(row)
    for rows in pools.values():
        rows.sort(key=lambda r: (-_num(r.get("semantic_score")), -_num(r.get("motion_score")), str(r.get("source_scene_file"))))
    return pools


def _pop_next_eligible(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    max_per_scene: int,
    max_per_sample: int,
    allow_reviewed_overlap: bool,
    allow_near_duplicates: bool,
    diversity_mode: str,
) -> dict[str, Any] | None:
    while rows:
        row = rows.pop(0)
        reason = _novel_reject_reason(row, state, max_per_scene, max_per_sample, allow_reviewed_overlap, allow_near_duplicates, diversity_mode)
        if reason:
            state["rejected"][reason] += 1
            continue
        _mark_novel_selected(row, state)
        return row
    return None


def _novel_reject_reason(
    row: dict[str, Any],
    state: dict[str, Any],
    max_per_scene: int,
    max_per_sample: int,
    allow_reviewed_overlap: bool,
    allow_near_duplicates: bool,
    diversity_mode: str,
) -> str | None:
    if not allow_reviewed_overlap and row.get("previously_reviewed"):
        return "previously_reviewed"
    if not allow_near_duplicates and row.get("duplicate_status") == "near_duplicate":
        return "near_duplicate"
    scene = str(row.get("source_scene_file") or "unknown")
    if max_per_scene >= 0 and state["scene"][scene] >= max_per_scene:
        return "scene_cap"
    sample = str(row.get("sample_id") or "")
    if sample and max_per_sample >= 0 and state["sample"][sample] >= max_per_sample:
        return "sample_cap"
    near = _candidate_near_group(row)
    if diversity_mode == "strict" and near in state["near"]:
        return "near_group_cap"
    return None


def _mark_novel_selected(row: dict[str, Any], state: dict[str, Any]) -> None:
    state["scene"][str(row.get("source_scene_file") or "unknown")] += 1
    if row.get("sample_id"):
        state["sample"][str(row["sample_id"])] += 1
    state["near"].add(_candidate_near_group(row))


def _review_row(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("category") or row.get("semantic_family") or "unknown")
    return {
        "review_id": f"review_{idx:03d}",
        "review_label": f"{idx:03d}_{_safe_label(category)}",
        "window_id": row.get("window_id"),
        "pair_window_id": row.get("pair_window_id"),
        "semantic_family": row.get("semantic_family") or row.get("semantic_family_guess") or "unknown",
        "category": category,
        "cowgirl_bucket": category,
        "pose_family": row.get("pose_family"),
        "pose_subtype": row.get("pose_subtype"),
        "motion_subtype": row.get("motion_subtype"),
        "phase": row.get("phase"),
        "partner_relation": row.get("partner_relation") or [],
        "contact_support": row.get("contact_support") or "unknown",
        "generation_safe": bool(row.get("generation_safe")),
        "why_selected": category,
        "source_scene_file": row.get("source_scene_file"),
        "source_scene_path": row.get("source_scene_path"),
        "technical_atom_id": row.get("technical_actor_id") or row.get("technical_atom_id"),
        "source_id": row.get("source_id"),
        "sample_id": row.get("sample_id"),
        "start_seconds": row.get("start_seconds"),
        "end_seconds": row.get("end_seconds"),
        "duration_seconds": row.get("duration_seconds") or _duration(row.get("start_seconds"), row.get("end_seconds")),
        "duplicate_status": row.get("duplicate_status") or "unique",
        "previously_reviewed": bool(row.get("previously_reviewed")),
        "duplicate_group_id": row.get("duplicate_group_id") or "",
        "duplicate_reason": row.get("duplicate_reason") or "",
        "overlaps_with_review_ids": row.get("overlaps_with_review_ids") or [],
        "review_trust_warning": row.get("review_trust_warning") or "",
        "is_human_ground_truth": False,
        "is_training_label": False,
    }


def _write_answer_sheet(path: Path, rows: list[dict[str, Any]]) -> None:
    data = {"metadata": {"audit_only": True, "is_training_label_file": False}, "reviews": {}}
    for row in rows:
        data["reviews"][row["review_id"]] = {
            "semantic_family_correct": "unknown",
            "pose_correct": "unknown",
            "motion_correct": "unknown",
            "partner_relation_correct": "unknown",
            "contact_support_correct": "unknown",
            "generation_safe_correct": "unknown",
            "notes": "",
        }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_review_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Strict Novel Semantic Review", "", "Audit-only review. No ML training labels.", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['review_label']}",
                "",
                f"- Scene: `{row.get('source_scene_file')}`",
                f"- Actor: `{row.get('technical_atom_id')}`",
                f"- Time: `{row.get('start_seconds')}` to `{row.get('end_seconds')}`",
                f"- Category: `{row.get('category')}`",
                f"- Duplicate status: `{row.get('duplicate_status')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_html(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = ["<!doctype html><meta charset='utf-8'><title>Strict Novel Review</title><h1>Strict Novel Review</h1>"]
    for row in rows:
        warn = ""
        if row.get("duplicate_status") != "unique":
            warn = "<p style='color:#b42318'>Duplicate warning</p>"
        parts.append(
            f"<section><h2>{html.escape(row['review_label'])}</h2>{warn}"
            f"<p>{html.escape(str(row.get('source_scene_file')))} "
            f"{html.escape(str(row.get('start_seconds')))}-{html.escape(str(row.get('end_seconds')))}s "
            f"{html.escape(str(row.get('category')))}</p></section>"
        )
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_quality_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Review Quality Report", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "review_folder",
        "review_id",
        "run_id",
        "source_scene_file",
        "technical_actor_id",
        "source_id",
        "sample_id",
        "window_id",
        "pair_window_id",
        "start_seconds",
        "end_seconds",
        "semantic_family_guess",
        "human_verdict",
        "human_labels",
        "duplicate_status",
        "duplicate_group_id",
        "duplicate_reason",
        "overlaps_with_review_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fields})


def _write_index_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Reviewed Window Index Report", ""]
    lines.append(f"- Reviewed records: {summary['records']}")
    lines.append(f"- Unique records: {summary['unique_count']}")
    lines.append(f"- Exact duplicates: {summary['exact_duplicate_count']}")
    lines.append(f"- Near duplicates: {summary['near_duplicate_count']}")
    lines.append(f"- Previously reviewed overlaps: {summary['previously_reviewed_count']}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {w}" for w in summary.get("warnings", [])) or lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_duplicate_audit(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Review Duplicate Audit Report",
        "",
        f"- Total reviewed items: {summary['total']}",
        f"- Unique windows/items: {summary['unique_count']}",
        f"- Exact duplicate count: {summary['exact_duplicate_count']}",
        f"- Near duplicate count: {summary['near_duplicate_count']}",
        f"- Repeated sample count: {summary['repeated_sample_count']}",
        "",
        "## Worst Review Folders",
        "",
    ]
    for folder, count in Counter(summary.get("duplicate_folder_counts", {})).most_common(10):
        lines.append(f"- `{folder}`: {count} duplicate/overlap records")
    lines.extend(["", "## Example Duplicate Groups", ""])
    for group, members in summary.get("worst_duplicate_groups", [])[:10]:
        lines.append(f"- `{group}`: {members}")
    lines.extend(["", "## Trust Guidance", ""])
    lines.append("- Reviews with exact or near duplicate groups should be treated with caution.")
    lines.append("- Future batches should use `--exclude-reviewed-index` and strict novelty caps.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _duplicate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(str(r.get("duplicate_status") or "unknown") for r in rows)
    sample_counts = Counter(str(r.get("sample_id")) for r in rows if r.get("sample_id"))
    groups = defaultdict(list)
    for r in rows:
        if r.get("duplicate_status") != "unique":
            groups[str(r.get("duplicate_group_id") or "unknown")].append(_global_review_id(r, 0))
    return {
        "total": len(rows),
        "unique_count": status.get("unique", 0),
        "exact_duplicate_count": status.get("exact_duplicate", 0),
        "near_duplicate_count": status.get("near_duplicate", 0),
        "previously_reviewed_count": status.get("previously_reviewed", 0),
        "repeated_sample_count": sum(1 for c in sample_counts.values() if c > 1),
        "status_counts": dict(status),
        "worst_duplicate_groups": sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True),
    }


def _exact_duplicate_key(
    rec: dict[str, Any],
    seen_window: dict[str, str],
    seen_pair: dict[str, str],
    seen_sample_time: dict[str, str],
    seen_source_time: dict[str, str],
) -> tuple[str | None, str | None]:
    keys = [
        ("same_window_id", str(rec.get("window_id") or ""), seen_window),
        ("same_pair_window_id", str(rec.get("pair_window_id") or ""), seen_pair),
        ("same_sample_time", _sample_time_key(rec), seen_sample_time),
        ("same_source_time", _source_time_key(rec), seen_source_time),
    ]
    for label, key, seen in keys:
        if key and key in seen:
            return label, seen[key]
    return None, None


def _mark_seen(
    rec: dict[str, Any],
    rid: str,
    seen_window: dict[str, str],
    seen_pair: dict[str, str],
    seen_sample_time: dict[str, str],
    seen_source_time: dict[str, str],
) -> None:
    if rec.get("window_id"):
        seen_window.setdefault(str(rec["window_id"]), rid)
    if rec.get("pair_window_id"):
        seen_pair.setdefault(str(rec["pair_window_id"]), rid)
    if _sample_time_key(rec):
        seen_sample_time.setdefault(_sample_time_key(rec), rid)
    if _source_time_key(rec):
        seen_source_time.setdefault(_source_time_key(rec), rid)


def _candidate_exact_overlap(row: dict[str, Any], reviewed: list[dict[str, Any]]) -> list[str]:
    out = []
    for rec in reviewed:
        if row.get("window_id") and row.get("window_id") == rec.get("window_id"):
            out.append(_global_review_id(rec, 0))
        elif row.get("pair_window_id") and row.get("pair_window_id") == rec.get("pair_window_id"):
            out.append(_global_review_id(rec, 0))
        elif _sample_time_key(row) and _sample_time_key(row) == _sample_time_key(rec):
            out.append(_global_review_id(rec, 0))
        elif _source_time_key(row) and _source_time_key(row) == _source_time_key(rec):
            out.append(_global_review_id(rec, 0))
    return _dedupe(out)


def _previously_reviewed_by_source(rec: dict[str, Any], previous: list[dict[str, Any]]) -> bool:
    return bool(_overlap_review_ids(rec, previous))


def _overlap_review_ids(rec: dict[str, Any], previous: list[dict[str, Any]]) -> list[str]:
    out = []
    for other in previous:
        if rec.get("sample_id") and rec.get("sample_id") == other.get("sample_id") and _time_overlap_ratio(rec, other) > 0.0:
            out.append(_global_review_id(other, 0))
        elif rec.get("source_id") and rec.get("source_id") == other.get("source_id") and _time_overlap_ratio(rec, other) > 0.0:
            out.append(_global_review_id(other, 0))
    return _dedupe(out)


def _near_duplicate_ids(rec: dict[str, Any], previous: list[dict[str, Any]]) -> list[str]:
    out = []
    for other in previous:
        if not _same_scene_actor(rec, other):
            continue
        same_sourceish = (rec.get("sample_id") and rec.get("sample_id") == other.get("sample_id")) or (
            rec.get("source_id") and rec.get("source_id") == other.get("source_id")
        )
        time_close = abs(_num(rec.get("start_seconds")) - _num(other.get("start_seconds"))) <= 2.0
        overlap = _time_overlap_ratio(rec, other)
        same_semantic = _same_semantic_shape(rec, other)
        if same_sourceish and (overlap > 0.5 or time_close or same_semantic):
            out.append(_global_review_id(other, 0))
        elif overlap > 0.5 and same_semantic:
            out.append(_global_review_id(other, 0))
    return _dedupe(out)


def _same_scene_actor(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(a.get("source_scene_file") and a.get("source_scene_file") == b.get("source_scene_file")) and (
        not a.get("technical_actor_id") or not b.get("technical_actor_id") or a.get("technical_actor_id") == b.get("technical_actor_id")
    )


def _same_semantic_shape(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("pose_subtype") or "") == str(b.get("pose_subtype") or "")
        and str(a.get("motion_subtype") or "") == str(b.get("motion_subtype") or "")
        and str(a.get("category") or "") == str(b.get("category") or "")
    )


def _time_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = _num_or_none(a.get("start_seconds")), _num_or_none(a.get("end_seconds"))
    b0, b1 = _num_or_none(b.get("start_seconds")), _num_or_none(b.get("end_seconds"))
    if a0 is None or a1 is None or b0 is None or b1 is None or a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = min(a1 - a0, b1 - b0)
    return inter / denom if denom > 0 else 0.0


def _sample_time_key(rec: dict[str, Any]) -> str:
    if not rec.get("sample_id"):
        return ""
    return f"{rec.get('sample_id')}|{_rounded_time(rec.get('start_seconds'))}|{_rounded_time(rec.get('end_seconds'))}"


def _source_time_key(rec: dict[str, Any]) -> str:
    if not rec.get("source_id"):
        return ""
    return f"{rec.get('source_id')}|{_rounded_time(rec.get('start_seconds'))}|{_rounded_time(rec.get('end_seconds'))}"


def _candidate_near_group(row: dict[str, Any]) -> str:
    start = round(_num(row.get("start_seconds")) / 2.0) * 2
    return "|".join(
        [
            str(row.get("source_scene_file") or ""),
            str(row.get("technical_actor_id") or row.get("technical_atom_id") or ""),
            str(row.get("source_id") or row.get("sample_id") or ""),
            str(row.get("pose_subtype") or ""),
            str(row.get("motion_subtype") or ""),
            str(start),
        ]
    )


def _duplicate_group_id(rec: dict[str, Any], status: str, reason: str, overlaps: list[str]) -> str:
    if status == "unique":
        key = _candidate_near_group(rec) or _global_review_id(rec, 0)
    else:
        key = "|".join([reason, _sample_time_key(rec), _source_time_key(rec), ",".join(overlaps)])
    return "dup_" + hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _global_review_id(row: dict[str, Any], index: int) -> str:
    folder = str(row.get("review_folder") or "")
    rid = str(row.get("review_id") or f"review_{index:05d}")
    if folder:
        return f"{Path(folder).name}:{rid}"
    return rid


def _parse_include_runs(value: str | list[str | Path]) -> list[Path]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    else:
        parts = [str(p) for p in value]
    return [Path(p) for p in parts]


def _rounded_time(value: Any) -> str:
    parsed = _num_or_none(value)
    if parsed is None:
        return ""
    return f"{parsed:.3f}"


def _num_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _num(value: Any, default: float = 0.0) -> float:
    parsed = _num_or_none(value)
    return default if parsed is None else parsed


def _duration(start: Any, end: Any) -> float | None:
    a = _num_or_none(start)
    b = _num_or_none(end)
    if a is None or b is None:
        return None
    return b - a


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, dict, set, tuple)) and not value:
        return False
    return True


def _dedupe(values: list[Any]) -> list[Any]:
    out = []
    seen = set()
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value).lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "review"


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    return value

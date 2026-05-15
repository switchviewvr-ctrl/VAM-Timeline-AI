"""Build aligned pair-window records from possible context pair candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.identity import make_pair_window_id
from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def build_pair_windows_v1(pair_candidates: str | Path, windows: str | Path, sample_index: str | Path, out: str | Path, report: str | Path) -> list[dict[str, Any]]:
    pairs = load_jsonl(pair_candidates)
    window_rows = load_jsonl(windows)
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        if row.get("sample_id"):
            by_sample[str(row["sample_id"])].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda r: (float(r.get("start_seconds") or 0.0), float(r.get("end_seconds") or 0.0)))

    out_rows: list[dict[str, Any]] = []
    for pair in pairs:
        a_id = str(pair.get("sample_id_a"))
        b_id = str(pair.get("sample_id_b"))
        a_windows = by_sample.get(a_id, [])
        b_windows = by_sample.get(b_id, [])
        if not a_windows or not b_windows:
            continue
        b_by_key = _windows_by_time_key(b_windows)
        for aw in a_windows:
            key = _time_key(aw)
            candidates = b_by_key.get(key) or _overlapping_candidates(aw, b_windows)
            for bw in candidates[:2]:
                rec = _pair_window_record(pair, aw, bw, samples.get(a_id, {}), samples.get(b_id, {}))
                if rec["time_overlap_seconds"] > 0:
                    out_rows.append(rec)
    write_jsonl(out, out_rows)
    _write_report(out_rows, report)
    return out_rows


def _windows_by_time_key(rows: list[dict[str, Any]]) -> dict[tuple[float, float, float], list[dict[str, Any]]]:
    out: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[_time_key(row)].append(row)
    return out


def _time_key(row: dict[str, Any]) -> tuple[float, float, float]:
    start = round(float(row.get("start_seconds") or 0.0), 3)
    end = round(float(row.get("end_seconds") or 0.0), 3)
    dur = round(float(row.get("duration_seconds") or (end - start)), 3)
    return start, end, dur


def _overlapping_candidates(a: dict[str, Any], b_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a0 = float(a.get("start_seconds") or 0.0)
    a1 = float(a.get("end_seconds") or 0.0)
    ad = max(float(a.get("duration_seconds") or (a1 - a0)), 1e-6)
    candidates = []
    for b in b_rows:
        b0 = float(b.get("start_seconds") or 0.0)
        b1 = float(b.get("end_seconds") or 0.0)
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        if overlap <= 0:
            continue
        bd = max(float(b.get("duration_seconds") or (b1 - b0)), 1e-6)
        score = overlap / max(ad, bd)
        if score >= 0.75:
            candidates.append((score, b))
    return [b for _, b in sorted(candidates, key=lambda item: item[0], reverse=True)]


def _pair_window_record(pair: dict[str, Any], aw: dict[str, Any], bw: dict[str, Any], sample_a: dict[str, Any], sample_b: dict[str, Any]) -> dict[str, Any]:
    a0 = float(aw.get("start_seconds") or 0.0)
    a1 = float(aw.get("end_seconds") or 0.0)
    b0 = float(bw.get("start_seconds") or 0.0)
    b1 = float(bw.get("end_seconds") or 0.0)
    start = max(a0, b0)
    end = min(a1, b1)
    overlap = max(0.0, end - start)
    warnings = []
    if abs((a1 - a0) - (b1 - b0)) > 0.1:
        warnings.append("paired windows have different durations")
    if sample_a.get("fps") and sample_b.get("fps") and abs(float(sample_a["fps"]) - float(sample_b["fps"])) > 0.01:
        warnings.append("paired samples have different fps")
    pid = str(pair.get("pair_id"))
    pair_window_id = make_pair_window_id(pid, str(aw.get("window_id")), str(bw.get("window_id")), start, end)
    return {
        "pair_window_id": pair_window_id,
        "pair_id": pair.get("pair_id"),
        "source_scene_file": pair.get("source_scene_file"),
        "sample_id_a": pair.get("sample_id_a"),
        "sample_id_b": pair.get("sample_id_b"),
        "technical_atom_id_a": pair.get("technical_atom_id_a"),
        "technical_atom_id_b": pair.get("technical_atom_id_b"),
        "window_id_a": aw.get("window_id"),
        "window_id_b": bw.get("window_id"),
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": overlap,
        "time_overlap_seconds": overlap,
        "frame_start_a": aw.get("frame_start"),
        "frame_end_a": aw.get("frame_end"),
        "frame_start_b": bw.get("frame_start"),
        "frame_end_b": bw.get("frame_end"),
        "clip_name_a": pair.get("clip_name_a"),
        "clip_name_b": pair.get("clip_name_b"),
        "pair_confidence": pair.get("pair_confidence"),
        "pairing_reasons": pair.get("pairing_reasons", []),
        "semantic_role_a": "unknown",
        "semantic_role_b": "unknown",
        "warnings": warnings,
    }


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    scene_counts = Counter(r.get("source_scene_file") for r in rows)
    pair_counts = Counter(r.get("pair_id") for r in rows)
    overlaps = np.asarray([r.get("time_overlap_seconds", 0.0) for r in rows], dtype=np.float64)
    lines = [
        "# Pair Windows v1",
        "",
        "Pair windows align two technical actor/sample windows in time. Rider/receiver roles remain unknown.",
        "",
        f"- Total pair windows: {len(rows)}",
        f"- Unique sample pairs: {len(pair_counts)}",
        f"- Median overlap seconds: {float(np.median(overlaps)) if overlaps.size else 0.0:.3f}",
        f"- High-quality pairs (confidence >= 0.8): {sum(1 for r in rows if float(r.get('pair_confidence') or 0.0) >= 0.8)}",
        "",
        "## Pair Windows Per Scene",
        "",
    ]
    for scene, count in scene_counts.most_common(30):
        lines.append(f"- `{scene}`: {count}")
    lines.extend(["", "## Top Sample Pairs", ""])
    for pair_id, count in pair_counts.most_common(30):
        lines.append(f"- `{pair_id}`: {count}")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

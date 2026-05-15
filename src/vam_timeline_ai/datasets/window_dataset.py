"""Build movement-window records from baked motion samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.identity import make_window_id
from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.motion.windows import make_default_window_set


def build_movement_windows(sample_index: str | Path, out: str | Path) -> list[dict[str, Any]]:
    samples = _load_jsonl(sample_index)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        ok = sample.get("bake_status") == "ok" and sample.get("baked_npz_path")
        if not ok:
            continue
        duration = float(sample.get("duration_seconds") or 0.0)
        fps = float(sample.get("fps") or 60.0)
        for group_name, windows in make_default_window_set(duration).items():
            window_size, stride = _parse_group(group_name)
            for idx, (start, end) in enumerate(windows):
                rows.append(
                    {
                        "window_id": make_window_id(sample["sample_id"], start, end, window_size, stride, fps),
                        "sample_id": sample["sample_id"],
                        "source_id": sample.get("source_id"),
                        "source_scene_file": sample.get("source_scene_file"),
                        "source_scene_path": sample.get("source_scene_path"),
                        "technical_atom_id": sample.get("technical_atom_id"),
                        "start_seconds": start,
                        "end_seconds": end,
                        "duration_seconds": round(end - start, 3),
                        "window_size_seconds": window_size,
                        "stride_seconds": stride,
                        "window_group": group_name,
                        "window_index": idx,
                        "fps": fps,
                        "frame_start": int(round(start * fps)),
                        "frame_end": int(round(end * fps)),
                        "semantic_role_guess": "unknown",
                        "labels": [],
                        "needs_manual_review": True,
                        "include_for_ml": True,
                        "warnings": [],
                    }
                )
    write_jsonl(out, rows)
    return rows


def _parse_group(group_name: str) -> tuple[float, float]:
    if group_name == "2s_stride_1s":
        return 2.0, 1.0
    if group_name == "4s_stride_2s":
        return 4.0, 2.0
    if group_name == "8s_stride_4s":
        return 8.0, 4.0
    return 0.0, 0.0


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

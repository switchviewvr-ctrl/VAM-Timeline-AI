"""Audit baked NPZ samples before trusting them for ML."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import write_jsonl


def audit_baked_samples(sample_index: str | Path, out_jsonl: str | Path, report: str | Path) -> list[dict[str, Any]]:
    samples = _load_jsonl(sample_index)
    rows = [_audit_one(sample) for sample in samples if sample.get("bake_status") == "ok"]
    write_jsonl(out_jsonl, rows)
    _write_report(rows, report, len(samples))
    return rows


def _audit_one(sample: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    base = {
        "sample_id": sample.get("sample_id"),
        "source_id": sample.get("source_id"),
        "source_scene_file": sample.get("source_scene_file"),
        "technical_atom_id": sample.get("technical_atom_id"),
        "baked_npz_path": sample.get("baked_npz_path"),
        "fps": sample.get("fps"),
        "frame_count": sample.get("frame_count"),
        "duration_seconds": sample.get("duration_seconds"),
        "controller_count": 0,
        "controller_names": [],
        "positions_shape": None,
        "rotations_shape": None,
        "has_positions": False,
        "has_rotations": False,
        "nan_count": 0,
        "inf_count": 0,
        "static_position_channel_count": 0,
        "static_rotation_channel_count": 0,
        "moving_controller_count": 0,
        "total_position_range": None,
        "total_rotation_range_proxy": None,
        "mean_position_speed": None,
        "max_position_speed": None,
        "suspiciously_static": False,
        "suspiciously_huge_motion": False,
        "suspiciously_short": False,
        "audit_status": "unknown",
        "warnings": warnings,
    }
    path = sample.get("baked_npz_path")
    if not path or not Path(path).exists():
        base["audit_status"] = "missing_npz"
        warnings.append("baked_npz_path missing or unreadable")
        return base
    try:
        with np.load(path, allow_pickle=True) as data:
            positions = data["positions"] if "positions" in data else np.zeros((0, 0, 3), dtype=np.float32)
            rotations = data["rotations"] if "rotations" in data else np.zeros((0, 0, 4), dtype=np.float32)
            times = data["times"] if "times" in data else np.zeros((0,), dtype=np.float32)
            controller_names = [str(v) for v in data["controller_names"].tolist()] if "controller_names" in data else []
        base.update(
            {
                "positions_shape": list(positions.shape),
                "rotations_shape": list(rotations.shape),
                "has_positions": positions.ndim == 3 and positions.shape[-1] == 3 and positions.size > 0,
                "has_rotations": rotations.ndim == 3 and rotations.shape[-1] == 4 and rotations.size > 0,
                "controller_count": len(controller_names),
                "controller_names": controller_names,
                "frame_count": int(len(times)),
            }
        )
        all_arrays = [positions, rotations, times]
        base["nan_count"] = int(sum(np.isnan(a).sum() for a in all_arrays if np.issubdtype(a.dtype, np.number)))
        base["inf_count"] = int(sum(np.isinf(a).sum() for a in all_arrays if np.issubdtype(a.dtype, np.number)))
        if base["has_positions"]:
            pos_range_by_controller = np.linalg.norm(np.ptp(positions, axis=0), axis=1)
            base["static_position_channel_count"] = int(np.sum(pos_range_by_controller < 1e-4))
            base["moving_controller_count"] = int(np.sum(pos_range_by_controller >= 1e-4))
            base["total_position_range"] = float(np.linalg.norm(np.ptp(positions.reshape(-1, 3), axis=0)))
            speed = _speed(positions, times)
            base["mean_position_speed"] = float(np.mean(speed)) if speed.size else 0.0
            base["max_position_speed"] = float(np.max(speed)) if speed.size else 0.0
        if base["has_rotations"]:
            rot_range_by_controller = np.linalg.norm(np.ptp(rotations, axis=0), axis=1)
            base["static_rotation_channel_count"] = int(np.sum(rot_range_by_controller < 1e-4))
            base["total_rotation_range_proxy"] = float(np.linalg.norm(np.ptp(rotations.reshape(-1, 4), axis=0)))
        base["suspiciously_static"] = bool((base["moving_controller_count"] or 0) == 0 or (base["mean_position_speed"] or 0.0) < 1e-4)
        base["suspiciously_huge_motion"] = bool((base["total_position_range"] or 0.0) > 25.0 or (base["max_position_speed"] or 0.0) > 25.0)
        base["suspiciously_short"] = bool((base["duration_seconds"] or 0.0) < 0.25 or (base["frame_count"] or 0) < 5)
        if base["nan_count"] or base["inf_count"]:
            base["audit_status"] = "error"
            warnings.append("arrays contain NaN or Inf")
        elif base["suspiciously_static"] or base["suspiciously_huge_motion"] or base["suspiciously_short"]:
            base["audit_status"] = "warning"
        else:
            base["audit_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        base["audit_status"] = "read_error"
        warnings.append(str(exc))
    return base


def _speed(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(times) < 2 or positions.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    dt = np.diff(times.astype(np.float64))
    dt = np.where(dt <= 0, 1.0 / 60.0, dt)
    vel = np.diff(positions.astype(np.float64), axis=0) / dt[:, None, None]
    return np.linalg.norm(vel, axis=2).reshape(-1)


def _write_report(rows: list[dict[str, Any]], report: str | Path, total_sample_records: int) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["audit_status"] for row in rows)
    controller_counts = Counter(row.get("controller_count", 0) for row in rows)
    controller_names = Counter(name for row in rows for name in row.get("controller_names", []))
    real_motion = sum(1 for row in rows if row.get("moving_controller_count", 0) > 0 and not row.get("suspiciously_static"))
    lines = [
        "# Baked Sample Audit",
        "",
        f"- Sample index records: {total_sample_records}",
        f"- Successfully baked samples audited: {len(rows)}",
        f"- Readable/motion-positive samples: {real_motion}",
        f"- Audit statuses: {dict(status_counts)}",
        f"- Suspicious static samples: {sum(1 for r in rows if r.get('suspiciously_static'))}",
        f"- Suspicious huge-motion samples: {sum(1 for r in rows if r.get('suspiciously_huge_motion'))}",
        "",
        "## Common Controller Counts",
        "",
    ]
    for count, freq in controller_counts.most_common(12):
        lines.append(f"- {count} controllers: {freq} samples")
    lines.extend(["", "## Top Controller Names", ""])
    for name, freq in controller_names.most_common(30):
        lines.append(f"- `{name}`: {freq}")
    lines.extend(["", "## Samples Needing Manual Inspection", ""])
    for row in [r for r in rows if r.get("audit_status") != "ok"][:50]:
        lines.append(f"- `{row.get('sample_id')}` ({row.get('audit_status')}): {', '.join(row.get('warnings', []))}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

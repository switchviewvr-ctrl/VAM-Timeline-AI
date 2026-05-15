"""Cowgirl/Riding feature extraction v0.

The first version computes real numeric root/pelvis-like motion features and
reports missing higher-level semantics honestly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import write_jsonl


ROOT_CONTROLLER_CANDIDATES = ("pelvisControl", "hipControl", "abdomen2Control", "chestControl")
FEATURE_NAMES = [
    "vertical_amplitude",
    "forward_back_amplitude",
    "lateral_amplitude",
    "movement_energy",
    "mean_speed",
    "max_speed",
    "speed_std",
    "acceleration_peak_count",
    "rhythm_regularity_proxy",
    "pause_ratio",
    "tempo_proxy",
]


def extract_cowgirl_features_v0(windows_path: str | Path, sample_index_path: str | Path, out_jsonl: str | Path, out_npz: str | Path, report: str | Path) -> list[dict[str, Any]]:
    windows = _load_jsonl(windows_path)
    samples = {row["sample_id"]: row for row in _load_jsonl(sample_index_path) if row.get("sample_id")}
    rows: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    window_ids: list[str] = []
    sample_ids: list[str] = []
    data_cache: dict[str, Any] = {}
    for window in windows:
        row = _extract_one(window, samples.get(window.get("sample_id")), data_cache)
        rows.append(row)
        matrix.append([row["features"].get(name, np.nan) for name in FEATURE_NAMES])
        window_ids.append(row["window_id"])
        sample_ids.append(str(row.get("sample_id")))
    write_jsonl(out_jsonl, rows)
    X = np.asarray(matrix, dtype=np.float32) if matrix else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=X,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        window_ids=np.asarray(window_ids, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
        metadata_json=json.dumps({"feature_version": "cowgirl_v0", "row_count": len(rows)}, ensure_ascii=False),
    )
    _write_report(rows, report)
    return rows


def _extract_one(window: dict[str, Any], sample: dict[str, Any] | None, data_cache: dict[str, Any] | None = None) -> dict[str, Any]:
    warnings = list(window.get("warnings", []))
    features = {name: np.nan for name in FEATURE_NAMES}
    used_controllers: list[str] = []
    quality = {"has_numeric_features": False, "mapping_confidence": "none"}
    if sample is None or sample.get("bake_status") != "ok" or not sample.get("baked_npz_path"):
        warnings.append("no baked sample available")
    else:
        try:
            data = _load_sample_arrays(sample, window.get("sample_id"), data_cache=data_cache)
            positions = data["positions"]
            times = data["times"]
            controller_names = data["controller_names"]
            idx, confidence = _root_controller_index(controller_names)
            if idx is None:
                warnings.append("no pelvis/root-like controller found")
            else:
                start = int(window.get("frame_start") or 0)
                end = int(window.get("frame_end") or 0)
                start = max(0, min(start, positions.shape[0] - 1))
                end = max(start + 1, min(end, positions.shape[0]))
                pos = positions[start:end, idx, :]
                t = times[start:end]
                used_controllers = [controller_names[idx]]
                features.update(_root_features(pos, t))
                quality = {"has_numeric_features": True, "mapping_confidence": confidence}
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"feature extraction failed: {exc}")
    return {
        "window_id": window.get("window_id"),
        "sample_id": window.get("sample_id"),
        "source_id": window.get("source_id"),
        "source_scene_file": window.get("source_scene_file"),
        "technical_atom_id": window.get("technical_atom_id"),
        "start_seconds": window.get("start_seconds"),
        "end_seconds": window.get("end_seconds"),
        "feature_version": "cowgirl_v0",
        "features": features,
        "feature_quality": quality,
        "controller_mapping": {"root_like_controller_used": used_controllers},
        "missing_feature_groups": ["torso", "hands", "legs", "head_gaze"],
        "warnings": warnings + ["torso/head/hand/leg semantic features are schema-only in v0"],
    }


def _load_sample_arrays(sample: dict[str, Any], sample_id: str | None, data_cache: dict[str, Any] | None) -> dict[str, Any]:
    cache = data_cache if data_cache is not None else {}
    key = str(sample_id or sample.get("sample_id") or sample.get("baked_npz_path"))
    if key in cache:
        return cache[key]
    loaded = np.load(sample["baked_npz_path"], allow_pickle=True)
    arrays = {
        "positions": loaded["positions"],
        "times": loaded["times"],
        "controller_names": [str(v) for v in loaded["controller_names"].tolist()],
    }
    loaded.close()
    if data_cache is not None:
        if len(cache) > 8:
            cache.clear()
        cache[key] = arrays
    return arrays


def _root_features(pos: np.ndarray, times: np.ndarray) -> dict[str, float]:
    if len(pos) < 2:
        return {name: np.nan for name in FEATURE_NAMES}
    span = np.ptp(pos, axis=0)
    dt = np.diff(times.astype(np.float64))
    dt = np.where(dt <= 0.0, 1.0 / 60.0, dt)
    vel = np.diff(pos, axis=0) / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)
    accel = np.diff(speed) / np.where(dt[1:] <= 0.0, 1.0 / 60.0, dt[1:]) if len(speed) > 1 else np.zeros((0,))
    speed_mean = float(np.mean(speed)) if len(speed) else 0.0
    speed_std = float(np.std(speed)) if len(speed) else 0.0
    pause_ratio = float(np.mean(speed < max(0.03, speed_mean * 0.15))) if len(speed) else 1.0
    energy = float(np.mean(speed**2)) if len(speed) else 0.0
    peaks = int(np.sum(accel > (np.mean(accel) + np.std(accel)))) if len(accel) else 0
    regularity = float(1.0 / (1.0 + (speed_std / (speed_mean + 1e-6))))
    tempo_proxy = float(_tempo_proxy(pos[:, 1], times))
    return {
        "vertical_amplitude": float(span[1]),
        "forward_back_amplitude": float(span[2]),
        "lateral_amplitude": float(span[0]),
        "movement_energy": energy,
        "mean_speed": speed_mean,
        "max_speed": float(np.max(speed)) if len(speed) else 0.0,
        "speed_std": speed_std,
        "acceleration_peak_count": float(peaks),
        "rhythm_regularity_proxy": regularity,
        "pause_ratio": pause_ratio,
        "tempo_proxy": tempo_proxy,
    }


def _tempo_proxy(y: np.ndarray, times: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    centered = y - np.mean(y)
    signs = np.sign(centered)
    crossings = np.sum((signs[1:] * signs[:-1]) < 0)
    duration = float(times[-1] - times[0]) if len(times) else 0.0
    if duration <= 0:
        return 0.0
    cycles = crossings / 2.0
    return float((cycles / duration) * 60.0)


def _root_controller_index(controller_names: list[str]) -> tuple[int | None, str]:
    for name in ROOT_CONTROLLER_CANDIDATES:
        if name in controller_names:
            return controller_names.index(name), "medium" if name != "pelvisControl" else "high"
    return (0, "low") if controller_names else (None, "none")


def _write_report(rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    numeric = sum(1 for row in rows if row.get("feature_quality", {}).get("has_numeric_features"))
    lines = [
        "# Cowgirl Feature Report v0",
        "",
        f"- Feature rows: {len(rows)}",
        f"- Rows with numeric root/pelvis-like features: {numeric}",
        "- Torso/head/hands/legs: not computed in v0; null/NaN with warnings.",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

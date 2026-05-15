"""Technical 60 Hz baking utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.identity import make_sample_id
from vam_timeline_ai.io.json_utils import load_json, safe_id_for_path, to_jsonable
from vam_timeline_ai.motion.native_motion import bake_native_source
from vam_timeline_ai.motion.quaternion_utils import angular_deltas, ensure_quat_continuity, quat_slerp
from vam_timeline_ai.timeline.parser import bake_timeline_source


EXTRACTION_VERSION = "extract_v2"


def make_time_grid(duration: float, fps: float = 60.0, loop: bool = False) -> np.ndarray:
    if duration <= 0:
        return np.zeros((0,), dtype=np.float32)
    step = 1.0 / fps
    times = np.arange(0.0, duration, step, dtype=np.float64)
    if not loop and (times.size == 0 or not np.isclose(times[-1], duration, rtol=0.0, atol=1e-6)):
        times = np.append(times, duration)
    return times.astype(np.float32)


def interpolate_positions(source_times: np.ndarray, positions: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    src_t = np.asarray(source_times, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    target = np.asarray(target_times, dtype=np.float64)
    if len(src_t) == 0:
        return np.zeros((len(target), 3), dtype=np.float32)
    if len(src_t) == 1:
        return np.repeat(pos[:1], len(target), axis=0).astype(np.float32)
    out = np.empty((len(target), 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(target, src_t, pos[:, axis], left=pos[0, axis], right=pos[-1, axis])
    return out


def resample_quaternions(source_times: np.ndarray, rotations: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    src_t = np.asarray(source_times, dtype=np.float64)
    rots = ensure_quat_continuity(np.asarray(rotations, dtype=np.float64))
    target = np.asarray(target_times, dtype=np.float64)
    if len(src_t) == 0:
        return np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (len(target), 1))
    if len(src_t) == 1:
        return np.repeat(rots[:1], len(target), axis=0).astype(np.float32)
    out = np.empty((len(target), 4), dtype=np.float32)
    for i, t in enumerate(target):
        if t <= src_t[0]:
            out[i] = rots[0]
        elif t >= src_t[-1]:
            out[i] = rots[-1]
        else:
            hi = int(np.searchsorted(src_t, t, side="right"))
            lo = hi - 1
            alpha = float((t - src_t[lo]) / (src_t[hi] - src_t[lo]))
            out[i] = quat_slerp(rots[lo], rots[hi], alpha)
    return ensure_quat_continuity(out)


def compute_velocities(positions: np.ndarray, times: np.ndarray, loop: bool = False) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float32)
    t = np.asarray(times, dtype=np.float64)
    out = np.zeros_like(pos, dtype=np.float32)
    if len(t) < 2:
        return out
    dt = np.diff(t)
    dt = np.where(dt <= 0.0, 1.0, dt)
    out[1:] = (pos[1:] - pos[:-1]) / dt[:, None, None]
    out[0] = out[1]
    return out


def compute_angular_delta_array(rotations: np.ndarray, loop: bool = False) -> np.ndarray:
    rots = np.asarray(rotations, dtype=np.float32)
    out = np.zeros_like(rots, dtype=np.float32)
    if rots.ndim != 3:
        return out
    for controller_idx in range(rots.shape[1]):
        out[:, controller_idx, :] = angular_deltas(rots[:, controller_idx, :], loop=loop)
    return out


def phase_from_times(times: np.ndarray, duration: float) -> np.ndarray:
    if duration <= 0:
        return np.zeros_like(times, dtype=np.float32)
    return (times.astype(np.float64) / duration).astype(np.float32)


def extract_motion_samples(source_index_path: str | Path, out_dir: str | Path, index_out: str | Path, fps: float = 60.0) -> list[dict[str, Any]]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    source_rows = _load_jsonl(source_index_path)
    scene_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    index_path = Path(index_out)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        for source in source_rows:
            record = _base_sample_record(source, fps)
            if source.get("source_type") not in {"timeline_controller_motion", "native_motion_animation"}:
                record["bake_status"] = "not_bakeable"
                record["warnings"].append("source type is metadata/scalar/trigger-only for this extractor")
            else:
                existing = _existing_sample_record(source, out_path, fps)
                if existing is not None:
                    record.update(existing)
                elif _too_large_for_v0(source, fps):
                    record["bake_status"] = "failed"
                    record["warnings"].append("source exceeds v0 full-clip bake budget; keep for future chunked extraction")
                else:
                    try:
                        scene_path = source.get("source_scene_path")
                        if scene_path not in scene_cache:
                            scene_cache[scene_path] = load_json(scene_path)
                        scene_data = scene_cache[scene_path]
                        if source["source_type"] == "timeline_controller_motion":
                            baked = bake_timeline_source(scene_data, source, fps=fps)
                        else:
                            baked = bake_native_source(scene_data, source, fps=fps)
                        sample_id = record["sample_id"]
                        baked["sample_id"] = sample_id
                        npz_path = out_path / f"{safe_id_for_path(sample_id)}.npz"
                        np.savez_compressed(
                            npz_path,
                            times=baked["times"],
                            positions=baked["positions"],
                            rotations=baked["rotations"],
                            velocities=baked["velocities"],
                            angular_deltas=baked["angular_deltas"],
                            controller_names=np.asarray(baked["controller_names"], dtype=object),
                            metadata_json=json.dumps(baked["metadata"], ensure_ascii=False),
                        )
                        record.update(_record_from_npz(sample_id, npz_path, baked["duration_seconds"], fps))
                        record["warnings"] = record["warnings"] + baked.get("warnings", [])
                    except Exception as exc:  # noqa: BLE001 - extraction should report failures
                        record["bake_status"] = "failed"
                        record["warnings"].append(str(exc))
            rows.append(record)
            json.dump(to_jsonable(record), f, ensure_ascii=False)
            f.write("\n")
    return rows


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _base_sample_record(source: dict[str, Any], fps: float) -> dict[str, Any]:
    sample_id = _expected_sample_id(source, fps=fps)
    return {
        "sample_id": sample_id,
        "source_id": source.get("source_id"),
        "source_type": source.get("source_type"),
        "source_scene_file": source.get("source_scene_file"),
        "source_scene_path": source.get("source_scene_path"),
        "technical_atom_id": source.get("technical_atom_id"),
        "storable_id": source.get("storable_id"),
        "clip_name": source.get("clip_name"),
        "clip_index": source.get("clip_index"),
        "extraction_version": EXTRACTION_VERSION,
        "fps": float(fps),
        "frame_count": 0,
        "duration_seconds": source.get("duration_seconds"),
        "controller_names": source.get("controller_names", []),
        "channels_available": {},
        "baked_npz_path": None,
        "parse_status": source.get("parse_status", "unknown"),
        "bake_status": "pending",
        "warnings": list(source.get("warnings", [])),
    }


def _existing_sample_record(source: dict[str, Any], out_path: Path, fps: float) -> dict[str, Any] | None:
    sample_id = _expected_sample_id(source, fps=fps)
    if not sample_id:
        return None
    npz_path = out_path / f"{safe_id_for_path(sample_id)}.npz"
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            duration = float(data["times"][-1]) if len(data["times"]) else float(source.get("duration_seconds") or 0.0)
            return _record_from_npz(sample_id, npz_path, duration, fps, data=data)
    except Exception:
        return None


def _expected_sample_id(source: dict[str, Any], fps: float) -> str:
    return make_sample_id(
        str(source.get("source_id") or "unknown_source"),
        fps=fps,
        extraction_version=EXTRACTION_VERSION,
        technical_atom_id=source.get("technical_atom_id"),
        clip_name=source.get("clip_name") or source.get("source_type"),
        clip_index=source.get("clip_index"),
    )


def _record_from_npz(sample_id: str, npz_path: Path, duration: float, fps: float, data: Any | None = None) -> dict[str, Any]:
    close_data = False
    if data is None:
        data = np.load(npz_path, allow_pickle=True)
        close_data = True
    try:
        controller_names = [str(x) for x in data["controller_names"].tolist()] if "controller_names" in data else []
        frame_count = int(len(data["times"])) if "times" in data else 0
    finally:
        if close_data:
            data.close()
    return {
        "sample_id": sample_id,
        "fps": float(fps),
        "frame_count": frame_count,
        "duration_seconds": float(duration),
        "controller_names": controller_names,
        "channels_available": {
            "positions": True,
            "rotations": True,
            "velocities": True,
            "angular_deltas": True,
        },
        "baked_npz_path": str(npz_path),
        "bake_status": "ok",
    }


def _too_large_for_v0(source: dict[str, Any], fps: float) -> bool:
    duration = source.get("duration_seconds")
    controller_count = int(source.get("controller_count") or 1)
    if duration is None:
        return False
    duration = float(duration)
    if duration > 180.0:
        return True
    estimated_values = duration * fps * max(controller_count, 1)
    return estimated_values > 175_000

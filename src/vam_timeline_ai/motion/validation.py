"""Basic technical validation for baked motion arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from vam_timeline_ai.motion.quaternion_utils import quat_norm_stats


def validate_baked_arrays(positions: np.ndarray, rotations: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    warnings: list[str] = []
    status = "ok"
    if positions.ndim != 3 or positions.shape[-1] != 3:
        status = "error"
        warnings.append("positions must have shape [frames, controllers, 3]")
    if rotations.ndim != 3 or rotations.shape[-1] != 4:
        status = "error"
        warnings.append("rotations must have shape [frames, controllers, 4]")
    if positions.shape[:2] != rotations.shape[:2]:
        status = "error"
        warnings.append("positions/rotations frame-controller dimensions differ")
    if len(times) != positions.shape[0]:
        status = "error"
        warnings.append("times length differs from frame count")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        status = "error"
        warnings.append("arrays contain NaN or Inf")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        status = "error"
        warnings.append("times are not strictly increasing")
    qstats = quat_norm_stats(rotations) if rotations.size else {"min": 0.0, "max": 0.0, "mean": 0.0}
    if qstats["min"] < 0.98 or qstats["max"] > 1.02:
        if status == "ok":
            status = "warning"
        warnings.append("quaternion norms outside [0.98, 1.02]")
    return {
        "status": status,
        "frame_count": int(positions.shape[0]) if positions.ndim else 0,
        "controller_count": int(positions.shape[1]) if positions.ndim >= 2 else 0,
        "quat_norm_stats": qstats,
        "warnings": warnings,
    }

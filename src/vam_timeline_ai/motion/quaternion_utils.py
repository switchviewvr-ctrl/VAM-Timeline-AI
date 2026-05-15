"""Quaternion utilities for technical motion baking."""

from __future__ import annotations

import math

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def quat_dot(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(np.dot(np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64)))


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    return np.asarray([-arr[0], -arr[1], -arr[2], arr[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64)
    return quat_normalize(
        np.asarray(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    norm_sq = float(np.dot(arr, arr))
    if norm_sq == 0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat_conjugate(arr) / norm_sq).astype(np.float32)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0n = quat_normalize(q0).astype(np.float64)
    q1n = quat_normalize(q1).astype(np.float64)
    dot = float(np.dot(q0n, q1n))
    if dot < 0.0:
        q1n = -q1n
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return quat_normalize(q0n + alpha * (q1n - q0n))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return quat_normalize((s0 * q0n) + (s1 * q1n))


def ensure_quat_continuity(quats: np.ndarray) -> np.ndarray:
    arr = quat_normalize(np.asarray(quats, dtype=np.float64)).astype(np.float32, copy=True)
    for i in range(1, len(arr)):
        if float(np.dot(arr[i], arr[i - 1])) < 0.0:
            arr[i] *= -1.0
    return arr


def angular_delta_quat(q_prev: np.ndarray, q_curr: np.ndarray) -> np.ndarray:
    return quat_multiply(q_curr, quat_inverse(q_prev))


def angular_deltas(quats: np.ndarray, loop: bool = False) -> np.ndarray:
    arr = ensure_quat_continuity(quats)
    out = np.zeros_like(arr, dtype=np.float32)
    if len(arr) == 0:
        return out
    if len(arr) == 1:
        out[0] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return out
    for i in range(1, len(arr)):
        out[i] = angular_delta_quat(arr[i - 1], arr[i])
    out[0] = angular_delta_quat(arr[-1], arr[0]) if loop else out[1]
    return out


def quat_norm_stats(quats: np.ndarray) -> dict[str, float]:
    arr = np.asarray(quats, dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    norms = np.linalg.norm(arr.reshape(-1, 4), axis=1)
    return {"min": float(norms.min()), "max": float(norms.max()), "mean": float(norms.mean())}

"""Parametric relative trajectory synthesis."""

from __future__ import annotations

from typing import Any

import numpy as np


def synthesize_oval_grind(
    duration: float,
    fps: float,
    cycles: float,
    amplitude_forward_back: float,
    amplitude_lateral: float,
    amplitude_vertical: float,
    phase_offset: float = 0.0,
    smoothness: float = 1.0,
    irregularity: float = 0.02,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    times, theta = _time_theta(duration, fps, cycles, phase_offset)
    x = amplitude_lateral * np.cos(theta)
    z = amplitude_forward_back * np.sin(theta)
    y = amplitude_vertical * np.sin(theta + np.pi / 3.0) * 0.45
    path = np.stack([x, y, z], axis=1)
    return times, _center(_add_smooth_irregularity(path, irregularity, seed, smoothness))


def synthesize_circular_grind(duration: float, fps: float, cycles: float, amplitude_forward_back: float, amplitude_lateral: float, amplitude_vertical: float, phase_offset: float = 0.0, smoothness: float = 1.0, irregularity: float = 0.015, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    amp = (float(amplitude_forward_back) + float(amplitude_lateral)) / 2.0
    return synthesize_oval_grind(duration, fps, cycles, amp, amp, amplitude_vertical, phase_offset, smoothness, irregularity, seed)


def synthesize_vertical_bounce(duration: float, fps: float, cycles: float, amplitude_forward_back: float, amplitude_lateral: float, amplitude_vertical: float, phase_offset: float = 0.0, smoothness: float = 1.0, irregularity: float = 0.015, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    times, theta = _time_theta(duration, fps, cycles, phase_offset)
    y = amplitude_vertical * np.sin(theta)
    z = amplitude_forward_back * 0.18 * np.sin(theta + np.pi / 2.0)
    x = amplitude_lateral * 0.10 * np.sin(theta * 0.5)
    path = np.stack([x, y, z], axis=1)
    return times, _center(_add_smooth_irregularity(path, irregularity, seed, smoothness))


def synthesize_forward_back_rock(duration: float, fps: float, cycles: float, amplitude_forward_back: float, amplitude_lateral: float, amplitude_vertical: float, phase_offset: float = 0.0, smoothness: float = 1.0, irregularity: float = 0.015, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    times, theta = _time_theta(duration, fps, cycles, phase_offset)
    z = amplitude_forward_back * np.sin(theta)
    y = amplitude_vertical * 0.30 * np.sin(theta + np.pi / 4.0)
    x = amplitude_lateral * 0.12 * np.sin(theta * 0.5)
    path = np.stack([x, y, z], axis=1)
    return times, _center(_add_smooth_irregularity(path, irregularity, seed, smoothness))


def synthesize_riding_general(duration: float, fps: float, cycles: float, amplitude_forward_back: float, amplitude_lateral: float, amplitude_vertical: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    times, oval = synthesize_oval_grind(duration, fps, cycles, amplitude_forward_back * 0.55, amplitude_lateral * 0.55, amplitude_vertical * 0.45, seed=seed)
    _, bounce = synthesize_vertical_bounce(duration, fps, cycles, amplitude_forward_back, amplitude_lateral, amplitude_vertical, seed=(seed or 0) + 17)
    return times, _center(oval * 0.45 + bounce * 0.55)


def synthesize_path(subtype: str, duration: float, fps: float, cycles: float, amplitude_forward_back: float, amplitude_lateral: float, amplitude_vertical: float, seed: int | None = None, irregularity: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    if subtype in {"oval_grind", "grinding"}:
        return synthesize_oval_grind(duration, fps, cycles, amplitude_forward_back, amplitude_lateral, amplitude_vertical, seed=seed, irregularity=irregularity)
    if subtype == "circular_grind":
        return synthesize_circular_grind(duration, fps, cycles, amplitude_forward_back, amplitude_lateral, amplitude_vertical, seed=seed, irregularity=irregularity)
    if subtype == "vertical_bounce":
        return synthesize_vertical_bounce(duration, fps, cycles, amplitude_forward_back, amplitude_lateral, amplitude_vertical, seed=seed, irregularity=irregularity)
    if subtype == "forward_back_rock":
        return synthesize_forward_back_rock(duration, fps, cycles, amplitude_forward_back, amplitude_lateral, amplitude_vertical, seed=seed, irregularity=irregularity)
    return synthesize_riding_general(duration, fps, cycles, amplitude_forward_back, amplitude_lateral, amplitude_vertical, seed=seed)


def _time_theta(duration: float, fps: float, cycles: float, phase_offset: float) -> tuple[np.ndarray, np.ndarray]:
    frames = max(2, int(round(float(duration) * float(fps))) + 1)
    times = np.linspace(0.0, float(duration), frames, dtype=np.float32)
    theta = (times / max(float(duration), 1e-6)) * (2.0 * np.pi * float(cycles)) + float(phase_offset)
    return times, theta


def _center(path: np.ndarray) -> np.ndarray:
    return (path - np.mean(path, axis=0, keepdims=True)).astype(np.float32)


def _add_smooth_irregularity(path: np.ndarray, irregularity: float, seed: int | None, smoothness: float) -> np.ndarray:
    if irregularity <= 0:
        return path
    rng = np.random.default_rng(seed)
    n = path.shape[0]
    anchors = max(4, min(16, n // 20))
    xp = np.linspace(0, n - 1, anchors)
    noise = rng.normal(0.0, irregularity, size=(anchors, 3))
    x = np.arange(n)
    smooth = np.stack([np.interp(x, xp, noise[:, i]) for i in range(3)], axis=1)
    return path + smooth * max(0.0, min(1.0, smoothness))


def path_stats(path: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    diffs = np.diff(path, axis=0)
    dt = np.diff(times)
    speeds = np.linalg.norm(diffs, axis=1) / np.maximum(dt, 1e-6)
    return {
        "max_abs_delta": float(np.max(np.abs(path))) if path.size else 0.0,
        "path_length": float(np.sum(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0,
        "speed_mean": float(np.mean(speeds)) if len(speeds) else 0.0,
        "speed_max": float(np.max(speeds)) if len(speeds) else 0.0,
    }

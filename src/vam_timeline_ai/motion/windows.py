"""Movement-window utilities.

Generative semantic analysis works on windows, not only on full clips or loops.
These helpers intentionally keep non-loop transition windows.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


WindowRange = tuple[float, float]


def make_windows(duration_seconds: float, window_seconds: float, stride_seconds: float) -> list[WindowRange]:
    """Create overlapping windows for a duration.

    If the source is shorter than the requested window, one short window is
    returned so useful transition or adjustment material is not discarded.
    """

    duration = _positive_or_zero(duration_seconds, "duration_seconds")
    window = _strictly_positive(window_seconds, "window_seconds")
    stride = _strictly_positive(stride_seconds, "stride_seconds")
    if duration == 0:
        return []
    if duration <= window:
        return [(0.0, _round_time(duration))]

    windows: list[WindowRange] = []
    start = 0.0
    epsilon = 1e-9
    while start + window <= duration + epsilon:
        end = min(start + window, duration)
        windows.append((_round_time(start), _round_time(end)))
        start += stride

    if windows and windows[-1][1] < duration and duration - windows[-1][0] >= min(window * 0.5, stride):
        start = max(0.0, duration - window)
        candidate = (_round_time(start), _round_time(duration))
        if candidate not in windows:
            windows.append(candidate)
    return windows


def make_default_window_set(duration_seconds: float) -> dict[str, list[WindowRange]]:
    """Create the default 2/4/8 second semantic window sets."""

    return {
        "2s_stride_1s": make_windows(duration_seconds, 2.0, 1.0),
        "4s_stride_2s": make_windows(duration_seconds, 4.0, 2.0),
        "8s_stride_4s": make_windows(duration_seconds, 8.0, 4.0),
    }


def window_id(sample_id: str, start: float, end: float) -> str:
    """Stable movement-window ID for manual labels and semantic records."""

    if not sample_id:
        raise ValueError("sample_id is required")
    if end <= start:
        raise ValueError("window end must be greater than start")
    return f"{sample_id}:{_round_time(start):.3f}-{_round_time(end):.3f}"


def _strictly_positive(value: float, name: str) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_or_zero(value: float, name: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _round_time(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

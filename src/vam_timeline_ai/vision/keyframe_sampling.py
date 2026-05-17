"""Small helpers for visual-judge keyframe and contact-sheet sampling."""

from __future__ import annotations

from typing import Any, Iterable


def choose_frame_times(start: float, end: float, count: int) -> list[float]:
    """Return evenly spaced timestamps including both ends when possible."""

    start_f = float(start)
    end_f = float(end)
    if end_f < start_f:
        start_f, end_f = end_f, start_f
    count_i = max(1, int(count))
    if count_i == 1 or abs(end_f - start_f) < 1e-9:
        return [round(start_f, 6)]
    step = (end_f - start_f) / float(count_i - 1)
    return [round(start_f + step * i, 6) for i in range(count_i)]


def choose_adaptive_frame_times(
    start: float,
    end: float,
    count: int,
    motion_deltas: Iterable[dict[str, Any]] | None = None,
) -> list[float]:
    """Prefer high-motion timestamps while preserving coverage across the window.

    ``motion_deltas`` may contain rows with ``time`` and one of ``delta``,
    ``motion_delta``, ``motion_strength``, or ``score``. If usable deltas are
    missing, this falls back to evenly spaced times.
    """

    base = choose_frame_times(start, end, count)
    rows = []
    for row in motion_deltas or []:
        try:
            t = float(row.get("time"))
            score = float(
                row.get("delta")
                if row.get("delta") is not None
                else row.get("motion_delta")
                if row.get("motion_delta") is not None
                else row.get("motion_strength")
                if row.get("motion_strength") is not None
                else row.get("score")
            )
        except (AttributeError, TypeError, ValueError):
            continue
        if float(start) <= t <= float(end):
            rows.append((score, t))
    if not rows:
        return base

    # Keep the first/last frame for context, then fill with strongest motion
    # moments. A tiny de-dupe radius avoids selecting several nearly identical
    # timestamps from the same burst.
    selected = {base[0], base[-1]}
    min_gap = max((float(end) - float(start)) / max(count * 2, 1), 0.001)
    for _score, t in sorted(rows, reverse=True):
        if len(selected) >= max(1, int(count)):
            break
        if all(abs(t - existing) >= min_gap for existing in selected):
            selected.add(round(t, 6))

    if len(selected) < max(1, int(count)):
        for t in base:
            selected.add(t)
            if len(selected) >= max(1, int(count)):
                break
    return sorted(selected)[: max(1, int(count))]


def build_contact_sheet_metadata(
    review_id: str,
    start: float,
    end: float,
    frame_times: list[float],
    visual_input_path: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "vam_timeline_ai_contact_sheet_metadata_v0",
        "review_id": review_id,
        "start_seconds": float(start),
        "end_seconds": float(end),
        "frame_count": len(frame_times),
        "frame_times": frame_times,
        "visual_input_path": visual_input_path,
        "recommended_for_vlm": True,
        "reason": "A single contact sheet gives the VLM pose and motion context without per-frame API calls.",
    }

"""Deterministic identity helpers for clean pipeline artifacts."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


def stable_hash(parts: list[str], length: int = 12) -> str:
    """Return a stable short hash for a sequence of identity parts."""

    text = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8", errors="surrogatepass")).hexdigest()[:length]


def sanitize_id_part(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[#/:\\\"'`<>|?*\x00-\x1f]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "none"


def format_seconds(value: float | int | str | None) -> str:
    if value is None:
        return "none"
    dec = Decimal(str(float(value))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    sign = "-" if dec < 0 else ""
    dec = abs(dec)
    whole = int(dec)
    frac = int((dec - whole) * 1000)
    return f"{sign}{whole:04d}.{frac:03d}"


def make_source_id(
    source_scene_relative_path: str | None,
    source_type: str | None,
    technical_atom_id: str | None,
    storable_id: str | None,
    plugin_id: str | None,
    clip_name: str | None,
    clip_index: int | str | None,
    track_or_controller: str | None = None,
) -> str:
    parts = [
        source_scene_relative_path,
        source_type,
        technical_atom_id,
        storable_id,
        plugin_id,
        clip_name,
        "" if clip_index is None else str(clip_index),
        track_or_controller,
    ]
    readable = "_".join(
        sanitize_id_part(p)
        for p in [source_scene_relative_path, technical_atom_id, storable_id or plugin_id, clip_name or track_or_controller or source_type, clip_index]
        if p not in {None, ""}
    )
    return f"src_{readable[:120]}_{stable_hash(parts)}"


def make_sample_id(
    source_id: str,
    fps: float,
    extraction_version: str,
    technical_atom_id: str | None = None,
    clip_name: str | None = None,
    clip_index: int | str | None = None,
) -> str:
    parts = [source_id, f"{float(fps):.6f}", extraction_version, technical_atom_id, clip_name, "" if clip_index is None else str(clip_index)]
    readable = "_".join(sanitize_id_part(p) for p in [technical_atom_id, clip_name, f"{float(fps):.0f}hz"] if p not in {None, ""})
    return f"sample_{readable[:96]}_{stable_hash(parts)}"


def make_window_id(sample_id: str, start_seconds: float, end_seconds: float, window_size_seconds: float, stride_seconds: float, fps: float) -> str:
    parts = [sample_id, format_seconds(start_seconds), format_seconds(end_seconds), format_seconds(window_size_seconds), format_seconds(stride_seconds), f"{float(fps):.6f}"]
    return f"win_{sanitize_id_part(sample_id)[:80]}_{format_seconds(start_seconds)}_{format_seconds(end_seconds)}_{stable_hash(parts)}"


def make_pair_id(
    source_scene_relative_path: str | None,
    sample_id_a: str,
    sample_id_b: str,
    clip_name_a: str | None = None,
    clip_name_b: str | None = None,
    pair_version: str = "pair_v1",
) -> str:
    ordered = sorted([str(sample_id_a), str(sample_id_b)])
    parts = [source_scene_relative_path, ordered[0], ordered[1], clip_name_a, clip_name_b, pair_version]
    readable = "_".join(sanitize_id_part(p) for p in [source_scene_relative_path, ordered[0][:24], ordered[1][:24]] if p)
    return f"pair_{readable[:100]}_{stable_hash(parts)}"


def make_pair_window_id(pair_id: str, window_id_a: str, window_id_b: str, start_seconds: float, end_seconds: float) -> str:
    parts = [pair_id, window_id_a, window_id_b, format_seconds(start_seconds), format_seconds(end_seconds)]
    return f"pwin_{sanitize_id_part(pair_id)[:80]}_{format_seconds(start_seconds)}_{format_seconds(end_seconds)}_{stable_hash(parts)}"


def make_feature_record_id(window_id: str, feature_version: str) -> str:
    parts = [window_id, feature_version]
    return f"feat_{sanitize_id_part(window_id)[:80]}_{stable_hash(parts)}"


def make_review_id(window_id: str, batch_name: str = "review_batch") -> str:
    parts = [batch_name, window_id]
    return f"review_{sanitize_id_part(batch_name)}_{sanitize_id_part(window_id)[:80]}_{stable_hash(parts)}"


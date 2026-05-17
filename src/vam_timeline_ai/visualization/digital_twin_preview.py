"""Render simple digital-twin review previews from controller tracks.

The output is review assistance only: it is not playback, not generation, and
not a visual-model truth source.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import html
import math
import shutil
import subprocess

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, safe_id_for_path, write_jsonl
from vam_timeline_ai.ui.review_ui import build_static_review_ui
from vam_timeline_ai.visualization.visual_judge_schema import visual_judge_schema_v0


VIEW_AXES = {
    "front": (0, 1, "X lateral", "Y vertical"),
    "side": (2, 1, "Z forward/back", "Y vertical"),
    "top": (0, 2, "X lateral", "Z forward/back"),
}

CONTROLLER_ALIASES = {
    "pelvis": ["pelvisControl", "hipControl"],
    "hip": ["hipControl", "pelvisControl"],
    "abdomen": ["abdomen2Control", "abdomenControl"],
    "chest": ["chestControl"],
    "head": ["headControl"],
    "lHand": ["lHandControl"],
    "rHand": ["rHandControl"],
    "lElbow": ["lElbowControl"],
    "rElbow": ["rElbowControl"],
    "lKnee": ["lKneeControl"],
    "rKnee": ["rKneeControl"],
    "lFoot": ["lFootControl"],
    "rFoot": ["rFootControl"],
    "lThigh": ["lThighControl"],
    "rThigh": ["rThighControl"],
}

SKELETON_EDGES = [
    ("pelvis", "abdomen"),
    ("pelvis", "chest"),
    ("abdomen", "chest"),
    ("chest", "head"),
    ("chest", "lElbow"),
    ("lElbow", "lHand"),
    ("chest", "rElbow"),
    ("rElbow", "rHand"),
    ("pelvis", "lThigh"),
    ("lThigh", "lKnee"),
    ("lKnee", "lFoot"),
    ("pelvis", "lKnee"),
    ("pelvis", "rThigh"),
    ("rThigh", "rKnee"),
    ("rKnee", "rFoot"),
    ("pelvis", "rKnee"),
]


def render_digital_twin_previews_v1(
    run_dir: str | Path,
    review_dir: str | Path,
    out_dir: str | Path,
    fps: int = 12,
    width: int = 960,
    height: int = 720,
    frames: int = 32,
    make_gif: bool = True,
    make_mp4: str | bool = "auto",
    view: str = "side",
) -> dict[str, Any]:
    """Render animated digital-twin review previews.

    This is a review aid only. It reconstructs a simplified skeleton from
    controller positions and never calls visual models or external APIs.
    """

    run = Path(run_dir)
    review = Path(review_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(review / "semantic_review_010.jsonl")
    samples = {str(r.get("sample_id")): r for r in load_jsonl(run / "baked" / "motion_sample_index.jsonl") if r.get("sample_id")}
    manifest: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        plt = None
        warnings.append(f"matplotlib unavailable: {exc}")

    resolved_view = view
    if view == "three_quarter":
        resolved_view = "side"
        warnings.append("three_quarter view is not implemented for v1; fell back to side view")
    if resolved_view not in VIEW_AXES:
        warnings.append(f"unknown view {view}; fell back to side view")
        resolved_view = "side"

    for row in rows:
        entry = _render_item_v1(
            row,
            samples.get(str(row.get("sample_id"))),
            run,
            out,
            plt,
            fps=max(1, int(fps)),
            width=max(320, int(width)),
            height=max(240, int(height)),
            frames=max(2, int(frames)),
            make_gif=make_gif,
            make_mp4=make_mp4,
            view=resolved_view,
        )
        manifest.append(entry)

    write_jsonl(out / "digital_twin_preview_manifest_v1.jsonl", manifest)
    dump_json(out / "visual_judge_schema_v0.json", visual_judge_schema_v0())
    _write_report_v1(out / "digital_twin_preview_report_v1.md", rows, manifest, warnings)
    _write_index_v1(out / "index.html", manifest)

    static_dir = review / "review_ui_static"
    static_summary = None
    if static_dir.exists():
        static_summary = build_static_review_ui(run, review, static_dir)

    return {
        "status": "ok",
        "review_items": len(rows),
        "previews_rendered": sum(1 for row in manifest if row.get("status") == "rendered"),
        "gif_created": sum(1 for row in manifest if row.get("gif_path")),
        "mp4_created": sum(1 for row in manifest if row.get("mp4_path")),
        "contact_sheets_created": sum(1 for row in manifest if row.get("contact_sheet_large_path")),
        "frames_created": sum(int(row.get("frame_count") or 0) for row in manifest),
        "failed_items": sum(1 for row in manifest if row.get("status") != "rendered"),
        "out_dir": str(out),
        "manifest": str(out / "digital_twin_preview_manifest_v1.jsonl"),
        "report": str(out / "digital_twin_preview_report_v1.md"),
        "index": str(out / "index.html"),
        "static_review_ui": static_summary,
        "warnings": warnings,
        "visual_judgments_are_ground_truth": False,
        "ml_training_performed": False,
        "external_api_calls": False,
    }


def render_digital_twin_review_previews_v0(run_dir: str | Path, review_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    review = Path(review_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(review / "semantic_review_010.jsonl")
    samples = {str(r.get("sample_id")): r for r in load_jsonl(run / "baked" / "motion_sample_index.jsonl") if r.get("sample_id")}
    manifest: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        plt = None
        warnings.append(f"matplotlib unavailable: {exc}")

    for row in rows:
        entry = _render_item(row, samples.get(str(row.get("sample_id"))), run, out, plt)
        manifest.append(entry)

    write_jsonl(out / "digital_twin_preview_manifest.jsonl", manifest)
    dump_json(out / "visual_judge_schema_v0.json", visual_judge_schema_v0())
    _write_report(out / "digital_twin_preview_report.md", rows, manifest, warnings)
    _write_index(out / "index.html", manifest)

    static_dir = review / "review_ui_static"
    static_summary = None
    if static_dir.exists():
        static_summary = build_static_review_ui(run, review, static_dir)

    counts = Counter(row.get("status") for row in manifest)
    return {
        "status": "ok",
        "review_items": len(rows),
        "previews_rendered": counts.get("rendered", 0),
        "could_not_visualize": len(rows) - counts.get("rendered", 0),
        "out_dir": str(out),
        "manifest": str(out / "digital_twin_preview_manifest.jsonl"),
        "report": str(out / "digital_twin_preview_report.md"),
        "index": str(out / "index.html"),
        "static_review_ui": static_summary,
        "warnings": warnings,
        "visual_judgments_are_ground_truth": False,
        "ml_training_performed": False,
    }


def _render_item(row: dict[str, Any], sample: dict[str, Any] | None, run: Path, out: Path, plt: Any) -> dict[str, Any]:
    rid = str(row.get("review_id") or row.get("window_id") or "review_item")
    safe = safe_id_for_path(rid)
    item_dir = out / safe
    item_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "review_id": row.get("review_id"),
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_actor_id": row.get("technical_actor_id") or row.get("technical_atom_id"),
        "status": "unavailable",
        "preview_dir": str(item_dir),
        "contact_sheet_path": None,
        "front_view_path": None,
        "side_view_path": None,
        "top_view_path": None,
        "partner_reference_status": "unavailable",
        "warnings": [],
    }
    if sample is None:
        entry["warnings"].append("sample not found in motion_sample_index")
        _write_item_meta(item_dir, row, entry)
        return entry
    path = _resolve_path(sample.get("baked_npz_path"), run)
    if not path or not path.exists():
        entry["warnings"].append(f"baked npz missing: {sample.get('baked_npz_path')}")
        _write_item_meta(item_dir, row, entry)
        return entry
    if plt is None:
        entry["warnings"].append("matplotlib unavailable; image previews not rendered")
        _write_item_meta(item_dir, row, entry)
        return entry

    try:
        window = _load_window(path, row)
        if window["positions"].shape[0] < 1:
            entry["warnings"].append("window contains no frames")
            _write_item_meta(item_dir, row, entry)
            return entry
        selected = _select_frames(window["positions"].shape[0], 8)
        controller_points = _controller_points(window["positions"], window["names"])
        missing = _missing_core_controllers(controller_points)
        if missing:
            entry["warnings"].append("missing controllers: " + ", ".join(missing))
        entry["warnings"].append("partner reference markers are proxy/unavailable unless pair tracks are present")

        _plot_contact_sheet(item_dir / "contact_sheet.png", row, window, controller_points, selected, plt)
        _plot_view(item_dir / "front_view.png", row, window, controller_points, selected[-1], "front", plt)
        _plot_view(item_dir / "side_view.png", row, window, controller_points, selected[-1], "side", plt)
        _plot_view(item_dir / "top_view.png", row, window, controller_points, selected[-1], "top", plt)
        entry.update(
            {
                "status": "rendered",
                "contact_sheet_path": str(item_dir / "contact_sheet.png"),
                "front_view_path": str(item_dir / "front_view.png"),
                "side_view_path": str(item_dir / "side_view.png"),
                "top_view_path": str(item_dir / "top_view.png"),
                "frame_indices": selected,
                "controller_count": len(window["names"]),
                "missing_core_controllers": missing,
            }
        )
    except Exception as exc:  # noqa: BLE001
        entry["warnings"].append(f"render failed: {exc}")
    _write_item_meta(item_dir, row, entry)
    return entry


def _render_item_v1(
    row: dict[str, Any],
    sample: dict[str, Any] | None,
    run: Path,
    out: Path,
    plt: Any,
    fps: int,
    width: int,
    height: int,
    frames: int,
    make_gif: bool,
    make_mp4: str | bool,
    view: str,
) -> dict[str, Any]:
    rid = str(row.get("review_id") or row.get("window_id") or "review_item")
    safe = safe_id_for_path(rid)
    item_dir = out / "items" / safe
    frame_dir = item_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "schema": "vam_timeline_ai_digital_twin_preview_item_v1",
        "review_id": row.get("review_id"),
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_actor_id": row.get("technical_actor_id") or row.get("technical_atom_id"),
        "status": "unavailable",
        "preview_dir": str(item_dir),
        "frames_dir": str(frame_dir),
        "preview_gif_path": None,
        "gif_path": None,
        "preview_mp4_path": None,
        "mp4_path": None,
        "contact_sheet_large_path": None,
        "primary_visual_path": None,
        "primary_visual_type": None,
        "visual_quality": "unavailable",
        "view": view,
        "partner_reference_status": "proxy_or_unavailable",
        "warnings": [],
        "visual_judgments_are_ground_truth": False,
    }
    if sample is None:
        entry["warnings"].append("sample not found in motion_sample_index")
        _write_item_meta_v1(item_dir, row, entry)
        return entry
    path = _resolve_path(sample.get("baked_npz_path"), run)
    if not path or not path.exists():
        entry["warnings"].append(f"baked npz missing: {sample.get('baked_npz_path')}")
        _write_item_meta_v1(item_dir, row, entry)
        return entry
    if plt is None:
        entry["warnings"].append("matplotlib unavailable; frame previews not rendered")
        _write_item_meta_v1(item_dir, row, entry)
        return entry

    try:
        window = _load_window(path, row)
        if window["positions"].shape[0] < 1:
            entry["warnings"].append("window contains no frames")
            _write_item_meta_v1(item_dir, row, entry)
            return entry
        selected = _select_frames(window["positions"].shape[0], frames)
        points = _controller_points(window["positions"], window["names"])
        missing = _missing_core_controllers(points)
        if missing:
            entry["warnings"].append("missing controllers: " + ", ".join(missing))
        entry["warnings"].append("partner reference markers are proxy/unavailable unless pair tracks are present")
        axis_limits = _axis_limits(window["positions"], points, view, selected)
        trail_roles = [role for role in ["pelvis", "lHand", "rHand"] if role in points]
        frame_paths: list[str] = []
        for out_idx, frame_idx in enumerate(selected):
            frame_path = frame_dir / f"frame_{out_idx:03d}.png"
            _plot_animation_frame(
                frame_path,
                row,
                window,
                points,
                frame_idx,
                selected[: out_idx + 1],
                trail_roles,
                view,
                axis_limits,
                plt,
                width,
                height,
            )
            frame_paths.append(str(frame_path))
        sheet_frames = _select_frames(len(selected), min(16, len(selected)))
        sheet_indices = [selected[i] for i in sheet_frames]
        sheet_path = item_dir / "contact_sheet_large.png"
        _plot_large_contact_sheet(sheet_path, row, window, points, sheet_indices, view, axis_limits, plt)
        gif_path = item_dir / "preview.gif"
        gif_created = _make_gif(gif_path, [Path(p) for p in frame_paths], fps, entry["warnings"]) if make_gif else False
        mp4_path = item_dir / "preview.mp4"
        mp4_created = _make_mp4(mp4_path, frame_dir, fps, make_mp4, entry["warnings"])

        entry.update(
            {
                "status": "rendered",
                "frame_paths": frame_paths,
                "frame_count": len(frame_paths),
                "contact_sheet_large_path": str(sheet_path),
                "preview_gif_path": str(gif_path) if gif_created else None,
                "gif_path": str(gif_path) if gif_created else None,
                "preview_mp4_path": str(mp4_path) if mp4_created else None,
                "mp4_path": str(mp4_path) if mp4_created else None,
                "controller_count": len(window["names"]),
                "missing_core_controllers": missing,
                "axis_limits": axis_limits,
            }
        )
        _set_primary_visual(entry)
    except Exception as exc:  # noqa: BLE001
        entry["warnings"].append(f"render failed: {exc}")
    _write_item_meta_v1(item_dir, row, entry)
    return entry


def _resolve_path(value: Any, run: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, run / path, run.parent.parent.parent / path]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), (Path.cwd() / path).resolve())


def _load_window(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        positions = np.asarray(data["positions"], dtype=float)
        times = np.asarray(data["times"], dtype=float) if "times" in data else np.arange(positions.shape[0], dtype=float) / 60.0
        names = [str(x) for x in data["controller_names"].tolist()] if "controller_names" in data else [f"controller_{i}" for i in range(positions.shape[1])]
    start = _time_to_frame(row.get("start_seconds"), times, 0)
    end = _time_to_frame(row.get("end_seconds"), times, len(times))
    start = max(0, min(start, max(0, len(times) - 1)))
    end = max(start + 1, min(end, len(times)))
    return {"positions": positions[start:end], "times": times[start:end] - times[start], "names": names}


def _time_to_frame(value: Any, times: np.ndarray, default: int) -> int:
    if value is None or not len(times):
        return default
    return int(np.searchsorted(times, float(value), side="left"))


def _select_frames(frame_count: int, count: int) -> list[int]:
    if frame_count <= count:
        return list(range(frame_count))
    return [int(round(x)) for x in np.linspace(0, frame_count - 1, count)]


def _controller_points(positions: np.ndarray, names: list[str]) -> dict[str, int]:
    by_name = {name: idx for idx, name in enumerate(names)}
    points = {}
    for role, aliases in CONTROLLER_ALIASES.items():
        for alias in aliases:
            if alias in by_name:
                points[role] = by_name[alias]
                break
    return points


def _missing_core_controllers(points: dict[str, int]) -> list[str]:
    required = ["pelvis", "chest", "head", "lHand", "rHand", "lKnee", "rKnee", "lFoot", "rFoot"]
    return [name for name in required if name not in points]


def _plot_contact_sheet(path: Path, row: dict[str, Any], window: dict[str, Any], points: dict[str, int], frames: list[int], plt: Any) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    for ax, frame in zip(axes.ravel(), frames):
        _draw_skeleton(ax, window["positions"][frame], points, "side", row, title=f"t={window['times'][frame]:.2f}s")
    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")
    fig.suptitle(_overlay_title(row), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_view(path: Path, row: dict[str, Any], window: dict[str, Any], points: dict[str, int], frame: int, view: str, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_skeleton(ax, window["positions"][frame], points, view, row, title=f"{view.title()} view")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_animation_frame(
    path: Path,
    row: dict[str, Any],
    window: dict[str, Any],
    points: dict[str, int],
    frame: int,
    trail_frames: list[int],
    trail_roles: list[str],
    view: str,
    limits: dict[str, float],
    plt: Any,
    width: int,
    height: int,
) -> None:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    _draw_skeleton_v1(ax, window["positions"], points, frame, trail_frames, trail_roles, view, row, limits)
    ax.text(
        0.012,
        0.988,
        _overlay_title(row) + f"\ntime={float(window['times'][frame]):.2f}s | visual preview only",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#172033",
        bbox={"facecolor": "white", "edgecolor": "#d7dce5", "alpha": 0.86, "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_large_contact_sheet(path: Path, row: dict[str, Any], window: dict[str, Any], points: dict[str, int], frames: list[int], view: str, limits: dict[str, float], plt: Any) -> None:
    cols = 4
    rows = max(1, int(math.ceil(len(frames) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), dpi=120, squeeze=False)
    trail_roles = [role for role in ["pelvis", "lHand", "rHand"] if role in points]
    for out_idx, (ax, frame) in enumerate(zip(axes.ravel(), frames)):
        earlier = frames[: out_idx + 1]
        _draw_skeleton_v1(ax, window["positions"], points, frame, earlier, trail_roles, view, row, limits, title=f"t={float(window['times'][frame]):.2f}s")
    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")
    fig.suptitle(_overlay_title(row) + " | large contact sheet | review-assist only", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _draw_skeleton_v1(
    ax: Any,
    positions: np.ndarray,
    points: dict[str, int],
    frame: int,
    trail_frames: list[int],
    trail_roles: list[str],
    view: str,
    row: dict[str, Any],
    limits: dict[str, float],
    title: str | None = None,
) -> None:
    frame_pos = positions[frame]
    a0, a1, xlabel, ylabel = VIEW_AXES[view]
    ax.set_facecolor("#f8fafc")
    for role in trail_roles:
        idx = points[role]
        trail = positions[trail_frames, idx, :]
        ax.plot(trail[:, a0], trail[:, a1], color=_role_color(role), linewidth=2.3, alpha=0.45, zorder=1)
    for left, right in SKELETON_EDGES:
        if left in points and right in points:
            p0 = frame_pos[points[left]]
            p1 = frame_pos[points[right]]
            ax.plot([p0[a0], p1[a0]], [p0[a1], p1[a1]], color="#263244", linewidth=3.0, alpha=0.92, zorder=2)
    for role, idx in points.items():
        p = frame_pos[idx]
        color = _role_color(role)
        size = 95 if role in {"pelvis", "chest", "head"} else 68
        ax.scatter([p[a0]], [p[a1]], s=size, color=color, edgecolor="white", linewidth=1.2, zorder=4)
        if role in {"pelvis", "chest", "head", "lHand", "rHand", "lFoot", "rFoot"}:
            ax.text(p[a0], p[a1], role, fontsize=9, color=color, weight="bold", zorder=5)
    _draw_partner_proxies(ax, frame_pos, points, view, row)
    ax.set_xlim(limits["xmin"], limits["xmax"])
    ax.set_ylim(limits["ymin"], limits["ymax"])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title or view.title(), fontsize=10)
    ax.grid(True, color="#cbd5e1", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def _axis_limits(positions: np.ndarray, points: dict[str, int], view: str, frames: list[int]) -> dict[str, float]:
    a0, a1, _, _ = VIEW_AXES[view]
    if points:
        idxs = list(points.values())
        arr = positions[np.asarray(frames, dtype=int)][:, idxs, :]
    else:
        arr = positions[np.asarray(frames, dtype=int)]
    xs = arr[..., a0].reshape(-1)
    ys = arr[..., a1].reshape(-1)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        return {"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0}
    xmin = float(np.min(xs[finite]))
    xmax = float(np.max(xs[finite]))
    ymin = float(np.min(ys[finite]))
    ymax = float(np.max(ys[finite]))
    dx = max(0.25, xmax - xmin)
    dy = max(0.25, ymax - ymin)
    # Leave room for proxy partner/reference markers and text labels.
    return {
        "xmin": xmin - dx * 0.28 - 0.08,
        "xmax": xmax + dx * 0.42 + 0.08,
        "ymin": ymin - dy * 0.32 - 0.08,
        "ymax": ymax + dy * 0.45 + 0.08,
    }


def _make_gif(path: Path, frame_paths: list[Path], fps: int, warnings: list[str]) -> bool:
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Pillow unavailable; GIF skipped: {exc}")
        return False
    try:
        adaptive = getattr(Image, "ADAPTIVE", 1)
        images = [Image.open(frame).convert("P", palette=adaptive) for frame in frame_paths if frame.exists()]
        if not images:
            warnings.append("GIF skipped: no frame PNGs")
            return False
        images[0].save(path, save_all=True, append_images=images[1:], duration=max(20, int(1000 / max(1, fps))), loop=0, optimize=False)
        for image in images:
            image.close()
        return True
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"GIF creation failed: {exc}")
        return False


def _make_mp4(path: Path, frame_dir: Path, fps: int, make_mp4: str | bool, warnings: list[str]) -> bool:
    requested = make_mp4 is True or str(make_mp4).lower() in {"1", "true", "yes", "on", "auto"}
    if not requested:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if str(make_mp4).lower() not in {"false", "0", "no", "off"}:
            warnings.append("ffmpeg unavailable; MP4 skipped")
        return False
    try:
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(max(1, fps)),
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return path.exists()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"MP4 creation failed: {exc}")
        return False


def _set_primary_visual(entry: dict[str, Any]) -> None:
    for path_key, visual_type, quality in [
        ("mp4_path", "mp4", "high"),
        ("gif_path", "gif", "high"),
        ("contact_sheet_large_path", "contact_sheet", "medium"),
    ]:
        if entry.get(path_key):
            entry["primary_visual_path"] = entry[path_key]
            entry["primary_visual_type"] = visual_type
            entry["visual_quality"] = quality
            return
    entry["visual_quality"] = "low"


def _write_item_meta_v1(item_dir: Path, row: dict[str, Any], entry: dict[str, Any]) -> None:
    dump_json(
        item_dir / "metadata.json",
        {
            "schema": "vam_timeline_ai_digital_twin_preview_item_v1",
            "review_only": True,
            "visual_judgments_are_ground_truth": False,
            "visual_model_output_is_truth": False,
            "row": row,
            "preview": entry,
        },
    )


def _draw_skeleton(ax: Any, frame_pos: np.ndarray, points: dict[str, int], view: str, row: dict[str, Any], title: str) -> None:
    a0, a1, xlabel, ylabel = VIEW_AXES[view]
    for left, right in SKELETON_EDGES:
        if left in points and right in points:
            p0 = frame_pos[points[left]]
            p1 = frame_pos[points[right]]
            ax.plot([p0[a0], p1[a0]], [p0[a1], p1[a1]], color="#3b4758", linewidth=1.8, alpha=0.9)
    for role, idx in points.items():
        p = frame_pos[idx]
        color = _role_color(role)
        size = 44 if role in {"pelvis", "chest", "head"} else 30
        ax.scatter([p[a0]], [p[a1]], s=size, color=color, edgecolor="white", linewidth=0.5, zorder=4)
        if role in {"pelvis", "chest", "head", "lHand", "rHand", "lFoot", "rFoot"}:
            ax.text(p[a0], p[a1], role, fontsize=7, color=color)
    _draw_partner_proxies(ax, frame_pos, points, view, row)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    _pad_limits(ax)


def _draw_partner_proxies(ax: Any, frame_pos: np.ndarray, points: dict[str, int], view: str, row: dict[str, Any]) -> None:
    if "pelvis" not in points:
        return
    a0, a1, _, _ = VIEW_AXES[view]
    pelvis = frame_pos[points["pelvis"]]
    chest = frame_pos[points.get("chest", points["pelvis"])]
    # These are visible reference placeholders only. They are not learned target coordinates.
    proxies = {
        "partner pelvis proxy": pelvis + np.array([0.0, -0.25, -0.12]),
        "partner chest proxy": chest + np.array([0.0, -0.18, -0.32]),
        "partner head proxy": chest + np.array([0.0, 0.18, -0.48]),
    }
    contact = str(row.get("contact_support") or "")
    if "hip" in contact:
        proxies["partner hips target"] = proxies["partner pelvis proxy"]
    if "chest" in contact:
        proxies["partner chest target"] = proxies["partner chest proxy"]
    if "leg" in contact or "thigh" in contact:
        proxies["partner legs/thighs proxy"] = pelvis + np.array([0.0, -0.45, 0.18])
    for label, p in proxies.items():
        ax.scatter([p[a0]], [p[a1]], marker="x", s=42, color="#d97706", linewidth=1.5, zorder=3)
        ax.text(p[a0], p[a1], label, fontsize=6.5, color="#9a3412")


def _role_color(role: str) -> str:
    if role in {"lHand", "rHand"}:
        return "#d97706"
    if role in {"lFoot", "rFoot", "lKnee", "rKnee"}:
        return "#2563eb"
    if role in {"pelvis", "hip"}:
        return "#16a34a"
    if role in {"chest", "head"}:
        return "#9333ea"
    return "#64748b"


def _pad_limits(ax: Any) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not all(math.isfinite(v) for v in [xmin, xmax, ymin, ymax]):
        return
    dx = max(0.05, (xmax - xmin) * 0.18)
    dy = max(0.05, (ymax - ymin) * 0.18)
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)


def _overlay_title(row: dict[str, Any]) -> str:
    pieces = [
        f"{row.get('review_id')}",
        f"family={row.get('semantic_family')}",
        f"pose={row.get('pose_subtype') or row.get('pose_family')}",
        f"motion={row.get('motion_subtype')}/{row.get('phase')}",
        f"contact={row.get('contact_support')}",
    ]
    scores = []
    for key in ["model_cowgirl_probability", "model_clean_motion_probability", "model_generation_safe_probability", "contact_support_confidence"]:
        if row.get(key) is not None:
            try:
                scores.append(f"{key.replace('model_', '').replace('_probability', '')}={float(row[key]):.2f}")
            except Exception:  # noqa: BLE001
                pass
    if scores:
        pieces.append(", ".join(scores))
    return " | ".join(str(p) for p in pieces if p)


def _write_item_meta(item_dir: Path, row: dict[str, Any], entry: dict[str, Any]) -> None:
    dump_json(
        item_dir / "preview_meta.json",
        {
            "schema": "vam_timeline_ai_digital_twin_preview_item_v0",
            "review_only": True,
            "visual_judgments_are_ground_truth": False,
            "row": row,
            "preview": entry,
        },
    )


def _write_report(path: Path, rows: list[dict[str, Any]], manifest: list[dict[str, Any]], warnings: list[str]) -> None:
    rendered = [m for m in manifest if m.get("status") == "rendered"]
    missing = [m for m in manifest if m.get("status") != "rendered"]
    warn_count = sum(len(m.get("warnings") or []) for m in manifest)
    lines = [
        "# Digital Twin Preview Report V0",
        "",
        "Review-assist only. Visual previews and any future visual-model fields are not ground truth without human confirmation.",
        "",
        f"- Review items: {len(rows)}",
        f"- Previews rendered: {len(rendered)}",
        f"- Could not visualize: {len(missing)}",
        f"- Item warnings: {warn_count}",
        f"- ML training performed: false",
        f"- Adult video generation performed: false",
        "",
        "## Global Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    lines.extend(["", "## Could Not Visualize", ""])
    lines.extend([f"- `{m.get('review_id')}`: {'; '.join(m.get('warnings') or [])}" for m in missing] or ["- none"])
    lines.extend(["", "## Rendered Items", ""])
    for m in rendered:
        lines.append(f"- `{m.get('review_id')}` contact sheet: `{m.get('contact_sheet_path')}` warnings: {'; '.join(m.get('warnings') or [])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v1(path: Path, rows: list[dict[str, Any]], manifest: list[dict[str, Any]], warnings: list[str]) -> None:
    rendered = [m for m in manifest if m.get("status") == "rendered"]
    missing = [m for m in manifest if m.get("status") != "rendered"]
    gif_count = sum(1 for m in manifest if m.get("gif_path"))
    mp4_count = sum(1 for m in manifest if m.get("mp4_path"))
    sheet_count = sum(1 for m in manifest if m.get("contact_sheet_large_path"))
    frame_count = sum(int(m.get("frame_count") or 0) for m in manifest)
    warn_count = sum(len(m.get("warnings") or []) for m in manifest)
    lines = [
        "# Digital Twin Animated Preview Report V1",
        "",
        "Review-assist only. GIF/MP4/contact-sheet previews are not ground truth and are not VaM rendering.",
        "",
        f"- Review items: {len(rows)}",
        f"- Previews rendered: {len(rendered)}",
        f"- Could not visualize: {len(missing)}",
        f"- GIFs created: {gif_count}",
        f"- MP4s created: {mp4_count}",
        f"- Large contact sheets created: {sheet_count}",
        f"- Frame PNGs created: {frame_count}",
        f"- Item warnings: {warn_count}",
        f"- ML training performed: false",
        f"- External API calls: false",
        f"- Adult video generation performed: false",
        "",
        "## Global Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    lines.extend(["", "## Missing / Failed Items", ""])
    lines.extend([f"- `{m.get('review_id')}`: {'; '.join(m.get('warnings') or [])}" for m in missing] or ["- none"])
    lines.extend(["", "## Rendered Items", ""])
    for m in rendered:
        lines.append(
            f"- `{m.get('review_id')}` primary={m.get('primary_visual_type')} "
            f"gif=`{m.get('gif_path')}` mp4=`{m.get('mp4_path')}` sheet=`{m.get('contact_sheet_large_path')}` "
            f"warnings: {'; '.join(m.get('warnings') or [])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index(path: Path, manifest: list[dict[str, Any]]) -> None:
    cards = []
    root = path.parent
    for item in manifest:
        rid = html.escape(str(item.get("review_id") or "item"))
        safe = safe_id_for_path(str(item.get("review_id") or item.get("window_id") or "review_item"))
        if item.get("status") == "rendered":
            img = f'<img src="{safe}/contact_sheet.png" alt="{rid} contact sheet">'
        else:
            img = f'<p class="warn">{html.escape("; ".join(item.get("warnings") or ["not rendered"]))}</p>'
        cards.append(f"<section><h2>{rid}</h2>{img}<p><a href=\"{safe}/preview_meta.json\">metadata</a></p></section>")
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Digital Twin Review Previews</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#f8fafc;color:#172033}}section{{background:white;border:1px solid #d7dce5;border-radius:8px;padding:12px;margin:0 0 16px}}img{{max-width:100%;border:1px solid #d7dce5;border-radius:6px}}.warn{{color:#b42318}}</style></head>
<body><h1>Digital Twin Review Previews V0</h1>
<p>Audit-only skeleton previews. These are not ground truth and not generated adult videos.</p>
<p><a href="digital_twin_preview_report.md">Report</a> | <a href="digital_twin_preview_manifest.jsonl">Manifest</a></p>
{''.join(cards)}
</body></html>
"""
    path.write_text(html_doc, encoding="utf-8")


def _write_index_v1(path: Path, manifest: list[dict[str, Any]]) -> None:
    cards = []
    for item in manifest:
        rid = html.escape(str(item.get("review_id") or "item"))
        safe = safe_id_for_path(str(item.get("review_id") or item.get("window_id") or "review_item"))
        if item.get("gif_path"):
            media = f'<img src="items/{safe}/preview.gif" alt="{rid} animated preview">'
        elif item.get("mp4_path"):
            media = f'<video controls src="items/{safe}/preview.mp4"></video>'
        elif item.get("contact_sheet_large_path"):
            media = f'<img src="items/{safe}/contact_sheet_large.png" alt="{rid} large contact sheet">'
        else:
            media = f'<p class="warn">{html.escape("; ".join(item.get("warnings") or ["not rendered"]))}</p>'
        links = []
        for label, name in [("GIF", "preview.gif"), ("MP4", "preview.mp4"), ("large sheet", "contact_sheet_large.png"), ("metadata", "metadata.json")]:
            if (path.parent / "items" / safe / name).exists():
                links.append(f'<a href="items/{safe}/{name}">{label}</a>')
        cards.append(
            f"<section><h2>{rid}</h2>{media}<p>primary: <code>{html.escape(str(item.get('primary_visual_type')))}</code> "
            f"quality: <code>{html.escape(str(item.get('visual_quality')))}</code></p>"
            f"<p>{' | '.join(links)}</p><p class=\"warn\">{html.escape('; '.join(item.get('warnings') or []))}</p></section>"
        )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Digital Twin Animated Previews V1</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#f8fafc;color:#172033}}section{{background:white;border:1px solid #d7dce5;border-radius:8px;padding:12px;margin:0 0 18px}}img,video{{max-width:100%;border:1px solid #d7dce5;border-radius:6px;background:white}}.warn{{color:#b42318}}a{{margin-right:10px}}</style></head>
<body><h1>Digital Twin Animated Previews V1</h1>
<p>Audit-only skeleton animations. Visual model judgments remain review-assist only until human-confirmed.</p>
<p><a href="digital_twin_preview_report_v1.md">Report</a> | <a href="digital_twin_preview_manifest_v1.jsonl">Manifest</a></p>
{''.join(cards)}
</body></html>
"""
    path.write_text(html_doc, encoding="utf-8")

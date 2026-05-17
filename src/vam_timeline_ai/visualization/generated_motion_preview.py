"""Render technical previews for synthesized relative motion flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_json


def render_generated_motion_preview_v0(flow: str | Path, out_dir: str | Path) -> dict[str, Any]:
    data = load_json(flow)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    driver = _find_driver(data)
    if driver is None:
        manifest = {"status": "no_driver_track", "files": []}
        _write_index(target, data, manifest, None)
        return manifest
    times = np.asarray(driver.get("times") or [], dtype=float)
    path = np.asarray(driver.get("position_deltas") or [], dtype=float)
    files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        files.append(_plot_trajectory(target / "hip_trajectory_top.png", path[:, 0], path[:, 2], "Top View: lateral vs forward/back", "lateral", "forward/back", plt))
        files.append(_plot_trajectory(target / "hip_trajectory_side.png", path[:, 2], path[:, 1], "Side View: forward/back vs vertical", "forward/back", "vertical", plt))
        files.append(_plot_trajectory(target / "hip_trajectory_front.png", path[:, 0], path[:, 1], "Front View: lateral vs vertical", "lateral", "vertical", plt))
        files.append(_plot_timeseries(target / "hip_xyz_timeseries.png", times, path, plt))
        files.append(_plot_speed(target / "hip_speed_curve.png", times, path, plt))
    except Exception as exc:
        (target / "preview_warning.txt").write_text(f"Matplotlib preview failed: {exc}\n", encoding="utf-8")
        files.append("preview_warning.txt")
    summary_path = target / "controller_track_summary.md"
    _write_track_summary(data, summary_path)
    files.append(summary_path.name)
    manifest = {
        "status": "ok",
        "flow_id": data.get("flow_id"),
        "driver_controller": driver.get("controller_name"),
        "trajectory_shape": data.get("trajectory_shape"),
        "files": files,
        "is_vam_playback": False,
        "timeline_export_performed": False,
    }
    _write_index(target, data, manifest, driver)
    return manifest


def _find_driver(data: dict[str, Any]) -> dict[str, Any] | None:
    tracks = data.get("controller_tracks", []) or []
    for track in tracks:
        if track.get("role") == "driver":
            return track
    for track in tracks:
        if track.get("controller_name") in {"hipControl", "pelvisControl"}:
            return track
    return tracks[0] if tracks else None


def _plot_trajectory(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, plt: Any) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(x, y, color="#1f77b4", linewidth=2.0)
    ax.scatter([x[0]], [y[0]], color="#2ca02c", label="start", s=35)
    ax.scatter([x[-1]], [y[-1]], color="#d62728", label="end", s=35)
    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.axvline(0, color="#999999", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_timeseries(path: Path, times: np.ndarray, positions: np.ndarray, plt: Any) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["lateral_x", "vertical_y", "forward_back_z"]
    for idx, label in enumerate(labels):
        ax.plot(times, positions[:, idx], linewidth=1.8, label=label)
    ax.set_title("Relative Driver Deltas Over Time")
    ax.set_xlabel("seconds")
    ax.set_ylabel("relative delta")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_speed(path: Path, times: np.ndarray, positions: np.ndarray, plt: Any) -> str:
    diffs = np.diff(positions, axis=0)
    dt = np.maximum(np.diff(times), 1e-6)
    speed = np.linalg.norm(diffs, axis=1) / dt if len(diffs) else np.asarray([])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times[1:], speed, color="#9467bd", linewidth=1.8)
    ax.set_title("Relative Driver Speed")
    ax.set_xlabel("seconds")
    ax.set_ylabel("relative delta / second")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _write_track_summary(data: dict[str, Any], path: Path) -> None:
    lines = [
        "# Generated Motion Track Summary",
        "",
        f"- Flow: `{data.get('flow_id')}`",
        f"- Coordinate space: `{data.get('coordinate_space')}`",
        f"- Export ready: `{data.get('export_ready')}`",
        f"- Clip stitching used: `{data.get('clip_stitching_used')}`",
        "",
        "## Tracks",
        "",
    ]
    for track in data.get("controller_tracks", []) or []:
        positions = np.asarray(track.get("position_deltas") or [], dtype=float)
        max_delta = float(np.max(np.abs(positions))) if positions.size else 0.0
        lines.append(f"- `{track.get('controller_name')}` role={track.get('role')} bodypart={track.get('bodypart')} max_abs_delta={max_delta:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index(target: Path, data: dict[str, Any], manifest: dict[str, Any], driver: dict[str, Any] | None) -> None:
    image_links = []
    for file_name in manifest.get("files", []) or []:
        if str(file_name).endswith(".png"):
            image_links.append(f"<figure><img src=\"{file_name}\" alt=\"{file_name}\"><figcaption>{file_name}</figcaption></figure>")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Generated Motion Flow V0 Preview</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #222; }}
    img {{ max-width: 760px; width: 100%; border: 1px solid #ccc; }}
    figure {{ margin: 0 0 24px 0; }}
    code {{ background: #f3f3f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Generated Motion Flow V0 Preview</h1>
  <p>This is a technical preview of relative motion deltas, not VaM playback.</p>
  <ul>
    <li>Flow: <code>{data.get('flow_id')}</code></li>
    <li>Driver: <code>{(driver or {}).get('controller_name')}</code></li>
    <li>Trajectory: <code>{data.get('trajectory_shape')}</code></li>
    <li>Coordinate space: <code>{data.get('coordinate_space')}</code></li>
    <li>Timeline export performed: <code>{data.get('timeline_export_performed')}</code></li>
  </ul>
  {''.join(image_links)}
  <p><a href="controller_track_summary.md">Controller track summary</a></p>
</body>
</html>
"""
    (target / "index.html").write_text(html, encoding="utf-8")

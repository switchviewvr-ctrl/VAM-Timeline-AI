"""Render technical previews for retargeted generated motion flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_json


def render_retargeted_motion_preview_v0(retargeted_flow: str | Path, out_dir: str | Path) -> dict[str, Any]:
    data = load_json(retargeted_flow)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    tracks = {track.get("controller_name"): track for track in data.get("controller_tracks", []) or []}
    driver = tracks.get("pelvisControl") or tracks.get("hipControl")
    files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if driver:
            path = _arr(driver)
            files.append(_plot_pose_view(target / "retarget_top_view.png", tracks, 0, 2, "Top View", "lateral", "forward/back", plt))
            files.append(_plot_pose_view(target / "retarget_side_view.png", tracks, 2, 1, "Side View", "forward/back", "vertical", plt))
            files.append(_plot_pose_view(target / "retarget_front_view.png", tracks, 0, 1, "Front View", "lateral", "vertical", plt))
            files.append(_plot_trajectory(target / "retarget_pelvis_trajectory.png", path, plt))
            files.append(_plot_anchor_stability(target / "retarget_anchor_stability.png", tracks, plt))
            files.append(_plot_distances(target / "retarget_controller_distances.png", tracks, plt))
    except Exception as exc:
        (target / "preview_warning.txt").write_text(f"Retarget preview failed: {exc}\n", encoding="utf-8")
        files.append("preview_warning.txt")
    summary = target / "retargeted_controller_summary.md"
    _write_summary(data, summary)
    files.append(summary.name)
    manifest = {
        "status": "ok",
        "flow_id": data.get("flow_id"),
        "files": files,
        "timeline_export_performed": False,
    }
    _write_index(target, data, manifest)
    return manifest


def render_retargeted_motion_preview_v1(retargeted_flow: str | Path, out_dir: str | Path) -> dict[str, Any]:
    manifest = render_retargeted_motion_preview_v0(retargeted_flow, out_dir)
    data = load_json(retargeted_flow)
    target = Path(out_dir)
    tracks = {track.get("controller_name"): track for track in data.get("controller_tracks", []) or []}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pelvis = _arr(tracks.get("pelvisControl") or tracks.get("hipControl"))
        chest = _arr(tracks.get("chestControl"))
        if len(pelvis) and len(chest):
            fig, ax = plt.subplots(figsize=(7, 4))
            n = min(len(pelvis), len(chest))
            ax.plot(pelvis[:n, 2] - np.mean(pelvis[:n, 2]), label="pelvis forward/back")
            ax.plot(chest[:n, 2] - np.mean(chest[:n, 2]), label="chest forward/back")
            ax.set_title("Pelvis vs Chest Phase")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(target / "retarget_pelvis_chest_phase.png", dpi=140)
            plt.close(fig)
            manifest["files"].append("retarget_pelvis_chest_phase.png")
        if len(pelvis):
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = ["lateral", "vertical", "forward/back"]
            for idx, label in enumerate(labels):
                ax.plot(pelvis[:, idx] - np.mean(pelvis[:, idx]), label=label)
            ax.set_title("Pelvis Axis Components")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(target / "retarget_pelvis_axis_components.png", dpi=140)
            plt.close(fig)
            manifest["files"].append("retarget_pelvis_axis_components.png")
    except Exception:
        pass
    _write_index(target, data, manifest)
    return manifest


def _arr(track: dict[str, Any] | None) -> np.ndarray:
    return np.asarray((track or {}).get("retargeted_positions") or [], dtype=float)


def _plot_pose_view(path: Path, tracks: dict[str, dict[str, Any]], i: int, j: int, title: str, xlabel: str, ylabel: str, plt: Any) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, track in tracks.items():
        arr = _arr(track)
        if not len(arr):
            continue
        if track.get("role") == "driver":
            ax.plot(arr[:, i], arr[:, j], linewidth=2.0, label=name)
        else:
            ax.scatter([arr[0, i]], [arr[0, j]], s=35, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_trajectory(path: Path, positions: np.ndarray, plt: Any) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(positions[:, 0], positions[:, 2], linewidth=2)
    ax.scatter([positions[0, 0]], [positions[0, 2]], color="green", s=35, label="start")
    ax.scatter([positions[-1, 0]], [positions[-1, 2]], color="red", s=35, label="end")
    ax.set_title("Retargeted Pelvis/Hip Trajectory")
    ax.set_xlabel("lateral")
    ax.set_ylabel("forward/back")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_anchor_stability(path: Path, tracks: dict[str, dict[str, Any]], plt: Any) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["lFootControl", "rFootControl", "lKneeControl", "rKneeControl"]:
        arr = _arr(tracks.get(name))
        if len(arr):
            baseline = arr[0:1]
            drift = np.linalg.norm(arr - baseline, axis=1)
            ax.plot(drift, label=name)
    ax.set_title("Anchor Drift")
    ax.set_xlabel("frame")
    ax.set_ylabel("distance from first frame")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_distances(path: Path, tracks: dict[str, dict[str, Any]], plt: Any) -> str:
    pelvis = _arr(tracks.get("pelvisControl") or tracks.get("hipControl"))
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(pelvis):
        for name in ["chestControl", "lFootControl", "rFootControl", "lKneeControl", "rKneeControl"]:
            arr = _arr(tracks.get(name))
            if len(arr):
                n = min(len(arr), len(pelvis))
                ax.plot(np.linalg.norm(arr[:n] - pelvis[:n], axis=1), label=f"{name} to pelvis")
    ax.set_title("Controller Distances")
    ax.set_xlabel("frame")
    ax.set_ylabel("distance")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _write_summary(data: dict[str, Any], path: Path) -> None:
    lines = [
        "# Retargeted Controller Summary",
        "",
        f"- Flow: `{data.get('flow_id')}`",
        f"- Coordinate space: `{data.get('coordinate_space')}`",
        f"- Source world coords used: `{data.get('source_world_coords_used')}`",
        f"- Clip stitching used: `{data.get('clip_stitching_used')}`",
        "",
        "## Controllers",
        "",
    ]
    for track in data.get("controller_tracks", []) or []:
        arr = _arr(track)
        first = arr[0].tolist() if len(arr) else []
        lines.append(f"- `{track.get('controller_name')}` role={track.get('role')} first_position={first}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index(target: Path, data: dict[str, Any], manifest: dict[str, Any]) -> None:
    figures = []
    for file_name in manifest.get("files", []) or []:
        if str(file_name).endswith(".png"):
            figures.append(f"<figure><img src=\"{file_name}\" alt=\"{file_name}\"><figcaption>{file_name}</figcaption></figure>")
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Retargeted Motion Preview V0</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px}}img{{max-width:760px;width:100%;border:1px solid #ccc}}figure{{margin:0 0 24px}}</style></head>
<body>
<h1>Retargeted Motion Preview V0</h1>
<p>Review preview only. This is not VaM playback and not production Timeline export.</p>
<ul>
<li>Flow: <code>{data.get('flow_id')}</code></li>
<li>Coordinate space: <code>{data.get('coordinate_space')}</code></li>
<li>Source world coords used: <code>{data.get('source_world_coords_used')}</code></li>
<li>Clip stitching used: <code>{data.get('clip_stitching_used')}</code></li>
</ul>
{''.join(figures)}
<p><a href="retargeted_controller_summary.md">Retargeted controller summary</a></p>
</body></html>
"""
    (target / "index.html").write_text(html, encoding="utf-8")

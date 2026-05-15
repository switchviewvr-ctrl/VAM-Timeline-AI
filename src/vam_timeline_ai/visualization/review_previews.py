"""Render static review previews for manual labeling batches."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_json, load_jsonl, safe_id_for_path, write_jsonl


CORE_PARTS = ["pelvis", "hip", "chest", "head", "left_hand", "right_hand", "left_foot", "right_foot"]


def render_review_previews_v1(review_batch: str | Path, sample_index: str | Path, controller_map: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(review_batch)
    samples = {r.get("sample_id"): r for r in load_jsonl(sample_index) if r.get("sample_id")}
    mappings = load_json(controller_map).get("controller_mappings", {})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        matplotlib_available = True
    except Exception as exc:  # noqa: BLE001
        plt = None
        matplotlib_available = False
        missing_reason = str(exc)
    manifest = []
    warnings = []
    if not matplotlib_available:
        warnings.append(f"matplotlib unavailable: {missing_reason}")
    for row in rows:
        item_dir = out / safe_id_for_path(row.get("review_id") or row.get("window_id"))
        item_dir.mkdir(parents=True, exist_ok=True)
        meta = _metadata(row, samples.get(row.get("sample_id")), mappings)
        if matplotlib_available and plt is not None:
            try:
                _render_one(row, samples.get(row.get("sample_id")), mappings, item_dir, plt)
                _render_pair(row, samples, mappings, item_dir, plt)
            except Exception as exc:  # noqa: BLE001
                meta["warnings"].append(f"plot rendering failed: {exc}")
        dump_json(item_dir / "metadata.json", meta)
        manifest.append({"review_id": row.get("review_id"), "window_id": row.get("window_id"), "preview_dir": str(item_dir), "warnings": meta["warnings"]})
    write_jsonl(out / "preview_manifest.jsonl", manifest)
    _write_index(rows, out, manifest)
    _write_report(out / "preview_report.md", rows, manifest, warnings)
    return {"items": len(rows), "matplotlib_available": matplotlib_available, "warnings": warnings}


def _render_one(row: dict[str, Any], sample: dict[str, Any] | None, mappings: dict[str, Any], out: Path, plt: Any) -> None:
    if not sample:
        return
    with np.load(sample["baked_npz_path"], allow_pickle=True) as data:
        pos = np.asarray(data["positions"], dtype=np.float32)
        times = np.asarray(data["times"], dtype=np.float32)
        names = [str(x) for x in data["controller_names"].tolist()]
    start = int(row.get("frame_start") or _time_to_frame(row.get("start_seconds"), times))
    end = int(row.get("frame_end") or _time_to_frame(row.get("end_seconds"), times, default=len(times)))
    start = max(0, min(start, len(times) - 1))
    end = max(start + 1, min(end, len(times)))
    p = pos[start:end]
    t = times[start:end] - times[start]
    indices = _core_indices(names, mappings)
    _plot_trajectory(p, names, indices, 0, 2, "X lateral", "Z forward/back (axis uncertain)", out / "trajectory_top.png", plt)
    _plot_trajectory(p, names, indices, 2, 1, "Z forward/back (axis uncertain)", "Y vertical", out / "trajectory_side.png", plt)
    _plot_trajectory(p, names, indices, 0, 1, "X lateral", "Y vertical", out / "trajectory_front.png", plt)
    _plot_speed(p, t, names, indices, out / "pelvis_speed.png", plt, pelvis_only=True)
    _plot_speed(p, t, names, indices, out / "key_controller_motion.png", plt, pelvis_only=False)


def _render_pair(row: dict[str, Any], samples: dict[str, dict[str, Any]], mappings: dict[str, Any], out: Path, plt: Any) -> None:
    pair = row.get("pair_context_summary") or {}
    sid_a = pair.get("sample_id_a")
    sid_b = pair.get("sample_id_b")
    if not sid_a or not sid_b or sid_a not in samples or sid_b not in samples:
        return
    a = _load_window(samples[sid_a], int(pair.get("frame_start_a") or 0), int(pair.get("frame_end_a") or 0))
    b = _load_window(samples[sid_b], int(pair.get("frame_start_b") or 0), int(pair.get("frame_end_b") or 0))
    ia = _root_index(a["names"], mappings)
    ib = _root_index(b["names"], mappings)
    if ia is None or ib is None:
        return
    n = min(len(a["pos"]), len(b["pos"]))
    if n < 2:
        return
    for ax0, ax1, name, xlabel, ylabel in [
        (0, 2, "pair_trajectory_top.png", "X lateral", "Z forward/back (axis uncertain)"),
        (2, 1, "pair_trajectory_side.png", "Z forward/back (axis uncertain)", "Y vertical"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(a["pos"][:n, ia, ax0], a["pos"][:n, ia, ax1], label=f"A {a['names'][ia]}")
        ax.plot(b["pos"][:n, ib, ax0], b["pos"][:n, ib, ax1], label=f"B {b['names'][ib]}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Pair root trajectory preview")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    dist = np.linalg.norm(a["pos"][:n, ia, :] - b["pos"][:n, ib, :], axis=1)
    t = a["times"][:n] - a["times"][0]
    ax.plot(t, dist, label="root distance")
    ax.set_xlabel("Window time seconds")
    ax.set_ylabel("Distance proxy")
    ax.set_title("Pair relative distance preview")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pair_relative_distances.png", dpi=120)
    plt.close(fig)


def _load_window(sample: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    with np.load(sample["baked_npz_path"], allow_pickle=True) as data:
        pos = np.asarray(data["positions"], dtype=np.float32)
        times = np.asarray(data["times"], dtype=np.float32)
        names = [str(x) for x in data["controller_names"].tolist()]
    start = max(0, min(start, pos.shape[0] - 1))
    end = pos.shape[0] if end <= 0 else end
    end = max(start + 1, min(end, pos.shape[0]))
    return {"pos": pos[start:end], "times": times[start:end], "names": names}


def _plot_trajectory(pos: np.ndarray, names: list[str], indices: list[int], ax0: int, ax1: int, xlabel: str, ylabel: str, path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx in indices:
        ax.plot(pos[:, idx, ax0], pos[:, idx, ax1], label=names[idx])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Controller trajectory preview")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_speed(pos: np.ndarray, t: np.ndarray, names: list[str], indices: list[int], path: Path, plt: Any, pelvis_only: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if pelvis_only:
        indices = indices[:1]
    for idx in indices:
        if len(pos) < 2:
            continue
        dt = np.diff(t.astype(np.float64))
        dt = np.where(dt <= 0, 1.0 / 60.0, dt)
        speed = np.linalg.norm(np.diff(pos[:, idx, :].astype(np.float64), axis=0) / dt[:, None], axis=1)
        ax.plot(t[1:], speed, label=names[idx])
    ax.set_xlabel("Window time seconds")
    ax.set_ylabel("Speed proxy m/s")
    ax.set_title("Controller speed preview")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _metadata(row: dict[str, Any], sample: dict[str, Any] | None, mappings: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    if sample is None:
        warnings.append("sample not found in sample index")
        names = []
    else:
        names = list(sample.get("controller_names", []))
    mapped = {name: mappings.get(name, {}).get("body_part", "unknown") for name in names}
    return {
        "review_id": row.get("review_id"),
        "window_id": row.get("window_id"),
        "sample_id": row.get("sample_id"),
        "source_scene_file": row.get("source_scene_file"),
        "technical_atom_id": row.get("technical_atom_id"),
        "time_range": [row.get("start_seconds"), row.get("end_seconds")],
        "weak_labels_v2": row.get("weak_labels_v2", []),
        "machine_proposals": row.get("machine_proposals", []),
        "silver_labels": row.get("silver_labels", {}),
        "machine_label_warning": row.get("machine_label_warning"),
        "top_features": row.get("top_features", {}),
        "mapped_controllers": mapped,
        "warnings": warnings,
    }


def _core_indices(names: list[str], mappings: dict[str, Any]) -> list[int]:
    out = []
    for idx, name in enumerate(names):
        if mappings.get(name, {}).get("body_part") in CORE_PARTS:
            out.append(idx)
    return out[:8] or list(range(min(len(names), 4)))


def _root_index(names: list[str], mappings: dict[str, Any]) -> int | None:
    for wanted in ["pelvis", "hip", "root", "abdomen"]:
        for idx, name in enumerate(names):
            if mappings.get(name, {}).get("body_part") == wanted:
                return idx
    return None


def _time_to_frame(value: Any, times: np.ndarray, default: int = 0) -> int:
    if value is None or len(times) == 0:
        return default
    return int(np.searchsorted(times, float(value), side="left"))


def _write_index(rows: list[dict[str, Any]], out: Path, manifest: list[dict[str, Any]]) -> None:
    cards = []
    manifest_by_review = {m.get("review_id"): m for m in manifest}
    for row in rows:
        safe = safe_id_for_path(row.get("review_id") or row.get("window_id"))
        image_tags = []
        for name in ["trajectory_top.png", "trajectory_side.png", "trajectory_front.png", "pelvis_speed.png", "key_controller_motion.png"]:
            if (out / safe / name).exists():
                image_tags.append(f'<img src="{safe}/{name}" alt="{name}" style="max-width:420px">')
        yaml_block = html.escape(_stub_for(row))
        machine = ", ".join(
            f"{item.get('label','')} ({item.get('confidence','')})"
            for item in row.get("machine_proposals", [])[:8]
        )
        silver = row.get("silver_labels", {}) or {}
        silver_text = ", ".join(
            [*(silver.get("positive_labels", []) or []), *(silver.get("role_candidates", []) or []), *(silver.get("contact_candidates", []) or [])]
        )
        cards.append(
            f"<section><h2>{html.escape(str(row.get('review_id')))}</h2>"
            f"<p><b>Window:</b> {html.escape(str(row.get('window_id')))}<br>"
            f"<b>Scene:</b> {html.escape(str(row.get('source_scene_file')))}<br>"
            f"<b>Atom:</b> {html.escape(str(row.get('technical_atom_id')))}<br>"
            f"<b>Time:</b> {row.get('start_seconds')} - {row.get('end_seconds')}</p>"
            f"<p><b>Weak hints:</b> {html.escape(', '.join(item.get('label','') for item in row.get('weak_labels_v2', [])[:8]))}</p>"
            f"<p><b>Machine proposals:</b> {html.escape(machine)}<br>"
            f"<b>Silver hints:</b> {html.escape(silver_text)}<br>"
            f"<b>Warning:</b> {html.escape(str(row.get('machine_label_warning') or 'Hints are not human truth.'))}</p>"
            + "".join(image_tags)
            + f"<pre>{yaml_block}</pre></section>"
        )
    html_text = "<!doctype html><meta charset='utf-8'><title>Review previews</title><style>body{font-family:sans-serif}section{border-bottom:1px solid #ddd;padding:1rem}img{margin:.25rem}</style><h1>Review previews</h1>" + "\n".join(cards)
    (out / "index.html").write_text(html_text, encoding="utf-8")


def _stub_for(row: dict[str, Any]) -> str:
    return "\n".join([
        "windows:",
        f"  \"{row.get('window_id')}\":",
        "    labels: []",
        "    negative_labels: []",
        "    uncertain_labels: []",
        "    semantic_role: \"unknown\"",
        "    focus_actor: \"unknown\"",
        "    movement_quality: \"questionable\"",
        "    include_for_ml: false",
        "    confidence: 0.0",
        "    notes: \"\"",
    ])


def _write_report(out: Path, rows: list[dict[str, Any]], manifest: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = [
        "# Review Preview Report",
        "",
        "These are offline technical previews, not VaM playback and not semantic proof.",
        "",
        f"- Review items: {len(rows)}",
        f"- Preview manifest rows: {len(manifest)}",
        f"- Items with warnings: {sum(1 for m in manifest if m.get('warnings'))}",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- None")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Clustering v1 for richer features.

Uses sklearn when available. If unavailable, writes an honest dependency report.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def cluster_ml_v1(dataset: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        _write_missing(out, exc)
        return {"ran": False, "reason": str(exc)}
    data = np.load(dataset, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    window_ids = [str(x) for x in data["window_ids"].tolist()]
    sample_ids = [str(x) for x in data["sample_ids"].tolist()]
    scenes = [str(x) for x in data["source_scene_files"].tolist()]
    usable = np.isfinite(X).any(axis=1)
    X_use = _fill_nan(X[usable])
    if len(X_use) < 20:
        _write_missing(out, "not enough usable rows for clustering")
        return {"ran": False, "reason": "not enough usable rows"}
    X_scaled = StandardScaler().fit_transform(X_use)
    components = min(10, X_scaled.shape[1], max(2, len(X_scaled) - 1))
    X_reduced = PCA(n_components=components, random_state=13).fit_transform(X_scaled)
    k = min(12, max(3, int(np.sqrt(len(X_reduced)) // 2)))
    labels = MiniBatchKMeans(n_clusters=k, random_state=13, batch_size=2048, n_init=5).fit_predict(X_reduced)
    usable_indices = np.where(usable)[0]
    assignments = []
    for source_idx, cluster_id in zip(usable_indices, labels, strict=False):
        assignments.append({"window_id": window_ids[source_idx], "sample_id": sample_ids[source_idx], "source_scene_file": scenes[source_idx], "cluster_id": int(cluster_id)})
    _write_jsonl(out / "cowgirl_cluster_assignments_v1.jsonl", assignments)
    _write_cluster_report(out / "cowgirl_cluster_report_v1.md", assignments, k)
    _write_feature_summary(out / "cowgirl_cluster_feature_summary_v1.md", X[usable], labels, feature_names)
    return {"ran": True, "assignments": len(assignments), "k": int(k)}


def _fill_nan(X: np.ndarray) -> np.ndarray:
    out = X.astype(np.float64, copy=True)
    means = np.nanmean(out, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(means, inds[1])
    return out.astype(np.float32)


def _write_missing(out: Path, reason: Any) -> None:
    text = f"# Cowgirl Clustering v1\n\nClustering v1 did not run.\n\nReason: `{reason}`\n"
    (out / "cowgirl_cluster_report_v1.md").write_text(text, encoding="utf-8")
    (out / "cowgirl_cluster_assignments_v1.jsonl").write_text("", encoding="utf-8")
    (out / "cowgirl_cluster_feature_summary_v1.md").write_text(text, encoding="utf-8")


def _write_cluster_report(path: Path, assignments: list[dict[str, Any]], k: int) -> None:
    counts = Counter(a["cluster_id"] for a in assignments)
    scene_by_cluster: dict[int, Counter] = defaultdict(Counter)
    for a in assignments:
        scene_by_cluster[a["cluster_id"]][a["source_scene_file"]] += 1
    lines = ["# Cowgirl Cluster Report v1", "", f"- Clusters: {k}", f"- Assignments: {len(assignments)}", ""]
    for cluster_id, count in sorted(counts.items()):
        top_scene, top_count = scene_by_cluster[cluster_id].most_common(1)[0]
        dominance = top_count / count
        warning = " WARNING: one scene dominates" if dominance > 0.65 else ""
        lines.append(f"- Cluster {cluster_id}: {count} windows; top scene `{top_scene}` {dominance:.1%}{warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_summary(path: Path, X: np.ndarray, labels: np.ndarray, feature_names: list[str]) -> None:
    lines = ["# Cowgirl Cluster Feature Summary v1", ""]
    for cluster_id in sorted(set(labels.tolist())):
        members = X[labels == cluster_id]
        means = np.nanmean(members, axis=0)
        top = sorted(zip(feature_names, means, strict=False), key=lambda item: abs(item[1]) if np.isfinite(item[1]) else -1, reverse=True)[:8]
        lines.append(f"## Cluster {cluster_id}")
        for name, value in top:
            lines.append(f"- `{name}`: {float(value):.4f}" if np.isfinite(value) else f"- `{name}`: NaN")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

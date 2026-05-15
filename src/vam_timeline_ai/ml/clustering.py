"""NumPy-only clustering baseline for ML readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def analyze_ml_v0(dataset: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    window_ids = [str(x) for x in data["window_ids"].tolist()]
    sample_ids = [str(x) for x in data["sample_ids"].tolist()]
    labels = [str(x) for x in data["label_names"].tolist()]
    usable_mask = np.isfinite(X).any(axis=1)
    X_use = X[usable_mask]
    window_use = [wid for wid, keep in zip(window_ids, usable_mask, strict=False) if keep]
    sample_use = [sid for sid, keep in zip(sample_ids, usable_mask, strict=False) if keep]
    assignments: list[dict[str, Any]] = []
    cluster_summary: dict[str, Any] = {"ran": False, "reason": "not enough usable rows"}
    if len(X_use) >= 4:
        X_fill = _fill_nan_with_column_means(X_use)
        X_std, mean, std = _standardize(X_fill)
        k = min(8, max(2, int(np.sqrt(len(X_std)))))
        cluster_ids = _kmeans(X_std, k=k, seed=13)
        for wid, sid, cluster_id in zip(window_use, sample_use, cluster_ids, strict=False):
            assignments.append({"window_id": wid, "sample_id": sid, "cluster_id": int(cluster_id)})
        cluster_summary = {"ran": True, "k": int(k), "usable_rows": int(len(X_use)), "feature_count": int(X.shape[1])}
    _write_jsonl(out_path / "cowgirl_cluster_assignments_v0.jsonl", assignments)
    _write_cluster_report(out_path / "cowgirl_cluster_report_v0.md", cluster_summary, assignments)
    readiness = _readiness_report(X, feature_names, labels, sample_ids, cluster_summary)
    (out_path / "ml_readiness_report.md").write_text(readiness, encoding="utf-8")
    return {"assignments": len(assignments), "cluster_summary": cluster_summary}


def _fill_nan_with_column_means(X: np.ndarray) -> np.ndarray:
    out = X.astype(np.float64, copy=True)
    means = np.nanmean(out, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(means, inds[1])
    return out.astype(np.float32)


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((X - mean) / std).astype(np.float32), mean, std


def _kmeans(X: np.ndarray, k: int, seed: int = 0, iterations: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(X) < k:
        return np.zeros((len(X),), dtype=np.int32)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros((len(X),), dtype=np.int32)
    for _ in range(iterations):
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        for idx in range(k):
            members = X[labels == idx]
            if len(members):
                centroids[idx] = members.mean(axis=0)
    return labels


def _readiness_report(X: np.ndarray, feature_names: list[str], labels: list[str], sample_ids: list[str], cluster_summary: dict[str, Any]) -> str:
    numeric_rows = int(np.sum(np.isfinite(X).any(axis=1))) if len(X) else 0
    missing_rates = []
    for idx, name in enumerate(feature_names):
        missing = float(np.mean(~np.isfinite(X[:, idx]))) if len(X) else 1.0
        missing_rates.append((name, missing))
    sample_counts: dict[str, int] = {}
    for sid in sample_ids:
        sample_counts[sid] = sample_counts.get(sid, 0) + 1
    lines = [
        "# ML Readiness Report",
        "",
        f"- Windows in dataset: {len(X)}",
        f"- Windows with numeric features: {numeric_rows}",
        f"- Manual label classes: {len(labels)}",
        f"- Enough labels for supervised classification: {len(labels) >= 2}",
        f"- Enough windows for clustering: {numeric_rows >= 4}",
        f"- Clustering ran: {cluster_summary.get('ran')}",
        "",
        "## Mostly Missing Features",
        "",
    ]
    for name, rate in sorted(missing_rates, key=lambda item: item[1], reverse=True)[:10]:
        lines.append(f"- `{name}`: {rate:.1%} missing")
    lines.extend(["", "## Dominant Samples", ""])
    for sid, count in sorted(sample_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
        lines.append(f"- `{sid}`: {count} windows")
    lines.extend(["", "## Next Manual Review", "", "Prioritize windows with numeric features but no labels, especially from samples that dominate the dataset."])
    return "\n".join(lines) + "\n"


def _write_cluster_report(path: Path, summary: dict[str, Any], assignments: list[dict[str, Any]]) -> None:
    lines = ["# Cowgirl Clustering Report v0", "", f"- Ran: {summary.get('ran')}", f"- Assignments: {len(assignments)}"]
    if summary.get("ran"):
        lines.append(f"- k: {summary.get('k')}")
    else:
        lines.append(f"- Reason: {summary.get('reason')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

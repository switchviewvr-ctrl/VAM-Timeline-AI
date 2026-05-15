"""Balanced silver v2 proxy baseline with sklearn or NumPy fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def train_silver_baseline_v1(
    dataset: str | Path,
    readiness: str | Path,
    out_dir: str | Path,
    report: str | Path,
    allow_numpy_fallback: bool = True,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        feature_names = [str(x) for x in data["feature_names"].tolist()]
        window_labels = [str(x) for x in data["silver_window_label_names"].tolist()]
        window_y = data["silver_v2_window_y_multilabel"].astype(np.int8)
        silver_labels = [str(x) for x in data["silver_label_names"].tolist()]
        default_mask = data["default_trainable_silver_label_mask"].astype(bool)
        scenes = np.asarray([str(x) for x in data["group_scene"].tolist()], dtype=object)
        samples = np.asarray([str(x) for x in data["group_sample"].tolist()], dtype=object)
    trainable = [label for label, keep in zip(silver_labels, default_mask.tolist()) if keep and label in window_labels]
    if not trainable:
        summary = _blocked("no default trainable silver v2 window labels", readiness, sklearn_used=False, numpy_used=False)
        _write_report(summary, report)
        return summary
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_available = True
    except Exception as exc:  # noqa: BLE001
        sklearn_available = False
        sklearn_error = str(exc)
    if not sklearn_available and not allow_numpy_fallback:
        summary = _blocked(f"scikit-learn unavailable and numpy fallback disabled: {sklearn_error}", readiness, sklearn_used=False, numpy_used=False)
        _write_report(summary, report)
        return summary

    split = _scene_split(scenes)
    labels_trained: list[str] = []
    labels_skipped: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    model_notes: list[str] = []
    for label in trainable:
        idx = window_labels.index(label)
        y_full = (window_y[:, idx] > 0).astype(np.int8)
        train_idx, test_idx = _balanced_indices(y_full, split, scenes, samples)
        if len(train_idx) < 40 or len(test_idx) < 10:
            labels_skipped[label] = "not enough balanced grouped train/test examples"
            continue
        if len(set(y_full[train_idx].tolist())) < 2 or len(set(y_full[test_idx].tolist())) < 2:
            labels_skipped[label] = "grouped split lacks both classes"
            continue
        if sklearn_available:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
            model.fit(X[train_idx], y_full[train_idx])
            pred = model.predict(X[test_idx])
            weights = _sklearn_weights(model, feature_names)
            sklearn_used = True
            numpy_used = False
        else:
            weights_raw, prep = _fit_numpy_ridge(X[train_idx], y_full[train_idx])
            pred = _predict_numpy_ridge(X[test_idx], weights_raw, prep)
            weights = _top_weights(weights_raw[1:], feature_names)
            sklearn_used = False
            numpy_used = True
        stats = _metrics(y_full[test_idx], pred)
        stats.update(
            {
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_positive": int(y_full[train_idx].sum()),
                "train_negative": int(len(train_idx) - y_full[train_idx].sum()),
                "test_positive": int(y_full[test_idx].sum()),
                "test_negative": int(len(test_idx) - y_full[test_idx].sum()),
                "top_feature_weights": weights,
            }
        )
        metrics[label] = stats
        note_path = out_path / f"{_safe(label)}.silver_v2_proxy_model_note.json"
        note_path.write_text(json.dumps({"label": label, "metrics": stats, "is_human_supervised": False}, indent=2, ensure_ascii=False), encoding="utf-8")
        model_notes.append(str(note_path))
        labels_trained.append(label)
    summary = {
        "trained": bool(labels_trained),
        "reason": "ok" if labels_trained else "all labels skipped after grouped balancing",
        "labels_attempted": trainable,
        "labels_trained": labels_trained,
        "labels_skipped": labels_skipped,
        "sklearn_used": bool(sklearn_available and labels_trained),
        "numpy_fallback_used": bool((not sklearn_available) and labels_trained),
        "model_notes": model_notes,
        "metrics": metrics,
        "readiness_report": str(readiness),
        "is_human_supervised": False,
        "warning": "Metrics are proxy/rule reproducibility metrics, not human semantic accuracy.",
    }
    _write_report(summary, report)
    return summary


def _blocked(reason: str, readiness: str | Path, sklearn_used: bool, numpy_used: bool) -> dict[str, Any]:
    return {
        "trained": False,
        "reason": reason,
        "labels_attempted": [],
        "labels_trained": [],
        "labels_skipped": {},
        "sklearn_used": sklearn_used,
        "numpy_fallback_used": numpy_used,
        "metrics": {},
        "readiness_report": str(readiness),
        "is_human_supervised": False,
        "warning": "No human semantic training occurred.",
    }


def _scene_split(scenes: np.ndarray) -> dict[str, np.ndarray]:
    unique = sorted(set(str(x) for x in scenes.tolist()))
    if len(unique) < 3:
        return {"train": np.ones(len(scenes), dtype=bool), "test": np.zeros(len(scenes), dtype=bool)}
    train_cut = max(1, int(len(unique) * 0.7))
    train_scenes = set(unique[:train_cut])
    return {"train": np.asarray([scene in train_scenes for scene in scenes], dtype=bool), "test": np.asarray([scene not in train_scenes for scene in scenes], dtype=bool)}


def _balanced_indices(y: np.ndarray, split: dict[str, np.ndarray], scenes: np.ndarray, samples: np.ndarray, max_per_scene: int = 500, max_per_sample: int = 80) -> tuple[np.ndarray, np.ndarray]:
    train = _balanced_for_mask(y, split["train"], scenes, samples, max_per_scene, max_per_sample)
    test = _balanced_for_mask(y, split["test"], scenes, samples, max_per_scene, max_per_sample)
    return train, test


def _balanced_for_mask(y: np.ndarray, mask: np.ndarray, scenes: np.ndarray, samples: np.ndarray, max_per_scene: int, max_per_sample: int) -> np.ndarray:
    pos = [idx for idx in np.where(mask & (y > 0))[0].tolist()]
    neg = [idx for idx in np.where(mask & (y == 0))[0].tolist()]
    pos = _cap(pos, scenes, samples, max_per_scene, max_per_sample)
    neg = _cap(neg, scenes, samples, max_per_scene, max_per_sample)
    n = min(len(pos), len(neg))
    if n == 0:
        return np.asarray([], dtype=np.int64)
    selected = sorted(pos[:n] + neg[:n])
    return np.asarray(selected, dtype=np.int64)


def _cap(indices: list[int], scenes: np.ndarray, samples: np.ndarray, max_per_scene: int, max_per_sample: int) -> list[int]:
    per_scene: dict[str, int] = {}
    per_sample: dict[str, int] = {}
    out: list[int] = []
    for idx in sorted(indices, key=lambda i: (str(scenes[i]), str(samples[i]), i)):
        scene = str(scenes[idx])
        sample = str(samples[idx])
        if per_scene.get(scene, 0) >= max_per_scene or per_sample.get(sample, 0) >= max_per_sample:
            continue
        out.append(idx)
        per_scene[scene] = per_scene.get(scene, 0) + 1
        per_sample[sample] = per_sample.get(sample, 0) + 1
    return out


def _fit_numpy_ridge(X: np.ndarray, y: np.ndarray, reg: float = 1.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_imp = np.where(np.isfinite(X), X, med)
    mean = X_imp.mean(axis=0)
    std = X_imp.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    X_std = (X_imp - mean) / std
    X_aug = np.concatenate([np.ones((X_std.shape[0], 1), dtype=np.float32), X_std], axis=1)
    y2 = np.where(y > 0, 1.0, -1.0).astype(np.float32)
    eye = np.eye(X_aug.shape[1], dtype=np.float32)
    eye[0, 0] = 0.0
    lhs = X_aug.T @ X_aug + reg * eye
    rhs = X_aug.T @ y2
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    weights = np.where(np.isfinite(weights), weights, 0.0)
    return weights.astype(np.float32), {"median": med.astype(np.float32), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _predict_numpy_ridge(X: np.ndarray, weights: np.ndarray, prep: dict[str, np.ndarray]) -> np.ndarray:
    X_imp = np.where(np.isfinite(X), X, prep["median"])
    X_std = (X_imp - prep["mean"]) / prep["std"]
    X_aug = np.concatenate([np.ones((X_std.shape[0], 1), dtype=np.float32), X_std], axis=1)
    return (X_aug @ weights >= 0).astype(np.int8)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {"accuracy": round(accuracy, 5), "precision": round(precision, 5), "recall": round(recall, 5), "f1": round(f1, 5), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _sklearn_weights(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    try:
        coefs = model.steps[-1][1].coef_[0]
        return _top_weights(coefs, feature_names)
    except Exception:
        return []


def _top_weights(weights: np.ndarray, feature_names: list[str], limit: int = 12) -> list[dict[str, Any]]:
    ranked = sorted(enumerate(weights.tolist()), key=lambda item: abs(float(item[1])), reverse=True)[:limit]
    return [{"feature": feature_names[idx], "weight": round(float(value), 6)} for idx, value in ranked]


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Silver Baseline v1",
        "",
        "This is a balanced silver-v2 proxy baseline. It is not human-supervised semantic learning.",
        "",
        f"- Trained: {summary['trained']}",
        f"- Reason: {summary['reason']}",
        f"- sklearn_used: {summary['sklearn_used']}",
        f"- numpy_fallback_used: {summary['numpy_fallback_used']}",
        f"- Labels attempted: {summary.get('labels_attempted') or 'None'}",
        f"- Labels trained: {summary.get('labels_trained') or 'None'}",
        f"- Warning: {summary['warning']}",
        "",
    ]
    if summary.get("labels_skipped"):
        lines.extend(["## Labels Skipped", ""])
        for label, reason in summary["labels_skipped"].items():
            lines.append(f"- `{label}`: {reason}")
    if summary.get("metrics"):
        lines.extend(["", "## Proxy Metrics", ""])
        for label, stats in summary["metrics"].items():
            compact = {k: v for k, v in stats.items() if k != "top_feature_weights"}
            lines.append(f"- `{label}`: {compact}")
    lines.extend(["", "Metrics measure rule/proxy reproducibility only, not human semantic correctness."])
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Small Cowgirl-only supervised baseline for review assistance."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl


TARGETS = ["label_cowgirl_candidate", "label_clean_motion", "label_generation_safe"]


def train_cowgirl_ml_baseline_v1(
    feature_table: str | Path,
    metadata: str | Path,
    splits: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        summary = _train_numpy_fallback(feature_table, metadata, splits, out, f"scikit-learn unavailable: {exc}")
        _write_reports(out, summary)
        return summary

    with np.load(feature_table, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int8)
        label_names = [str(x) for x in data["label_names"].tolist()]
        feature_names = [str(x) for x in data["feature_names"].tolist()]
    train_idx = _load_indices(Path(splits) / "train_indices.json")
    test_idx = _load_indices(Path(splits) / "test_indices.json")
    if not test_idx:
        test_idx = _load_indices(Path(splits) / "val_indices.json")
    meta = load_jsonl(metadata)
    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    top_features: dict[str, list[dict[str, Any]]] = {}

    for target in TARGETS:
        if target not in label_names:
            metrics[target] = {"trained": False, "reason": "target missing"}
            continue
        j = label_names.index(target)
        train = [i for i in train_idx if int(y[i, j]) in {0, 1}]
        test = [i for i in test_idx if int(y[i, j]) in {0, 1}]
        counts = Counter(int(y[i, j]) for i in train)
        if len(counts) < 2 or min(counts.values()) < 3:
            metrics[target] = {"trained": False, "reason": f"insufficient train labels/classes: {dict(counts)}"}
            continue
        if len(test) < 2 or len({int(y[i, j]) for i in test}) < 2:
            metrics[target] = {"trained": False, "reason": "test split lacks both classes; refusing overclaimed metric"}
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(X[train], y[train, j])
        pred = model.predict(X[test])
        prob = model.predict_proba(X[test])[:, 1]
        precision, recall, f1, support = precision_recall_fscore_support(y[test, j], pred, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y[test, j], pred, labels=[0, 1]).tolist()
        model_file = out / f"{target}.pkl"
        with model_file.open("wb") as f:
            pickle.dump(model, f)
        models[target] = str(model_file)
        metrics[target] = {
            "trained": True,
            "model_type": "sklearn LogisticRegression pipeline",
            "train_rows": len(train),
            "test_rows": len(test),
            "train_class_counts": dict(counts),
            "precision": {"false": float(precision[0]), "true": float(precision[1])},
            "recall": {"false": float(recall[0]), "true": float(recall[1])},
            "f1": {"false": float(f1[0]), "true": float(f1[1])},
            "support": {"false": int(support[0]), "true": int(support[1])},
            "confusion_matrix_labels_false_true": cm,
            "probability_range": [float(np.min(prob)), float(np.max(prob))],
        }
        top_features[target] = _top_features(model, feature_names)

    feature_file = out / "feature_names.json"
    feature_file.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    summary = {
        "schema": "cowgirl_baseline_model_v1",
        "trained": bool(models),
        "review_assist_only": True,
        "automatic_labeling_allowed": False,
        "generative_model": False,
        "model_type": "sklearn LogisticRegression" if models else "blocked",
        "models": models,
        "feature_names": str(feature_file),
        "metrics": metrics,
        "top_features": top_features,
        "metadata_rows": len(meta),
        "warnings": ["Small human-reviewed dataset; use only for ranking review candidates.", "No weak/silver/machine labels were used as targets."],
    }
    _write_reports(out, summary)
    return summary


def _train_numpy_fallback(feature_table: str | Path, metadata: str | Path, splits: str | Path, out: Path, reason: str) -> dict[str, Any]:
    with np.load(feature_table, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int8)
        label_names = [str(x) for x in data["label_names"].tolist()]
        feature_names = [str(x) for x in data["feature_names"].tolist()]
    train_idx = _load_indices(Path(splits) / "train_indices.json")
    test_idx = _load_indices(Path(splits) / "test_indices.json") or _load_indices(Path(splits) / "val_indices.json")
    models = {}
    metrics = {}
    top_features = {}
    for target in TARGETS:
        if target not in label_names:
            metrics[target] = {"trained": False, "reason": "target missing"}
            continue
        j = label_names.index(target)
        train = [i for i in train_idx if int(y[i, j]) in {0, 1}]
        test = [i for i in test_idx if int(y[i, j]) in {0, 1}]
        counts = Counter(int(y[i, j]) for i in train)
        if len(counts) < 2 or min(counts.values()) < 3:
            metrics[target] = {"trained": False, "reason": f"insufficient train labels/classes: {dict(counts)}"}
            continue
        if len(test) < 2 or len({int(y[i, j]) for i in test}) < 2:
            metrics[target] = {"trained": False, "reason": "test split lacks both classes; refusing overclaimed metric"}
            continue
        model = _fit_numpy_logistic(X[train], y[train, j])
        prob = _predict_numpy(model, X[test])
        pred = (prob >= 0.5).astype(np.int8)
        stats = _binary_metrics(y[test, j], pred, prob)
        model_file = out / f"{target}.numpy_logistic.npz"
        np.savez_compressed(model_file, weights=model["weights"], bias=model["bias"], median=model["median"], scale=model["scale"])
        models[target] = str(model_file)
        metrics[target] = {"trained": True, "model_type": "NumPy logistic regression fallback", "train_rows": len(train), "test_rows": len(test), "train_class_counts": dict(counts), **stats}
        top_features[target] = _top_features_numpy(model["weights"], feature_names)
    feature_file = out / "feature_names.json"
    feature_file.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    return {
        "schema": "cowgirl_baseline_model_v1",
        "trained": bool(models),
        "reason": reason,
        "review_assist_only": True,
        "automatic_labeling_allowed": False,
        "generative_model": False,
        "model_type": "NumPy logistic regression fallback" if models else "blocked",
        "models": models,
        "feature_names": str(feature_file),
        "metrics": metrics,
        "top_features": top_features,
        "metadata_rows": len(load_jsonl(metadata)),
        "warnings": [reason, "NumPy fallback is a tiny baseline ranker; use only for review prioritization.", "No weak/silver/machine labels were used as targets."],
    }


def _fit_numpy_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 600, lr: float = 0.08, l2: float = 0.01) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    median = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    X = np.where(np.isfinite(X), X, median)
    scale = np.std(X, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    Xs = (X - median) / scale
    w = np.zeros(Xs.shape[1], dtype=np.float32)
    b = np.float32(0.0)
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    weights = np.where(y > 0.5, neg / pos, 1.0).astype(np.float32)
    for _ in range(epochs):
        z = np.clip(Xs @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * weights
        grad_w = (Xs.T @ err) / len(y) + l2 * w
        grad_b = np.asarray(err.mean(), dtype=np.float32)
        w -= lr * grad_w.astype(np.float32)
        b -= lr * grad_b
    return {"weights": w, "bias": np.asarray(b, dtype=np.float32), "median": median, "scale": scale}


def _predict_numpy(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, model["median"])
    Xs = (X - model["median"]) / model["scale"]
    z = np.clip(Xs @ model["weights"] + float(model["bias"]), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)
    out = {}
    for label, value in [("false", 0), ("true", 1)]:
        tp = int(((y_true == value) & (y_pred == value)).sum())
        fp = int(((y_true != value) & (y_pred == value)).sum())
        fn = int(((y_true == value) & (y_pred != value)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.setdefault("precision", {})[label] = precision
        out.setdefault("recall", {})[label] = recall
        out.setdefault("f1", {})[label] = f1
        out.setdefault("support", {})[label] = int((y_true == value).sum())
    out["confusion_matrix_labels_false_true"] = [
        [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
        [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
    ]
    out["probability_range"] = [float(np.min(prob)), float(np.max(prob))]
    return out


def _top_features_numpy(weights: np.ndarray, feature_names: list[str], limit: int = 15) -> list[dict[str, Any]]:
    pairs = sorted(enumerate(weights), key=lambda p: abs(float(p[1])), reverse=True)[:limit]
    return [{"feature": feature_names[i], "coefficient": float(c)} for i, c in pairs]


def _load_indices(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [int(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def _top_features(model: Any, feature_names: list[str], limit: int = 15) -> list[dict[str, Any]]:
    try:
        coefs = model.named_steps["logisticregression"].coef_[0]
    except Exception:
        return []
    pairs = sorted(enumerate(coefs), key=lambda p: abs(float(p[1])), reverse=True)[:limit]
    return [{"feature": feature_names[i], "coefficient": float(c)} for i, c in pairs]


def _blocked(out: Path, reason: str) -> dict[str, Any]:
    return {
        "schema": "cowgirl_baseline_model_v1",
        "trained": False,
        "reason": reason,
        "review_assist_only": True,
        "automatic_labeling_allowed": False,
        "generative_model": False,
        "models": {},
        "metrics": {},
        "warnings": [reason],
    }


def _write_reports(out: Path, summary: dict[str, Any]) -> None:
    dump_json(out / "training_summary.json", summary)
    lines = [
        "# Cowgirl Baseline Model v1",
        "",
        f"- Trained: `{summary.get('trained')}`",
        f"- Model type: `{summary.get('model_type', 'blocked')}`",
        f"- Review assist only: `{summary.get('review_assist_only')}`",
        f"- Automatic labeling allowed: `{summary.get('automatic_labeling_allowed')}`",
        f"- Generative model: `{summary.get('generative_model')}`",
        "",
        "## Metrics",
        "",
    ]
    if summary.get("metrics"):
        for target, stats in summary["metrics"].items():
            lines.append(f"- `{target}`: `{stats}`")
    else:
        lines.append(f"- blocked: {summary.get('reason')}")
    if summary.get("top_features"):
        lines.extend(["", "## Top Feature Signals", ""])
        for target, feats in summary["top_features"].items():
            lines.append(f"### {target}")
            for feat in feats[:10]:
                lines.append(f"- `{feat['feature']}`: {feat['coefficient']:.4f}")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {w}" for w in summary.get("warnings", []))
    (out / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_readiness(out / "cowgirl_ml_readiness_v1.md", summary)


def _write_readiness(path: Path, summary: dict[str, Any]) -> None:
    trained_targets = [k for k, v in (summary.get("metrics") or {}).items() if v.get("trained")]
    lines = [
        "# Cowgirl ML Readiness v1",
        "",
        f"- Enough labels for at least one review-assist model: `{bool(trained_targets)}`",
        f"- Trained targets: `{trained_targets}`",
        "- Use for review ranking: `yes`" if trained_targets else "- Use for review ranking: `blocked`",
        "- Use for candidate DB scoring: `experimental review-assist only`" if trained_targets else "- Use for candidate DB scoring: `no`",
        "- Use for automatic labels: `no`",
        "- Use for generation-safe selection: `no`",
        "",
        "This model is not ground truth and must not write `manual_labels.yaml`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

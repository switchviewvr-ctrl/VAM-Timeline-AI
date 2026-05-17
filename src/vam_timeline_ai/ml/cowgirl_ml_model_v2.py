"""Cowgirl ML v2 review-ranker training and scoring."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl


TARGETS = [
    "label_cowgirl_semantic_family",
    "label_cowgirl_clean_motion",
    "label_cowgirl_pose_context",
    "label_not_cowgirl_bj_oral",
    "label_generation_safe_or_complete",
]


def train_cowgirl_ml_v2(feature_table: str | Path, metadata: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(feature_table, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int8)
        label_names = [str(x) for x in data["label_names"].tolist()]
        feature_names = [str(x) for x in data["feature_names"].tolist()]
    meta = load_jsonl(metadata)

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_available = True
        sklearn_error = ""
    except Exception as exc:
        sklearn_available = False
        sklearn_error = str(exc)

    models: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    top_features: dict[str, list[dict[str, Any]]] = {}
    splits: dict[str, Any] = {}

    for target in TARGETS:
        if target not in label_names:
            metrics[target] = {"trained": False, "reason": "target missing"}
            continue
        j = label_names.index(target)
        labeled = [i for i in range(len(meta)) if int(y[i, j]) in {0, 1}]
        counts = Counter(int(y[i, j]) for i in labeled)
        if len(counts) < 2 or min(counts.values()) < 2:
            metrics[target] = {"trained": False, "reason": f"insufficient labeled classes: {dict(counts)}", "labeled_rows": len(labeled)}
            continue
        train_idx, test_idx, split_note = _grouped_split(labeled, y[:, j], meta)
        if len({int(y[i, j]) for i in train_idx}) < 2:
            metrics[target] = {"trained": False, "reason": f"grouped train split lacks both classes: {split_note}", "labeled_rows": len(labeled)}
            continue

        if sklearn_available:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            model.fit(X[train_idx], y[train_idx, j])
            prob = model.predict_proba(X[test_idx])[:, 1] if test_idx else np.asarray([], dtype=np.float32)
            pred = model.predict(X[test_idx]) if test_idx else np.asarray([], dtype=np.int8)
            model_file = out / f"{target}.pkl"
            with model_file.open("wb") as f:
                pickle.dump(model, f)
            models[target] = str(model_file)
            top_features[target] = _top_features_sklearn(model, feature_names)
            model_type = "sklearn LogisticRegression pipeline"
        else:
            model = _fit_numpy_logistic(X[train_idx], y[train_idx, j])
            prob = _predict_numpy(model, X[test_idx]) if test_idx else np.asarray([], dtype=np.float32)
            pred = (prob >= 0.5).astype(np.int8) if test_idx else np.asarray([], dtype=np.int8)
            model_file = out / f"{target}.numpy_logistic.npz"
            np.savez_compressed(model_file, weights=model["weights"], bias=model["bias"], median=model["median"], scale=model["scale"])
            models[target] = str(model_file)
            top_features[target] = _top_features_numpy(model["weights"], feature_names)
            model_type = f"NumPy logistic fallback ({sklearn_error})"

        metric_values = (
            _metrics(y[test_idx, j], pred, prob)
            if len(test_idx) and len({int(y[i, j]) for i in test_idx}) >= 2
            else {"heldout_metric_status": "not_available_or_single_class_test"}
        )
        metrics[target] = {
            "trained": True,
            "model_type": model_type,
            "labeled_rows": len(labeled),
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_class_counts": dict(Counter(int(y[i, j]) for i in train_idx)),
            "test_class_counts": dict(Counter(int(y[i, j]) for i in test_idx)),
            "grouped_split_note": split_note,
            **metric_values,
        }
        splits[target] = {"train_indices": train_idx, "test_indices": test_idx, "note": split_note}

    (out / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    dump_json(out / "splits_v2.json", splits)
    summary = {
        "schema": "cowgirl_ml_model_v2",
        "trained": bool(models),
        "review_ranker_only": True,
        "not_auto_labeler": True,
        "generative_model": False,
        "sklearn_used": sklearn_available and bool(models),
        "numpy_fallback_used": (not sklearn_available) and bool(models),
        "models": models,
        "metrics": metrics,
        "top_features": top_features,
        "metadata_rows": len(meta),
        "trust_verdict": "review_ranker_only" if models else "blocked",
        "warnings": ["Small human-reviewed dataset. Use probabilities only for review prioritization.", "No unreviewed gate output was used as target truth."],
    }
    _write_training_reports(out, summary)
    return summary


def score_new_scenes_cowgirl_ml_v2(
    model_dir: str | Path,
    feature_table: str | Path,
    metadata: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    with np.load(feature_table, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        label_names = [str(x) for x in data["label_names"].tolist()]
    meta = load_jsonl(metadata)
    models = _load_models(model_path)
    rows = []
    for i, m in enumerate(meta):
        probs = {target: _predict_model(model, X[i : i + 1]) for target, model in models.items()}
        row = dict(m)
        row.update(
            {
                "model_cowgirl_family_probability": probs.get("label_cowgirl_semantic_family"),
                "model_cowgirl_clean_motion_probability": probs.get("label_cowgirl_clean_motion"),
                "model_pose_context_probability": probs.get("label_cowgirl_pose_context"),
                "model_bj_oral_negative_probability": probs.get("label_not_cowgirl_bj_oral"),
                "model_generation_incomplete_probability": _incomplete_probability(probs.get("label_generation_safe_or_complete")),
                "uncertainty_score": _uncertainty(probs),
                "disagreement_with_gates": _disagreement(m, probs),
                "recommended_review_bucket": _bucket(m, probs),
                "review_ranker_only": True,
                "not_auto_labeler": True,
            }
        )
        rows.append(row)
    write_jsonl(out_jsonl, rows)
    summary = {
        "status": "ok" if models else "blocked",
        "rows": len(rows),
        "models": sorted(models),
        "bucket_counts": dict(Counter(r.get("recommended_review_bucket") for r in rows)),
        "score_ranges": _score_ranges(rows),
        "review_ranker_only": True,
    }
    _write_score_report(report, summary)
    return summary


def _grouped_split(indices: list[int], y: np.ndarray, meta: list[dict[str, Any]]) -> tuple[list[int], list[int], str]:
    groups: dict[str, list[int]] = {}
    for i in indices:
        group = str(meta[i].get("sample_id") or meta[i].get("source_id") or meta[i].get("source_scene_file") or i)
        groups.setdefault(group, []).append(i)
    group_items = sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
    negatives = [(g, xs) for g, xs in group_items if any(int(y[i]) == 0 for i in xs)]
    positives = [(g, xs) for g, xs in group_items if any(int(y[i]) == 1 for i in xs)]
    # Prefer a two-group holdout with both classes while keeping both classes in train.
    for pos_group, pos_indices in positives:
        for neg_group, neg_indices in negatives:
            if pos_group == neg_group:
                continue
            test_groups = {pos_group, neg_group}
            test = [i for g, xs in group_items if g in test_groups for i in xs]
            train = [i for g, xs in group_items if g not in test_groups for i in xs]
            if len({int(y[i]) for i in train}) >= 2 and len({int(y[i]) for i in test}) >= 2:
                return train, test, f"held out groups {sorted(test_groups)}"
    for test_group, test_indices in group_items:
        train = [i for g, xs in group_items if g != test_group for i in xs]
        if len({int(y[i]) for i in train}) >= 2:
            return train, test_indices, f"held out group {test_group}; test may be single-class"
    # Last resort: grouped split impossible, train all and report no held-out metric.
    return list(indices), [], "grouped held-out split unavailable; trained on all labeled rows for review ranking"


def _fit_numpy_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 700, lr: float = 0.08, l2: float = 0.01) -> dict[str, np.ndarray]:
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
        w -= lr * ((Xs.T @ err) / len(y) + l2 * w).astype(np.float32)
        b -= lr * np.asarray(err.mean(), dtype=np.float32)
    return {"weights": w, "bias": np.asarray(b, dtype=np.float32), "median": median, "scale": scale}


def _predict_numpy(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, model["median"])
    Xs = (X - model["median"]) / model["scale"]
    z = np.clip(Xs @ model["weights"] + float(model["bias"]), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _load_models(model_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for target in TARGETS:
        pkl = model_dir / f"{target}.pkl"
        if pkl.exists():
            with pkl.open("rb") as f:
                out[target] = {"kind": "sklearn", "model": pickle.load(f)}
            continue
        npz = model_dir / f"{target}.numpy_logistic.npz"
        if npz.exists():
            with np.load(npz, allow_pickle=True) as data:
                out[target] = {"kind": "numpy", "weights": data["weights"], "bias": data["bias"], "median": data["median"], "scale": data["scale"]}
    return out


def _predict_model(model: dict[str, Any], x: np.ndarray) -> float:
    if model["kind"] == "sklearn":
        return float(model["model"].predict_proba(x)[:, 1][0])
    return float(_predict_numpy(model, x)[0])


def _metrics(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int8)
    pred = pred.astype(np.int8)
    out: dict[str, Any] = {"precision": {}, "recall": {}, "f1": {}, "support": {}}
    for label, name in [(0, "false"), (1, "true")]:
        tp = int(((y_true == label) & (pred == label)).sum())
        fp = int(((y_true != label) & (pred == label)).sum())
        fn = int(((y_true == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out["precision"][name] = float(precision)
        out["recall"][name] = float(recall)
        out["f1"][name] = float(f1)
        out["support"][name] = int((y_true == label).sum())
    out["confusion_matrix_labels_false_true"] = [
        [int(((y_true == 0) & (pred == 0)).sum()), int(((y_true == 0) & (pred == 1)).sum())],
        [int(((y_true == 1) & (pred == 0)).sum()), int(((y_true == 1) & (pred == 1)).sum())],
    ]
    out["probability_range"] = [float(np.min(prob)), float(np.max(prob))]
    return out


def _top_features_sklearn(model: Any, names: list[str], limit: int = 12) -> list[dict[str, Any]]:
    try:
        coefs = model.named_steps["logisticregression"].coef_[0]
    except Exception:
        return []
    return [{"feature": names[i], "coefficient": float(c)} for i, c in sorted(enumerate(coefs), key=lambda p: abs(float(p[1])), reverse=True)[:limit]]


def _top_features_numpy(weights: np.ndarray, names: list[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"feature": names[i], "coefficient": float(c)} for i, c in sorted(enumerate(weights), key=lambda p: abs(float(p[1])), reverse=True)[:limit]]


def _incomplete_probability(gen_safe_probability: float | None) -> float | None:
    return None if gen_safe_probability is None else float(1.0 - gen_safe_probability)


def _uncertainty(probs: dict[str, float]) -> float:
    vals = [p for p in probs.values() if p is not None]
    if not vals:
        return 1.0
    return float(max(0.0, 1.0 - min(abs(p - 0.5) for p in vals) * 2.0))


def _disagreement(meta: dict[str, Any], probs: dict[str, float]) -> list[str]:
    flags = []
    category = str(meta.get("category") or "")
    p = probs.get("label_cowgirl_semantic_family")
    clean = probs.get("label_cowgirl_clean_motion")
    if p is not None and category.startswith("cowgirl") and p < 0.4:
        flags.append("gate_cowgirl_model_negative")
    if p is not None and not category.startswith("cowgirl") and p > 0.65:
        flags.append("model_cowgirl_gate_non_cowgirl")
    if clean is not None and category == "cowgirl_clean_cyclic_motion" and clean < 0.45:
        flags.append("clean_gate_model_low")
    return flags


def _bucket(meta: dict[str, Any], probs: dict[str, float]) -> str:
    flags = _disagreement(meta, probs)
    p = probs.get("label_cowgirl_semantic_family")
    clean = probs.get("label_cowgirl_clean_motion")
    bj = probs.get("label_not_cowgirl_bj_oral")
    incomplete = _incomplete_probability(probs.get("label_generation_safe_or_complete"))
    category = str(meta.get("category") or "")
    final_gate = str(meta.get("final_clean_motion_gate") or "")
    driver = str(meta.get("primary_driver_controller") or "")
    gate_clean_cowgirl = category == "cowgirl_clean_cyclic_motion" and final_gate == "pass" and driver == "hipControl"
    if incomplete is not None and incomplete > 0.65 and p is not None and p > 0.6:
        return "incomplete_pose_but_semantic_cowgirl"
    if flags:
        return "model_gate_disagreement"
    if bj is not None and bj > 0.7:
        return "likely_bj_or_hj_negative"
    if gate_clean_cowgirl and p is not None and clean is not None and p > 0.75 and clean > 0.65:
        return "high_confidence_clean_cowgirl"
    if "pose_context" in category or (p is not None and p > 0.55 and clean is not None and clean < 0.45):
        return "cowgirl_pose_context"
    return "uncertain_boundary"


def _score_ranges(rows: list[dict[str, Any]]) -> dict[str, list[float | None]]:
    out = {}
    for key in ["model_cowgirl_family_probability", "model_cowgirl_clean_motion_probability", "model_bj_oral_negative_probability"]:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = [float(min(vals)), float(max(vals))] if vals else [None, None]
    return out


def _write_training_reports(out: Path, summary: dict[str, Any]) -> None:
    dump_json(out / "training_summary.json", summary)
    lines = [
        "# Cowgirl ML v2",
        "",
        f"- Trained: `{summary['trained']}`",
        f"- Trust verdict: `{summary['trust_verdict']}`",
        f"- Review ranker only: `{summary['review_ranker_only']}`",
        f"- Not auto-labeler: `{summary['not_auto_labeler']}`",
        "",
        "## Metrics",
        "",
    ]
    for target, stats in summary["metrics"].items():
        lines.append(f"- `{target}`: `{stats}`")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {w}" for w in summary["warnings"])
    (out / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_score_report(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cowgirl ML Scores v2",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Models: `{summary['models']}`",
        f"- Bucket counts: `{summary['bucket_counts']}`",
        f"- Score ranges: `{summary['score_ranges']}`",
        "- Review ranker only: `true`",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

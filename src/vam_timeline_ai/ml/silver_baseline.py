"""Silver-supervised baseline for feature sanity checks.

This is deliberately framed as rule/proxy imitation. It is not human-supervised
semantic learning and does not prove semantic accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.ml.silver_readiness import analyze_silver_readiness


def train_silver_baseline_v0(dataset: str | Path, out_dir: str | Path, report: str | Path) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    readiness_report = Path(report).with_name("silver_baseline_v0_readiness_check.md")
    silver_labels_guess = Path(dataset).parents[2] / "labels" / "machine_proposals" / "silver_labels_v1.jsonl"
    readiness = analyze_silver_readiness(dataset, silver_labels_guess, readiness_report) if silver_labels_guess.exists() else {"eligible_labels": []}
    eligible = readiness.get("eligible_labels", [])
    if not eligible:
        summary = {
            "trained": False,
            "reason": "readiness blocked: no silver labels met proxy-training coverage thresholds",
            "labels_trained": [],
            "labels_skipped": {},
            "is_human_supervised": False,
            "warning": "Silver baseline did not run; no human semantic training occurred.",
        }
        _write_report(summary, report)
        return summary
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        summary = {
            "trained": False,
            "reason": f"scikit-learn unavailable: {exc}",
            "labels_trained": [],
            "labels_skipped": {label: "scikit-learn unavailable" for label in eligible},
            "is_human_supervised": False,
            "warning": "No model was trained. Install optional ml dependencies for proxy baselines.",
        }
        _write_report(summary, report)
        return summary

    with np.load(dataset, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        label_names = [str(x) for x in data["silver_label_names"].tolist()]
        y_pos = data["silver_y_positive_multilabel"]
        y_neg = data["silver_y_negative_multilabel"]
        scenes = np.asarray([str(x) for x in data["group_scene"].tolist()], dtype=object)
        feature_names = [str(x) for x in data["feature_names"].tolist()]
    split = _scene_split(scenes)
    labels_trained: list[str] = []
    labels_skipped: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    for label in eligible:
        if label not in label_names:
            labels_skipped[label] = "label not present in dataset"
            continue
        idx = label_names.index(label)
        mask = (y_pos[:, idx] > 0) | (y_neg[:, idx] > 0)
        if int(mask.sum()) < 100:
            labels_skipped[label] = "not enough positive+negative silver rows"
            continue
        y = (y_pos[:, idx] > 0).astype(np.int8)
        train_mask = mask & split["train"]
        test_mask = mask & split["test"]
        if len(set(y[train_mask].tolist())) < 2 or int(test_mask.sum()) == 0:
            labels_skipped[label] = "grouped split did not preserve both classes"
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
        model.fit(X[train_mask], y[train_mask])
        score = float(model.score(X[test_mask], y[test_mask])) if int(test_mask.sum()) else float("nan")
        model_note = out_path / f"{_safe(label)}.silver_proxy_model_note.json"
        model_note.write_text(
            json.dumps(
                {
                    "label": label,
                    "is_human_supervised": False,
                    "note": "Model fit was a silver-label proxy sanity check; model serialization omitted.",
                    "top_feature_coefficients": _top_coefficients(model, feature_names),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        labels_trained.append(label)
        metrics[label] = {
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "test_score_rule_proxy_only": score,
            "model_note": str(model_note),
        }
    summary = {
        "trained": bool(labels_trained),
        "reason": "ok" if labels_trained else "all eligible labels skipped after grouped split checks",
        "labels_trained": labels_trained,
        "labels_skipped": labels_skipped,
        "metrics": metrics,
        "is_human_supervised": False,
        "warning": "This is weak-supervised rule/proxy imitation. Metrics are not semantic accuracy.",
    }
    _write_report(summary, report)
    return summary


def _scene_split(scenes: np.ndarray) -> dict[str, np.ndarray]:
    unique = sorted(set(str(x) for x in scenes.tolist()))
    if len(unique) < 3:
        return {"train": np.ones(len(scenes), dtype=bool), "test": np.zeros(len(scenes), dtype=bool)}
    train_cut = max(1, int(len(unique) * 0.7))
    train_scenes = set(unique[:train_cut])
    return {"train": np.asarray([scene in train_scenes for scene in scenes], dtype=bool), "test": np.asarray([scene not in train_scenes for scene in scenes], dtype=bool)}


def _top_coefficients(model: Any, feature_names: list[str], limit: int = 12) -> list[dict[str, Any]]:
    try:
        clf = model.steps[-1][1]
        coefs = clf.coef_[0]
    except Exception:
        return []
    ranked = sorted(enumerate(coefs), key=lambda item: abs(float(item[1])), reverse=True)[:limit]
    return [{"feature": feature_names[idx], "coefficient": float(value)} for idx, value in ranked]


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Silver Baseline v0",
        "",
        "This baseline uses silver labels only. It is weak-supervised rule/proxy imitation, not human-supervised semantic training.",
        "",
        f"- Trained: {summary['trained']}",
        f"- Reason: {summary['reason']}",
        f"- Human-supervised: {summary['is_human_supervised']}",
        f"- Labels trained: {summary.get('labels_trained') or 'None'}",
        f"- Warning: {summary['warning']}",
        "",
    ]
    if summary.get("labels_skipped"):
        lines.extend(["## Labels Skipped", ""])
        for label, reason in summary["labels_skipped"].items():
            lines.append(f"- `{label}`: {reason}")
    if summary.get("metrics"):
        lines.extend(["", "## Proxy Fit Summary", ""])
        for label, stats in summary["metrics"].items():
            lines.append(f"- `{label}`: {stats}")
    lines.extend(["", "Do not use these metrics as semantic accuracy; the model can learn the same rules that generated the silver labels."])
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

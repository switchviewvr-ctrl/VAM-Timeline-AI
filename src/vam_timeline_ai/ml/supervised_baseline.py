"""Optional supervised baseline guarded by readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.ml.supervised_readiness import analyze_supervised_readiness


def train_supervised_baseline_v0(dataset: str | Path, split_plan: str | Path, out_dir: str | Path, report: str | Path) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    readiness_report = Path(report).with_name("supervised_baseline_v0_readiness_check.md")
    readiness = analyze_supervised_readiness(dataset, Path(dataset).with_name("manual_labels.yaml"), split_plan, readiness_report)
    if not readiness.get("eligible_labels"):
        summary = {
            "trained": False,
            "reason": "readiness blocked: no eligible manual label classes",
            "eligible_labels": [],
            "models_written": [],
        }
        _write_report(summary, report)
        return summary
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        summary = {
            "trained": False,
            "reason": f"scikit-learn unavailable: {exc}",
            "eligible_labels": readiness.get("eligible_labels", []),
            "models_written": [],
        }
        _write_report(summary, report)
        return summary

    with np.load(dataset, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        names = [str(x) for x in data["manual_label_names"].tolist()]
        y_pos = data["manual_y_positive_multilabel"]
        y_neg = data["manual_y_negative_multilabel"]
        include = data["include_for_ml"].astype(bool)
    models_written: list[str] = []
    metrics: dict[str, Any] = {}
    for label in readiness["eligible_labels"]:
        idx = names.index(label)
        mask = include & ((y_pos[:, idx] > 0) | (y_neg[:, idx] > 0))
        if int(mask.sum()) < 40:
            metrics[label] = {"skipped": "not enough positive+negative include_for_ml rows after masking"}
            continue
        y = (y_pos[mask, idx] > 0).astype(np.int8)
        if len(set(y.tolist())) < 2:
            metrics[label] = {"skipped": "only one class present after masking"}
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[mask], y)
        model_path = out_path / f"{_safe(label)}.model_unserialized_note.json"
        model_path.write_text(json.dumps({"label": label, "note": "Model fit in-memory only; serialization intentionally omitted in MVP."}, indent=2), encoding="utf-8")
        models_written.append(str(model_path))
        metrics[label] = {"trained_rows": int(mask.sum()), "positive_rows": int(y.sum()), "negative_rows": int((1 - y).sum())}
    summary = {"trained": bool(models_written), "reason": "ok" if models_written else "no labels trained after final masking", "eligible_labels": readiness["eligible_labels"], "models_written": models_written, "metrics": metrics}
    _write_report(summary, report)
    return summary


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _write_report(summary: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Supervised Baseline v0",
        "",
        f"- Trained: {summary['trained']}",
        f"- Reason: {summary['reason']}",
        f"- Eligible labels: {summary.get('eligible_labels') or 'None'}",
        f"- Models written: {len(summary.get('models_written', []))}",
        "",
        "This baseline uses manual positive/negative labels only. Weak labels, filenames, and atom names are not targets.",
    ]
    if summary.get("metrics"):
        lines.extend(["", "## Metrics / Fit Summary", ""])
        for label, stats in summary["metrics"].items():
            lines.append(f"- `{label}`: {stats}")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

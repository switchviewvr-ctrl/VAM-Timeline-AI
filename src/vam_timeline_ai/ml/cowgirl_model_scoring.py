"""Score candidates with the Cowgirl review-assist baseline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ml.supervised_feature_table import build_all_candidate_feature_matrix


def score_clean_v3_with_cowgirl_model_v1(
    run_dir: str | Path,
    model_dir: str | Path,
    feature_source: str,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    feature_file = model_path / "feature_names.json"
    if not feature_file.exists():
        summary = {"status": "blocked", "reason": "feature_names.json missing", "rows": 0}
        _write_report(report, summary)
        write_jsonl(out_jsonl, [])
        return summary
    feature_names = json.loads(feature_file.read_text(encoding="utf-8"))
    X, records = build_all_candidate_feature_matrix(run_dir, feature_names)
    models = _load_models(model_path)
    if not models:
        summary = {"status": "blocked", "reason": "no trained model files found", "rows": len(records)}
        _write_report(report, summary)
        write_jsonl(out_jsonl, [])
        return summary
    rows = []
    for i, record in enumerate(records):
        scored = _score_record(record, X[i : i + 1], models)
        rows.append(scored)
    write_jsonl(out_jsonl, rows)
    summary = {
        "status": "ok",
        "rows": len(rows),
        "feature_source": feature_source,
        "models": sorted(models),
        "priority_counts": _counts(rows, "recommended_review_priority"),
        "disagreement_count": sum(1 for r in rows if r.get("disagreement_flags")),
        "review_assist_only": True,
    }
    _write_report(report, summary)
    return summary


def _load_models(model_dir: Path) -> dict[str, Any]:
    models = {}
    for name in ["label_cowgirl_candidate", "label_clean_motion", "label_generation_safe"]:
        path = model_dir / f"{name}.pkl"
        if path.exists():
            with path.open("rb") as f:
                models[name] = {"kind": "sklearn", "model": pickle.load(f)}
            continue
        np_path = model_dir / f"{name}.numpy_logistic.npz"
        if np_path.exists():
            with np.load(np_path, allow_pickle=True) as data:
                models[name] = {
                    "kind": "numpy_logistic",
                    "weights": data["weights"],
                    "bias": data["bias"],
                    "median": data["median"],
                    "scale": data["scale"],
                }
    return models


def _score_record(record: dict[str, Any], x: np.ndarray, models: dict[str, Any]) -> dict[str, Any]:
    probs = {}
    for target, model in models.items():
        try:
            probs[target] = float(_predict_model(model, x)[0])
        except Exception:
            probs[target] = None
    heuristic_category = str(record.get("category") or "")
    heuristic_cowgirl = heuristic_category.startswith("cowgirl_") or str(record.get("semantic_family") or "") == "cowgirl"
    cowgirl_p = probs.get("label_cowgirl_candidate")
    clean_p = probs.get("label_clean_motion")
    gen_p = probs.get("label_generation_safe")
    flags = []
    if cowgirl_p is not None:
        if heuristic_cowgirl and cowgirl_p < 0.4:
            flags.append("heuristic_cowgirl_model_negative")
        if (not heuristic_cowgirl) and cowgirl_p > 0.6:
            flags.append("heuristic_negative_model_cowgirl")
    uncertainty = min(abs((cowgirl_p or 0.5) - 0.5), abs((clean_p or 0.5) - 0.5))
    priority = _priority(cowgirl_p, clean_p, gen_p, flags)
    return {
        "window_id": record.get("window_id") or "",
        "sample_id": record.get("sample_id") or "",
        "source_id": record.get("source_id") or "",
        "source_scene_file": record.get("source_scene_file") or "",
        "technical_actor_id": record.get("technical_actor_id") or record.get("technical_atom_id") or "",
        "start_seconds": record.get("start_seconds"),
        "end_seconds": record.get("end_seconds"),
        "model_cowgirl_probability": cowgirl_p,
        "model_clean_motion_probability": clean_p,
        "model_generation_safe_probability": gen_p,
        "heuristic_category": heuristic_category,
        "heuristic_semantic_family": record.get("semantic_family") or "",
        "disagreement_flags": flags,
        "uncertainty_score": float(1.0 - min(0.5, uncertainty) * 2.0),
        "recommended_review_priority": priority,
        "pose_subtype": record.get("pose_subtype") or "",
        "motion_subtype": record.get("motion_subtype") or "",
        "phase": record.get("phase") or "",
        "contact_support": record.get("contact_support") or "",
        "generation_safe": record.get("generation_safe"),
        "review_assist_only": True,
    }


def _predict_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if model.get("kind") == "sklearn":
        return model["model"].predict_proba(x)[:, 1]
    if model.get("kind") == "numpy_logistic":
        X = np.asarray(x, dtype=np.float32)
        X = np.where(np.isfinite(X), X, model["median"])
        Xs = (X - model["median"]) / model["scale"]
        z = np.clip(Xs @ model["weights"] + float(model["bias"]), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("unknown model kind")


def _priority(cowgirl_p: float | None, clean_p: float | None, gen_p: float | None, flags: list[str]) -> str:
    if flags:
        return "model_heuristic_disagreement"
    if cowgirl_p is None:
        return "insufficient_features"
    if 0.4 <= cowgirl_p <= 0.6 or (clean_p is not None and 0.4 <= clean_p <= 0.6):
        return "uncertain_boundary"
    if cowgirl_p >= 0.75 and (clean_p is None or clean_p >= 0.6):
        return "high_confidence_cowgirl"
    if cowgirl_p <= 0.25:
        return "high_confidence_negative"
    if gen_p is not None and gen_p >= 0.6:
        return "generation_safe_candidate_check"
    return "uncertain_boundary"


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Cowgirl Model Scores v1",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Rows scored: `{summary.get('rows')}`",
        f"- Models: `{summary.get('models', [])}`",
        f"- Priority counts: `{summary.get('priority_counts', {})}`",
        f"- Disagreement count: `{summary.get('disagreement_count', 0)}`",
        "",
        "Scores are review-assist probabilities, not automatic labels.",
    ]
    if summary.get("reason"):
        lines.append(f"- Blocked reason: {summary['reason']}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

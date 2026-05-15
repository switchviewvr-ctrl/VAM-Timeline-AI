"""Orchestrate machine-label audit, aggregation, silver v2, and proxy baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_machine_labeling_v2(run_dir: str | Path, min_silver_score: float = 0.78, train_silver_baseline: bool = True, allow_numpy_fallback: bool = True) -> dict[str, Any]:
    from vam_timeline_ai.ml.dataset_v4 import build_ml_dataset_v4
    from vam_timeline_ai.ml.silver_baseline_v1 import train_silver_baseline_v1
    from vam_timeline_ai.ml.silver_readiness_v2 import analyze_silver_readiness_v2
    from vam_timeline_ai.semantics.machine_label_aggregation import aggregate_machine_labels_v2
    from vam_timeline_ai.semantics.machine_label_audit import audit_machine_labels_v1
    from vam_timeline_ai.semantics.machine_proposal_review_batch_v2 import build_machine_proposal_review_batch_v2
    from vam_timeline_ai.semantics.silver_labels_v2 import build_silver_labels_v2
    from vam_timeline_ai.visualization.review_previews import render_review_previews_v1

    run = Path(run_dir)
    proposal_dir = run / "labels" / "machine_proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "proposals": proposal_dir / "machine_label_proposals_v1.jsonl",
        "silver_v1": proposal_dir / "silver_labels_v1.jsonl",
        "windows": run / "semantic" / "movement_windows.jsonl",
        "windows_labeled": run / "semantic" / "movement_windows_labeled.jsonl",
        "pair_windows": run / "semantic" / "pair_windows_v1.jsonl",
        "features": run / "features" / "cowgirl_window_features_v1.jsonl",
        "weak": run / "semantic" / "weak_labels_v2.jsonl",
        "manual_labels": run / "labels" / "manual_labels.yaml",
        "sample_index": run / "baked" / "motion_sample_index.jsonl",
        "controller_map": run / "semantic" / "controller_bodypart_map.json",
        "audit_md": proposal_dir / "machine_label_audit_v1.md",
        "audit_json": proposal_dir / "machine_label_audit_v1.json",
        "window_scores": proposal_dir / "machine_window_label_scores_v2.jsonl",
        "pair_scores": proposal_dir / "machine_pair_label_scores_v2.jsonl",
        "aggregation_report": proposal_dir / "machine_label_aggregation_report_v2.md",
        "silver_window": proposal_dir / "silver_window_labels_v2.jsonl",
        "silver_pair": proposal_dir / "silver_pair_labels_v2.jsonl",
        "silver_yaml": proposal_dir / "silver_labels_v2.yaml",
        "silver_report": proposal_dir / "silver_labels_report_v2.md",
        "dataset_v4": run / "ml" / "datasets" / "cowgirl_ml_dataset_v4.npz",
        "dataset_report": run / "ml" / "reports" / "cowgirl_ml_dataset_v4_report.md",
        "readiness": run / "ml" / "reports" / "silver_readiness_report_v2.md",
        "baseline_dir": run / "ml" / "models" / "silver_baseline_v1",
        "baseline_report": run / "ml" / "reports" / "silver_baseline_v1_report.md",
        "review_batch": run / "labels" / "batches" / "batch_machine_review_002",
        "summary": proposal_dir / "machine_labeling_v2_summary.md",
    }
    missing = [str(paths[key]) for key in ["proposals", "silver_v1", "windows", "pair_windows", "features", "weak"] if not paths[key].exists()]
    if missing:
        summary = {"status": "blocked_missing_inputs", "missing_inputs": missing}
        _write_summary(summary, paths["summary"])
        return summary
    windows_for_dataset = paths["windows_labeled"] if paths["manual_labels"].exists() and paths["windows_labeled"].exists() else paths["windows"]
    audit = audit_machine_labels_v1(run, paths["proposals"], paths["silver_v1"], paths["windows"], paths["pair_windows"], paths["audit_md"], paths["audit_json"])
    aggregation = aggregate_machine_labels_v2(paths["proposals"], paths["window_scores"], paths["pair_scores"], paths["aggregation_report"])
    silver = build_silver_labels_v2(paths["window_scores"], paths["pair_scores"], paths["silver_window"], paths["silver_pair"], paths["silver_yaml"], paths["silver_report"], min_score=min_silver_score)
    dataset = build_ml_dataset_v4(paths["features"], windows_for_dataset, paths["weak"], paths["manual_labels"], paths["silver_window"], paths["silver_pair"], paths["dataset_v4"], paths["dataset_report"])
    readiness = analyze_silver_readiness_v2(paths["dataset_v4"], paths["silver_window"], paths["silver_pair"], paths["readiness"])
    baseline = {"trained": False, "reason": "not requested"}
    if train_silver_baseline:
        baseline = train_silver_baseline_v1(paths["dataset_v4"], paths["readiness"], paths["baseline_dir"], paths["baseline_report"], allow_numpy_fallback=allow_numpy_fallback)
    review_rows = build_machine_proposal_review_batch_v2(run, paths["window_scores"], paths["pair_scores"], paths["silver_window"], paths["silver_pair"], paths["review_batch"])
    if paths["sample_index"].exists() and paths["controller_map"].exists():
        preview = render_review_previews_v1(paths["review_batch"] / "review_batch.jsonl", paths["sample_index"], paths["controller_map"], paths["review_batch"] / "previews")
    else:
        preview = {"items": 0, "matplotlib_available": False, "warnings": ["sample index or controller map missing; previews skipped"]}
    summary = {
        "status": "ok",
        "audit_duplicate_keys": audit.get("duplicate_proposal_key_count"),
        "audit_conflict_counts": audit.get("conflict_counts", {}),
        "raw_proposals": aggregation.get("raw_proposals"),
        "window_score_rows": aggregation.get("window_score_rows"),
        "pair_score_rows": aggregation.get("pair_score_rows"),
        "silver_v2_window_records": silver.get("v2_silver_window_records"),
        "silver_v2_pair_records": silver.get("v2_silver_pair_records"),
        "dataset_v4_shape": dataset.get("shape"),
        "readiness_trainable_labels": readiness.get("labels_trainable_by_default", []),
        "baseline_trained": baseline.get("trained", False),
        "baseline_sklearn_used": baseline.get("sklearn_used", False),
        "baseline_numpy_fallback_used": baseline.get("numpy_fallback_used", False),
        "review_batch_size": len(review_rows),
        "preview_items": preview.get("items", 0),
        "manual_labels_modified": False,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    _write_summary(summary, paths["summary"])
    return summary


def _write_summary(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Machine Labeling v2 Summary",
        "",
        "This workflow audits, aggregates, and balances machine/silver labels. It does not modify manual_labels.yaml.",
        "",
    ]
    for key, value in summary.items():
        if key == "paths":
            continue
        lines.append(f"- {key}: {value}")
    if summary.get("paths"):
        lines.extend(["", "## Outputs", ""])
        for key, value in summary["paths"].items():
            lines.append(f"- `{key}`: `{value}`")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")

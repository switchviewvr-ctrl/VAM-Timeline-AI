"""Safe orchestration for ingesting the latest edited review batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.ml.dataset_v2 import build_ml_dataset_v2
from vam_timeline_ai.ml.splits import plan_ml_splits_v1
from vam_timeline_ai.ml.supervised_baseline import train_supervised_baseline_v0
from vam_timeline_ai.ml.supervised_readiness import analyze_supervised_readiness
from vam_timeline_ai.semantics.active_review_batch import build_active_review_batch_v3
from vam_timeline_ai.semantics.edited_label_batch import inspect_edited_label_batch
from vam_timeline_ai.semantics.labeling_next_step import write_labeling_next_step
from vam_timeline_ai.semantics.manual_label_merge import merge_manual_label_batch
from vam_timeline_ai.semantics.manual_label_summary import summarize_manual_labels
from vam_timeline_ai.semantics.manual_label_validation import validate_manual_labels_v2
from vam_timeline_ai.semantics.manual_labels import apply_manual_labels
from vam_timeline_ai.semantics.review_batch_discovery import find_latest_review_batch


def ingest_latest_edited_batch(run_dir: str | Path, schema: str | Path, stop_if_missing: bool = True) -> dict[str, Any]:
    run = Path(run_dir)
    discovery = find_latest_review_batch(run, run / "labels" / "latest_review_batch_report.md")
    latest = discovery.get("latest_batch")
    if not latest:
        next_step = write_labeling_next_step(run, run / "labels" / "human_labeling_next_step.md")
        return {"status": "no_valid_batch", "message": "No valid review batch found.", "discovery": discovery, "next_step": next_step}

    batch_name = latest["batch_name"]
    batch_path = Path(latest["path"])
    edited = batch_path / "manual_labels.edited.yaml"
    stub = batch_path / "manual_labels.stub.yaml"
    windows = run / "semantic" / "movement_windows.jsonl"
    pair_windows = run / "semantic" / "pair_windows_v1.jsonl"

    if not edited.exists():
        next_step = write_labeling_next_step(run, run / "labels" / "human_labeling_next_step.md")
        return {
            "status": "waiting_for_human_labels",
            "message": "No edited labels found. Human labeling required.",
            "batch": batch_name,
            "edited_path": str(edited),
            "next_step": next_step,
        }

    inspection = inspect_edited_label_batch(stub, edited, windows, pair_windows, batch_path / "edited_label_batch_inspection.md")
    if not inspection["safe_to_merge"]:
        return {"status": "edited_batch_rejected", "batch": batch_name, "inspection": inspection}

    manual_labels = run / "labels" / "manual_labels.yaml"
    merge = merge_manual_label_batch(
        manual_labels,
        edited,
        manual_labels,
        backup=True,
        report=run / "labels" / f"manual_label_merge_report_{batch_name}.md",
    )
    if merge.get("status") == "error":
        return {"status": "merge_failed", "batch": batch_name, "merge": merge}

    validation = validate_manual_labels_v2(manual_labels, schema, windows, pair_windows, run / "labels" / "manual_label_validation_report.md")
    if validation["status"] == "error":
        return {"status": "validation_failed", "batch": batch_name, "merge": merge, "validation": validation}

    summary = summarize_manual_labels(manual_labels, windows, pair_windows, run / "labels" / "manual_label_summary.md")
    apply_manual_labels(windows, manual_labels, run / "semantic" / "movement_windows_labeled.jsonl", run / "semantic" / "manual_label_report.md")
    dataset = build_ml_dataset_v2(
        run / "features" / "cowgirl_window_features_v1.jsonl",
        run / "semantic" / "movement_windows_labeled.jsonl",
        run / "semantic" / "weak_labels_v2.jsonl",
        manual_labels,
        run / "ml" / "datasets" / "cowgirl_ml_dataset_v2.npz",
        run / "ml" / "reports" / "cowgirl_ml_dataset_v2_report.md",
    )
    split = plan_ml_splits_v1(
        run / "ml" / "datasets" / "cowgirl_ml_dataset_v2.npz",
        manual_labels,
        run / "ml" / "datasets" / "split_plan_v1.json",
        run / "ml" / "reports" / "split_plan_v1_report.md",
    )
    readiness = analyze_supervised_readiness(
        run / "ml" / "datasets" / "cowgirl_ml_dataset_v2.npz",
        manual_labels,
        run / "ml" / "datasets" / "split_plan_v1.json",
        run / "ml" / "reports" / "supervised_readiness_report.md",
    )
    baseline: dict[str, Any] | None = None
    if readiness.get("eligible_labels") and split.get("can_plan_supervised_split"):
        baseline = train_supervised_baseline_v0(
            run / "ml" / "datasets" / "cowgirl_ml_dataset_v2.npz",
            run / "ml" / "datasets" / "split_plan_v1.json",
            run / "ml" / "models" / "supervised_baseline_v0",
            run / "ml" / "reports" / "supervised_baseline_v0_report.md",
        )

    next_batch_num = (latest.get("batch_number") or 0) + 1
    next_batch = run / "labels" / "batches" / f"batch_{next_batch_num:03d}"
    build_active_review_batch_v3(
        windows,
        run / "features" / "cowgirl_window_features_v1.jsonl",
        run / "semantic" / "weak_labels_v2.jsonl",
        pair_windows,
        run / "features" / "cowgirl_pair_features_v0.jsonl",
        manual_labels,
        run / "ml" / "reports" / "supervised_readiness_report.md",
        next_batch,
        batch_size=120,
        max_per_scene=15,
        max_per_sample=3,
        prefer_coverage_gaps=True,
    )
    return {
        "status": "ingested",
        "batch": batch_name,
        "inspection": inspection,
        "merge": merge,
        "validation": validation,
        "summary": summary,
        "dataset": dataset,
        "split": split,
        "readiness": readiness,
        "baseline": baseline,
        "next_batch": str(next_batch),
    }

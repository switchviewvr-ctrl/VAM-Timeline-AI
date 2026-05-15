"""One-command orchestration for machine proposal / silver label workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_machine_labeling_v1(run_dir: str | Path, min_silver_confidence: float = 0.75, train_silver_baseline: bool = True) -> dict[str, Any]:
    from vam_timeline_ai.ml.dataset_v3 import build_ml_dataset_v3
    from vam_timeline_ai.ml.silver_baseline import train_silver_baseline_v0
    from vam_timeline_ai.ml.silver_readiness import analyze_silver_readiness
    from vam_timeline_ai.semantics.label_comparison import compare_machine_labels_to_manual
    from vam_timeline_ai.semantics.machine_label_proposals import generate_machine_label_proposals_v1
    from vam_timeline_ai.semantics.machine_proposal_review_batch import build_machine_proposal_review_batch
    from vam_timeline_ai.semantics.silver_labels import build_silver_labels_v1
    from vam_timeline_ai.visualization.review_previews import render_review_previews_v1

    run = Path(run_dir)
    proposal_dir = run / "labels" / "machine_proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": run / "features" / "cowgirl_window_features_v1.jsonl",
        "pair_features": run / "features" / "cowgirl_pair_features_v0.jsonl",
        "weak": run / "semantic" / "weak_labels_v2.jsonl",
        "windows": run / "semantic" / "movement_windows.jsonl",
        "windows_labeled": run / "semantic" / "movement_windows_labeled.jsonl",
        "pair_windows": run / "semantic" / "pair_windows_v1.jsonl",
        "manual_labels": run / "labels" / "manual_labels.yaml",
        "sample_index": run / "baked" / "motion_sample_index.jsonl",
        "controller_map": run / "semantic" / "controller_bodypart_map.json",
        "proposals_jsonl": proposal_dir / "machine_label_proposals_v1.jsonl",
        "proposals_yaml": proposal_dir / "machine_label_proposals_v1.yaml",
        "proposals_report": proposal_dir / "machine_label_proposals_report_v1.md",
        "silver_jsonl": proposal_dir / "silver_labels_v1.jsonl",
        "silver_yaml": proposal_dir / "silver_labels_v1.yaml",
        "silver_report": proposal_dir / "silver_labels_report_v1.md",
        "comparison": proposal_dir / "machine_vs_manual_comparison.md",
        "dataset_v3": run / "ml" / "datasets" / "cowgirl_ml_dataset_v3.npz",
        "dataset_report": run / "ml" / "reports" / "cowgirl_ml_dataset_v3_report.md",
        "silver_readiness": run / "ml" / "reports" / "silver_readiness_report.md",
        "silver_model_dir": run / "ml" / "models" / "silver_baseline_v0",
        "silver_baseline_report": run / "ml" / "reports" / "silver_baseline_v0_report.md",
        "review_batch": run / "labels" / "batches" / "batch_machine_review_001",
        "summary": proposal_dir / "machine_labeling_v1_summary.md",
    }
    missing_inputs = [str(path) for key, path in paths.items() if key in {"features", "weak", "windows"} and not path.exists()]
    if missing_inputs:
        summary = {"status": "blocked_missing_inputs", "missing_inputs": missing_inputs}
        _write_summary(summary, paths["summary"])
        return summary
    if not paths["manual_labels"].exists() or not paths["windows_labeled"].exists():
        paths["windows_labeled"] = paths["windows"]

    proposals = generate_machine_label_proposals_v1(
        run,
        paths["features"],
        paths["pair_features"],
        paths["weak"],
        paths["windows"],
        paths["pair_windows"],
        paths["proposals_jsonl"],
        paths["proposals_yaml"],
        paths["proposals_report"],
    )
    silver = build_silver_labels_v1(paths["proposals_jsonl"], paths["silver_jsonl"], paths["silver_yaml"], paths["silver_report"], min_confidence=min_silver_confidence)
    comparison = compare_machine_labels_to_manual(paths["manual_labels"], paths["silver_jsonl"], paths["comparison"])
    dataset = build_ml_dataset_v3(paths["features"], paths["windows_labeled"], paths["weak"], paths["manual_labels"], paths["silver_jsonl"], paths["dataset_v3"], paths["dataset_report"])
    readiness = analyze_silver_readiness(paths["dataset_v3"], paths["silver_jsonl"], paths["silver_readiness"])
    baseline: dict[str, Any] = {"trained": False, "reason": "not requested"}
    if train_silver_baseline:
        baseline = train_silver_baseline_v0(paths["dataset_v3"], paths["silver_model_dir"], paths["silver_baseline_report"])
    review_rows = build_machine_proposal_review_batch(run, paths["proposals_jsonl"], paths["silver_jsonl"], paths["review_batch"])
    preview_summary: dict[str, Any]
    if paths["sample_index"].exists() and paths["controller_map"].exists():
        preview_summary = render_review_previews_v1(paths["review_batch"] / "review_batch.jsonl", paths["sample_index"], paths["controller_map"], paths["review_batch"] / "previews")
    else:
        preview_summary = {"items": 0, "matplotlib_available": False, "warnings": ["sample index or controller map missing; previews skipped"]}
    summary = {
        "status": "ok",
        "proposal_count": len(proposals),
        "silver_record_count": len(silver),
        "dataset_v3_shape": dataset.get("shape"),
        "silver_readiness_eligible_labels": readiness.get("eligible_labels", []),
        "silver_baseline_trained": baseline.get("trained", False),
        "silver_baseline_reason": baseline.get("reason"),
        "machine_review_batch_size": len(review_rows),
        "preview_items": preview_summary.get("items", 0),
        "manual_labels_modified": False,
        "comparison_status": comparison.get("status"),
        "paths": {key: str(value) for key, value in paths.items() if key not in {"features", "pair_features", "weak", "windows", "windows_labeled", "pair_windows"}},
    }
    _write_summary(summary, paths["summary"])
    return summary


def _write_summary(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Machine Labeling v1 Summary",
        "",
        "This workflow generated machine proposals and silver labels only. It did not modify manual_labels.yaml.",
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

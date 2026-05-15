"""Cross-check pipeline artifacts for count consistency and stale outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import load_jsonl


def audit_data_integrity(
    source_index: str | Path,
    sample_index: str | Path,
    windows: str | Path,
    features: str | Path,
    dataset: str | Path,
    out: str | Path,
    strict: bool = False,
    pair_windows: str | Path | None = None,
    pair_features: str | Path | None = None,
    review_batch: str | Path | None = None,
) -> dict[str, Any]:
    sources = load_jsonl(source_index)
    samples = load_jsonl(sample_index)
    window_rows = load_jsonl(windows)
    feature_rows = load_jsonl(features)
    dataset_summary = _dataset_summary(dataset)
    pair_window_rows = load_jsonl(pair_windows) if pair_windows else []
    pair_feature_rows = load_jsonl(pair_features) if pair_features else []
    review_rows = load_jsonl(review_batch) if review_batch else []

    source_ids = [r.get("source_id") for r in sources if r.get("source_id")]
    sample_ids = [r.get("sample_id") for r in samples if r.get("sample_id")]
    ok_samples = [r for r in samples if r.get("bake_status") == "ok"]
    ok_sample_ids = {r.get("sample_id") for r in ok_samples if r.get("sample_id")}
    window_ids = [r.get("window_id") for r in window_rows if r.get("window_id")]
    feature_window_ids = [r.get("window_id") for r in feature_rows if r.get("window_id")]
    window_sample_ids = [r.get("sample_id") for r in window_rows if r.get("sample_id")]
    feature_sample_ids = [r.get("sample_id") for r in feature_rows if r.get("sample_id")]

    duplicate_windows = [wid for wid, count in Counter(window_ids).items() if count > 1]
    duplicate_features = [wid for wid, count in Counter(feature_window_ids).items() if count > 1]
    duplicate_sample_ids = [sid for sid, count in Counter(sample_ids).items() if count > 1]
    duplicate_source_ids = [sid for sid, count in Counter(source_ids).items() if count > 1]
    pair_window_ids = [r.get("pair_window_id") for r in pair_window_rows if r.get("pair_window_id")]
    pair_feature_ids = [r.get("pair_window_id") for r in pair_feature_rows if r.get("pair_window_id")]
    duplicate_pair_window_ids = [pid for pid, count in Counter(pair_window_ids).items() if count > 1]
    duplicate_pair_feature_ids = [pid for pid, count in Counter(pair_feature_ids).items() if count > 1]
    pair_feature_without_pair_window = sorted(set(pair_feature_ids) - set(pair_window_ids))[:200]
    pair_window_without_pair_feature = sorted(set(pair_window_ids) - set(pair_feature_ids))[:200] if pair_feature_rows else []
    missing_feature_windows = sorted(set(window_ids) - set(feature_window_ids))[:200]
    feature_without_window = sorted(set(feature_window_ids) - set(window_ids))[:200]
    windows_from_non_ok = sorted(set(window_sample_ids) - ok_sample_ids)
    feature_samples_not_ok = sorted(set(feature_sample_ids) - ok_sample_ids)
    dataset_window_ids = set(dataset_summary.get("window_ids", []))
    dataset_without_feature = sorted(dataset_window_ids - set(feature_window_ids))[:200]
    feature_without_dataset = sorted(set(feature_window_ids) - dataset_window_ids)[:200] if dataset_summary.get("exists") else []
    missing_npz = [r.get("sample_id") for r in ok_samples if not r.get("baked_npz_path") or not Path(str(r.get("baked_npz_path"))).exists()]
    review_missing_windows = sorted({r.get("window_id") for r in review_rows if r.get("window_id") and r.get("window_id") not in set(window_ids)})[:200]
    review_pair_ids = {r.get("pair_window_id") for r in review_rows if r.get("pair_window_id")}
    review_missing_pair_windows = sorted(review_pair_ids - set(pair_window_ids))[:200] if pair_window_rows else []

    summary = {
        "source_records": len(sources),
        "source_types": dict(Counter(r.get("source_type", "unknown") for r in sources)),
        "sample_records": len(samples),
        "sample_statuses": dict(Counter(r.get("bake_status", "unknown") for r in samples)),
        "successful_baked_samples": len(ok_samples),
        "failed_or_unbakeable_samples": len(samples) - len(ok_samples),
        "unique_source_ids": len(set(source_ids)),
        "unique_sample_ids_in_sample_index": len(set(sample_ids)),
        "unique_sample_ids_ok": len(ok_sample_ids),
        "windows": len(window_rows),
        "unique_window_ids": len(set(window_ids)),
        "unique_sample_ids_in_windows": len(set(window_sample_ids)),
        "feature_rows": len(feature_rows),
        "unique_window_ids_in_features": len(set(feature_window_ids)),
        "unique_sample_ids_in_features": len(set(feature_sample_ids)),
        "dataset": dataset_summary,
        "strict": bool(strict),
        "duplicate_source_ids": duplicate_source_ids[:200],
        "duplicate_window_ids": duplicate_windows[:200],
        "duplicate_feature_window_ids": duplicate_features[:200],
        "duplicate_sample_ids": duplicate_sample_ids[:200],
        "duplicate_pair_window_ids": duplicate_pair_window_ids[:200],
        "duplicate_pair_feature_window_ids": duplicate_pair_feature_ids[:200],
        "missing_feature_window_ids": missing_feature_windows,
        "feature_window_ids_without_window": feature_without_window,
        "dataset_window_ids_without_feature": dataset_without_feature,
        "feature_window_ids_without_dataset": feature_without_dataset,
        "missing_baked_npz_sample_ids": missing_npz[:200],
        "window_sample_ids_not_baked_ok_count": len(windows_from_non_ok),
        "feature_sample_ids_not_baked_ok_count": len(feature_samples_not_ok),
        "pair_windows": len(pair_window_rows),
        "unique_pair_window_ids": len(set(pair_window_ids)),
        "pair_features": len(pair_feature_rows),
        "pair_feature_ids_without_pair_window": pair_feature_without_pair_window,
        "pair_window_ids_without_pair_feature": pair_window_without_pair_feature,
        "review_rows": len(review_rows),
        "review_missing_window_ids": review_missing_windows,
        "review_missing_pair_window_ids": review_missing_pair_windows,
        "window_count_per_sample_top": Counter(window_sample_ids).most_common(20),
        "v0_v1_outputs_present": _v0_v1_presence(Path(out).parents[1] if len(Path(out).parents) > 1 else Path.cwd()),
        "findings": [],
        "errors": [],
        "warnings": [],
        "info": [],
    }
    summary["errors"], summary["warnings"], summary["info"] = _classify_findings(summary, strict)
    summary["findings"] = summary["errors"] + summary["warnings"] + summary["info"]
    _write_report(summary, out)
    return summary


def _dataset_summary(dataset: str | Path) -> dict[str, Any]:
    path = Path(dataset)
    if not path.exists():
        return {"exists": False}
    with np.load(path, allow_pickle=True) as data:
        X = data["X"] if "X" in data else np.zeros((0, 0))
        sample_ids = [str(x) for x in data["sample_ids"].tolist()] if "sample_ids" in data else []
        window_ids = [str(x) for x in data["window_ids"].tolist()] if "window_ids" in data else []
        scenes = [str(x) for x in data["group_scene"].tolist()] if "group_scene" in data else []
        sources = [str(x) for x in data["group_source"].tolist()] if "group_source" in data else []
        metadata = str(data["metadata_json"].tolist()) if "metadata_json" in data else ""
        return {
            "exists": True,
            "shape": list(X.shape),
            "unique_sample_ids": len(set(sample_ids)),
            "unique_window_ids": len(set(window_ids)),
            "window_ids": sorted(set(window_ids)),
            "unique_scenes": len(set(scenes)),
            "unique_sources": len(set(sources)),
            "metadata_json": metadata,
        }


def _v0_v1_presence(root: Path) -> dict[str, bool]:
    return {
        "features_v0": (root / "features" / "cowgirl_window_features_v0.jsonl").exists(),
        "features_v1": (root / "features" / "cowgirl_window_features_v1.jsonl").exists(),
        "dataset_v0": (root / "ml" / "datasets" / "cowgirl_ml_dataset_v0.npz").exists(),
        "dataset_v1": (root / "ml" / "datasets" / "cowgirl_ml_dataset_v1.npz").exists(),
    }


def _classify_findings(summary: dict[str, Any], strict: bool) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    def add(message: str, strict_error: bool = False) -> None:
        if strict and strict_error:
            errors.append(message)
        else:
            warnings.append(message)

    if summary["duplicate_source_ids"]:
        add("Duplicate source IDs exist.", strict_error=True)
    if summary["windows"] != summary["feature_rows"]:
        add("Feature row count differs from movement-window count.", strict_error=True)
    if summary["dataset"].get("exists") and summary["dataset"].get("shape", [0])[0] != summary["feature_rows"]:
        add("ML dataset row count differs from feature row count.", strict_error=True)
    if summary["duplicate_window_ids"]:
        add("Duplicate movement window IDs exist.", strict_error=True)
    if summary["duplicate_feature_window_ids"]:
        add("Duplicate feature window IDs exist.", strict_error=True)
    if summary["duplicate_sample_ids"]:
        add("Duplicate sample IDs exist in motion_sample_index. This explains why baked-ok records can exceed unique baked sample IDs.", strict_error=True)
    if summary["feature_window_ids_without_window"]:
        add("Some feature rows reference windows that are missing from movement_windows.", strict_error=True)
    if summary["window_sample_ids_not_baked_ok_count"]:
        add("Some windows reference sample IDs that are not baked-ok in the current sample index; this can indicate stale windows or failed/unbakeable rows carried forward.", strict_error=True)
    if summary["feature_sample_ids_not_baked_ok_count"]:
        add("Some feature rows reference sample IDs that are not baked-ok in the current sample index.", strict_error=True)
    if summary["dataset_window_ids_without_feature"]:
        add("Some ML dataset window IDs do not exist in feature rows.", strict_error=True)
    if summary["feature_window_ids_without_dataset"]:
        add("Some feature window IDs do not exist in the ML dataset.", strict_error=True)
    if summary["missing_baked_npz_sample_ids"]:
        add("Some baked-ok sample records are missing NPZ files.", strict_error=True)
    if summary["duplicate_pair_window_ids"]:
        add("Duplicate pair_window_id values exist.", strict_error=True)
    if summary["duplicate_pair_feature_window_ids"]:
        add("Duplicate pair feature pair_window_id values exist.", strict_error=True)
    if summary["pair_feature_ids_without_pair_window"]:
        add("Some pair feature rows reference missing pair windows.", strict_error=True)
    if summary["pair_window_ids_without_pair_feature"]:
        add("Some pair windows do not have pair feature rows.", strict_error=True)
    if summary["review_missing_window_ids"]:
        add("Review batch references missing window IDs.", strict_error=True)
    if summary["review_missing_pair_window_ids"]:
        add("Review batch references missing pair_window IDs.", strict_error=True)
    if summary["dataset"].get("unique_sample_ids", 0) > summary["unique_sample_ids_ok"]:
        add("The ML dataset has more unique sample IDs than baked-ok samples. This is expected only if feature/window artifacts include stale, failed, or metadata-only sample IDs; otherwise rebuild windows/features from the current sample index.", strict_error=True)
    if summary["v0_v1_outputs_present"].get("features_v0") and summary["v0_v1_outputs_present"].get("features_v1"):
        warnings.append("Both v0 and v1 feature outputs are present. Use explicit paths to avoid mixing versions.")
    if not errors and not warnings:
        info.append("No obvious count mismatch detected.")
    return errors, warnings, info


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Data Integrity Report",
        "",
        "This report checks pipeline counts and flags stale or mixed outputs. It does not infer semantic roles.",
        "",
        f"- Source records: {summary['source_records']}",
        f"- Sample records: {summary['sample_records']}",
        f"- Successful baked samples: {summary['successful_baked_samples']}",
        f"- Failed/unbakeable samples: {summary['failed_or_unbakeable_samples']}",
        f"- Unique sample IDs in sample index: {summary['unique_sample_ids_in_sample_index']}",
        f"- Unique baked-ok sample IDs: {summary['unique_sample_ids_ok']}",
        f"- Duplicate sample IDs in sample index: {len(summary['duplicate_sample_ids'])}",
        f"- Movement windows: {summary['windows']}",
        f"- Unique sample IDs in windows: {summary['unique_sample_ids_in_windows']}",
        f"- Feature rows: {summary['feature_rows']}",
        f"- Unique sample IDs in features: {summary['unique_sample_ids_in_features']}",
        f"- ML dataset shape: {summary['dataset'].get('shape')}",
        f"- Unique sample IDs in ML dataset: {summary['dataset'].get('unique_sample_ids')}",
        f"- Pair windows: {summary['pair_windows']}",
        f"- Pair feature rows: {summary['pair_features']}",
        f"- Review rows: {summary['review_rows']}",
        f"- Strict mode: {summary['strict']}",
        "",
        "## Source Types",
        "",
    ]
    for key, count in sorted(summary["source_types"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Sample Statuses", ""])
    for key, count in sorted(summary["sample_statuses"].items()):
        lines.append(f"- `{key}`: {count}")
    for heading, key in [("Errors", "errors"), ("Warnings", "warnings"), ("Info", "info")]:
        lines.extend(["", f"## {heading}", ""])
        if summary[key]:
            lines.extend(f"- {finding}" for finding in summary[key])
        else:
            lines.append("- None")
    if summary["duplicate_sample_ids"]:
        lines.extend(["", "## Duplicate Sample IDs", ""])
        for sample_id in summary["duplicate_sample_ids"][:50]:
            lines.append(f"- `{sample_id}`")
    lines.extend(["", "## Top Window Counts Per Sample", ""])
    for sample_id, count in summary["window_count_per_sample_top"]:
        lines.append(f"- `{sample_id}`: {count}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")

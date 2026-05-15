"""Human operator local status summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vam_timeline_ai import __version__
from vam_timeline_ai.semantics.review_batch_discovery import find_latest_review_batch


def write_local_status(run_dir: str | Path, out: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    import_ok = True
    clean_exists = run.exists()
    discovery = find_latest_review_batch(run) if clean_exists else {"status": "missing_run", "latest_batch": None}
    latest = discovery.get("latest_batch")
    manual_labels = run / "labels" / "manual_labels.yaml"
    manual_count = _manual_count(manual_labels)
    dataset_v2 = run / "ml" / "datasets" / "cowgirl_ml_dataset_v2.npz"
    dataset_shape = _dataset_shape(dataset_v2)
    readiness = _read_status_line(run / "ml" / "reports" / "supervised_readiness_report.md", "Eligible labels:")
    repo_safety = _read_status_line(run / "audits" / "repo_safety_report.md", "Status:")
    if latest and not latest.get("has_edited"):
        next_command = "write/edit manual_labels.edited.yaml for the latest batch"
    elif latest and latest.get("has_edited"):
        next_command = "python -m vam_timeline_ai.cli ingest-latest-edited-batch --run-dir data\\runs\\clean_v2 --schema data\\labels\\manual_labels.schema_v2.yaml --stop-if-missing true"
    else:
        next_command = "build a review batch"
    result = {
        "package_version": __version__,
        "package_import_ok": import_ok,
        "run_dir": str(run),
        "clean_v2_exists": clean_exists,
        "latest_review_batch": latest,
        "manual_labels_exists": manual_labels.exists(),
        "manual_label_count": manual_count,
        "ml_dataset_v2_exists": dataset_v2.exists(),
        "ml_dataset_v2_shape": dataset_shape,
        "supervised_readiness_status": readiness,
        "repo_safety_status": repo_safety,
        "next_recommended_command": next_command,
    }
    _write_report(result, out)
    return result


def _manual_count(path: Path) -> int:
    if not path.exists() or "template" in path.name.lower():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        windows = data.get("windows", {}) if isinstance(data.get("windows", {}), dict) else {}
        pairs = data.get("pair_windows", {}) if isinstance(data.get("pair_windows", {}), dict) else {}
        return len(windows) + len(pairs)
    except Exception:
        return 0


def _dataset_shape(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return list(data["X"].shape)
    except Exception:
        return None


def _read_status_line(path: Path, prefix: str) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            normalized = stripped[2:].strip() if stripped.startswith("- ") else stripped
            if normalized.startswith(prefix) or prefix in normalized:
                return stripped
    except Exception:
        return None
    return None


def _write_report(result: dict[str, Any], out: str | Path) -> None:
    latest = result.get("latest_review_batch") or {}
    lines = [
        "# Local Status Report",
        "",
        f"- Package import works: {result['package_import_ok']}",
        f"- Package version: {result['package_version']}",
        f"- Run dir: `{result['run_dir']}`",
        f"- clean_v2 exists: {result['clean_v2_exists']}",
        f"- Latest batch: `{latest.get('batch_name')}`" if latest else "- Latest batch: None",
        f"- Latest batch has edited labels: {latest.get('has_edited')}" if latest else "- Latest batch has edited labels: False",
        f"- Manual labels exist: {result['manual_labels_exists']}",
        f"- Manual label entry count: {result['manual_label_count']}",
        f"- ML dataset v2 exists: {result['ml_dataset_v2_exists']}",
        f"- ML dataset v2 shape: {result['ml_dataset_v2_shape']}",
        f"- Supervised readiness status: {result['supervised_readiness_status']}",
        f"- Repo safety status: {result['repo_safety_status']}",
        "",
        "## Next Recommended Action",
        "",
        result["next_recommended_command"],
    ]
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

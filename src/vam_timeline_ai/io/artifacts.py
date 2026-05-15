"""Clean-run artifact manifest helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json
from vam_timeline_ai.io.path_utils import default_reference_paths


RUN_SUBDIRS = [
    "audits",
    "semantic",
    "baked",
    "baked/samples",
    "features",
    "ml",
    "ml/datasets",
    "ml/reports",
    "labels",
    "labels/batches",
]


def prepare_clean_run(
    data_root: str | Path,
    run_name: str,
    backup_existing: bool,
    out_manifest: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    data_root_path = Path(data_root)
    run_root = data_root_path / "runs" / run_name
    for subdir in RUN_SUBDIRS:
        (run_root / subdir).mkdir(parents=True, exist_ok=True)
    old_artifacts = _find_existing_artifacts(data_root_path)
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(default_reference_paths().project_root),
        "data_root": str(data_root_path),
        "run_root": str(run_root),
        "backup_existing": bool(backup_existing),
        "old_artifact_locations": old_artifacts if backup_existing else [],
        "artifact_paths": {
            "audits": str(run_root / "audits"),
            "semantic": str(run_root / "semantic"),
            "baked": str(run_root / "baked"),
            "features": str(run_root / "features"),
            "ml": str(run_root / "ml"),
            "labels": str(run_root / "labels"),
        },
        "pipeline_stage_status": {
            "prepared": True,
            "source_index": False,
            "motion_samples": False,
            "movement_windows": False,
            "features_v1": False,
            "weak_labels": False,
            "pair_windows": False,
            "pair_features": False,
            "ml_dataset": False,
            "strict_integrity": False,
            "review_batch": False,
        },
        "warnings": _stale_warnings(old_artifacts),
        "command_suggestions": _command_suggestions(run_root),
    }
    dump_json(out_manifest, manifest)
    _write_report(manifest, report)
    return manifest


def _find_existing_artifacts(data_root: Path) -> list[str]:
    patterns = [
        "semantic/*.jsonl",
        "features/*.jsonl",
        "features/*.npz",
        "baked/*.jsonl",
        "ml/datasets/*.npz",
        "labels/batches/*",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(p) for p in data_root.glob(pattern))
    return sorted(found)


def _stale_warnings(old_artifacts: list[str]) -> list[str]:
    warnings = []
    if any("cowgirl_window_features_v0" in p for p in old_artifacts) and any("cowgirl_window_features_v1" in p for p in old_artifacts):
        warnings.append("top-level v0 and v1 feature outputs both exist; use clean run paths explicitly")
    if any("batch_001" in p for p in old_artifacts):
        warnings.append("batch_001 predates identity-clean IDs and should be treated as experimental/stale")
    return warnings


def _command_suggestions(run_root: Path) -> list[str]:
    return [
        f"python -m vam_timeline_ai.cli build-motion-source-index --raw-dir \"G:\\VAM\\Research\\MocapResearch\" --out {run_root}\\semantic\\motion_source_index.jsonl --report {run_root}\\semantic\\motion_source_index_report.md --recursive",
        f"python -m vam_timeline_ai.cli extract-motion-samples --source-index {run_root}\\semantic\\motion_source_index.jsonl --out-dir {run_root}\\baked\\samples --index-out {run_root}\\baked\\motion_sample_index.jsonl --fps 60",
        f"python -m vam_timeline_ai.cli audit-data-integrity --source-index {run_root}\\semantic\\motion_source_index.jsonl --sample-index {run_root}\\baked\\motion_sample_index.jsonl --windows {run_root}\\semantic\\movement_windows.jsonl --features {run_root}\\features\\cowgirl_window_features_v1.jsonl --dataset {run_root}\\ml\\datasets\\cowgirl_ml_dataset_v1.npz --strict true --out {run_root}\\audits\\data_integrity_report.md",
    ]


def _write_report(manifest: dict[str, Any], report: str | Path) -> None:
    lines = [
        "# Prepare Clean Run Report",
        "",
        f"- Run name: `{manifest['run_name']}`",
        f"- Run root: `{manifest['run_root']}`",
        f"- Backup existing recorded: {manifest['backup_existing']}",
        f"- Old artifact locations recorded: {len(manifest['old_artifact_locations'])}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {w}" for w in manifest["warnings"]) if manifest["warnings"] else lines.append("- None")
    lines.extend(["", "## Rebuild Command Suggestions", ""])
    lines.extend(f"```powershell\n{cmd}\n```" for cmd in manifest["command_suggestions"])
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")


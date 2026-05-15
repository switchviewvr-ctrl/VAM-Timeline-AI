"""Validate real manual label YAML against known windows and schema."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl
from vam_timeline_ai.semantics.manual_labels import ALLOWED_MANUAL_LABELS, ALLOWED_QUALITY, ALLOWED_ROLES


def validate_manual_labels_v2(labels: str | Path, schema: str | Path, windows: str | Path, pair_windows: str | Path, out: str | Path) -> dict[str, Any]:
    labels_path = Path(labels)
    window_rows = {r.get("window_id"): r for r in load_jsonl(windows) if r.get("window_id")}
    pair_window_rows = {r.get("pair_window_id"): r for r in load_jsonl(pair_windows) if r.get("pair_window_id")}
    window_ids = set(window_rows)
    pair_window_ids = set(pair_window_rows)
    errors: list[str] = []
    warnings: list[str] = []
    if not labels_path.exists():
        warnings.append(f"manual labels file does not exist: {labels_path}")
        data: dict[str, Any] = {}
    elif "template" in labels_path.name.lower():
        errors.append("template file was passed as real manual labels")
        data = _load_yaml(labels_path)
    else:
        data = _load_yaml(labels_path)
    if not Path(schema).exists():
        warnings.append(f"schema file does not exist: {schema}")

    window_entries = data.get("windows", {}) if isinstance(data.get("windows", {}), dict) else {}
    pair_entries = data.get("pair_windows", {}) if isinstance(data.get("pair_windows", {}), dict) else {}
    sample_counts = Counter()
    scene_counts = Counter()
    for wid, entry in window_entries.items():
        if wid not in window_ids:
            errors.append(f"unknown window_id: {wid}")
        wrow = window_rows.get(wid, {})
        sample_counts[wrow.get("sample_id", str(wid).split(":")[0])] += 1
        scene = wrow.get("source_scene_file")
        if scene:
            scene_counts[scene] += 1
        for label in _all_window_labels(entry):
            if str(label).startswith("weak_"):
                errors.append(f"weak label `{label}` used as manual label in window `{wid}`")
            if label not in ALLOWED_MANUAL_LABELS:
                errors.append(f"unknown manual label `{label}` in window `{wid}`")
        role = entry.get("semantic_role")
        if role and role not in ALLOWED_ROLES:
            errors.append(f"invalid semantic_role `{role}` in window `{wid}`")
        quality = entry.get("movement_quality")
        if quality and quality not in ALLOWED_QUALITY:
            errors.append(f"invalid movement_quality `{quality}` in window `{wid}`")
        _validate_confidence(entry.get("confidence"), f"window `{wid}`", errors)
        conf = _float_or_none(entry.get("confidence"))
        if entry.get("include_for_ml") is True and (conf is None or conf < 0.6):
            warnings.append(f"window `{wid}` has include_for_ml true with low/missing confidence")
        if role and role != "unknown" and (conf is None or conf < 0.5):
            warnings.append(f"window `{wid}` assigns semantic_role `{role}` with low/missing confidence")
        labels_set = set(entry.get("labels", []) or [])
        neg_set = set(entry.get("negative_labels", []) or [])
        if labels_set & neg_set:
            warnings.append(f"window `{wid}` has labels also listed as negative: {sorted(labels_set & neg_set)}")
    for pid, entry in pair_entries.items():
        if pid not in pair_window_ids:
            errors.append(f"unknown pair_window_id: {pid}")
        for label in (entry.get("pair_labels", []) or []) + (entry.get("contact_labels", []) or []):
            if str(label).startswith("weak_"):
                errors.append(f"weak label `{label}` used as manual pair/contact label in pair window `{pid}`")
            if label not in ALLOWED_MANUAL_LABELS:
                errors.append(f"unknown pair/contact label `{label}` in pair window `{pid}`")
        _validate_confidence(entry.get("confidence"), f"pair_window `{pid}`", errors)

    for scene, actors in (data.get("actors", {}) or {}).items():
        if not isinstance(actors, dict):
            continue
        scene_counts[scene] += len(actors)
        for atom_id, entry in actors.items():
            role = entry.get("semantic_role")
            if role and role not in ALLOWED_ROLES:
                errors.append(f"invalid actor semantic_role `{role}` for `{scene}` / `{atom_id}`")
            _validate_confidence(entry.get("confidence"), f"actor `{scene}` / `{atom_id}`", errors)
    concentrated_samples = [sample for sample, count in sample_counts.items() if count > 50]
    if concentrated_samples:
        warnings.append(f"many labels are concentrated in these samples: {concentrated_samples[:10]}")
    if len(scene_counts) < 2 and window_entries:
        warnings.append("manual labels are concentrated in fewer than two scenes")
    negative_count = sum(len((entry or {}).get("negative_labels", []) or []) for entry in window_entries.values())
    if window_entries and negative_count == 0:
        warnings.append("no negative/control window labels found yet")
    status = "error" if errors else ("warning" if warnings else "ok")
    result = {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "window_labels": len(window_entries),
        "pair_window_labels": len(pair_entries),
        "sample_label_counts": dict(sample_counts),
        "scene_label_counts": dict(scene_counts),
    }
    _write_report(result, out)
    return result


def _all_window_labels(entry: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ["labels", "negative_labels", "uncertain_labels"]:
        labels.extend(entry.get(key, []) or [])
    return labels


def _validate_confidence(value: Any, context: str, errors: list[str]) -> None:
    if value is None or value == "":
        return
    try:
        parsed = float(value)
    except Exception:
        errors.append(f"invalid confidence `{value}` in {context}")
        return
    if parsed < 0.0 or parsed > 1.0:
        errors.append(f"confidence outside [0,1] in {context}: {value}")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_report(result: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Manual Label Validation v2",
        "",
        f"- Status: {result['status']}",
        f"- Window label entries: {result['window_labels']}",
        f"- Pair-window label entries: {result['pair_window_labels']}",
        "",
        "## Errors",
        "",
    ]
    if result["errors"]:
        lines.extend(f"- {e}" for e in result["errors"])
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if result["warnings"]:
        lines.extend(f"- {w}" for w in result["warnings"])
    else:
        lines.append("- None")
    lines.extend(["", "## Group Leakage Risk", "", "Random window splits are invalid; future splits must group by scene/sample/source."])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")

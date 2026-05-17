"""Estimate numeric parameter profiles after ontology meaning is defined."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl


def calibrate_motion_parameters_v1(
    run_dir: str | Path,
    ontology: str | Path,
    resolved: str | Path,
    relative_features: str | Path,
    trajectory_features: str | Path,
    human_ledger: str | Path,
    out_json: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    ledger_rows = load_jsonl(human_ledger)
    human_windows = _human_positive_windows(ledger_rows)
    resolved_rows = load_jsonl(resolved)
    rel_by_window = {str(r.get("window_id")): r for r in load_jsonl(relative_features)}
    usable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    if ledger_rows and not human_windows:
        output = {
            "schema_version": "motion_parameter_profiles_v1",
            "meaning_source": "top_down_ontology",
            "profiles": {},
            "insufficient_samples": {},
            "blocked_reason": "human_review_ledger_exists_but_no_positive_window_ids_could_be_linked",
            "not_training_truth": True,
        }
        dump_json(out_json, output)
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(
            "# Motion Parameter Calibration V1\n\n"
            "Blocked: a human review ledger exists, but no positive reviewed records could be linked to window IDs. "
            "No numeric ranges were invented from weak/heuristic labels.\n",
            encoding="utf-8",
        )
        return {"status": "blocked", "profiles": 0, "insufficient": {}, "skipped": 0, "out_json": str(out_json), "report": str(report), "blocked_reason": output["blocked_reason"]}
    for row in resolved_rows:
        wid = str(row.get("window_id") or "")
        if human_windows and wid not in human_windows:
            skipped += 1
            continue
        if row.get("clean_motion_gate") not in {"pass", "soft_pass_short"}:
            skipped += 1
            continue
        if row.get("conflict_flags"):
            skipped += 1
            continue
        rel = rel_by_window.get(wid)
        if not rel:
            skipped += 1
            continue
        usable[str(row.get("resolved_motion_subtype") or row.get("resolved_semantic_family") or "unknown")].append(rel)

    profiles = {}
    insufficient = {}
    for subtype, rows in sorted(usable.items()):
        if len(rows) < 3:
            insufficient[subtype] = len(rows)
            continue
        profiles[subtype] = _profile(rows)

    output = {
        "schema_version": "motion_parameter_profiles_v1",
        "meaning_source": "top_down_ontology",
        "uses_human_review_only_when_available": True,
        "profiles": profiles,
        "insufficient_samples": insufficient,
        "not_training_truth": True,
    }
    dump_json(out_json, output)
    lines = [
        "# Motion Parameter Calibration V1",
        "",
        "Parameter calibration does not define meaning. It estimates ranges from ontology-consistent, human-reviewed data when available.",
        "",
        f"- Profiles written: {len(profiles)}",
        f"- Insufficient sample groups: {insufficient}",
        f"- Skipped records: {skipped}",
        f"- Human reviewed windows available: {len(human_windows)}",
    ]
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "profiles": len(profiles), "insufficient": insufficient, "skipped": skipped, "out_json": str(out_json), "report": str(report)}


def _human_positive_windows(rows: list[dict[str, Any]]) -> set[str]:
    positives = set()
    for row in rows:
        labels = " ".join(str(x) for x in (row.get("human_labels") or row.get("error_tags") or row.get("actual_labels") or []))
        family = str(row.get("human_semantic_family") or row.get("semantic_family") or "")
        verdict = str(row.get("verdict") or row.get("user_verdict") or "")
        if "cowgirl_true_segment" in labels or (family == "cowgirl" and verdict in {"correct", "partially_correct", "correct_low_confidence"}):
            if row.get("window_id"):
                positives.add(str(row["window_id"]))
    return positives


def _profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "relative_pelvis_vertical_amplitude",
        "relative_pelvis_forward_back_amplitude",
        "relative_pelvis_lateral_amplitude",
        "local_path_length",
        "local_motion_energy",
        "local_velocity_mean",
        "local_grind_score",
        "local_bounce_score",
        "torso_relative_to_pelvis_motion",
        "head_relative_to_chest_motion",
    ]
    out = {"sample_count": len(rows), "ranges": {}}
    for key in keys:
        values = []
        for row in rows:
            value = (row.get("feature_values") or {}).get(key)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        if values:
            values.sort()
            out["ranges"][key] = {"min": values[0], "p25": values[len(values) // 4], "median": values[len(values) // 2], "p75": values[(len(values) * 3) // 4], "max": values[-1]}
    return out

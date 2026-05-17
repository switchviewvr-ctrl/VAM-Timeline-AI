"""Rig anatomy feature extraction.

This module maps existing controller cycle features into semantic body-region
features. It is analysis-only: no labels, no ML training, no Timeline export.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.semantics.ontology_loader import load_yaml


def extract_rig_anatomy_features_v1(
    run_dir: str | Path,
    anatomy: str | Path,
    roles: str | Path,
    out_jsonl: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    anatomy_data = load_yaml(anatomy)
    roles_data = load_yaml(roles)
    region_map = _controller_region_map(anatomy_data)
    role_map = roles_data.get("families", {})
    cycle_rows = load_jsonl(run / "features" / "motion_cycle_features_v1.jsonl")
    rows = [_row_from_cycle(row, region_map, role_map) for row in cycle_rows]
    write_jsonl(out_jsonl, rows)
    summary = {
        "status": "ok",
        "records": len(rows),
        "dominant_regions": dict(Counter(r.get("dominant_region") for r in rows)),
        "primary_driver_guesses": dict(Counter(r.get("primary_driver_region_guess") for r in rows)),
        "has_hipControl": sum(1 for r in rows if r.get("has_hipControl")),
        "has_hands": sum(1 for r in rows if r.get("has_hands")),
        "has_feet": sum(1 for r in rows if r.get("has_feet")),
        "out_jsonl": str(out_jsonl),
        "report": str(report),
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }
    _write_report(Path(report), summary, anatomy, roles)
    return summary


def _controller_region_map(anatomy: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for region, cfg in (anatomy.get("regions") or {}).items():
        for key in ["primary_vam_controllers", "secondary_vam_controllers", "related_vam_controllers"]:
            for controller in cfg.get(key) or []:
                out.setdefault(str(controller), set()).add(str(region))
    return out


def _row_from_cycle(cycle: dict[str, Any], region_map: dict[str, set[str]], role_map: dict[str, Any]) -> dict[str, Any]:
    metrics = cycle.get("controller_metrics") if isinstance(cycle.get("controller_metrics"), dict) else {}
    region_metrics: dict[str, dict[str, float]] = {}
    controller_regions: dict[str, list[str]] = {}
    for controller, metric in metrics.items():
        regions = sorted(region_map.get(str(controller), {_fallback_region(str(controller))}))
        controller_regions[str(controller)] = regions
        for region in regions:
            target = region_metrics.setdefault(region, _empty_region_metrics())
            _accumulate(target, metric)
    for metric in region_metrics.values():
        count = max(metric.pop("_count", 0.0), 1.0)
        metric["mean_cyclicity_score"] = round(metric["mean_cyclicity_score"] / count, 6)
        metric["mean_transition_score"] = round(metric["mean_transition_score"] / count, 6)
        metric["mean_pose_hold_score"] = round(metric["mean_pose_hold_score"] / count, 6)
    dominant = _dominant_region(region_metrics)
    row = {
        "schema": "rig_anatomy_features_v1",
        "window_id": cycle.get("window_id"),
        "sample_id": cycle.get("sample_id"),
        "source_id": cycle.get("source_id"),
        "source_scene_file": cycle.get("source_scene_file"),
        "technical_atom_id": cycle.get("technical_atom_id"),
        "start_seconds": cycle.get("start_seconds"),
        "end_seconds": cycle.get("end_seconds"),
        "duration_seconds": cycle.get("duration_seconds"),
        "controller_regions": controller_regions,
        "region_metrics": region_metrics,
        "dominant_region": dominant,
        "primary_driver_region_guess": dominant if _region_score(region_metrics.get(dominant)) > 0.0 else "unknown",
        "region_static_scores": {region: _static_score(metric) for region, metric in region_metrics.items()},
        "anchor_stability_by_region": _anchor_stability(cycle, region_metrics),
        "has_hipControl": "hipControl" in metrics,
        "has_pelvisControl": "pelvisControl" in metrics,
        "has_hands": bool({"lHandControl", "rHandControl"} & set(metrics)),
        "has_feet": bool({"lFootControl", "rFootControl"} & set(metrics)),
        "has_knees": bool({"lKneeControl", "rKneeControl"} & set(metrics)),
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }
    row.update(_flat_region_features(region_metrics))
    row.update(_family_conflict_features(row, region_metrics, role_map))
    return row


def _empty_region_metrics() -> dict[str, float]:
    return {
        "motion_score": 0.0,
        "path_length": 0.0,
        "max_displacement_range": 0.0,
        "max_cycle_count": 0.0,
        "max_frequency_hz": 0.0,
        "mean_cyclicity_score": 0.0,
        "mean_transition_score": 0.0,
        "mean_pose_hold_score": 0.0,
        "_count": 0.0,
    }


def _accumulate(target: dict[str, float], metric: dict[str, Any]) -> None:
    rng = _num(metric.get("max_displacement_range"))
    path = _num(metric.get("total_path_length"))
    cyc = _num(metric.get("cyclicity_score"))
    trans = _num(metric.get("transition_score"))
    hold = _num(metric.get("pose_hold_score"))
    cycles = _num(metric.get("estimated_cycle_count"))
    freq = _num(metric.get("estimated_frequency_hz"))
    score = max(rng / 0.20, path / 1.0, cyc)
    target["motion_score"] = round(max(target["motion_score"], min(1.0, score)), 6)
    target["path_length"] = round(target["path_length"] + path, 6)
    target["max_displacement_range"] = round(max(target["max_displacement_range"], rng), 6)
    target["max_cycle_count"] = round(max(target["max_cycle_count"], cycles), 6)
    target["max_frequency_hz"] = round(max(target["max_frequency_hz"], freq), 6)
    target["mean_cyclicity_score"] += cyc
    target["mean_transition_score"] += trans
    target["mean_pose_hold_score"] += hold
    target["_count"] += 1.0


def _dominant_region(metrics: dict[str, dict[str, float]]) -> str:
    if not metrics:
        return "unknown"
    return max(metrics, key=lambda region: _region_score(metrics[region]))


def _region_score(metric: dict[str, float] | None) -> float:
    if not metric:
        return 0.0
    return max(float(metric.get("motion_score") or 0.0), min(float(metric.get("path_length") or 0.0) / 1.0, 1.0))


def _static_score(metric: dict[str, float]) -> float:
    return round(max(0.0, min(1.0, 1.0 - float(metric.get("motion_score") or 0.0))), 6)


def _anchor_stability(cycle: dict[str, Any], region_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    anchor = cycle.get("anchor_summary") if isinstance(cycle.get("anchor_summary"), dict) else {}
    return {
        "feet": "stable" if anchor.get("feet_stable") is not False else "unstable",
        "knees": "stable" if anchor.get("knees_stable") is not False else "unstable",
        "hands": "stable" if anchor.get("hands_stable") is not False else "unstable",
        "lower_body_core_static_score": _static_score(region_metrics.get("lower_body_core", _empty_region_metrics())),
        "pelvis_static_score": _static_score(region_metrics.get("pelvis", _empty_region_metrics())),
    }


def _flat_region_features(region_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for region in [
        "lower_body_core",
        "pelvis",
        "thighs",
        "knees",
        "feet",
        "upper_body_core",
        "head_neck",
        "hands",
        "arms",
    ]:
        metric = region_metrics.get(region, _empty_region_metrics())
        out[f"{region}_motion_score"] = float(metric.get("motion_score") or 0.0)
        out[f"{region}_path_length"] = float(metric.get("path_length") or 0.0)
        out[f"{region}_cycle_count"] = float(metric.get("max_cycle_count") or 0.0)
        out[f"{region}_static_score"] = _static_score(metric)
    return out


def _family_conflict_features(row: dict[str, Any], region_metrics: dict[str, dict[str, float]], role_map: dict[str, Any]) -> dict[str, Any]:
    lower = _region_score(region_metrics.get("lower_body_core"))
    pelvis = _region_score(region_metrics.get("pelvis"))
    head = _region_score(region_metrics.get("head_neck"))
    hands = _region_score(region_metrics.get("hands"))
    feet = _region_score(region_metrics.get("feet"))
    knees = _region_score(region_metrics.get("knees"))
    return {
        "cowgirl_head_driver_conflict": head > max(lower, pelvis, 0.10) * 1.20,
        "cowgirl_hand_driver_conflict": hands > max(lower, pelvis, 0.10) * 1.20,
        "bj_lower_body_driver_conflict": lower > max(head, 0.10) * 1.20 or pelvis > max(head, 0.10) * 1.20,
        "hj_lower_body_driver_conflict": lower > max(hands, 0.10) * 1.20 or pelvis > max(hands, 0.10) * 1.20,
        "doggy_hand_locomotion_conflict": hands > 0.35 and (feet > 0.20 or knees > 0.25),
        "roles_schema_loaded": bool(role_map),
    }


def _fallback_region(controller: str) -> str:
    lowered = controller.lower()
    if "hip" in lowered:
        return "lower_body_core"
    if "pelvis" in lowered:
        return "pelvis"
    if "thigh" in lowered:
        return "thighs"
    if "knee" in lowered:
        return "knees"
    if "foot" in lowered or "toe" in lowered:
        return "feet"
    if "hand" in lowered:
        return "hands"
    if "elbow" in lowered or "shoulder" in lowered:
        return "arms"
    if "head" in lowered or "neck" in lowered:
        return "head_neck"
    if "chest" in lowered or "abdomen" in lowered:
        return "upper_body_core"
    return "unknown"


def _num(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_report(path: Path, summary: dict[str, Any], anatomy: str | Path, roles: str | Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Rig Anatomy Features v1",
        "",
        f"- Anatomy: `{anatomy}`",
        f"- Roles: `{roles}`",
        f"- Records: `{summary['records']}`",
        f"- Dominant regions: `{summary['dominant_regions']}`",
        f"- Primary driver guesses: `{summary['primary_driver_guesses']}`",
        f"- Has hipControl: `{summary['has_hipControl']}`",
        f"- Has hands: `{summary['has_hands']}`",
        f"- Has feet: `{summary['has_feet']}`",
        "- ML training performed: `false`",
        "- Timeline generation performed: `false`",
        "- manual_labels.yaml modified: `false`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

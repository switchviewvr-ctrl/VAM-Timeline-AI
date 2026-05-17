"""Human-readable reports for imported manual pose captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl


def report_manual_pose_captures_v1(captures: str | Path, out: str | Path) -> dict[str, Any]:
    rows = load_jsonl(captures)
    family_counts: dict[str, int] = {}
    subtype_counts: dict[str, int] = {}
    controller_counts: list[int] = []
    missing_counts: dict[str, int] = {}
    distances_by_subtype: dict[str, list[float]] = {}
    facing_counts: dict[str, int] = {}
    pose_hint_counts: dict[str, int] = {}

    for row in rows:
        labels = row.get("human_labels") or {}
        metrics = row.get("metrics") or {}
        family = str(labels.get("pose_family") or "unknown")
        subtype = str(labels.get("pose_subtype") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        distance = metrics.get("rider_pelvis_to_partner_pelvis_distance")
        if isinstance(distance, (int, float)):
            distances_by_subtype.setdefault(subtype, []).append(float(distance))
        facing = str(metrics.get("rider_facing_relative_to_partner") or "unknown")
        facing_counts[facing] = facing_counts.get(facing, 0) + 1
        pose_hint = str(metrics.get("pose_hint") or "unknown")
        pose_hint_counts[pose_hint] = pose_hint_counts.get(pose_hint, 0) + 1
        for role in ("rider", "partner"):
            atom = ((row.get("atoms") or {}).get(role) or {})
            controller_counts.append(int(atom.get("controller_count") or 0))
            for name in atom.get("missing_controllers") or []:
                key = f"{role}:{name}"
                missing_counts[key] = missing_counts.get(key, 0) + 1

    distance_summary = {
        subtype: {
            "count": len(values),
            "mean": round(sum(values) / len(values), 5) if values else None,
            "min": round(min(values), 5) if values else None,
            "max": round(max(values), 5) if values else None,
        }
        for subtype, values in sorted(distances_by_subtype.items())
    }
    summary = {
        "status": "ok",
        "captures": len(rows),
        "out": str(out),
        "family_counts": family_counts,
        "subtype_counts": subtype_counts,
        "controller_count_min": min(controller_counts) if controller_counts else 0,
        "controller_count_max": max(controller_counts) if controller_counts else 0,
        "missing_controller_counts": missing_counts,
        "distance_summary_by_subtype": distance_summary,
        "facing_counts": facing_counts,
        "pose_hint_counts": pose_hint_counts,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    _write_report(summary, out)
    return summary


def _write_report(summary: dict[str, Any], out: str | Path) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Pose Capture Report V1",
        "",
        f"- Captures: `{summary['captures']}`",
        f"- Families: `{summary['family_counts']}`",
        f"- Subtypes: `{summary['subtype_counts']}`",
        f"- Controller count range: `{summary['controller_count_min']}` to `{summary['controller_count_max']}`",
        f"- Facing hints: `{summary['facing_counts']}`",
        f"- Pose hints: `{summary['pose_hint_counts']}`",
        "- ML training performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Alignment Distances By Subtype",
        "",
    ]
    if summary["distance_summary_by_subtype"]:
        for subtype, stats in summary["distance_summary_by_subtype"].items():
            lines.append(f"- `{subtype}`: count `{stats['count']}`, mean `{stats['mean']}`, min `{stats['min']}`, max `{stats['max']}`")
    else:
        lines.append("- No alignment distances available yet.")
    lines.append("")
    lines.append("## Missing Controllers")
    if summary["missing_controller_counts"]:
        for key, count in sorted(summary["missing_controller_counts"].items()):
            lines.append(f"- `{key}`: {count}")
    else:
        lines.append("- None reported.")
    lines.extend([
        "",
        "## Suggested Ontology Corrections",
        "",
        "Use these captures to adjust ontology geometry only after human inspection. Do not treat missing data, ML output, or VLM output as ground truth.",
    ])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Build a real manual VaM pose ground-truth dataset from captures and notes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

from vam_timeline_ai.datasets.manual_pose_extraction import capture_id_from_name
from vam_timeline_ai.datasets.manual_pose_measurements import compute_manual_pose_measurements
from vam_timeline_ai.io.json_utils import dump_json, load_json, write_jsonl


def build_manual_pose_ground_truth_v1(
    capture_dir: str | Path,
    human_labels: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    source = Path(capture_dir)
    label_payload = load_json(human_labels)
    labels = {str(row.get("capture_id")): row for row in (label_payload.get("labels") or [])}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for json_path in sorted(source.glob("*.json")):
        capture_id = capture_id_from_name(json_path.name)
        if not capture_id:
            continue
        seen_ids.add(capture_id)
        raw = load_json(json_path)
        label = labels.get(capture_id)
        screenshot = _find_screenshot(source, capture_id)
        row_warnings = []
        if label is None:
            row_warnings.append("label_missing")
        if screenshot is None:
            row_warnings.append("screenshot_missing")
        if raw.get("schema_version") != "pose_capture_v1":
            row_warnings.append("unexpected_schema")
        measurements = compute_manual_pose_measurements(raw, label)
        rows.append(
            {
                "schema_version": "manual_pose_ground_truth_v1",
                "capture_id": capture_id,
                "raw_capture_path": str(json_path),
                "screenshot_path": str(screenshot) if screenshot is not None else "",
                "human_labels": label or {"capture_id": capture_id, "family": "unknown", "pose_subtype": "unknown", "warnings": ["label_missing"]},
                "rider_atom": ((raw.get("atoms") or {}).get("rider") or {}).get("atom_uid") or ((raw.get("atoms") or {}).get("rider") or {}).get("atom_name"),
                "partner_atom": ((raw.get("atoms") or {}).get("partner") or {}).get("atom_uid") or ((raw.get("atoms") or {}).get("partner") or {}).get("atom_name"),
                "atoms": raw.get("atoms") or {},
                "derived": raw.get("derived") or {},
                "measurements": measurements,
                "quality_flags": raw.get("pose_quality_flags") or {},
                "warnings": row_warnings,
                "ml_training_run": False,
                "manual_labels_yaml_modified": False,
                "auto_labeling_run": False,
            }
        )

    orphan_labels = sorted(set(labels) - seen_ids)
    warnings.extend(f"orphan_label:{capture_id}" for capture_id in orphan_labels)
    write_jsonl(out_jsonl, rows)
    _write_csv(rows, out_csv)
    patch_path = Path(out_jsonl).with_name("manual_pose_ontology_patches_v1.yaml")
    _write_ontology_patch(rows, patch_path)
    summary = _summary(rows, orphan_labels, warnings, source, human_labels, out_jsonl, out_csv, report, patch_path)
    _write_report(summary, rows, report)
    return summary


def _find_screenshot(source: Path, capture_id: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = source / f"{capture_id}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(source.glob(f"{capture_id}*.png"))
    return matches[0] if matches else None


def _write_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    target = Path(out_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "capture_id",
        "family",
        "pose_subtype",
        "motion_intent",
        "primary_driver",
        "screenshot_path",
        "raw_capture_path",
        "pelvis_distance",
        "head_to_partner_pelvis_distance",
        "lhand_nearest_target",
        "rhand_nearest_target",
        "warnings",
    ]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            labels = row.get("human_labels") or {}
            partner = ((row.get("measurements") or {}).get("partner_relative") or {})
            hands = ((row.get("measurements") or {}).get("hand_target_candidates") or {})
            writer.writerow(
                {
                    "capture_id": row.get("capture_id"),
                    "family": labels.get("family"),
                    "pose_subtype": labels.get("pose_subtype"),
                    "motion_intent": labels.get("motion_intent"),
                    "primary_driver": labels.get("primary_driver"),
                    "screenshot_path": row.get("screenshot_path"),
                    "raw_capture_path": row.get("raw_capture_path"),
                    "pelvis_distance": partner.get("rider_pelvis_to_partner_pelvis_distance"),
                    "head_to_partner_pelvis_distance": partner.get("rider_head_to_partner_pelvis_distance"),
                    "lhand_nearest_target": (hands.get("lHandControl") or {}).get("nearest_target"),
                    "rhand_nearest_target": (hands.get("rHandControl") or {}).get("nearest_target"),
                    "warnings": ",".join(row.get("warnings") or []),
                }
            )


def _summary(
    rows: list[dict[str, Any]],
    orphan_labels: list[str],
    warnings: list[str],
    capture_dir: Path,
    human_labels: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
    patch_path: Path,
) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    subtype_counts: dict[str, int] = {}
    matched = 0
    screenshots = 0
    missing_labels = 0
    distances: list[float] = []
    for row in rows:
        labels = row.get("human_labels") or {}
        family = str(labels.get("family") or "unknown")
        subtype = str(labels.get("pose_subtype") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        if "label_missing" not in (row.get("warnings") or []):
            matched += 1
        else:
            missing_labels += 1
        if row.get("screenshot_path"):
            screenshots += 1
        distance = (((row.get("measurements") or {}).get("partner_relative") or {}).get("rider_pelvis_to_partner_pelvis_distance"))
        if isinstance(distance, (int, float)):
            distances.append(float(distance))
    return {
        "status": "ok",
        "capture_dir": str(capture_dir),
        "human_labels": str(human_labels),
        "out_jsonl": str(out_jsonl),
        "out_csv": str(out_csv),
        "report": str(report),
        "ontology_patch": str(patch_path),
        "captures": len(rows),
        "matched_labels": matched,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "screenshots": screenshots,
        "family_counts": family_counts,
        "subtype_counts": subtype_counts,
        "mean_rider_pelvis_to_partner_pelvis_distance": round(sum(distances) / len(distances), 5) if distances else None,
        "warnings": warnings,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
        "auto_labeling_run": False,
    }


def _write_report(summary: dict[str, Any], rows: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Pose Ground Truth Build Report V1",
        "",
        f"- Capture dir: `{summary['capture_dir']}`",
        f"- Captures found: `{summary['captures']}`",
        f"- Captures matched to human labels: `{summary['matched_labels']}`",
        f"- Missing labels: `{summary['missing_labels']}`",
        f"- Orphan labels: `{len(summary['orphan_labels'])}`",
        f"- Screenshots linked: `{summary['screenshots']}`",
        f"- Family counts: `{summary['family_counts']}`",
        f"- Subtype counts: `{summary['subtype_counts']}`",
        f"- Mean rider pelvis to partner pelvis distance: `{summary['mean_rider_pelvis_to_partner_pelvis_distance']}`",
        f"- Ontology patch proposal: `{summary['ontology_patch']}`",
        "- ML training performed: `false`",
        "- Auto-labeling performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Captures",
        "",
    ]
    for row in rows:
        labels = row.get("human_labels") or {}
        partner = ((row.get("measurements") or {}).get("partner_relative") or {})
        lines.append(
            f"- `{row['capture_id']}`: `{labels.get('family')}` / `{labels.get('pose_subtype')}`, "
            f"driver `{labels.get('primary_driver')}`, pelvis distance `{partner.get('rider_pelvis_to_partner_pelvis_distance')}`"
        )
    if summary["orphan_labels"]:
        lines.extend(["", "## Orphan Labels", ""])
        lines.extend(f"- `{capture_id}`" for capture_id in summary["orphan_labels"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ontology_patch(rows: list[dict[str, Any]], out: str | Path) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    subtypes = sorted({str((row.get("human_labels") or {}).get("pose_subtype")) for row in rows if (row.get("human_labels") or {}).get("pose_subtype")})
    lines = [
        "# Proposed manual pose ontology patches v1.",
        "# Review manually before applying. This file is not automatically applied.",
        "new_pose_subtypes:",
    ]
    for subtype in subtypes:
        lines.append(f"  - {subtype}")
    lines.extend(
        [
            "family_rules:",
            "  cowgirl:",
            "    - feet mostly static",
            "    - knees may phase slightly",
            "    - pelvis_hip is the motion driver",
            "    - hand support varies and does not define family alone",
            "  doggy:",
            "    - female/front actor often passive",
            "    - body response follows partner driver",
            "    - feet usually static",
            "    - hands or front body provide support",
            "  bj_oral:",
            "    - head_neck is the primary driver",
            "    - pelvis remains mostly static",
            "    - hands may support or assist",
            "  handjob:",
            "    - hand is the primary driver",
            "    - not Cowgirl even when kneeling or visually close",
            "  missionary:",
            "    - receiver often passive",
            "    - legs and feet react to partner motion",
            "measurement_notes:",
            "  - Use partner-relative controller transforms as pose geometry evidence.",
            "  - Do not use these captures as automatic labels for unrelated windows.",
            "  - Do not train ML from this dataset in this step.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

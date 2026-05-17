"""Import manual VaM skeleton pose captures as pose ground-truth examples.

The importer is intentionally read-only with respect to labels and training:
it normalizes capture JSONs, preserves the human labels embedded by the VaM
plugin, and writes review/analysis artifacts only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

from vam_timeline_ai.io.json_utils import as_float, load_json, write_jsonl


CONTROLLER_ALIASES = {
    "hip": "hipControl",
    "pelvis": "pelvisControl",
    "abdomen": "abdomenControl",
    "abdomen2": "abdomen2Control",
    "chest": "chestControl",
    "head": "headControl",
    "lhand": "lHandControl",
    "rhand": "rHandControl",
    "lknee": "lKneeControl",
    "rknee": "rKneeControl",
    "lfoot": "lFootControl",
    "rfoot": "rFootControl",
}


def import_manual_pose_captures_v1(input_dir: str | Path, out_jsonl: str | Path, report: str | Path) -> dict[str, Any]:
    source = Path(input_dir)
    rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    if source.exists():
        for path in sorted(source.glob("*.json")):
            try:
                raw = load_json(path)
                row = normalize_pose_capture(raw, source_path=path)
                if row["schema_valid"]:
                    rows.append(row)
                else:
                    invalid.append(f"{path.name}: {row.get('schema_error')}")
            except Exception as exc:  # noqa: BLE001
                invalid.append(f"{path.name}: {exc}")

    write_jsonl(out_jsonl, rows)
    summary = _summary(rows, invalid, source, out_jsonl, report)
    _write_import_report(summary, rows, invalid, report)
    return summary


def normalize_pose_capture(raw: dict[str, Any], *, source_path: str | Path | None = None) -> dict[str, Any]:
    schema = str(raw.get("schema_version") or "")
    labels = dict(raw.get("human_labels") or {})
    atoms = dict(raw.get("atoms") or {})
    derived = dict(raw.get("derived") or {})
    valid = schema == "pose_capture_v1" and "rider" in atoms and "partner" in atoms
    row = {
        "schema_version": schema,
        "schema_valid": valid,
        "schema_error": "" if valid else "expected pose_capture_v1 with atoms.rider and atoms.partner",
        "source_path": str(source_path) if source_path is not None else "",
        "created_at": raw.get("created_at"),
        "source": raw.get("source"),
        "vam_version": raw.get("vam_version"),
        "scene_name": raw.get("scene_name"),
        "human_labels": {
            "pose_family": str(labels.get("pose_family") or "unknown"),
            "pose_subtype": str(labels.get("pose_subtype") or "unknown"),
            "motion_intent": str(labels.get("motion_intent") or ""),
            "human_notes": str(labels.get("human_notes") or ""),
        },
        "atoms": {
            "rider": _normalize_atom(atoms.get("rider") or {}),
            "partner": _normalize_atom(atoms.get("partner") or {}),
        },
        "derived": derived,
        "metrics": _metrics(raw),
        "pose_quality_flags": dict(raw.get("pose_quality_flags") or {}),
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    return row


def _normalize_atom(atom: dict[str, Any]) -> dict[str, Any]:
    controllers = {}
    for name, value in (atom.get("controllers") or {}).items():
        normalized = normalize_controller_name(str(name))
        controllers[normalized] = value
    return {
        "atom_uid": atom.get("atom_uid"),
        "atom_name": atom.get("atom_name"),
        "controllers": controllers,
        "missing_controllers": [normalize_controller_name(str(v)) for v in (atom.get("missing_controllers") or [])],
        "controller_count": len(controllers),
    }


def normalize_controller_name(name: str) -> str:
    if name in CONTROLLER_ALIASES.values():
        return name
    key = name.strip().replace("_", "").replace("-", "").lower()
    return CONTROLLER_ALIASES.get(key, name.strip())


def _metrics(raw: dict[str, Any]) -> dict[str, Any]:
    derived = raw.get("derived") or {}
    atoms = raw.get("atoms") or {}
    metrics = {
        "rider_pelvis_to_partner_pelvis_distance": _relation_distance(derived, "rider_pelvis_to_partner_pelvis"),
        "rider_head_to_partner_pelvis_distance": _relation_distance(derived, "rider_head_to_partner_pelvis"),
        "rider_lhand_to_partner_chest_distance": _relation_distance(derived, "rider_lhand_to_partner_chest"),
        "rider_rhand_to_partner_chest_distance": _relation_distance(derived, "rider_rhand_to_partner_chest"),
        "rider_lhand_to_partner_pelvis_distance": _relation_distance(derived, "rider_lhand_to_partner_pelvis"),
        "rider_rhand_to_partner_pelvis_distance": _relation_distance(derived, "rider_rhand_to_partner_pelvis"),
        "rider_facing_relative_to_partner": _nested(derived, ["orientation_hints", "rider_facing_relative_to_partner"], "unknown"),
        "pose_hint": _nested(derived, ["orientation_hints", "pose_hint"], "unknown"),
    }
    if metrics["rider_pelvis_to_partner_pelvis_distance"] is None:
        metrics["rider_pelvis_to_partner_pelvis_distance"] = _world_controller_distance(atoms, "rider", "pelvisControl", "partner", "pelvisControl")
    return metrics


def _relation_distance(derived: dict[str, Any], key: str) -> float | None:
    rel = derived.get(key) or {}
    return as_float(rel.get("distance"))


def _world_controller_distance(atoms: dict[str, Any], role_a: str, ctrl_a: str, role_b: str, ctrl_b: str) -> float | None:
    a = _controller_world_position(atoms, role_a, ctrl_a)
    b = _controller_world_position(atoms, role_b, ctrl_b)
    if a is None or b is None:
        return None
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _controller_world_position(atoms: dict[str, Any], role: str, controller: str) -> list[float] | None:
    ctrl = ((atoms.get(role) or {}).get("controllers") or {}).get(controller) or {}
    values = ctrl.get("world_position")
    if not isinstance(values, list) or len(values) < 3:
        return None
    parsed = [as_float(values[i]) for i in range(3)]
    if any(v is None for v in parsed):
        return None
    return [float(v) for v in parsed if v is not None]


def _nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _summary(rows: list[dict[str, Any]], invalid: list[str], source: Path, out_jsonl: str | Path, report: str | Path) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    subtype_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    distances: list[float] = []
    for row in rows:
        labels = row.get("human_labels") or {}
        family = str(labels.get("pose_family") or "unknown")
        subtype = str(labels.get("pose_subtype") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        for role in ("rider", "partner"):
            for name in (((row.get("atoms") or {}).get(role) or {}).get("missing_controllers") or []):
                key = f"{role}:{name}"
                missing_counts[key] = missing_counts.get(key, 0) + 1
        distance = (row.get("metrics") or {}).get("rider_pelvis_to_partner_pelvis_distance")
        if distance is not None:
            distances.append(float(distance))
    return {
        "status": "ok",
        "input_dir": str(source),
        "out_jsonl": str(out_jsonl),
        "report": str(report),
        "captures": len(rows),
        "invalid_files": len(invalid),
        "family_counts": family_counts,
        "subtype_counts": subtype_counts,
        "missing_controller_counts": missing_counts,
        "mean_rider_pelvis_to_partner_pelvis_distance": round(sum(distances) / len(distances), 5) if distances else None,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }


def _write_import_report(summary: dict[str, Any], rows: list[dict[str, Any]], invalid: list[str], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Pose Capture Import Report V1",
        "",
        f"- Input dir: `{summary['input_dir']}`",
        f"- Captures imported: `{summary['captures']}`",
        f"- Invalid files: `{summary['invalid_files']}`",
        f"- Output JSONL: `{summary['out_jsonl']}`",
        f"- Family counts: `{summary['family_counts']}`",
        f"- Subtype counts: `{summary['subtype_counts']}`",
        f"- Mean rider pelvis to partner pelvis distance: `{summary['mean_rider_pelvis_to_partner_pelvis_distance']}`",
        "- ML training performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
    ]
    if not rows:
        lines.extend([
            "No captures were imported yet. This is OK before the VaM plugin has saved snapshots.",
            "",
        ])
    if invalid:
        lines.append("## Invalid Files")
        lines.extend(f"- {msg}" for msg in invalid)
        lines.append("")
    if summary["missing_controller_counts"]:
        lines.append("## Missing Controllers")
        for key, count in sorted(summary["missing_controller_counts"].items()):
            lines.append(f"- `{key}`: {count}")
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")

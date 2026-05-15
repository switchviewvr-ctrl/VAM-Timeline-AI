"""Discover VaM controller names and conservatively map them to body parts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import dump_json


CANONICAL_BODY_PARTS = (
    "root",
    "hip",
    "pelvis",
    "abdomen",
    "chest",
    "neck",
    "head",
    "left_hand",
    "right_hand",
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "left_thigh",
    "right_thigh",
    "unknown",
)

PATTERNS: tuple[tuple[str, str], ...] = (
    ("pelvis", "pelvis"),
    ("hip", "hip"),
    ("abdomen", "abdomen"),
    ("chest", "chest"),
    ("neck", "neck"),
    ("head", "head"),
    ("eyetarget", "head"),
    ("lhand", "left_hand"),
    ("lefthand", "left_hand"),
    ("rhand", "right_hand"),
    ("righthand", "right_hand"),
    ("lelbow", "left_elbow"),
    ("leftelbow", "left_elbow"),
    ("relbow", "right_elbow"),
    ("rightelbow", "right_elbow"),
    ("lknee", "left_knee"),
    ("leftknee", "left_knee"),
    ("rknee", "right_knee"),
    ("rightknee", "right_knee"),
    ("lfoot", "left_foot"),
    ("leftfoot", "left_foot"),
    ("rfoot", "right_foot"),
    ("rightfoot", "right_foot"),
    ("lthigh", "left_thigh"),
    ("leftthigh", "left_thigh"),
    ("rthigh", "right_thigh"),
    ("rightthigh", "right_thigh"),
)


def discover_controller_map(sample_index: str | Path, out: str | Path, map_out: str | Path, report: str | Path) -> dict[str, Any]:
    rows = _load_jsonl(sample_index)
    freq = Counter(name for row in rows if row.get("bake_status") == "ok" for name in row.get("controller_names", []))
    mapping = {name: map_controller_name(name) for name in sorted(freq)}
    per_sample = []
    for row in rows:
        if row.get("bake_status") != "ok":
            continue
        per_sample.append(
            {
                "sample_id": row.get("sample_id"),
                "source_scene_file": row.get("source_scene_file"),
                "technical_atom_id": row.get("technical_atom_id"),
                "mapped_body_parts": sorted({mapping[name]["body_part"] for name in row.get("controller_names", []) if mapping.get(name)}),
            }
        )
    inventory = {
        "controller_name_frequency": dict(freq.most_common()),
        "controller_mappings": mapping,
        "per_sample_mapped_body_parts": per_sample,
        "unmapped_controller_names": [name for name, item in mapping.items() if item["body_part"] == "unknown"],
        "ambiguous_controller_names": [name for name, item in mapping.items() if item["mapping_confidence"] == "low"],
        "recommended_mapping_additions": [],
    }
    dump_json(out, inventory)
    dump_json(map_out, {"body_parts": list(CANONICAL_BODY_PARTS), "controller_mappings": mapping})
    _write_report(inventory, report)
    return inventory


def map_controller_name(name: str) -> dict[str, Any]:
    token = "".join(ch for ch in name.lower() if ch.isalnum())
    for pattern, body_part in PATTERNS:
        if pattern in token:
            return {"body_part": body_part, "mapping_confidence": "high", "matched_pattern": pattern}
    if token in {"control", "rootcontrol"}:
        return {"body_part": "root", "mapping_confidence": "medium", "matched_pattern": token}
    return {"body_part": "unknown", "mapping_confidence": "none", "matched_pattern": None}


def _write_report(inventory: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    mappings = inventory["controller_mappings"]
    part_counts = Counter(item["body_part"] for item in mappings.values())
    lines = [
        "# Controller Mapping Report",
        "",
        "Controller names are mapped to body parts conservatively. This is not actor role or gender inference.",
        "",
        "## Body Parts Measurable",
        "",
    ]
    for part, count in sorted(part_counts.items()):
        lines.append(f"- `{part}`: {count} controller names")
    lines.extend(["", "## Controller Frequencies", ""])
    for name, freq in list(inventory["controller_name_frequency"].items())[:40]:
        m = mappings[name]
        lines.append(f"- `{name}`: {freq} -> `{m['body_part']}` ({m['mapping_confidence']})")
    if inventory["unmapped_controller_names"]:
        lines.extend(["", "## Unmapped Controller Names", ""])
        for name in inventory["unmapped_controller_names"]:
            lines.append(f"- `{name}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

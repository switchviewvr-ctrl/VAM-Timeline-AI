"""Lightweight scene scanner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_json
from vam_timeline_ai.scene.source_discovery import discover_sources
from vam_timeline_ai.timeline.timeline_parser import looks_like_external_timeline_export


def scan_json_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    item: dict[str, Any] = {
        "file_name": p.name,
        "file_path": str(p),
        "file_size_bytes": p.stat().st_size if p.exists() else None,
        "parse_status": "unknown",
        "json_kind": "unknown",
        "is_vam_scene": False,
        "is_external_timeline_export": False,
        "atoms_count": 0,
        "person_atoms_count": 0,
        "person_atoms": [],
        "warnings": [],
    }
    try:
        data = load_json(p)
    except Exception as exc:  # noqa: BLE001 - scanner must not crash on bad JSON
        item["parse_status"] = "error"
        item["json_kind"] = "parse_failed"
        item["error"] = str(exc)
        return item

    item["parse_status"] = "ok"
    if not isinstance(data, dict):
        item["json_kind"] = type(data).__name__
        item["warnings"].append("top-level JSON is not an object")
        item.update(discover_sources(data))
        return item

    atoms = data.get("atoms")
    if isinstance(atoms, list):
        item["json_kind"] = "vam_scene"
        item["is_vam_scene"] = True
        item["atoms_count"] = len(atoms)
        persons = []
        for atom in atoms:
            if not isinstance(atom, dict):
                continue
            atom_type = str(atom.get("type") or atom.get("atomType") or atom.get("AtomType") or "")
            if atom_type == "Person":
                persons.append(str(atom.get("id", "")))
        item["person_atoms"] = persons
        item["person_atoms_count"] = len(persons)
    elif looks_like_external_timeline_export(data):
        item["json_kind"] = "external_timeline_export"
        item["is_external_timeline_export"] = True
    else:
        item["json_kind"] = "json_object"
        item["warnings"].append("no atoms[] found; not classified as a VaM scene")

    item.update(discover_sources(data))
    item["likely_filename_tags"] = filename_tags(p.name)
    return item


def filename_tags(file_name: str) -> list[str]:
    lower = file_name.lower()
    tags: list[str] = []
    for token, tag in [
        ("cow", "filename_cowgirl_hint"),
        ("riding", "filename_riding_hint"),
        ("ride", "filename_ride_hint"),
        ("couch", "filename_couch_hint"),
        ("voxta", "filename_voxta_hint"),
        ("toy", "filename_specialty_toy_hint"),
        ("bootstrapped", "filename_specialty_edgecase_hint"),
        ("knot", "filename_specialty_edgecase_hint"),
    ]:
        if token in lower and tag not in tags:
            tags.append(tag)
    return tags

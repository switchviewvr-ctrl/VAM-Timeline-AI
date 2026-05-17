"""Load top-down motion ontology YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised only when PyYAML is absent
    yaml = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8-sig")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return data or {}


def load_motion_families(path: str | Path) -> dict[str, Any]:
    data = load_yaml(path)
    return data.get("families", data)


def latest_existing(paths: list[str | Path]) -> Path | None:
    for path in paths:
        p = Path(path)
        if p.exists():
            return p
    return None

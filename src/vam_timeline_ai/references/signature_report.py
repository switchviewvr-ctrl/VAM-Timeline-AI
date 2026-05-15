"""Build family signatures from handmade reference features."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl


def build_handmade_reference_signatures(features: str | Path, out_json: str | Path, report: str | Path) -> dict[str, Any]:
    rows = load_jsonl(features)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("label_family") or "unknown")].append(row)
        if row.get("is_transition_or_realign"):
            by_family["transition"].append(row)
    signatures = {"families": {}}
    for family, family_rows in sorted(by_family.items()):
        signatures["families"][family] = _signature(family_rows)
    dump_json(out_json, signatures)
    _write_report(signatures, report)
    return signatures


def _signature(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controller_sets = Counter(tuple(sorted(r.get("controller_names", []) or [])) for r in rows)
    controller_freq = Counter(name for r in rows for name in r.get("controller_names", []) or [])
    family_freq = Counter(r.get("primary_controller_family", "unknown") for r in rows)
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in (row.get("feature_values") or {}).items():
            try:
                val = float(value)
            except Exception:
                continue
            if np.isfinite(val):
                numeric[key].append(val)
    return {
        "count": len(rows),
        "controller_sets": [list(k) for k, _ in controller_sets.most_common(5)],
        "typical_controllers": [name for name, _ in controller_freq.most_common(12)],
        "primary_controller_family_counts": dict(family_freq),
        "feature_medians": {key: float(np.median(vals)) for key, vals in numeric.items() if vals},
        "feature_ranges": {key: [float(np.min(vals)), float(np.max(vals))] for key, vals in numeric.items() if vals},
        "teleport_root_warning_count": sum(1 for r in rows if r.get("teleport_risk") in {"medium", "high"}),
    }


def _write_report(signatures: dict[str, Any], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Handmade Reference Signatures",
        "",
        "Cowgirl should generally show hipControl motion, often thigh/torso support, and should not be head-only or root-only.",
        "",
        "BJ/head references are often headControl-dominant and must not be misclassified as Cowgirl just because they are rhythmic.",
        "",
        "Doggy can overlap with hip/thigh motion and needs later pose/context separation. Hand/head gestures are useful false-positive guards. Realign/transition references are not clean rhythmic motion.",
        "",
    ]
    for family, sig in (signatures.get("families") or {}).items():
        lines.extend([f"## {family}", "", f"- Count: {sig.get('count', 0)}", f"- Typical controllers: {', '.join(sig.get('typical_controllers', [])[:10]) or 'none'}", f"- Primary families: {sig.get('primary_controller_family_counts', {})}", ""])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

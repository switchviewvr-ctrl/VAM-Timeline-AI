"""Build possible actor/context pair candidates without assigning roles."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.identity import make_pair_id
from vam_timeline_ai.io.json_utils import write_jsonl


def build_context_pair_candidates(sample_index: str | Path, out: str | Path, report: str | Path) -> list[dict[str, Any]]:
    samples = [r for r in _load_jsonl(sample_index) if r.get("bake_status") == "ok" and r.get("controller_names")]
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_scene[str(sample.get("source_scene_file"))].append(sample)
    pairs: list[dict[str, Any]] = []
    for scene, scene_samples in by_scene.items():
        for i, a in enumerate(scene_samples):
            for b in scene_samples[i + 1:]:
                if a.get("technical_atom_id") == b.get("technical_atom_id"):
                    continue
                pair = _pair_record(scene, a, b)
                if pair["pair_confidence"] > 0:
                    pairs.append(pair)
    write_jsonl(out, pairs)
    _write_report(pairs, report)
    return pairs


def _pair_record(scene: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    da = float(a.get("duration_seconds") or 0.0)
    db = float(b.get("duration_seconds") or 0.0)
    overlap = min(da, db) / max(da, db, 1e-6)
    reasons = []
    confidence = 0.0
    if a.get("clip_name") and a.get("clip_name") == b.get("clip_name"):
        confidence += 0.45
        reasons.append("same clip_name")
    if overlap > 0.8:
        confidence += 0.35
        reasons.append("similar duration")
    if _person_like(a) and _person_like(b):
        confidence += 0.2
        reasons.append("both have Person-like controller sets")
    return {
        "pair_id": make_pair_id(scene, str(a.get("sample_id")), str(b.get("sample_id")), a.get("clip_name"), b.get("clip_name")),
        "source_scene_file": scene,
        "sample_id_a": a.get("sample_id"),
        "technical_atom_id_a": a.get("technical_atom_id"),
        "sample_id_b": b.get("sample_id"),
        "technical_atom_id_b": b.get("technical_atom_id"),
        "clip_name_a": a.get("clip_name"),
        "clip_name_b": b.get("clip_name"),
        "duration_a": da,
        "duration_b": db,
        "duration_overlap_proxy": float(overlap),
        "pair_confidence": round(min(confidence, 1.0), 3),
        "pairing_reasons": reasons,
        "warnings": [],
        "semantic_role_a": "unknown",
        "semantic_role_b": "unknown",
    }


def _person_like(sample: dict[str, Any]) -> bool:
    names = set(sample.get("controller_names", []))
    return bool(names & {"hipControl", "pelvisControl"}) and bool(names & {"chestControl", "headControl"})


def _write_report(pairs: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Context Pair Candidates",
        "",
        "Pairs are possible actor/context pairings. No rider/receiver role is assigned here.",
        "",
        f"- Pair candidates: {len(pairs)}",
        "",
    ]
    for pair in sorted(pairs, key=lambda p: p["pair_confidence"], reverse=True)[:50]:
        lines.append(f"- `{pair['source_scene_file']}`: `{pair['sample_id_a']}` + `{pair['sample_id_b']}` confidence={pair['pair_confidence']} ({', '.join(pair['pairing_reasons'])})")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

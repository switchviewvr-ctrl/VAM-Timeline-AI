"""Plan larger human review batches without exporting them."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl


TARGETS = {
    "cowgirl_clean_motion_generation_safe": 20,
    "cowgirl_contact_support": 10,
    "cowgirl_context_or_intro": 8,
    "bj_oral": 8,
    "receiver_response": 5,
    "standing_hand_head": 5,
    "unknown_or_unusable": 4,
}


def plan_larger_review_batch_v1(
    run_dir: str | Path,
    semantic_db: str | Path,
    cowgirl_db: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    semantic = load_jsonl(semantic_db)
    cowgirl = load_jsonl(cowgirl_db)
    pools = _pools(semantic, cowgirl)
    selected = {name: _select_with_caps(rows, limit) for name, (rows, limit) in pools.items()}
    _write_report(selected, pools, out)
    return {
        "status": "ok",
        "planned_total": sum(len(v) for v in selected.values()),
        "target_total": sum(TARGETS.values()),
        "selected_counts": {k: len(v) for k, v in selected.items()},
        "out": str(out),
    }


def _pools(semantic: list[dict[str, Any]], cowgirl: list[dict[str, Any]]) -> dict[str, tuple[list[dict[str, Any]], int]]:
    return {
        "cowgirl_clean_motion_generation_safe": ([r for r in cowgirl if r.get("category") == "cowgirl_clean_motion_generation_safe"], 20),
        "cowgirl_contact_support": ([r for r in cowgirl if r.get("category") in {"cowgirl_hands_on_partner_chest", "cowgirl_hands_on_partner_hips", "cowgirl_hands_on_floor_or_bed", "cowgirl_ambiguous_partner_contact"}], 10),
        "cowgirl_context_or_intro": ([r for r in cowgirl if r.get("category") in {"cowgirl_pose_context_low_motion", "cowgirl_intro_alignment", "cowgirl_possible_insertion_setup"}], 8),
        "bj_oral": ([r for r in semantic if r.get("semantic_family") == "bj_oral"], 8),
        "receiver_response": ([r for r in semantic if r.get("semantic_family") == "receiver_response"], 5),
        "standing_hand_head": ([r for r in semantic if r.get("semantic_family") in {"hand_gesture", "head_gesture"}], 5),
        "unknown_or_unusable": ([r for r in semantic if r.get("semantic_family") == "unknown"], 4),
    }


def _select_with_caps(rows: list[dict[str, Any]], limit: int, scene_cap: int = 5) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_windows: set[str] = set()
    scene_counts: Counter[str] = Counter()
    for row in sorted(rows, key=lambda r: -float(r.get("semantic_score") or r.get("motion_score") or 0.0)):
        if len(selected) >= limit:
            break
        wid = str(row.get("window_id") or "")
        sample = str(row.get("sample_id") or "")
        scene = str(row.get("source_scene_file") or "unknown")
        if wid in seen_windows:
            continue
        if sample and sample in seen_samples:
            continue
        if scene_counts[scene] >= scene_cap:
            continue
        selected.append(row)
        seen_windows.add(wid)
        if sample:
            seen_samples.add(sample)
        scene_counts[scene] += 1
    return selected


def _write_report(selected: dict[str, list[dict[str, Any]]], pools: dict[str, tuple[list[dict[str, Any]], int]], out: str | Path) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Larger Review Batch Plan v1",
        "",
        "This is a plan only. No 60-item review package was generated.",
        "",
        "## Target Mix",
        "",
        "- 20 cowgirl_clean_motion_generation_safe",
        "- 10 Cowgirl contact/support candidates: partner chest, partner hips, floor/bed, hands free/ambiguous",
        "- 8 Cowgirl pose-context low-motion / intro alignment",
        "- 8 BJ/oral candidates",
        "- 5 receiver-response candidates",
        "- 5 standing/hand/head candidates",
        "- 4 unknown/unusable candidates",
        "",
        "## Planned Availability With Caps",
        "",
    ]
    for name, (rows, target_count) in pools.items():
        chosen = selected.get(name, [])
        lines.append(f"- `{name}`: target {target_count}, available {len(rows)}, planned {len(chosen)}")
    lines.extend(
        [
            "",
            "## Constraints For Future Export",
            "",
            "- Max 1 item per sample_id.",
            "- Max 5 items per scene for the 60-item batch.",
            "- Avoid near-duplicate windows from the same sample/time/phase.",
            "- Use the practical VaM review package builder after explicit user approval.",
            "",
            "## Expected Manual Review Time",
            "",
            "- Rough estimate: 60 to 120 minutes depending on scene load times and Timeline segment import success.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

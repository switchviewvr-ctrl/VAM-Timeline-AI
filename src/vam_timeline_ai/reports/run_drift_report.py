"""clean_v2 -> clean_v3 drift report."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl


def compare_clean_v2_clean_v3(clean_v2: str | Path, clean_v3: str | Path, out: str | Path) -> dict[str, Any]:
    v2 = Path(clean_v2)
    v3 = Path(clean_v3)
    v2_cow = _load_first(v2 / "datasets" / "cowgirl_candidate_db_v3.jsonl", v2 / "datasets" / "cowgirl_candidate_db_v2.jsonl", v2 / "audits" / "cowgirl_candidate_scores_v11.jsonl")
    v3_cow = _load_first(v3 / "datasets" / "cowgirl_candidate_db_v6.jsonl", v3 / "datasets" / "cowgirl_candidate_db_v5.jsonl")
    v3_sem = _load_first(v3 / "datasets" / "semantic_candidate_db_v1.jsonl", v3 / "datasets" / "semantic_candidate_db_v0.jsonl")
    by_v3 = {r.get("window_id"): r for r in v3_cow if r.get("window_id")}
    changed = Counter()
    for row in v2_cow:
        match = by_v3.get(row.get("window_id"))
        if not match:
            continue
        cat = str(match.get("category") or "")
        fam = str(match.get("semantic_family") or "")
        if fam == "bj_oral" or "bj_oral" in cat:
            changed["became_bj_oral"] += 1
        elif "receiver_response" in cat:
            changed["became_receiver_response"] += 1
        elif "standing" in cat:
            changed["became_standing_hand_head"] += 1
        elif fam == "unknown" or "unknown" in cat:
            changed["became_unknown"] += 1
        elif "missing_partner_context" in cat:
            changed["became_missing_partner_context"] += 1
        elif "low_motion" in cat:
            changed["became_low_motion_context"] += 1
    _write_report(v2, v3, v2_cow, v3_cow, v3_sem, changed, out)
    return {
        "status": "ok",
        "v2_cowgirl_records": len(v2_cow),
        "v3_cowgirl_records": len(v3_cow),
        "v3_semantic_records": len(v3_sem),
        "changed": dict(changed),
        "out": str(out),
    }


def _load_first(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def _write_report(
    v2: Path,
    v3: Path,
    v2_cow: list[dict[str, Any]],
    v3_cow: list[dict[str, Any]],
    v3_sem: list[dict[str, Any]],
    changed: Counter[str],
    out: str | Path,
) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# clean_v2 to clean_v3 Drift Report",
        "",
        f"- clean_v2: `{v2}`",
        f"- clean_v3: `{v3}`",
        f"- clean_v2 Cowgirl records found: {len(v2_cow)}",
        f"- clean_v3 Cowgirl records found: {len(v3_cow)}",
        f"- clean_v3 semantic records found: {len(v3_sem)}",
        "",
        "## Category Counts",
        "",
        "### clean_v2 Cowgirl",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in Counter(r.get("category") for r in v2_cow).most_common()) if v2_cow else lines.append("- Missing or unavailable")
    lines.extend(["", "### clean_v3 Cowgirl"])
    lines.extend(f"- `{k}`: {v}" for k, v in Counter(r.get("category") for r in v3_cow).most_common()) if v3_cow else lines.append("- Missing or unavailable")
    lines.extend(["", "## Matched Window Drift", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in changed.most_common()) if changed else lines.append("- No matched-window drift could be computed or no overlap found.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- clean_v3 is stricter because it separates pose, motion, partner relation, contact/support, phase, and generation safety.",
            "- clean_v2 Cowgirl-only outputs should be considered deprecated for contact/support and generation-safe decisions.",
            "- clean_v3 v16 should be reviewed before any larger batch or generation continuation.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

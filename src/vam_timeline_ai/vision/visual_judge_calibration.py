"""Build a small visual judge calibration set from available review artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


TARGETS = [
    ("clear doggy", "should_detect_doggy"),
    ("elevated doggy", "should_detect_doggy"),
    ("kneeling cowgirl", "should_detect_cowgirl"),
    ("reverse cowgirl", "should_detect_reverse_cowgirl_if_visible"),
    ("cowgirl lean-back supported", "should_detect_cowgirl"),
    ("standing hand/head", "should_detect_standing"),
    ("BJ/oral", "should_detect_bj_oral"),
    ("cowgirl pose but no motion", "should_not_guess_family_without_evidence"),
    ("unknown/broken pose", "should_mark_unknown_if_no_partner_motion"),
    ("single-frame no-partner", "should_not_guess_family_without_evidence"),
]


def build_visual_judge_calibration_set_v1(run_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    requests = []
    for path in sorted(run.glob("audits/**/visual_judge_requests.jsonl")):
        requests.extend(load_jsonl(path))
    items = []
    for label, behavior in TARGETS:
        match = _find_match(requests, label)
        items.append(
            {
                "calibration_id": label.replace("/", "_").replace(" ", "_"),
                "target_example": label,
                "image_or_contact_sheet_path": match.get("primary_visual_path") if match else None,
                "source_review_id": match.get("review_id") if match else None,
                "expected_coarse_class": _expected_class(label),
                "expected_behavior": behavior,
                "notes": "Calibration is audit-only; missing paths mean user should add a real VaM capture.",
            }
        )
    write_jsonl(out / "calibration_items.jsonl", items)
    (out / "calibration_report.md").write_text(
        "# Visual Judge Calibration Set V1\n\n"
        f"- Items: {len(items)}\n"
        f"- With visual path: {sum(1 for i in items if i.get('image_or_contact_sheet_path'))}\n"
        "- Default trust gate remains disabled until live calibration passes.\n",
        encoding="utf-8",
    )
    return {"status": "ok", "items": len(items), "with_visual_path": sum(1 for i in items if i.get("image_or_contact_sheet_path")), "out_dir": str(out), "items_path": str(out / "calibration_items.jsonl")}


def _find_match(requests: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    terms = label.lower().replace("-", " ").split()
    for row in requests:
        hay = " ".join(str(row.get(k) or "") for k in ["review_id", "primary_visual_path", "visual_quality", "system_guess"]).lower()
        if any(term in hay for term in terms):
            return row
    return requests[0] if requests else None


def _expected_class(label: str) -> str:
    lower = label.lower()
    if "doggy" in lower:
        return "doggy"
    if "cowgirl" in lower and "reverse" in lower:
        return "reverse_cowgirl"
    if "cowgirl" in lower:
        return "cowgirl"
    if "standing" in lower:
        return "standing_hand_head"
    if "bj" in lower or "oral" in lower:
        return "bj_oral"
    return "unknown"

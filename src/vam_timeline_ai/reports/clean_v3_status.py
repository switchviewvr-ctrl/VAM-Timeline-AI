"""Operator status report for clean_v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any


KEY_ARTIFACTS = {
    "semantic_actions_v1": "semantic_actions/semantic_actions_v1.jsonl",
    "semantic_candidate_db_v1": "datasets/semantic_candidate_db_v1.jsonl",
    "cowgirl_candidate_db_v6": "datasets/cowgirl_candidate_db_v6.jsonl",
    "v16_review": "audits/semantic_review_010_v16/semantic_review_010.jsonl",
    "v16_vam_package": "audits/semantic_review_010_v16/vam_review_package/vam_review_index.html",
    "dashboard": "reports/clean_v3_semantic_dashboard.md",
    "human_ledger": "audits/human_review_ledger.jsonl",
}


def clean_v3_status(run_dir: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    paths = {name: run / rel for name, rel in KEY_ARTIFACTS.items()}
    present = {name: path.exists() for name, path in paths.items()}
    latest_review = "semantic_review_010_v16" if present.get("v16_review") else "semantic_review_010_v15" if (run / "audits" / "semantic_review_010_v15").exists() else "none"
    blockers = _blockers(present)
    lines = [
        "# clean_v3 Status",
        "",
        f"- Run exists: `{run.exists()}`",
        f"- Latest review package: `{latest_review}`",
        "",
        "## Key Artifacts",
        "",
    ]
    lines.extend(f"- `{name}`: {'present' if ok else 'missing'} - `{paths[name]}`" for name, ok in present.items())
    lines.extend(["", "## Current Blockers", ""])
    lines.extend(f"- {b}" for b in blockers) if blockers else lines.append("- None blocking QA. v16 still needs human review before larger batch.")
    lines.extend(
        [
            "",
            "## Next Recommended Manual Action",
            "",
            "Open the v16 VaM review package and manually validate the 10 examples before exporting any larger review batch.",
        ]
    )
    out_path = Path(out) if out else run / "reports" / "clean_v3_status.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "run_exists": run.exists(), "present": present, "blockers": blockers, "out": str(out_path)}


def _blockers(present: dict[str, bool]) -> list[str]:
    blockers = []
    if not present.get("semantic_actions_v1"):
        blockers.append("semantic_actions_v1 is missing; calibration must run before QA conclusions are trusted.")
    if not present.get("cowgirl_candidate_db_v6"):
        blockers.append("cowgirl_candidate_db_v6 is missing; v16 selection cannot be trusted.")
    if not present.get("v16_vam_package"):
        blockers.append("v16 VaM review package is missing; human review is blocked.")
    blockers.append("Contact/support detection remains low confidence until v16 is manually reviewed.")
    blockers.append("Generation-safe classification remains experimental; do not continue generation from v16 yet.")
    return blockers

"""One-command clean_v3 overnight QA runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from vam_timeline_ai.audits.error_taxonomy import build_error_taxonomy_report
from vam_timeline_ai.audits.human_review_memory import build_human_review_ledger
from vam_timeline_ai.audits.repo_safety import audit_repo_safety
from vam_timeline_ai.audits.review_batch_planner import plan_larger_review_batch_v1
from vam_timeline_ai.datasets.db_invariant_validator import validate_semantic_dbs
from vam_timeline_ai.reports.clean_v3_status import clean_v3_status
from vam_timeline_ai.reports.prompt_capability_matrix import write_prompt_capability_matrix
from vam_timeline_ai.reports.run_drift_report import compare_clean_v2_clean_v3
from vam_timeline_ai.reports.semantic_qa_dashboard import write_clean_v3_dashboard


def run_clean_v3_overnight_qa(run_dir: str | Path, include_runs: str) -> dict[str, Any]:
    run = Path(run_dir)
    reports = run / "reports"
    audits = run / "audits"
    reports.mkdir(parents=True, exist_ok=True)
    audits.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    def step(name: str, fn: Callable[[], dict[str, Any]]) -> None:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"status": "blocked", "error": str(exc)}

    step(
        "human_review_ledger",
        lambda: build_human_review_ledger(
            run,
            include_runs,
            audits / "human_review_ledger.jsonl",
            audits / "human_review_ledger.csv",
            audits / "human_review_ledger_report.md",
        ),
    )
    step(
        "error_taxonomy",
        lambda: build_error_taxonomy_report(audits / "human_review_ledger.jsonl", audits / "error_taxonomy_report.md"),
    )
    step(
        "semantic_db_invariants",
        lambda: validate_semantic_dbs(
            run,
            run / "datasets" / "semantic_candidate_db_v1.jsonl",
            run / "datasets" / "cowgirl_candidate_db_v6.jsonl",
            reports / "semantic_db_invariant_report.md",
        ),
    )
    step(
        "dashboard",
        lambda: write_clean_v3_dashboard(
            run,
            reports / "clean_v3_semantic_dashboard.md",
            reports / "clean_v3_semantic_dashboard.html",
        ),
    )
    step(
        "drift_report",
        lambda: compare_clean_v2_clean_v3(run.parent / "clean_v2", run, reports / "clean_v2_to_clean_v3_drift_report.md"),
    )
    step(
        "larger_review_plan",
        lambda: plan_larger_review_batch_v1(
            run,
            run / "datasets" / "semantic_candidate_db_v1.jsonl",
            run / "datasets" / "cowgirl_candidate_db_v6.jsonl",
            reports / "larger_review_batch_plan_v1.md",
        ),
    )
    step("prompt_matrix", lambda: write_prompt_capability_matrix(run, reports / "prompt_capability_matrix.md"))
    step("status", lambda: clean_v3_status(run, reports / "clean_v3_status.md"))
    step("repo_safety", lambda: audit_repo_safety(Path.cwd(), audits / "repo_safety_report.md"))
    _write_summary(results, reports / "overnight_qa_summary.md")
    return {"status": "ok", "results": results, "summary": str(reports / "overnight_qa_summary.md")}


def _write_summary(results: dict[str, Any], out: Path) -> None:
    lines = [
        "# clean_v3 Overnight QA Summary",
        "",
        "This QA run does not train ML, does not generate animations, and does not modify manual labels.",
        "",
        "## Steps",
        "",
    ]
    for name, result in results.items():
        lines.append(f"- `{name}`: `{result.get('status')}`")
        if result.get("error"):
            lines.append(f"  - Error: {result['error']}")
    lines.extend(["", "## Key Outputs", ""])
    for name, result in results.items():
        for key in ["out", "out_md", "out_jsonl", "summary"]:
            if result.get(key):
                lines.append(f"- `{name}` {key}: `{result[key]}`")
    lines.extend(
        [
            "",
            "## Morning Action",
            "",
            "Open `reports/clean_v3_semantic_dashboard.md`, then review `audits/semantic_review_010_v16/vam_review_package/vam_review_index.html` before approving any larger batch.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

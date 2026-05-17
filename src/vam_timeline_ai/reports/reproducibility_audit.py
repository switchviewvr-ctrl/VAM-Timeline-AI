"""Schema, artifact, reproducibility, and operator reports for clean_v3."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import subprocess

from vam_timeline_ai.io.json_utils import dump_json, load_jsonl, write_jsonl
from vam_timeline_ai.reports.candidate_lineage import write_candidate_lineage_report


HASH_SIZE_LIMIT = 50_000_000


SCHEMA_REGISTRY: list[dict[str, Any]] = [
    {
        "artifact_name": "pose_features_v0",
        "schema_version": "pose_features_v0",
        "required_fields": ["window_id", "sample_id"],
        "optional_fields": [
            "pelvis_height_proxy",
            "chest_height_proxy",
            "head_height_proxy",
            "knee_height_proxy",
            "foot_height_proxy",
            "hand_height_proxy",
            "torso_forward_lean_proxy",
            "kneeling_score",
            "standing_score",
            "lying_on_back_score",
            "pose_anchor_completeness",
            "pose_controller_coverage",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": [
            "relative_motion/relative_motion_window_index.jsonl",
            "audits/body_motion_quality.jsonl",
            "audits/pose_anchor_completeness.jsonl",
            "audits/controller_validity.jsonl",
        ],
        "suitability": ["analysis_only", "not_training_truth"],
    },
    {
        "artifact_name": "pose_semantics_v0",
        "schema_version": "pose_semantics_v0",
        "required_fields": ["window_id", "sample_id", "pose_family", "pose_subtype", "pose_confidence"],
        "optional_fields": ["support_context", "anchor_requirements", "pose_generation_safe", "warnings"],
        "deprecated_fields": [],
        "source_artifact_dependencies": ["pose_semantics/pose_features_v0.jsonl"],
        "suitability": ["analysis_only", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "partner_relative_features_v0",
        "schema_version": "partner_relative_features_v0",
        "required_fields": ["window_id"],
        "optional_fields": [
            "pair_window_id",
            "rider_pelvis_to_partner_pelvis_offset",
            "pelvis_alignment_score",
            "rider_above_partner_score",
            "partner_lying_score",
            "hands_on_partner_chest_score",
            "hands_on_partner_hips_score",
            "partner_context_confidence",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": [
            "semantic/pair_windows_v1.jsonl",
            "features/cowgirl_pair_features_v0.jsonl",
            "pose_semantics/pose_semantics_v0.jsonl",
        ],
        "suitability": ["analysis_only", "not_training_truth"],
    },
    {
        "artifact_name": "interaction_semantics_v0",
        "schema_version": "interaction_semantics_v0",
        "required_fields": ["window_id", "interaction_family", "interaction_confidence"],
        "optional_fields": [
            "pair_window_id",
            "actor_role",
            "partner_role",
            "partner_relation",
            "contact_targets",
            "support_context",
            "contact_support_confidence",
            "warnings",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": [
            "interaction_semantics/partner_relative_features_v0.jsonl",
            "pose_semantics/pose_semantics_v0.jsonl",
        ],
        "suitability": ["analysis_only", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "semantic_actions_v0",
        "schema_version": "semantic_actions_v0",
        "required_fields": ["window_id", "semantic_family", "pose_family", "motion_subtype", "phase", "generation_safe"],
        "optional_fields": [
            "pair_window_id",
            "partner_relation",
            "contact_support",
            "semantic_score",
            "pose_score",
            "motion_score",
            "interaction_score",
            "warnings",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": [
            "datasets/cowgirl_candidate_db_v3.jsonl",
            "pose_semantics/pose_semantics_v0.jsonl",
            "interaction_semantics/interaction_semantics_v0.jsonl",
        ],
        "suitability": ["candidate_db", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "semantic_actions_v1",
        "schema_version": "semantic_actions_v1",
        "required_fields": ["window_id", "semantic_family", "pose_family", "motion_subtype", "phase", "generation_safe"],
        "optional_fields": [
            "motion_content_strength",
            "clean_motion_score",
            "low_motion_hold_score",
            "intro_alignment_score",
            "insertion_setup_score",
            "foot_anchor_motion_score",
            "lower_body_anchor_stability",
            "anchor_motion_weird",
            "contact_support_confidence",
            "contact_support_margin",
            "warnings",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": [
            "semantic_actions/semantic_actions_v0.jsonl",
            "audits/semantic_review_010_v15/semantic_review_010_human_notes.yaml",
        ],
        "suitability": ["candidate_db", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "semantic_candidate_db_v0",
        "schema_version": "semantic_candidate_db_v0",
        "required_fields": ["candidate_id", "window_id", "semantic_family", "category", "generation_safe"],
        "optional_fields": ["pose_family", "motion_subtype", "partner_relation", "contact_support", "preserve_for_future_dataset"],
        "deprecated_fields": [],
        "source_artifact_dependencies": ["semantic_actions/semantic_actions_v0.jsonl"],
        "suitability": ["candidate_db", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "semantic_candidate_db_v1",
        "schema_version": "semantic_candidate_db_v1",
        "required_fields": ["candidate_id", "window_id", "semantic_family", "category", "phase", "generation_safe"],
        "optional_fields": [
            "pose_family",
            "pose_subtype",
            "motion_subtype",
            "partner_relation",
            "contact_support",
            "safe_for_learning",
            "invalidity_reason",
            "preserve_for_future_dataset",
            "confidence_scores",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": ["semantic_actions/semantic_actions_v1.jsonl"],
        "suitability": ["candidate_db", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "cowgirl_candidate_db_v5",
        "schema_version": "cowgirl_candidate_db_v5",
        "required_fields": ["candidate_id", "window_id", "category", "semantic_family", "generation_safe"],
        "optional_fields": ["pose_family", "motion_subtype", "partner_relation", "contact_support", "invalidity_reason"],
        "deprecated_fields": [],
        "source_artifact_dependencies": ["datasets/semantic_candidate_db_v0.jsonl"],
        "suitability": ["candidate_db", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "cowgirl_candidate_db_v6",
        "schema_version": "cowgirl_candidate_db_v6",
        "required_fields": ["candidate_id", "window_id", "category", "semantic_family", "phase", "generation_safe"],
        "optional_fields": [
            "pose_family",
            "pose_subtype",
            "motion_subtype",
            "partner_relation",
            "contact_support",
            "contact_support_confidence",
            "anchor_motion_weird",
            "invalidity_reason",
        ],
        "deprecated_fields": ["semantic_cowgirl_generation_safe", "bj_oral_trap_negative"],
        "source_artifact_dependencies": ["datasets/semantic_candidate_db_v1.jsonl"],
        "suitability": ["candidate_db", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "motion_primitives_v1",
        "schema_version": "motion_primitives_v1",
        "required_fields": ["primitive_id", "semantic_family", "subtype", "source_window_ids", "relative_motion_summary"],
        "optional_fields": [
            "required_pose_family",
            "compatible_pose_subtypes",
            "required_partner_relation",
            "contact_support_requirements",
            "anchor_profile",
            "interaction_frame",
            "warnings",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": [
            "datasets/cowgirl_candidate_db_v5.jsonl",
            "relative_motion/relative_motion_features.jsonl",
            "pose_semantics/pose_semantics_v0.jsonl",
            "interaction_semantics/interaction_semantics_v0.jsonl",
        ],
        "suitability": ["generation_candidate", "not_training_truth"],
    },
    {
        "artifact_name": "review_manifest",
        "schema_version": "vam_review_manifest_v0",
        "required_fields": ["review_id", "window_id", "source_scene_path", "technical_atom_id", "start_seconds", "end_seconds"],
        "optional_fields": ["can_export_timeline_segment", "review_method", "warnings"],
        "deprecated_fields": [],
        "source_artifact_dependencies": ["audits/semantic_review_010_v*/semantic_review_010.jsonl"],
        "suitability": ["review_only", "not_training_truth"],
    },
    {
        "artifact_name": "review_answer_sheet",
        "schema_version": "vam_review_answer_sheet_v0",
        "required_fields": ["reviews"],
        "optional_fields": [
            "semantic_family_correct",
            "pose_correct",
            "motion_correct",
            "partner_relation_correct",
            "contact_support_correct",
            "generation_safe_correct",
            "notes",
        ],
        "deprecated_fields": [],
        "source_artifact_dependencies": ["review_manifest"],
        "suitability": ["review_only", "not_training_truth"],
    },
    {
        "artifact_name": "generated_motion_flow",
        "schema_version": "generated_motion_flow_v1",
        "required_fields": ["flow_id", "coordinate_space", "controller_tracks", "no_world_coordinates", "no_person_root_tracks"],
        "optional_fields": ["semantic_plan", "selected_primitive_group", "trajectory_shape", "rhythm_profile", "amplitude_profile"],
        "deprecated_fields": ["review_only_retargeted_flow_timeline_v0"],
        "source_artifact_dependencies": ["generation/cowgirl_motion_primitives_v1.jsonl"],
        "suitability": ["generation_candidate", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "retargeted_motion_flow",
        "schema_version": "retargeted_motion_flow_v1",
        "required_fields": ["flow_id", "source_generated_flow", "baseline_pose_id", "controller_tracks", "coordinate_space"],
        "optional_fields": ["safe_for_review_export_candidate", "safe_for_generation_template_candidate", "warnings"],
        "deprecated_fields": [],
        "source_artifact_dependencies": ["generated_motion_flow", "baseline_pose"],
        "suitability": ["generation_candidate", "review_only", "not_training_truth"],
    },
    {
        "artifact_name": "native_timeline_export",
        "schema_version": "native_timeline_export_v1",
        "required_fields": ["SerializeVersion", "Animations"],
        "optional_fields": ["generated_from_relative_flow", "baseline_pose_id", "experimental_native_timeline_export"],
        "deprecated_fields": ["review_only_timeline_v0", "review_only_retargeted_flow_timeline_v0"],
        "source_artifact_dependencies": ["retargeted_motion_flow", "baseline_pose"],
        "suitability": ["review_only", "not_training_truth"],
    },
]


IMPORTANT_ARTIFACTS: list[tuple[str, str, str, list[str]]] = [
    ("pose_semantics/pose_features_v0.jsonl", "pose_features_v0", "current", ["relative_motion/relative_motion_window_index.jsonl"]),
    ("pose_semantics/pose_semantics_v0.jsonl", "pose_semantics_v0", "current", ["pose_semantics/pose_features_v0.jsonl"]),
    ("interaction_semantics/partner_relative_features_v0.jsonl", "partner_relative_features_v0", "current", ["semantic/pair_windows_v1.jsonl"]),
    ("interaction_semantics/interaction_semantics_v0.jsonl", "interaction_semantics_v0", "current", ["interaction_semantics/partner_relative_features_v0.jsonl"]),
    ("semantic_actions/semantic_actions_v0.jsonl", "semantic_actions_v0", "deprecated", ["pose_semantics/pose_semantics_v0.jsonl"]),
    ("semantic_actions/semantic_actions_v1.jsonl", "semantic_actions_v1", "current", ["semantic_actions/semantic_actions_v0.jsonl"]),
    ("datasets/semantic_candidate_db_v0.jsonl", "semantic_candidate_db_v0", "deprecated", ["semantic_actions/semantic_actions_v0.jsonl"]),
    ("datasets/semantic_candidate_db_v1.jsonl", "semantic_candidate_db_v1", "current", ["semantic_actions/semantic_actions_v1.jsonl"]),
    ("datasets/cowgirl_candidate_db_v5.jsonl", "cowgirl_candidate_db_v5", "deprecated", ["datasets/semantic_candidate_db_v0.jsonl"]),
    ("datasets/cowgirl_candidate_db_v6.jsonl", "cowgirl_candidate_db_v6", "current", ["datasets/semantic_candidate_db_v1.jsonl"]),
    ("generation/cowgirl_motion_primitives_v1.jsonl", "motion_primitives_v1", "review_needed", ["datasets/cowgirl_candidate_db_v5.jsonl"]),
    ("audits/semantic_review_010_v16/vam_review_package/vam_review_manifest.jsonl", "review_manifest", "review_needed", ["audits/semantic_review_010_v16/semantic_review_010.jsonl"]),
    ("audits/semantic_review_010_v16/vam_review_package/vam_review_answer_sheet.yaml", "review_answer_sheet", "review_needed", ["audits/semantic_review_010_v16/vam_review_package/vam_review_manifest.jsonl"]),
    ("generation/cowgirl_motion_flow_v1_review/generated_motion_flow_v1.json", "generated_motion_flow", "deprecated", ["generation/cowgirl_motion_primitives_v0.jsonl"]),
    ("generation/cowgirl_motion_flow_v1_review/retargeted_motion_flow_v1.json", "retargeted_motion_flow", "deprecated", ["generation/cowgirl_motion_flow_v1_review/generated_motion_flow_v1.json"]),
    ("generation/native_timeline_export_v1/generated_cowgirl_motion_v1.timeline.json", "native_timeline_export", "deprecated", ["generation/cowgirl_motion_flow_v1_review/retargeted_motion_flow_v1.json"]),
]


def write_schema_registry(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out_json = reports / "schema_registry.json"
    out_md = reports / "schema_registry.md"
    data = {"run_dir": str(run), "schemas": SCHEMA_REGISTRY}
    dump_json(out_json, data)
    lines = [
        "# clean_v3 Schema Registry",
        "",
        "Candidate DBs are not manual ground truth. Review findings are audit labels unless explicitly promoted.",
        "Generated Timeline exports remain experimental unless manually confirmed in VaM.",
        "",
    ]
    for schema in SCHEMA_REGISTRY:
        lines.extend(
            [
                f"## {schema['artifact_name']}",
                "",
                f"- Schema version: `{schema['schema_version']}`",
                f"- Suitability: {', '.join(f'`{s}`' for s in schema['suitability'])}",
                f"- Required fields: {', '.join(f'`{f}`' for f in schema['required_fields'])}",
                f"- Optional fields: {', '.join(f'`{f}`' for f in schema['optional_fields']) or 'None'}",
                f"- Deprecated fields: {', '.join(f'`{f}`' for f in schema['deprecated_fields']) or 'None'}",
                f"- Dependencies: {', '.join(f'`{f}`' for f in schema['source_artifact_dependencies']) or 'None'}",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "ok", "schemas": len(SCHEMA_REGISTRY), "out_json": str(out_json), "out_md": str(out_md)}


def write_artifact_manifest(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    reports = run / "reports"
    rows = [_artifact_row(run, rel, schema, status, deps) for rel, schema, status, deps in IMPORTANT_ARTIFACTS]
    out_jsonl = reports / "artifact_manifest.jsonl"
    out_md = reports / "artifact_manifest.md"
    write_jsonl(out_jsonl, rows)
    counts = Counter(row["status"] for row in rows)
    lines = [
        "# clean_v3 Artifact Manifest",
        "",
        f"- Artifacts listed: {len(rows)}",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in counts.most_common())
    lines.extend(["", "## Artifacts", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['path']}",
                "",
                f"- Exists: `{row['exists']}`",
                f"- Status: `{row['status']}`",
                f"- Schema: `{row['schema_version']}`",
                f"- Size: {row['file_size']}",
                f"- Row count: {row['row_count']}",
                f"- SHA256: `{row['sha256']}`",
                f"- Created by: `{row['created_by_command']}`",
                f"- Dependencies: {', '.join(f'`{d}`' for d in row['source_dependencies']) or 'None'}",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "ok", "artifacts": len(rows), "status_counts": dict(counts), "out_jsonl": str(out_jsonl), "out": str(out_md)}


def write_deprecated_artifacts_report(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    clean_v2 = run.parent / "clean_v2"
    items = [
        ("Old weak labels", clean_v2 / "semantic" / "weak_labels_v2.jsonl", "reference only", "weak labels are proxies, not final truth"),
        ("Old silver labels", clean_v2 / "semantic" / "silver_labels_v2.jsonl", "reference only", "silver labels reproduce rule confidence, not human semantics"),
        ("clean_v2 Cowgirl DB v3", clean_v2 / "datasets" / "cowgirl_candidate_db_v3.jsonl", "deprecated as final truth", "lacks clean_v3 pose/interaction/contact semantics"),
        ("review_only_timeline_v0 custom schema", clean_v2 / "generation" / "timeline_export_v0" / "review_only_timeline_v0.json", "deprecated", "not native Timeline JSON"),
        ("Generated flow v0", clean_v2 / "generation" / "generated_motion_flow_v0.json", "reference only", "no pose/partner/contact context"),
        ("Retargeted flow v0", clean_v2 / "generation" / "retargeted_motion_flow_v0.json", "reference only", "review baseline only"),
        ("Old BJ/oral trap guard", clean_v2 / "audits" / "bj_oral_trap_guard.jsonl", "deprecated wording", "BJ/oral is a valid semantic family, not bad data"),
        ("Pre-clean_v3 generation-safe categories", clean_v2 / "datasets" / "cowgirl_candidate_db_v3.jsonl", "deprecated as generation truth", "generation-safe before partner/contact semantics is not enough"),
    ]
    out = reports / "deprecated_artifacts_report.md"
    lines = [
        "# Deprecated Artifact Report",
        "",
        "Do not delete these artifacts. Treat them as reference material only unless revalidated through clean_v3 semantic actions.",
        "",
    ]
    for title, path, status, reason in items:
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Path: `{path}`",
                f"- Exists: `{path.exists()}`",
                f"- Status: {status}",
                f"- Reason: {reason}",
                "- Replaced by: clean_v3 Semantic Action model with pose + motion + partner/contact semantics.",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "ok", "items": len(items), "out": str(out)}


def write_real_generation_input_requirements(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    report = run / "reports" / "real_generation_input_requirements.md"
    reference = _project_root(run) / "references" / "REAL_GENERATION_INPUT_REQUIREMENTS.md"
    text = """# Real Generation Input Requirements

The real final pipeline is:

```text
prompt -> semantic motion plan -> generated relative motion flow -> scene-aware retargeting -> native VaM Timeline animation
```

Synthetic review timelines are useful for debugging, but true scene-aware generation requires current-scene context from VaM.

## Required Current-Scene Inputs

- Target rider Person atom.
- Partner/receiver Person atom for interaction prompts.
- Current controller positions and rotations for the rider.
- Current controller positions and rotations for the partner.
- Partner pelvis reference.
- Partner chest reference.
- Partner head reference.
- Scene up/forward frame or partner-local frame.
- Body scale estimates for rider and partner.
- Controller availability and missing-controller policy.
- Controller physics/control state assumptions.
- Bed/floor/support reference if hands, knees, feet, or body support targets are relevant.

## Prompt Example

Prompt: `cowgirl grinding, hands on partner chest`

Required generation context:

- Rider atom.
- Partner atom.
- Partner chest target for both hands.
- Partner pelvis target for rider pelvis alignment.
- Rider baseline pose compatible with Cowgirl.
- Contact/support constraints that keep hands near partner chest.
- Knees/feet anchor constraints.
- No Person/root motion.
- No source-scene world-coordinate copying.

## Hard Boundary

Without current-scene partner references, the system can only create synthetic review timelines, not true scene-aware generation.
Candidate DBs and generated review flows are not enough to safely target a real VaM scene.
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    reference.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    reference.write_text(text, encoding="utf-8")
    return {"status": "ok", "report": str(report), "reference": str(reference)}


def write_morning_checklist(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    out = run / "reports" / "MORNING_CHECKLIST.md"
    v16 = run / "audits" / "semantic_review_010_v16" / "vam_review_package" / "vam_review_index.html"
    lines = [
        "# Morning Checklist",
        "",
        "1. Open `reports/overnight_qa_summary.md`.",
        "2. Open `reports/clean_v3_semantic_dashboard.html`.",
        "3. Check `reports/semantic_db_invariant_report.md` for errors first.",
        "4. Check the v16 review package if it exists.",
        "5. Do not trust candidate DBs if invariants failed.",
        "6. Review only the 10-item v16 batch first.",
        "7. If v16 passes, request the larger review batch.",
        "8. Do not train ML yet.",
        "9. Do not continue Timeline generation until pose/partner/contact semantics are validated.",
        "",
        "## First Files To Open",
        "",
        f"- `{run / 'reports' / 'overnight_qa_summary.md'}`",
        f"- `{run / 'reports' / 'clean_v3_semantic_dashboard.html'}`",
        f"- `{run / 'reports' / 'semantic_db_invariant_report.md'}`",
        f"- `{v16}`" if v16.exists() else "- v16 VaM review package: missing",
        f"- `{run / 'reports' / 'larger_review_batch_plan_v1.md'}`",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "out": str(out), "v16_package_exists": v16.exists()}


def write_repo_snapshot_report(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    project = _project_root(run)
    out = run / "reports" / "repo_snapshot_report.md"
    git_available = _git(["--version"], project)["ok"]
    is_repo = git_available and _git(["rev-parse", "--is-inside-work-tree"], project)["ok"]
    branch = _git_text(["branch", "--show-current"], project) if is_repo else ""
    commit = _git_text(["rev-parse", "HEAD"], project) if is_repo else ""
    status = _git_text(["status", "--short"], project).splitlines() if is_repo else []
    staged = _git_text(["diff", "--cached", "--name-only"], project).splitlines() if is_repo else []
    ignored = _git_text(["status", "--short", "--ignored", str(run.relative_to(project))], project).splitlines() if is_repo and _is_relative_to(run, project) else []
    warnings = _repo_snapshot_warnings(staged, status, run)
    lines = [
        "# Repository Snapshot Report",
        "",
        f"- Project root: `{project}`",
        f"- Git available: `{git_available}`",
        f"- Is git repo: `{is_repo}`",
        f"- Current branch: `{branch}`",
        f"- Latest commit hash: `{commit}`",
        f"- Working tree status entries: {len(status)}",
        f"- Staged entries: {len(staged)}",
        f"- Ignored clean_v3/run entries visible to git status: {len(ignored)}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {w}" for w in warnings) if warnings else lines.append("- None")
    lines.extend(["", "## Git Status Summary", ""])
    lines.extend(f"- `{line}`" for line in status[:200]) if status else lines.append("- Clean or unavailable")
    if len(status) > 200:
        lines.append(f"- ... {len(status) - 200} more")
    lines.extend(["", "## Ignored Run Artifact Summary", ""])
    lines.extend(f"- `{line}`" for line in ignored[:80]) if ignored else lines.append("- None visible or git unavailable")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "branch": branch, "commit": commit, "warnings": warnings, "out": str(out)}


def run_clean_v3_reproducibility_audit(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    def step(name: str, fn) -> None:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"status": "blocked", "error": str(exc)}

    step("schema_registry", lambda: write_schema_registry(run))
    step("artifact_manifest", lambda: write_artifact_manifest(run))
    step("candidate_lineage", lambda: write_candidate_lineage_report(run, reports / "candidate_lineage_report.md"))
    step("deprecated_artifacts", lambda: write_deprecated_artifacts_report(run))
    step("real_generation_input_requirements", lambda: write_real_generation_input_requirements(run))
    step("morning_checklist", lambda: write_morning_checklist(run))
    step("repo_snapshot", lambda: write_repo_snapshot_report(run))
    summary = reports / "reproducibility_audit_summary.md"
    _write_repro_summary(results, summary)
    return {"status": "ok", "results": results, "summary": str(summary)}


def _artifact_row(run: Path, rel: str, schema: str, default_status: str, deps: list[str]) -> dict[str, Any]:
    path = run / rel
    exists = path.exists()
    status = default_status if exists else "missing"
    return {
        "path": str(path),
        "relative_path": rel,
        "exists": exists,
        "file_size": path.stat().st_size if exists and path.is_file() else 0,
        "sha256": _sha256_or_skip(path),
        "row_count": _row_count(path),
        "schema_version": schema,
        "created_by_command": _created_by(schema),
        "source_dependencies": deps,
        "status": status,
    }


def _sha256_or_skip(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    if size > HASH_SIZE_LIMIT:
        return "skipped_large_file"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            return sum(1 for line in f if line.strip())
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = sum(1 for _ in csv.reader(f))
        return max(rows - 1, 0)
    return None


def _created_by(schema: str) -> str:
    mapping = {
        "pose_features_v0": "extract-pose-features-v0",
        "pose_semantics_v0": "classify-poses-v0",
        "partner_relative_features_v0": "extract-partner-relative-features-v0",
        "interaction_semantics_v0": "classify-interactions-v0",
        "semantic_actions_v0": "build-semantic-actions-v0",
        "semantic_actions_v1": "rebuild-clean-v3-semantic-actions-v1",
        "semantic_candidate_db_v0": "build-semantic-candidate-db-v0",
        "semantic_candidate_db_v1": "rebuild-clean-v3-semantic-actions-v1",
        "cowgirl_candidate_db_v5": "build-cowgirl-candidate-db-v5",
        "cowgirl_candidate_db_v6": "rebuild-clean-v3-semantic-actions-v1",
        "motion_primitives_v1": "extract-cowgirl-motion-primitives-v1",
        "review_manifest": "build-vam-review-package/export-semantic-review-v16",
    }
    return mapping.get(schema, "unknown")


def _write_repro_summary(results: dict[str, Any], out: Path) -> None:
    lines = [
        "# clean_v3 Reproducibility Audit Summary",
        "",
        "This audit does not train ML, does not generate new animations, and does not modify manual labels.",
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
        for key in ["out", "out_md", "out_json", "out_jsonl", "report", "reference", "summary"]:
            if result.get(key):
                lines.append(f"- `{name}` {key}: `{result[key]}`")
    lines.extend(["", "## Morning Start", "", "Open `reports/MORNING_CHECKLIST.md` first."])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _project_root(path: Path) -> Path:
    current = path.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return Path.cwd()


def _git(args: list[str], cwd: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stdout": "", "stderr": str(exc)}


def _git_text(args: list[str], cwd: Path) -> str:
    result = _git(args, cwd)
    return result["stdout"].strip() if result["ok"] else ""


def _repo_snapshot_warnings(staged: list[str], status: list[str], run: Path) -> list[str]:
    warnings: list[str] = []
    staged_norm = [s.replace("\\", "/") for s in staged]
    status_norm = [s.replace("\\", "/") for s in status]
    if any("data/runs/" in p for p in staged_norm):
        warnings.append("Generated run data appears staged; do not commit local generated data.")
    if any(p.endswith("manual_labels.yaml") for p in staged_norm) or (run / "labels" / "manual_labels.yaml").exists():
        warnings.append("manual_labels.yaml exists or is staged; do not modify or commit it without explicit promotion.")
    if any(Path(p).suffix.lower() in {".npz", ".npy", ".pkl", ".joblib", ".model", ".onnx"} for p in staged_norm):
        warnings.append("Generated arrays/models appear staged.")
    if any(p.lower().endswith(".json") and ("raw" in p.lower() or "scene" in p.lower()) for p in staged_norm):
        warnings.append("Raw VaM scene-like JSON may be staged.")
    if any("?? data/runs/" in p for p in status_norm):
        warnings.append("Untracked generated run data is visible; keep it ignored/uncommitted.")
    return warnings


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

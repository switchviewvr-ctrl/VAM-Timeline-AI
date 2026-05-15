"""Command line interface for VaM Timeline AI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vam_timeline_ai import __version__
from vam_timeline_ai.io.json_utils import dump_json
from vam_timeline_ai.io.path_utils import default_reference_paths
from vam_timeline_ai.motion.baker import extract_motion_samples
from vam_timeline_ai.motion.source_inventory import build_motion_source_index
from vam_timeline_ai.reports.report_writer import write_raw_scan_report
from vam_timeline_ai.scene.scene_parser import scan_json_file


IGNORED_SCAN_DIR_NAMES = {
    "out_dataset",
    "out_inspect",
    "out_km190",
    "out_voxta",
    "out_audit",
    "out_dataset_batch1",
    "vam_mocap_dataset_compiler",
    "vam-timeline-master",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vam_timeline_ai", description="Semantic VaM Timeline motion analysis tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Print project and reference path status.")

    scan = subparsers.add_parser("scan-raw-folder", help="Lightweight scan of raw VaM JSON files.")
    scan.add_argument("--raw-dir", required=True, help="Folder containing raw VaM scene JSON files.")
    scan.add_argument("--out", required=True, help="Output folder for raw_scan_index.json and raw_scan_report.md.")
    scan.add_argument("--recursive", action="store_true", help="Recursively scan JSON files, skipping known project/output folders.")

    state = subparsers.add_parser("audit-project-state", help="Write an honest capability/status report.")
    state.add_argument("--out", required=True)

    repo_safety = subparsers.add_parser("audit-repo-safety", help="Audit public GitHub repo safety.")
    repo_safety.add_argument("--project-root", required=True)
    repo_safety.add_argument("--out", required=True)

    local_status = subparsers.add_parser("local-status", help="Write local clean-run status for the human operator.")
    local_status.add_argument("--run-dir", required=True)
    local_status.add_argument("--out", required=True)

    reality = subparsers.add_parser("export-reality-audit-100", help="Export 100 reality-audit examples without training or labels.")
    reality.add_argument("--run-dir", required=True)
    reality.add_argument("--out-dir", required=True)
    reality.add_argument("--count", type=int, default=100)

    reality_summary = subparsers.add_parser("summarize-reality-audit", help="Summarize completed reality-audit annotations.")
    reality_summary.add_argument("--annotations", required=True)
    reality_summary.add_argument("--audit-batch", required=True)
    reality_summary.add_argument("--out", required=True)

    semantic_review = subparsers.add_parser("export-semantic-review-010", help="Export a 10-item VaM semantic review batch.")
    semantic_review.add_argument("--run-dir", required=True)
    semantic_review.add_argument("--out-dir", required=True)
    semantic_review.add_argument("--count", type=int, default=10)
    semantic_review.add_argument("--attempt-timeline-export", default="true")
    semantic_review.add_argument("--export-mode", default="motion_only", choices=["motion_only", "motion_plus_static_anchors", "generation_template_candidate"])
    semantic_review.add_argument("--candidate-db", default=None)
    semantic_review.add_argument("--use-cowgirl-candidate-db", default="false")
    semantic_review.add_argument("--use-body-motion-quality", default="false")
    semantic_review.add_argument("--prefer-clean-body-motion", default="false")
    semantic_review.add_argument("--use-handmade-reference-matches", default="false")
    semantic_review.add_argument("--prefer-longer-cowgirl-windows", default="false")
    semantic_review.add_argument("--min-cowgirl-window-seconds", type=float, default=4.0)
    semantic_review.add_argument("--use-cowgirl-candidate-score-v2", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v3", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v4", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v5", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v6", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v7", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v8", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v9", default="false")
    semantic_review.add_argument("--use-rider-receiver-discrimination", default="false")
    semantic_review.add_argument("--use-relative-motion-features", default="false")
    semantic_review.add_argument("--use-trajectory-shape-features", default="false")
    semantic_review.add_argument("--use-relative-reference-matches", default="false")
    semantic_review.add_argument("--use-pose-export-validity", default="false")
    semantic_review.add_argument("--use-controller-validity", default="false")
    semantic_review.add_argument("--use-pose-anchor-completeness", default="false")
    semantic_review.add_argument("--use-controller-orientation-validity", default="false")

    semantic_summary = subparsers.add_parser("summarize-semantic-review-010", help="Summarize user answers for the 10-item semantic review.")
    semantic_summary.add_argument("--answers", required=True)
    semantic_summary.add_argument("--review", required=True)
    semantic_summary.add_argument("--out", required=True)

    clean = subparsers.add_parser("prepare-clean-run", help="Create clean run folders and a run manifest.")
    clean.add_argument("--data-root", required=True)
    clean.add_argument("--run-name", required=True)
    clean.add_argument("--backup-existing", default="true")
    clean.add_argument("--out-manifest", required=True)
    clean.add_argument("--report", required=True)

    sources = subparsers.add_parser("build-motion-source-index", help="Build technical motion source inventory JSONL.")
    sources.add_argument("--raw-dir", required=True)
    sources.add_argument("--out", required=True)
    sources.add_argument("--report", required=True)
    sources.add_argument("--recursive", action="store_true")

    extract = subparsers.add_parser("extract-motion-samples", help="Bake technical motion sources to 60 Hz NPZ samples.")
    extract.add_argument("--source-index", required=True)
    extract.add_argument("--out-dir", required=True)
    extract.add_argument("--index-out", required=True)
    extract.add_argument("--fps", type=float, default=60.0)

    windows = subparsers.add_parser("build-movement-windows", help="Build semantic movement-window records from baked samples.")
    windows.add_argument("--sample-index", required=True)
    windows.add_argument("--out", required=True)

    features = subparsers.add_parser("extract-cowgirl-features-v0", help="Extract numeric Cowgirl/Riding feature rows v0.")
    features.add_argument("--windows", required=True)
    features.add_argument("--sample-index", required=True)
    features.add_argument("--out-jsonl", required=True)
    features.add_argument("--out-npz", required=True)
    features.add_argument("--report", required=True)

    labels = subparsers.add_parser("apply-manual-labels", help="Apply real manual labels to movement-window records.")
    labels.add_argument("--windows", required=True)
    labels.add_argument("--labels", required=True)
    labels.add_argument("--out", required=True)
    labels.add_argument("--report", required=True)

    dataset = subparsers.add_parser("build-ml-dataset-v0", help="Build ML-ready dataset v0 from features and labels.")
    dataset.add_argument("--features", required=True)
    dataset.add_argument("--windows", required=True)
    dataset.add_argument("--out", required=True)
    dataset.add_argument("--report", required=True)

    ml = subparsers.add_parser("analyze-ml-v0", help="Run ML readiness and clustering baseline.")
    ml.add_argument("--dataset", required=True)
    ml.add_argument("--out-dir", required=True)

    audit_baked = subparsers.add_parser("audit-baked-samples", help="Audit baked NPZ samples for real motion and suspicious data.")
    audit_baked.add_argument("--sample-index", required=True)
    audit_baked.add_argument("--out-jsonl", required=True)
    audit_baked.add_argument("--report", required=True)

    body_quality = subparsers.add_parser("audit-body-motion-quality", help="Audit body-controller vs root/whole-person motion quality.")
    body_quality.add_argument("--run-dir", required=True)
    body_quality.add_argument("--sample-index", required=True)
    body_quality.add_argument("--features", required=True)
    body_quality.add_argument("--controller-map", required=True)
    body_quality.add_argument("--out-jsonl", required=True)
    body_quality.add_argument("--report", required=True)

    cowgirl_score = subparsers.add_parser("score-cowgirl-candidates-v2", help="Score clean Cowgirl review candidates using body quality and reference matches.")
    cowgirl_score.add_argument("--run-dir", required=True)
    cowgirl_score.add_argument("--wild-reference-matches", required=True)
    cowgirl_score.add_argument("--body-quality", required=True)
    cowgirl_score.add_argument("--features", required=True)
    cowgirl_score.add_argument("--out-jsonl", required=True)
    cowgirl_score.add_argument("--report", required=True)

    rider_receiver = subparsers.add_parser("score-rider-receiver-v1", help="Score active rider vs receiver/body-response review candidates.")
    rider_receiver.add_argument("--run-dir", required=True)
    rider_receiver.add_argument("--features", required=True)
    rider_receiver.add_argument("--pair-features", required=True)
    rider_receiver.add_argument("--pair-windows", required=True)
    rider_receiver.add_argument("--body-quality", required=True)
    rider_receiver.add_argument("--wild-reference-matches", required=True)
    rider_receiver.add_argument("--out-jsonl", required=True)
    rider_receiver.add_argument("--report", required=True)

    cowgirl_score_v3 = subparsers.add_parser("score-cowgirl-candidates-v3", help="Score clean active-rider Cowgirl candidates with receiver/body-response penalties.")
    cowgirl_score_v3.add_argument("--run-dir", required=True)
    cowgirl_score_v3.add_argument("--wild-reference-matches", required=True)
    cowgirl_score_v3.add_argument("--body-quality", required=True)
    cowgirl_score_v3.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v3.add_argument("--features", required=True)
    cowgirl_score_v3.add_argument("--out-jsonl", required=True)
    cowgirl_score_v3.add_argument("--report", required=True)

    rel_windows = subparsers.add_parser("build-relative-motion-windows", help="Build safe relative/local body-controller window representations.")
    rel_windows.add_argument("--run-dir", required=True)
    rel_windows.add_argument("--sample-index", required=True)
    rel_windows.add_argument("--windows", required=True)
    rel_windows.add_argument("--controller-map", required=True)
    rel_windows.add_argument("--body-quality", required=True)
    rel_windows.add_argument("--out-dir", required=True)
    rel_windows.add_argument("--index-out", required=True)
    rel_windows.add_argument("--report", required=True)

    rel_features = subparsers.add_parser("extract-relative-motion-features", help="Extract features from relative/local motion windows.")
    rel_features.add_argument("--relative-index", required=True)
    rel_features.add_argument("--out-jsonl", required=True)
    rel_features.add_argument("--out-npz", required=True)
    rel_features.add_argument("--report", required=True)

    traj_shapes = subparsers.add_parser("analyze-trajectory-shapes", help="Analyze relative pelvis/hip trajectory shapes.")
    traj_shapes.add_argument("--relative-index", required=True)
    traj_shapes.add_argument("--relative-features", required=True)
    traj_shapes.add_argument("--out-jsonl", required=True)
    traj_shapes.add_argument("--out-npz", required=True)
    traj_shapes.add_argument("--report", required=True)

    cowgirl_score_v4 = subparsers.add_parser("score-cowgirl-candidates-v4", help="Score Cowgirl candidates using relative motion and trajectory shape.")
    cowgirl_score_v4.add_argument("--run-dir", required=True)
    cowgirl_score_v4.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v4.add_argument("--relative-features", required=True)
    cowgirl_score_v4.add_argument("--trajectory-features", required=True)
    cowgirl_score_v4.add_argument("--body-quality", required=True)
    cowgirl_score_v4.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v4.add_argument("--features", required=True)
    cowgirl_score_v4.add_argument("--out-jsonl", required=True)
    cowgirl_score_v4.add_argument("--report", required=True)

    pose_export = subparsers.add_parser("audit-pose-export-validity", help="Audit semantic review pose/export validity separately from semantic correctness.")
    pose_export.add_argument("--run-dir", required=True)
    pose_export.add_argument("--review-dir", required=True)
    pose_export.add_argument("--sample-index", required=True)
    pose_export.add_argument("--relative-index", required=True)
    pose_export.add_argument("--body-quality", required=True)
    pose_export.add_argument("--controller-validity", default=None)
    pose_export.add_argument("--pose-anchor-completeness", default=None)
    pose_export.add_argument("--controller-orientation-validity", default=None)
    pose_export.add_argument("--out-jsonl", required=True)
    pose_export.add_argument("--report", required=True)

    pose_anchor = subparsers.add_parser("audit-pose-anchor-completeness", help="Audit whether pose-critical static anchors such as feet/knees are present.")
    pose_anchor.add_argument("--run-dir", required=True)
    pose_anchor.add_argument("--relative-index", required=True)
    pose_anchor.add_argument("--sample-index", required=True)
    pose_anchor.add_argument("--controller-map", required=True)
    pose_anchor.add_argument("--body-quality", required=True)
    pose_anchor.add_argument("--out-jsonl", required=True)
    pose_anchor.add_argument("--report", required=True)

    controller_validity = subparsers.add_parser("audit-controller-validity", help="Audit anatomical/controller plausibility for relative motion windows.")
    controller_validity.add_argument("--run-dir", required=True)
    controller_validity.add_argument("--relative-index", required=True)
    controller_validity.add_argument("--sample-index", required=True)
    controller_validity.add_argument("--controller-map", required=True)
    controller_validity.add_argument("--pose-anchor-completeness", default=None)
    controller_validity.add_argument("--controller-orientation-validity", default=None)
    controller_validity.add_argument("--out-jsonl", required=True)
    controller_validity.add_argument("--report", required=True)

    orientation_validity = subparsers.add_parser("audit-controller-orientation-validity", help="Audit controller rotation/orientation twist validity for generation safety.")
    orientation_validity.add_argument("--run-dir", required=True)
    orientation_validity.add_argument("--relative-index", required=True)
    orientation_validity.add_argument("--sample-index", required=True)
    orientation_validity.add_argument("--controller-map", required=True)
    orientation_validity.add_argument("--pose-anchor-completeness", default=None)
    orientation_validity.add_argument("--out-jsonl", required=True)
    orientation_validity.add_argument("--report", required=True)

    distance_validity = subparsers.add_parser("audit-controller-distance-validity", help="Audit controller distance plausibility for generation safety.")
    distance_validity.add_argument("--run-dir", required=True)
    distance_validity.add_argument("--relative-index", required=True)
    distance_validity.add_argument("--sample-index", required=True)
    distance_validity.add_argument("--controller-map", required=True)
    distance_validity.add_argument("--pose-anchor-completeness", default=None)
    distance_validity.add_argument("--out-jsonl", required=True)
    distance_validity.add_argument("--report", required=True)

    core_controllers = subparsers.add_parser("audit-cowgirl-core-controllers", help="Audit core hip/pelvis and lower-body controller requirements for Cowgirl generation safety.")
    core_controllers.add_argument("--run-dir", required=True)
    core_controllers.add_argument("--relative-index", required=True)
    core_controllers.add_argument("--controller-map", required=True)
    core_controllers.add_argument("--body-quality", required=True)
    core_controllers.add_argument("--pose-anchor-completeness", required=True)
    core_controllers.add_argument("--out-jsonl", required=True)
    core_controllers.add_argument("--report", required=True)

    bj_oral_guard = subparsers.add_parser("audit-bj-oral-trap-guard", help="Audit head/BJ/oral-domain trap candidates that should not be generation-safe Cowgirl.")
    bj_oral_guard.add_argument("--run-dir", required=True)
    bj_oral_guard.add_argument("--relative-features", required=True)
    bj_oral_guard.add_argument("--trajectory-features", required=True)
    bj_oral_guard.add_argument("--relative-reference-matches", required=True)
    bj_oral_guard.add_argument("--cowgirl-core-controllers", required=True)
    bj_oral_guard.add_argument("--out-jsonl", required=True)
    bj_oral_guard.add_argument("--report", required=True)

    cowgirl_score_v5 = subparsers.add_parser("score-cowgirl-candidates-v5", help="Score semantic Cowgirl separately from generation/export usability.")
    cowgirl_score_v5.add_argument("--run-dir", required=True)
    cowgirl_score_v5.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v5.add_argument("--relative-features", required=True)
    cowgirl_score_v5.add_argument("--trajectory-features", required=True)
    cowgirl_score_v5.add_argument("--body-quality", required=True)
    cowgirl_score_v5.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v5.add_argument("--pose-export-validity", required=True)
    cowgirl_score_v5.add_argument("--features", required=True)
    cowgirl_score_v5.add_argument("--out-jsonl", required=True)
    cowgirl_score_v5.add_argument("--report", required=True)

    cowgirl_score_v6 = subparsers.add_parser("score-cowgirl-candidates-v6", help="Score Cowgirl semantics/context/clean motion separately from controller generation safety.")
    cowgirl_score_v6.add_argument("--run-dir", required=True)
    cowgirl_score_v6.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v6.add_argument("--relative-features", required=True)
    cowgirl_score_v6.add_argument("--trajectory-features", required=True)
    cowgirl_score_v6.add_argument("--body-quality", required=True)
    cowgirl_score_v6.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v6.add_argument("--pose-export-validity", required=True)
    cowgirl_score_v6.add_argument("--controller-validity", required=True)
    cowgirl_score_v6.add_argument("--features", required=True)
    cowgirl_score_v6.add_argument("--out-jsonl", required=True)
    cowgirl_score_v6.add_argument("--report", required=True)

    cowgirl_score_v7 = subparsers.add_parser("score-cowgirl-candidates-v7", help="Score Cowgirl candidates with pose-anchor completeness and controller validity.")
    cowgirl_score_v7.add_argument("--run-dir", required=True)
    cowgirl_score_v7.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v7.add_argument("--relative-features", required=True)
    cowgirl_score_v7.add_argument("--trajectory-features", required=True)
    cowgirl_score_v7.add_argument("--body-quality", required=True)
    cowgirl_score_v7.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v7.add_argument("--pose-export-validity", required=True)
    cowgirl_score_v7.add_argument("--controller-validity", required=True)
    cowgirl_score_v7.add_argument("--pose-anchor-completeness", required=True)
    cowgirl_score_v7.add_argument("--features", required=True)
    cowgirl_score_v7.add_argument("--out-jsonl", required=True)
    cowgirl_score_v7.add_argument("--report", required=True)

    cowgirl_score_v8 = subparsers.add_parser("score-cowgirl-candidates-v8", help="Score Cowgirl candidates with anchor, controller, and orientation validity.")
    cowgirl_score_v8.add_argument("--run-dir", required=True)
    cowgirl_score_v8.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v8.add_argument("--relative-features", required=True)
    cowgirl_score_v8.add_argument("--trajectory-features", required=True)
    cowgirl_score_v8.add_argument("--body-quality", required=True)
    cowgirl_score_v8.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v8.add_argument("--pose-export-validity", required=True)
    cowgirl_score_v8.add_argument("--controller-validity", required=True)
    cowgirl_score_v8.add_argument("--pose-anchor-completeness", required=True)
    cowgirl_score_v8.add_argument("--controller-orientation-validity", required=True)
    cowgirl_score_v8.add_argument("--features", required=True)
    cowgirl_score_v8.add_argument("--out-jsonl", required=True)
    cowgirl_score_v8.add_argument("--report", required=True)

    cowgirl_score_v9 = subparsers.add_parser("score-cowgirl-candidates-v9", help="Score Cowgirl candidates with distance/orientation/anchor generation safety.")
    cowgirl_score_v9.add_argument("--run-dir", required=True)
    cowgirl_score_v9.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v9.add_argument("--relative-features", required=True)
    cowgirl_score_v9.add_argument("--trajectory-features", required=True)
    cowgirl_score_v9.add_argument("--body-quality", required=True)
    cowgirl_score_v9.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v9.add_argument("--pose-export-validity", required=True)
    cowgirl_score_v9.add_argument("--controller-validity", required=True)
    cowgirl_score_v9.add_argument("--pose-anchor-completeness", required=True)
    cowgirl_score_v9.add_argument("--controller-orientation-validity", required=True)
    cowgirl_score_v9.add_argument("--controller-distance-validity", required=True)
    cowgirl_score_v9.add_argument("--features", required=True)
    cowgirl_score_v9.add_argument("--out-jsonl", required=True)
    cowgirl_score_v9.add_argument("--report", required=True)

    cowgirl_score_v10 = subparsers.add_parser("score-cowgirl-candidates-v10", help="Score Cowgirl candidates with core-controller and BJ/oral trap generation-safety gates.")
    cowgirl_score_v10.add_argument("--run-dir", required=True)
    cowgirl_score_v10.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v10.add_argument("--relative-features", required=True)
    cowgirl_score_v10.add_argument("--trajectory-features", required=True)
    cowgirl_score_v10.add_argument("--body-quality", required=True)
    cowgirl_score_v10.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v10.add_argument("--pose-export-validity", required=True)
    cowgirl_score_v10.add_argument("--controller-validity", required=True)
    cowgirl_score_v10.add_argument("--pose-anchor-completeness", required=True)
    cowgirl_score_v10.add_argument("--controller-orientation-validity", required=True)
    cowgirl_score_v10.add_argument("--controller-distance-validity", required=True)
    cowgirl_score_v10.add_argument("--cowgirl-core-controllers", required=True)
    cowgirl_score_v10.add_argument("--bj-oral-trap-guard", required=True)
    cowgirl_score_v10.add_argument("--features", required=True)
    cowgirl_score_v10.add_argument("--out-jsonl", required=True)
    cowgirl_score_v10.add_argument("--report", required=True)

    candidate_db = subparsers.add_parser("build-cowgirl-candidate-db-v1", help="Build curated Cowgirl candidate inventory for review, not training.")
    candidate_db.add_argument("--run-dir", required=True)
    candidate_db.add_argument("--candidate-scores", required=True)
    candidate_db.add_argument("--relative-features", required=True)
    candidate_db.add_argument("--trajectory-features", required=True)
    candidate_db.add_argument("--body-quality", required=True)
    candidate_db.add_argument("--pose-anchor-completeness", required=True)
    candidate_db.add_argument("--controller-validity", required=True)
    candidate_db.add_argument("--controller-orientation-validity", required=True)
    candidate_db.add_argument("--controller-distance-validity", required=True)
    candidate_db.add_argument("--out-jsonl", required=True)
    candidate_db.add_argument("--out-csv", required=True)
    candidate_db.add_argument("--report", required=True)

    candidate_db_v2 = subparsers.add_parser("build-cowgirl-candidate-db-v2", help="Build curated Cowgirl candidate inventory v2 with core/trap generation-safety gates.")
    candidate_db_v2.add_argument("--run-dir", required=True)
    candidate_db_v2.add_argument("--candidate-scores", required=True)
    candidate_db_v2.add_argument("--relative-features", required=True)
    candidate_db_v2.add_argument("--trajectory-features", required=True)
    candidate_db_v2.add_argument("--body-quality", required=True)
    candidate_db_v2.add_argument("--pose-anchor-completeness", required=True)
    candidate_db_v2.add_argument("--controller-validity", required=True)
    candidate_db_v2.add_argument("--controller-orientation-validity", required=True)
    candidate_db_v2.add_argument("--controller-distance-validity", required=True)
    candidate_db_v2.add_argument("--cowgirl-core-controllers", required=True)
    candidate_db_v2.add_argument("--bj-oral-trap-guard", required=True)
    candidate_db_v2.add_argument("--out-jsonl", required=True)
    candidate_db_v2.add_argument("--out-csv", required=True)
    candidate_db_v2.add_argument("--report", required=True)

    cmap = subparsers.add_parser("discover-controller-map", help="Discover controller names and conservative body-part mapping.")
    cmap.add_argument("--sample-index", required=True)
    cmap.add_argument("--out", required=True)
    cmap.add_argument("--map-out", required=True)
    cmap.add_argument("--report", required=True)

    fv1 = subparsers.add_parser("extract-cowgirl-features-v1", help="Extract richer Cowgirl/Riding feature proxies v1.")
    fv1.add_argument("--windows", required=True)
    fv1.add_argument("--sample-index", required=True)
    fv1.add_argument("--controller-map", required=True)
    fv1.add_argument("--out-jsonl", required=True)
    fv1.add_argument("--out-npz", required=True)
    fv1.add_argument("--report", required=True)

    handmade_import = subparsers.add_parser("import-handmade-reference-animations", help="Import handmade labeled reference Timeline animations.")
    handmade_import.add_argument("--zip", required=True)
    handmade_import.add_argument("--out-dir", required=True)

    handmade_features = subparsers.add_parser("extract-handmade-reference-features", help="Extract features from handmade reference animations.")
    handmade_features.add_argument("--manifest", required=True)
    handmade_features.add_argument("--sample-index", required=True)
    handmade_features.add_argument("--out-jsonl", required=True)
    handmade_features.add_argument("--out-npz", required=True)
    handmade_features.add_argument("--report", required=True)

    handmade_relative = subparsers.add_parser("build-handmade-relative-reference-features", help="Build relative and trajectory features for handmade references.")
    handmade_relative.add_argument("--handmade-sample-index", required=True)
    handmade_relative.add_argument("--controller-map", required=True)
    handmade_relative.add_argument("--out-jsonl", required=True)
    handmade_relative.add_argument("--out-npz", required=True)
    handmade_relative.add_argument("--report", required=True)

    handmade_sigs = subparsers.add_parser("build-handmade-reference-signatures", help="Build handmade reference family signatures.")
    handmade_sigs.add_argument("--features", required=True)
    handmade_sigs.add_argument("--out-json", required=True)
    handmade_sigs.add_argument("--report", required=True)

    handmade_match = subparsers.add_parser("compare-wild-to-handmade-references", help="Compare wild windows to handmade reference signatures.")
    handmade_match.add_argument("--wild-features", required=True)
    handmade_match.add_argument("--wild-body-quality", required=True)
    handmade_match.add_argument("--handmade-features", required=True)
    handmade_match.add_argument("--signatures", required=True)
    handmade_match.add_argument("--out-jsonl", required=True)
    handmade_match.add_argument("--report", required=True)

    relative_match = subparsers.add_parser("compare-relative-wild-to-handmade", help="Compare wild and handmade references in relative + trajectory feature space.")
    relative_match.add_argument("--wild-relative-features", required=True)
    relative_match.add_argument("--wild-trajectory-features", required=True)
    relative_match.add_argument("--handmade-relative-features", required=True)
    relative_match.add_argument("--handmade-trajectory-features", required=True)
    relative_match.add_argument("--out-jsonl", required=True)
    relative_match.add_argument("--report", required=True)

    pairs = subparsers.add_parser("build-context-pair-candidates", help="Build possible actor/context pair candidates without roles.")
    pairs.add_argument("--sample-index", required=True)
    pairs.add_argument("--out", required=True)
    pairs.add_argument("--report", required=True)

    weak = subparsers.add_parser("generate-weak-labels-v1", help="Generate weak_ review labels from numeric proxies.")
    weak.add_argument("--features", required=True)
    weak.add_argument("--out", required=True)
    weak.add_argument("--report", required=True)

    review = subparsers.add_parser("build-review-queue-v1", help="Build diverse manual labeling review queue.")
    review.add_argument("--features", required=True)
    review.add_argument("--weak-labels", required=True)
    review.add_argument("--clusters", required=True)
    review.add_argument("--windows", required=True)
    review.add_argument("--out", required=True)
    review.add_argument("--markdown", required=True)

    dsv1 = subparsers.add_parser("build-ml-dataset-v1", help="Build ML dataset v1 with manual/weak labels separated.")
    dsv1.add_argument("--features", required=True)
    dsv1.add_argument("--windows", required=True)
    dsv1.add_argument("--weak-labels", required=True)
    dsv1.add_argument("--out", required=True)
    dsv1.add_argument("--report", required=True)

    arv1 = subparsers.add_parser("analyze-ml-v1", help="Write leakage-aware ML readiness reports.")
    arv1.add_argument("--dataset", required=True)
    arv1.add_argument("--out-dir", required=True)

    clv1 = subparsers.add_parser("cluster-ml-v1", help="Cluster v1 features if sklearn is available.")
    clv1.add_argument("--dataset", required=True)
    clv1.add_argument("--out-dir", required=True)

    integrity = subparsers.add_parser("audit-data-integrity", help="Cross-check source/sample/window/feature/dataset counts.")
    integrity.add_argument("--source-index", required=True)
    integrity.add_argument("--sample-index", required=True)
    integrity.add_argument("--windows", required=True)
    integrity.add_argument("--features", required=True)
    integrity.add_argument("--dataset", required=True)
    integrity.add_argument("--out", required=True)
    integrity.add_argument("--pair-windows")
    integrity.add_argument("--pair-features")
    integrity.add_argument("--review-batch")
    integrity.add_argument("--strict", default="false")

    weak2 = subparsers.add_parser("calibrate-weak-labels-v2", help="Calibrate broad weak labels into weak_v2 review hints.")
    weak2.add_argument("--features", required=True)
    weak2.add_argument("--weak-labels", required=True)
    weak2.add_argument("--out", required=True)
    weak2.add_argument("--report", required=True)

    pair_windows = subparsers.add_parser("build-pair-windows-v1", help="Build aligned pair windows without role assignment.")
    pair_windows.add_argument("--pair-candidates", required=True)
    pair_windows.add_argument("--windows", required=True)
    pair_windows.add_argument("--sample-index", required=True)
    pair_windows.add_argument("--out", required=True)
    pair_windows.add_argument("--report", required=True)

    pair_features = subparsers.add_parser("extract-pair-features-v0", help="Extract pair/context feature proxies.")
    pair_features.add_argument("--pair-windows", required=True)
    pair_features.add_argument("--sample-index", required=True)
    pair_features.add_argument("--controller-map", required=True)
    pair_features.add_argument("--out-jsonl", required=True)
    pair_features.add_argument("--out-npz", required=True)
    pair_features.add_argument("--report", required=True)

    schema = subparsers.add_parser("write-manual-label-schema-v2", help="Write manual label schema/template/guide files.")
    schema.add_argument("--out", required=True)
    schema.add_argument("--template", required=True)
    schema.add_argument("--guide", required=True)

    validate_labels = subparsers.add_parser("validate-manual-labels-v2", help="Validate real manual labels.")
    validate_labels.add_argument("--labels", required=True)
    validate_labels.add_argument("--schema", required=True)
    validate_labels.add_argument("--windows", required=True)
    validate_labels.add_argument("--pair-windows", required=True)
    validate_labels.add_argument("--out", required=True)

    batch = subparsers.add_parser("build-review-batch-v2", help="Build balanced manual review batch.")
    batch.add_argument("--windows", required=True)
    batch.add_argument("--features", required=True)
    batch.add_argument("--weak-labels", required=True)
    batch.add_argument("--pair-windows", required=True)
    batch.add_argument("--pair-features", required=True)
    batch.add_argument("--clusters", required=True)
    batch.add_argument("--out-dir", required=True)
    batch.add_argument("--batch-size", type=int, default=120)
    batch.add_argument("--max-per-scene", type=int, default=15)
    batch.add_argument("--max-per-sample", type=int, default=3)
    batch.add_argument("--prefer-pair-context", default="true")

    previews = subparsers.add_parser("render-review-previews-v1", help="Render static review preview package.")
    previews.add_argument("--review-batch", required=True)
    previews.add_argument("--sample-index", required=True)
    previews.add_argument("--controller-map", required=True)
    previews.add_argument("--out-dir", required=True)

    merge = subparsers.add_parser("merge-manual-label-batch", help="Merge edited batch YAML into manual_labels.yaml.")
    merge.add_argument("--base", required=True)
    merge.add_argument("--batch", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--backup", default="true")
    merge.add_argument("--report", required=True)

    inspect_batch = subparsers.add_parser("inspect-edited-label-batch", help="Inspect an edited label batch before merging.")
    inspect_batch.add_argument("--stub", required=True)
    inspect_batch.add_argument("--edited", required=True)
    inspect_batch.add_argument("--windows", required=True)
    inspect_batch.add_argument("--pair-windows", required=True)
    inspect_batch.add_argument("--out", required=True)

    label_summary = subparsers.add_parser("summarize-manual-labels", help="Summarize manual label coverage.")
    label_summary.add_argument("--labels", required=True)
    label_summary.add_argument("--windows", required=True)
    label_summary.add_argument("--pair-windows", required=True)
    label_summary.add_argument("--out", required=True)

    split_plan = subparsers.add_parser("plan-ml-splits-v1", help="Plan leakage-safe future ML splits.")
    split_plan.add_argument("--dataset", required=True)
    split_plan.add_argument("--labels", required=True)
    split_plan.add_argument("--out", required=True)
    split_plan.add_argument("--report", required=True)

    dsv2 = subparsers.add_parser("build-ml-dataset-v2", help="Build ML dataset v2 with positive/negative/uncertain labels separated.")
    dsv2.add_argument("--features", required=True)
    dsv2.add_argument("--windows", required=True)
    dsv2.add_argument("--weak-labels", required=True)
    dsv2.add_argument("--manual-labels", required=True)
    dsv2.add_argument("--out", required=True)
    dsv2.add_argument("--report", required=True)

    sup_ready = subparsers.add_parser("analyze-supervised-readiness", help="Check whether real manual labels are sufficient for supervised ML.")
    sup_ready.add_argument("--dataset", required=True)
    sup_ready.add_argument("--labels", required=True)
    sup_ready.add_argument("--split-plan", required=True)
    sup_ready.add_argument("--out", required=True)

    baseline = subparsers.add_parser("train-supervised-baseline-v0", help="Train a guarded supervised baseline only if readiness allows.")
    baseline.add_argument("--dataset", required=True)
    baseline.add_argument("--split-plan", required=True)
    baseline.add_argument("--out-dir", required=True)
    baseline.add_argument("--report", required=True)

    active_batch = subparsers.add_parser("build-active-review-batch-v3", help="Build the next active labeling review batch from label gaps.")
    active_batch.add_argument("--windows", required=True)
    active_batch.add_argument("--features", required=True)
    active_batch.add_argument("--weak-labels", required=True)
    active_batch.add_argument("--pair-windows", required=True)
    active_batch.add_argument("--pair-features", required=True)
    active_batch.add_argument("--manual-labels", required=True)
    active_batch.add_argument("--supervised-readiness", required=True)
    active_batch.add_argument("--out-dir", required=True)
    active_batch.add_argument("--batch-size", type=int, default=120)
    active_batch.add_argument("--max-per-scene", type=int, default=15)
    active_batch.add_argument("--max-per-sample", type=int, default=3)
    active_batch.add_argument("--prefer-coverage-gaps", default="true")

    latest_batch = subparsers.add_parser("find-latest-review-batch", help="Discover latest valid review batch from a clean run.")
    latest_batch.add_argument("--run-dir", required=True)
    latest_batch.add_argument("--out", required=True)

    next_step = subparsers.add_parser("write-labeling-next-step", help="Write exact human next-step instructions for labeling.")
    next_step.add_argument("--run-dir", required=True)
    next_step.add_argument("--out", required=True)

    ingest = subparsers.add_parser("ingest-latest-edited-batch", help="Safely ingest the latest edited review batch if it exists.")
    ingest.add_argument("--run-dir", required=True)
    ingest.add_argument("--schema", required=True)
    ingest.add_argument("--stop-if-missing", default="true")

    machine = subparsers.add_parser("generate-machine-label-proposals-v1", help="Generate machine semantic label proposals from numeric proxies.")
    machine.add_argument("--run-dir", required=True)
    machine.add_argument("--features", required=True)
    machine.add_argument("--pair-features", required=True)
    machine.add_argument("--weak-labels", required=True)
    machine.add_argument("--windows", required=True)
    machine.add_argument("--pair-windows", required=True)
    machine.add_argument("--out-jsonl", required=True)
    machine.add_argument("--out-yaml", required=True)
    machine.add_argument("--report", required=True)

    silver = subparsers.add_parser("build-silver-labels-v1", help="Build high-confidence silver labels from machine proposals.")
    silver.add_argument("--proposals", required=True)
    silver.add_argument("--out-jsonl", required=True)
    silver.add_argument("--out-yaml", required=True)
    silver.add_argument("--report", required=True)
    silver.add_argument("--min-confidence", type=float, default=0.75)

    compare = subparsers.add_parser("compare-machine-labels-to-manual", help="Compare silver labels to real manual labels if any exist.")
    compare.add_argument("--manual-labels", required=True)
    compare.add_argument("--silver-labels", required=True)
    compare.add_argument("--out", required=True)

    dsv3 = subparsers.add_parser("build-ml-dataset-v3", help="Build ML dataset v3 with manual, weak, and silver labels separated.")
    dsv3.add_argument("--features", required=True)
    dsv3.add_argument("--windows", required=True)
    dsv3.add_argument("--weak-labels", required=True)
    dsv3.add_argument("--manual-labels", required=True)
    dsv3.add_argument("--silver-labels", required=True)
    dsv3.add_argument("--out", required=True)
    dsv3.add_argument("--report", required=True)

    silver_ready = subparsers.add_parser("analyze-silver-readiness", help="Analyze readiness for silver-supervised proxy baselines.")
    silver_ready.add_argument("--dataset", required=True)
    silver_ready.add_argument("--silver-labels", required=True)
    silver_ready.add_argument("--out", required=True)

    silver_baseline = subparsers.add_parser("train-silver-baseline-v0", help="Train weak-supervised silver proxy baseline if possible.")
    silver_baseline.add_argument("--dataset", required=True)
    silver_baseline.add_argument("--out-dir", required=True)
    silver_baseline.add_argument("--report", required=True)

    machine_batch = subparsers.add_parser("build-machine-proposal-review-batch", help="Build review batch focused on checking machine proposals.")
    machine_batch.add_argument("--run-dir", required=True)
    machine_batch.add_argument("--proposals", required=True)
    machine_batch.add_argument("--silver-labels", required=True)
    machine_batch.add_argument("--out-dir", required=True)
    machine_batch.add_argument("--batch-size", type=int, default=120)
    machine_batch.add_argument("--max-per-scene", type=int, default=15)
    machine_batch.add_argument("--max-per-sample", type=int, default=3)

    run_machine = subparsers.add_parser("run-machine-labeling-v1", help="Run machine proposal, silver label, dataset v3, and review batch workflow.")
    run_machine.add_argument("--run-dir", required=True)
    run_machine.add_argument("--min-silver-confidence", type=float, default=0.75)
    run_machine.add_argument("--train-silver-baseline", default="true")

    machine_audit = subparsers.add_parser("audit-machine-labels-v1", help="Audit raw machine proposals and silver v1 duplication/conflicts.")
    machine_audit.add_argument("--run-dir", required=True)
    machine_audit.add_argument("--proposals", required=True)
    machine_audit.add_argument("--silver-labels", required=True)
    machine_audit.add_argument("--windows", required=True)
    machine_audit.add_argument("--pair-windows", required=True)
    machine_audit.add_argument("--out", required=True)
    machine_audit.add_argument("--out-json", required=True)

    aggregate = subparsers.add_parser("aggregate-machine-labels-v2", help="Aggregate raw machine proposals into deduplicated score rows.")
    aggregate.add_argument("--proposals", required=True)
    aggregate.add_argument("--out-window-jsonl", required=True)
    aggregate.add_argument("--out-pair-jsonl", required=True)
    aggregate.add_argument("--report", required=True)

    silver2 = subparsers.add_parser("build-silver-labels-v2", help="Build silver labels v2 from aggregated machine scores.")
    silver2.add_argument("--window-scores", required=True)
    silver2.add_argument("--pair-scores", required=True)
    silver2.add_argument("--out-window-jsonl", required=True)
    silver2.add_argument("--out-pair-jsonl", required=True)
    silver2.add_argument("--out-yaml", required=True)
    silver2.add_argument("--report", required=True)
    silver2.add_argument("--min-score", type=float, default=0.78)

    dsv4 = subparsers.add_parser("build-ml-dataset-v4", help="Build ML dataset v4 from silver v2 labels.")
    dsv4.add_argument("--features", required=True)
    dsv4.add_argument("--windows", required=True)
    dsv4.add_argument("--weak-labels", required=True)
    dsv4.add_argument("--manual-labels", required=True)
    dsv4.add_argument("--silver-window-labels", required=True)
    dsv4.add_argument("--silver-pair-labels", required=True)
    dsv4.add_argument("--out", required=True)
    dsv4.add_argument("--report", required=True)

    silver_ready2 = subparsers.add_parser("analyze-silver-readiness-v2", help="Analyze silver v2 readiness and balance.")
    silver_ready2.add_argument("--dataset", required=True)
    silver_ready2.add_argument("--silver-window-labels", required=True)
    silver_ready2.add_argument("--silver-pair-labels", required=True)
    silver_ready2.add_argument("--out", required=True)

    silver_baseline1 = subparsers.add_parser("train-silver-baseline-v1", help="Train balanced silver v2 proxy baseline with sklearn or NumPy fallback.")
    silver_baseline1.add_argument("--dataset", required=True)
    silver_baseline1.add_argument("--readiness", required=True)
    silver_baseline1.add_argument("--out-dir", required=True)
    silver_baseline1.add_argument("--report", required=True)
    silver_baseline1.add_argument("--allow-numpy-fallback", default="true")

    machine_batch2 = subparsers.add_parser("build-machine-proposal-review-batch-v2", help="Build review batch from aggregated machine/silver v2 scores.")
    machine_batch2.add_argument("--run-dir", required=True)
    machine_batch2.add_argument("--window-scores", required=True)
    machine_batch2.add_argument("--pair-scores", required=True)
    machine_batch2.add_argument("--silver-window-labels", required=True)
    machine_batch2.add_argument("--silver-pair-labels", required=True)
    machine_batch2.add_argument("--out-dir", required=True)
    machine_batch2.add_argument("--batch-size", type=int, default=120)
    machine_batch2.add_argument("--max-per-scene", type=int, default=15)
    machine_batch2.add_argument("--max-per-sample", type=int, default=3)

    run_machine2 = subparsers.add_parser("run-machine-labeling-v2", help="Run audit, aggregation, silver v2, dataset v4, baseline, and review batch.")
    run_machine2.add_argument("--run-dir", required=True)
    run_machine2.add_argument("--min-silver-score", type=float, default=0.78)
    run_machine2.add_argument("--train-silver-baseline", default="true")
    run_machine2.add_argument("--allow-numpy-fallback", default="true")

    return parser


def cmd_info(_args: argparse.Namespace) -> int:
    refs = default_reference_paths()
    print(f"VaM Timeline AI {__version__}")
    print(f"Project root: {refs.project_root}")
    print("Configured reference paths:")
    for name, status in refs.as_status_dict().items():
        print(f"- {name}: {status['path']} (exists={status['exists']})")
    return 0


def cmd_scan_raw_folder(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(_iter_json_files(raw_dir, recursive=args.recursive))
    scanned = [scan_json_file(path) for path in json_files]
    scanned.sort(key=lambda item: item.get("file_name", "").lower())

    index: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "vam_timeline_ai raw scan",
        "version": __version__,
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "recursive": bool(args.recursive),
        "files": scanned,
        "totals": _scan_totals(scanned),
        "notes": [
            "This scan is technical and lightweight.",
            "It does not bake motion, infer actor roles, export Timeline clips, run VaM, or train ML.",
            "Filename hints and atom IDs are not semantic truth.",
        ],
    }

    index_path = out_dir / "raw_scan_index.json"
    report_path = out_dir / "raw_scan_report.md"
    dump_json(index_path, index)
    write_raw_scan_report(index, report_path)

    totals = index["totals"]
    print(f"Raw scan written: {index_path}")
    print(f"Raw scan report: {report_path}")
    print(
        "Scanned {total} JSON files: {scenes} VaM scenes, {external} external Timeline exports, {failures} parse failures.".format(
            total=totals["total_json_files"],
            scenes=totals["vam_scenes"],
            external=totals["external_timeline_exports"],
            failures=totals["parse_failures"],
        )
    )
    return 0


def cmd_audit_repo_safety(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.repo_safety import audit_repo_safety

    result = audit_repo_safety(args.project_root, args.out)
    print(f"Repository safety report written: {args.out}")
    print(f"Status: {result['status']}; errors={len(result['errors'])}; warnings={len(result['warnings'])}")
    return 1 if result["status"] == "ERROR" else 0


def cmd_local_status(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.local_status import write_local_status

    result = write_local_status(args.run_dir, args.out)
    print(f"Local status report written: {args.out}")
    latest = result.get("latest_review_batch") or {}
    print(f"Latest batch: {latest.get('batch_name')}; edited={latest.get('has_edited')}; manual labels={result['manual_label_count']}")
    return 0


def cmd_audit_project_state(args: argparse.Namespace) -> int:
    out = Path(args.out)
    root = Path.cwd()
    capabilities = {
        "raw scan": (root / "data" / "audits" / "raw_scan" / "raw_scan_index.json").exists(),
        "Timeline source inventory": (root / "src" / "vam_timeline_ai" / "motion" / "source_inventory.py").exists(),
        "native MotionAnimationMaster source inventory": (root / "src" / "vam_timeline_ai" / "motion" / "source_inventory.py").exists(),
        "Timeline decoder": (root / "src" / "vam_timeline_ai" / "timeline" / "codec.py").exists(),
        "native motion decoder": (root / "src" / "vam_timeline_ai" / "motion" / "native_motion.py").exists(),
        "60 Hz baker": (root / "src" / "vam_timeline_ai" / "motion" / "baker.py").exists(),
        "movement window generator": (root / "src" / "vam_timeline_ai" / "motion" / "windows.py").exists(),
        "cowgirl feature extractor": (root / "src" / "vam_timeline_ai" / "cowgirl" / "feature_extractor.py").exists(),
        "manual label loader": (root / "src" / "vam_timeline_ai" / "semantics" / "manual_labels.py").exists(),
        "ML dataset builder": (root / "src" / "vam_timeline_ai" / "ml" / "dataset.py").exists(),
        "clustering baseline": (root / "src" / "vam_timeline_ai" / "ml" / "clustering.py").exists(),
        "trained model files": any((root / "data" / "ml" / "models").glob("*")) if (root / "data" / "ml" / "models").exists() else False,
    }
    lines = ["# Project State Report", "", "Honest capability status. Missing means not implemented or no artifact exists yet.", ""]
    for name, exists in capabilities.items():
        lines.append(f"- {name}: {'present' if exists else 'missing'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Project state report written: {out}")
    return 0


def cmd_prepare_clean_run(args: argparse.Namespace) -> int:
    from vam_timeline_ai.io.artifacts import prepare_clean_run

    manifest = prepare_clean_run(args.data_root, args.run_name, backup_existing=_arg_bool(args.backup_existing), out_manifest=args.out_manifest, report=args.report)
    print(f"Clean run manifest written: {args.out_manifest}")
    print(f"Run root: {manifest['run_root']}")
    print(f"Warnings: {len(manifest.get('warnings', []))}")
    return 0


def cmd_build_motion_source_index(args: argparse.Namespace) -> int:
    rows = build_motion_source_index(args.raw_dir, args.out, args.report, recursive=bool(args.recursive))
    print(f"Motion source records written: {args.out}")
    print(f"Motion source report written: {args.report}")
    print(f"Motion sources found: {len(rows)}")
    return 0


def cmd_extract_motion_samples(args: argparse.Namespace) -> int:
    rows = extract_motion_samples(args.source_index, args.out_dir, args.index_out, fps=args.fps)
    ok = sum(1 for row in rows if row.get("bake_status") == "ok")
    failed = sum(1 for row in rows if row.get("bake_status") == "failed")
    not_bakeable = sum(1 for row in rows if row.get("bake_status") == "not_bakeable")
    print(f"Motion sample index written: {args.index_out}")
    print(f"Baked ok: {ok}; failed: {failed}; not bakeable: {not_bakeable}")
    return 0


def cmd_build_movement_windows(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.window_dataset import build_movement_windows

    rows = build_movement_windows(args.sample_index, args.out)
    usable = sum(1 for row in rows if row.get("include_for_ml"))
    print(f"Movement windows written: {args.out}")
    print(f"Windows: {len(rows)}; include_for_ml: {usable}")
    return 0


def cmd_extract_cowgirl_features(args: argparse.Namespace) -> int:
    from vam_timeline_ai.cowgirl.feature_extractor import extract_cowgirl_features_v0

    rows = extract_cowgirl_features_v0(args.windows, args.sample_index, args.out_jsonl, args.out_npz, args.report)
    numeric = sum(1 for row in rows if row.get("feature_quality", {}).get("has_numeric_features"))
    print(f"Cowgirl feature rows written: {args.out_jsonl}")
    print(f"Cowgirl feature matrix written: {args.out_npz}")
    print(f"Feature rows: {len(rows)}; numeric rows: {numeric}")
    return 0


def cmd_apply_manual_labels(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.manual_labels import apply_manual_labels

    rows = apply_manual_labels(args.windows, args.labels, args.out, args.report)
    labeled = sum(1 for row in rows if row.get("labels"))
    print(f"Labeled windows written: {args.out}")
    print(f"Windows: {len(rows)}; labeled: {labeled}")
    return 0


def cmd_build_ml_dataset(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.dataset import build_ml_dataset_v0

    summary = build_ml_dataset_v0(args.features, args.windows, args.out, args.report)
    print(f"ML dataset written: {args.out}")
    print(f"Rows: {summary['row_count']}; labels: {summary['label_count']}")
    return 0


def cmd_analyze_ml(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.clustering import analyze_ml_v0

    summary = analyze_ml_v0(args.dataset, args.out_dir)
    print(f"ML analysis reports written: {args.out_dir}")
    print(f"Cluster assignments: {summary['assignments']}")
    return 0


def cmd_audit_baked(args: argparse.Namespace) -> int:
    from vam_timeline_ai.motion.data_audit import audit_baked_samples

    rows = audit_baked_samples(args.sample_index, args.out_jsonl, args.report)
    ok = sum(1 for row in rows if row.get("audit_status") == "ok")
    print(f"Baked sample audit written: {args.out_jsonl}")
    print(f"Audited: {len(rows)}; ok: {ok}")
    return 0


def cmd_audit_body_motion_quality(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.body_motion_quality import audit_body_motion_quality

    rows = audit_body_motion_quality(args.run_dir, args.sample_index, args.features, args.controller_map, args.out_jsonl, args.report)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("body_motion_quality"))
        counts[key] = counts.get(key, 0) + 1
    print(f"Body motion quality audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; counts: {counts}")
    return 0


def cmd_score_cowgirl_candidates_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v2

    rows = score_cowgirl_candidates_v2(args.run_dir, args.wild_reference_matches, args.body_quality, args.features, args.out_jsonl, args.report)
    clean = sum(1 for row in rows if row.get("clean_cowgirl_candidate"))
    print(f"Cowgirl candidate scores v2 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; clean candidates: {clean}")
    return 0


def cmd_score_rider_receiver_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.rider_receiver_discrimination import score_rider_receiver_v1

    rows = score_rider_receiver_v1(
        args.run_dir,
        args.features,
        args.pair_features,
        args.pair_windows,
        args.body_quality,
        args.wild_reference_matches,
        args.out_jsonl,
        args.report,
    )
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("rider_receiver_status"))
        counts[key] = counts.get(key, 0) + 1
    print(f"Rider/receiver scores written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; status counts: {counts}")
    return 0


def cmd_score_cowgirl_candidates_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v3

    rows = score_cowgirl_candidates_v3(
        args.run_dir,
        args.wild_reference_matches,
        args.body_quality,
        args.rider_receiver_scores,
        args.features,
        args.out_jsonl,
        args.report,
    )
    clean = sum(1 for row in rows if row.get("clean_cowgirl_rider_candidate_v3"))
    receiver = sum(1 for row in rows if row.get("likely_receiver_false_positive"))
    print(f"Cowgirl candidate scores v3 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; clean active-rider candidates: {clean}; receiver false positives: {receiver}")
    return 0


def cmd_build_relative_motion_windows(args: argparse.Namespace) -> int:
    from vam_timeline_ai.motion.relative_motion import build_relative_motion_windows

    rows = build_relative_motion_windows(args.run_dir, args.sample_index, args.windows, args.controller_map, args.body_quality, args.out_dir, args.index_out, args.report)
    safe = sum(1 for row in rows if row.get("safe_for_learning"))
    print(f"Relative motion windows written: {args.index_out}")
    print(f"Rows: {len(rows)}; safe_for_learning: {safe}")
    return 0


def cmd_extract_relative_motion_features(args: argparse.Namespace) -> int:
    from vam_timeline_ai.features.relative_features import extract_relative_motion_features

    rows = extract_relative_motion_features(args.relative_index, args.out_jsonl, args.out_npz, args.report)
    safe = sum(1 for row in rows if row.get("feature_values", {}).get("safe_for_learning"))
    print(f"Relative motion features written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; safe_for_learning: {safe}")
    return 0


def cmd_analyze_trajectory_shapes(args: argparse.Namespace) -> int:
    from vam_timeline_ai.features.trajectory_shape import analyze_trajectory_shapes

    rows = analyze_trajectory_shapes(args.relative_index, args.relative_features, args.out_jsonl, args.out_npz, args.report)
    print(f"Trajectory shape features written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}")
    return 0


def cmd_score_cowgirl_candidates_v4(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v4

    rows = score_cowgirl_candidates_v4(
        args.run_dir,
        args.relative_reference_matches,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.rider_receiver_scores,
        args.features,
        args.out_jsonl,
        args.report,
    )
    clean = sum(1 for row in rows if row.get("clean_cowgirl_candidate_v4"))
    print(f"Cowgirl candidate scores v4 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; clean relative/trajectory candidates: {clean}")
    return 0


def cmd_audit_pose_export_validity(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.pose_export_validity import audit_pose_export_validity

    rows = audit_pose_export_validity(args.run_dir, args.review_dir, args.sample_index, args.relative_index, args.body_quality, args.out_jsonl, args.report, args.controller_validity, args.pose_anchor_completeness, args.controller_orientation_validity)
    safe = sum(1 for row in rows if row.get("generation_template_safe"))
    print(f"Pose/export validity audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; generation_template_safe: {safe}")
    return 0


def cmd_audit_pose_anchor_completeness(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.pose_anchor_completeness import audit_pose_anchor_completeness

    rows = audit_pose_anchor_completeness(args.run_dir, args.relative_index, args.sample_index, args.controller_map, args.body_quality, args.out_jsonl, args.report)
    complete = sum(1 for row in rows if row.get("generation_pose_anchor_status") == "complete")
    missing_foot = sum(1 for row in rows if row.get("missing_foot_controllers"))
    print(f"Pose anchor completeness audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; complete: {complete}; missing_foot: {missing_foot}")
    return 0


def cmd_audit_controller_validity(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.controller_validity import audit_controller_validity

    rows = audit_controller_validity(args.run_dir, args.relative_index, args.sample_index, args.controller_map, args.out_jsonl, args.report, args.pose_anchor_completeness, args.controller_orientation_validity)
    foot = sum(1 for row in rows if row.get("foot_controller_outlier"))
    invalid = sum(1 for row in rows if row.get("controller_validity_status") == "invalid")
    print(f"Controller validity audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; invalid: {invalid}; foot_outliers: {foot}")
    return 0


def cmd_audit_controller_orientation_validity(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.controller_orientation_validity import audit_controller_orientation_validity

    rows = audit_controller_orientation_validity(
        args.run_dir,
        args.relative_index,
        args.sample_index,
        args.controller_map,
        args.pose_anchor_completeness,
        args.out_jsonl,
        args.report,
    )
    invalid = sum(1 for row in rows if row.get("orientation_validity_status") == "invalid")
    foot = sum(1 for row in rows if row.get("foot_rotation_outlier"))
    print(f"Controller orientation validity audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; invalid: {invalid}; foot_rotation_outliers: {foot}")
    return 0


def cmd_audit_controller_distance_validity(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.controller_distance_validity import audit_controller_distance_validity

    rows = audit_controller_distance_validity(
        args.run_dir,
        args.relative_index,
        args.sample_index,
        args.controller_map,
        args.pose_anchor_completeness,
        args.out_jsonl,
        args.report,
    )
    invalid = sum(1 for row in rows if row.get("controller_distance_validity_status") == "invalid")
    outliers = sum(1 for row in rows if row.get("controller_distance_outlier"))
    print(f"Controller distance validity audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; invalid: {invalid}; distance_outliers: {outliers}")
    return 0


def cmd_audit_cowgirl_core_controllers(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.cowgirl_core_controller_requirements import audit_cowgirl_core_controllers

    rows = audit_cowgirl_core_controllers(
        args.run_dir,
        args.relative_index,
        args.controller_map,
        args.body_quality,
        args.pose_anchor_completeness,
        args.out_jsonl,
        args.report,
    )
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("cowgirl_core_controller_status"))
        counts[key] = counts.get(key, 0) + 1
    gate = sum(1 for row in rows if row.get("generation_safe_core_controller_gate") is True)
    print(f"Cowgirl core controller audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; gate_pass: {gate}; status_counts: {counts}")
    return 0


def cmd_audit_bj_oral_trap_guard(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.bj_oral_trap_guard import audit_bj_oral_trap_guard

    rows = audit_bj_oral_trap_guard(
        args.run_dir,
        args.relative_features,
        args.trajectory_features,
        args.relative_reference_matches,
        args.cowgirl_core_controllers,
        args.out_jsonl,
        args.report,
    )
    traps = sum(1 for row in rows if row.get("head_or_oral_domain_trap"))
    pose_false = sum(1 for row in rows if row.get("cowgirl_pose_false_positive"))
    print(f"BJ/oral trap guard audit written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; traps: {traps}; cowgirl_pose_false_positive: {pose_false}")
    return 0


def cmd_score_cowgirl_candidates_v5(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v5

    rows = score_cowgirl_candidates_v5(
        args.run_dir,
        args.relative_reference_matches,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.rider_receiver_scores,
        args.pose_export_validity,
        args.features,
        args.out_jsonl,
        args.report,
    )
    semantic = sum(1 for row in rows if row.get("semantic_cowgirl_candidate_v5"))
    generation = sum(1 for row in rows if row.get("generation_candidate_v5"))
    print(f"Cowgirl candidate scores v5 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; semantic candidates: {semantic}; generation candidates: {generation}")
    return 0


def cmd_score_cowgirl_candidates_v6(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v6

    rows = score_cowgirl_candidates_v6(
        args.run_dir,
        args.relative_reference_matches,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.rider_receiver_scores,
        args.pose_export_validity,
        args.controller_validity,
        args.features,
        args.out_jsonl,
        args.report,
    )
    semantic = sum(1 for row in rows if row.get("semantic_cowgirl_candidate_v6"))
    generation = sum(1 for row in rows if row.get("generation_candidate_v6"))
    invalid = sum(1 for row in rows if row.get("semantically_cowgirl_but_controller_invalid"))
    print(f"Cowgirl candidate scores v6 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; semantic candidates: {semantic}; generation candidates: {generation}; semantic-controller-invalid: {invalid}")
    return 0


def cmd_score_cowgirl_candidates_v7(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v7

    rows = score_cowgirl_candidates_v7(
        args.run_dir,
        args.relative_reference_matches,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.rider_receiver_scores,
        args.pose_export_validity,
        args.controller_validity,
        args.pose_anchor_completeness,
        args.features,
        args.out_jsonl,
        args.report,
    )
    semantic = sum(1 for row in rows if row.get("semantic_cowgirl_candidate_v7"))
    generation = sum(1 for row in rows if row.get("semantic_cowgirl_generation_safe"))
    anchor_incomplete = sum(1 for row in rows if row.get("semantic_cowgirl_anchor_incomplete"))
    print(f"Cowgirl candidate scores v7 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; semantic: {semantic}; generation_safe: {generation}; anchor_incomplete: {anchor_incomplete}")
    return 0


def cmd_score_cowgirl_candidates_v8(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v8

    rows = score_cowgirl_candidates_v8(
        args.run_dir,
        args.relative_reference_matches,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.rider_receiver_scores,
        args.pose_export_validity,
        args.controller_validity,
        args.pose_anchor_completeness,
        args.controller_orientation_validity,
        args.features,
        args.out_jsonl,
        args.report,
    )
    semantic = sum(1 for row in rows if row.get("semantic_cowgirl_candidate_v8"))
    generation = sum(1 for row in rows if row.get("semantic_cowgirl_generation_safe"))
    orientation_invalid = sum(1 for row in rows if row.get("semantic_cowgirl_orientation_invalid"))
    print(f"Cowgirl candidate scores v8 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; semantic: {semantic}; generation_safe: {generation}; orientation_invalid: {orientation_invalid}")
    return 0


def cmd_score_cowgirl_candidates_v9(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v9

    rows = score_cowgirl_candidates_v9(
        args.run_dir,
        args.relative_reference_matches,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.rider_receiver_scores,
        args.pose_export_validity,
        args.controller_validity,
        args.pose_anchor_completeness,
        args.controller_orientation_validity,
        args.controller_distance_validity,
        args.features,
        args.out_jsonl,
        args.report,
    )
    semantic = sum(1 for row in rows if row.get("semantic_cowgirl_candidate_v9"))
    generation = sum(1 for row in rows if row.get("semantic_cowgirl_generation_safe"))
    distance_invalid = sum(1 for row in rows if row.get("semantic_cowgirl_distance_invalid"))
    print(f"Cowgirl candidate scores v9 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; semantic: {semantic}; generation_safe: {generation}; distance_invalid: {distance_invalid}")
    return 0


def cmd_score_cowgirl_candidates_v10(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v10

    rows = score_cowgirl_candidates_v10(
        args.run_dir,
        args.relative_reference_matches,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.rider_receiver_scores,
        args.pose_export_validity,
        args.controller_validity,
        args.pose_anchor_completeness,
        args.controller_orientation_validity,
        args.controller_distance_validity,
        args.cowgirl_core_controllers,
        args.bj_oral_trap_guard,
        args.features,
        args.out_jsonl,
        args.report,
    )
    semantic = sum(1 for row in rows if row.get("semantic_cowgirl_candidate_v10"))
    generation = sum(1 for row in rows if row.get("semantic_cowgirl_generation_safe"))
    core_missing = sum(1 for row in rows if row.get("semantic_cowgirl_core_controller_missing"))
    traps = sum(1 for row in rows if row.get("bj_oral_trap_negative"))
    print(f"Cowgirl candidate scores v10 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; semantic: {semantic}; generation_safe: {generation}; core_missing: {core_missing}; bj_oral_traps: {traps}")
    return 0


def cmd_build_cowgirl_candidate_db_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.cowgirl_candidate_database import build_cowgirl_candidate_db_v1

    rows = build_cowgirl_candidate_db_v1(
        args.run_dir,
        args.candidate_scores,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.pose_anchor_completeness,
        args.controller_validity,
        args.controller_orientation_validity,
        args.controller_distance_validity,
        args.out_jsonl,
        args.out_csv,
        args.report,
    )
    generation = sum(1 for row in rows if row.get("category") == "semantic_cowgirl_generation_safe")
    print(f"Cowgirl candidate DB v1 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; generation_safe: {generation}")
    return 0


def cmd_build_cowgirl_candidate_db_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.cowgirl_candidate_database import build_cowgirl_candidate_db_v2

    rows = build_cowgirl_candidate_db_v2(
        args.run_dir,
        args.candidate_scores,
        args.relative_features,
        args.trajectory_features,
        args.body_quality,
        args.pose_anchor_completeness,
        args.controller_validity,
        args.controller_orientation_validity,
        args.controller_distance_validity,
        args.cowgirl_core_controllers,
        args.bj_oral_trap_guard,
        args.out_jsonl,
        args.out_csv,
        args.report,
    )
    generation = sum(1 for row in rows if row.get("category") == "semantic_cowgirl_generation_safe")
    core_missing = sum(1 for row in rows if row.get("category") == "semantic_cowgirl_core_controller_missing")
    traps = sum(1 for row in rows if row.get("category") == "bj_oral_trap_negative")
    print(f"Cowgirl candidate DB v2 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; generation_safe: {generation}; core_missing: {core_missing}; bj_oral_traps: {traps}")
    return 0


def cmd_discover_controller_map(args: argparse.Namespace) -> int:
    from vam_timeline_ai.motion.controller_mapping import discover_controller_map

    inventory = discover_controller_map(args.sample_index, args.out, args.map_out, args.report)
    print(f"Controller inventory written: {args.out}")
    print(f"Controller body-part map written: {args.map_out}")
    print(f"Controller names: {len(inventory['controller_mappings'])}")
    return 0


def cmd_extract_cowgirl_features_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.cowgirl.feature_extractor_v1 import extract_cowgirl_features_v1

    rows = extract_cowgirl_features_v1(args.windows, args.sample_index, args.controller_map, args.out_jsonl, args.out_npz, args.report)
    numeric = sum(1 for row in rows if row.get("feature_quality", {}).get("has_any_numeric_features"))
    print(f"Cowgirl v1 feature rows written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; numeric: {numeric}")
    return 0


def cmd_import_handmade_reference_animations(args: argparse.Namespace) -> int:
    from vam_timeline_ai.references.handmade_import import import_handmade_reference_animations

    summary = import_handmade_reference_animations(args.zip, args.out_dir)
    print(f"Handmade reference import written: {args.out_dir}")
    print(f"Status: {summary.get('status')}; JSON: {summary.get('json_count')}; JPG: {summary.get('jpg_count')}")
    return 0


def cmd_extract_handmade_reference_features(args: argparse.Namespace) -> int:
    from vam_timeline_ai.references.handmade_features import extract_handmade_reference_features

    rows = extract_handmade_reference_features(args.manifest, args.sample_index, args.out_jsonl, args.out_npz, args.report)
    print(f"Handmade reference features written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}")
    return 0


def cmd_build_handmade_relative_reference_features(args: argparse.Namespace) -> int:
    from vam_timeline_ai.references.handmade_relative import build_handmade_relative_reference_features

    rows = build_handmade_relative_reference_features(args.handmade_sample_index, args.controller_map, args.out_jsonl, args.out_npz, args.report)
    safe = sum(1 for row in rows if row.get("safe_for_learning"))
    print(f"Handmade relative features written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; safe_for_learning: {safe}")
    return 0


def cmd_build_handmade_reference_signatures(args: argparse.Namespace) -> int:
    from vam_timeline_ai.references.signature_report import build_handmade_reference_signatures

    sigs = build_handmade_reference_signatures(args.features, args.out_json, args.report)
    print(f"Handmade signatures written: {args.out_json}")
    print(f"Families: {len(sigs.get('families', {}))}")
    return 0


def cmd_compare_wild_to_handmade_references(args: argparse.Namespace) -> int:
    from vam_timeline_ai.references.reference_matcher import compare_wild_to_handmade_references

    rows = compare_wild_to_handmade_references(args.wild_features, args.wild_body_quality, args.handmade_features, args.signatures, args.out_jsonl, args.report)
    print(f"Wild/reference matches written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}")
    return 0


def cmd_compare_relative_wild_to_handmade(args: argparse.Namespace) -> int:
    from vam_timeline_ai.references.relative_matcher import compare_relative_wild_to_handmade

    rows = compare_relative_wild_to_handmade(
        args.wild_relative_features,
        args.wild_trajectory_features,
        args.handmade_relative_features,
        args.handmade_trajectory_features,
        args.out_jsonl,
        args.report,
    )
    print(f"Relative wild/reference matches written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}")
    return 0


def cmd_build_context_pairs(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.context_pairing import build_context_pair_candidates

    rows = build_context_pair_candidates(args.sample_index, args.out, args.report)
    print(f"Context pair candidates written: {args.out}")
    print(f"Pairs: {len(rows)}")
    return 0


def cmd_generate_weak_labels(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.weak_labels import generate_weak_labels_v1

    rows = generate_weak_labels_v1(args.features, args.out, args.report)
    labels = sum(len(row.get("weak_labels", [])) for row in rows)
    print(f"Weak labels written: {args.out}")
    print(f"Rows: {len(rows)}; weak label assignments: {labels}")
    return 0


def cmd_build_review_queue(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.review_queue import build_review_queue_v1

    rows = build_review_queue_v1(args.features, args.weak_labels, args.clusters, args.windows, args.out, args.markdown)
    print(f"Review queue written: {args.out}")
    print(f"Review queue markdown: {args.markdown}")
    print(f"Rows: {len(rows)}")
    return 0


def cmd_build_ml_dataset_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.dataset_v1 import build_ml_dataset_v1

    summary = build_ml_dataset_v1(args.features, args.windows, args.weak_labels, args.out, args.report)
    print(f"ML dataset v1 written: {args.out}")
    print(f"Shape: {summary['shape']}; weak labels: {summary['weak_label_count']}; manual labels: {summary['manual_label_count']}")
    return 0


def cmd_analyze_ml_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.readiness_v1 import analyze_ml_v1

    summary = analyze_ml_v1(args.dataset, args.out_dir)
    print(f"ML v1 readiness reports written: {args.out_dir}")
    print(f"Rows: {summary['rows']}; features: {summary['features']}")
    return 0


def cmd_cluster_ml_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.clustering_v1 import cluster_ml_v1

    summary = cluster_ml_v1(args.dataset, args.out_dir)
    print(f"Cluster v1 reports written: {args.out_dir}")
    print(f"Ran: {summary.get('ran')}; reason: {summary.get('reason', '')}")
    return 0


def cmd_audit_data_integrity(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.integrity import audit_data_integrity

    summary = audit_data_integrity(
        args.source_index,
        args.sample_index,
        args.windows,
        args.features,
        args.dataset,
        args.out,
        strict=_arg_bool(args.strict),
        pair_windows=args.pair_windows,
        pair_features=args.pair_features,
        review_batch=args.review_batch,
    )
    print(f"Data integrity report written: {args.out}")
    print(f"Errors: {len(summary.get('errors', []))}; warnings: {len(summary.get('warnings', []))}; windows={summary['windows']}; features={summary['feature_rows']}")
    return 1 if _arg_bool(args.strict) and summary.get("errors") else 0


def cmd_calibrate_weak_labels_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.weak_label_calibration import calibrate_weak_labels_v2

    rows = calibrate_weak_labels_v2(args.features, args.weak_labels, args.out, args.report)
    assignments = sum(len(row.get("weak_labels", [])) for row in rows)
    print(f"Weak labels v2 written: {args.out}")
    print(f"Rows: {len(rows)}; assignments: {assignments}")
    return 0


def cmd_build_pair_windows_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.pair_windows import build_pair_windows_v1

    rows = build_pair_windows_v1(args.pair_candidates, args.windows, args.sample_index, args.out, args.report)
    print(f"Pair windows written: {args.out}")
    print(f"Pair windows: {len(rows)}")
    return 0


def cmd_extract_pair_features_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.cowgirl.pair_feature_extractor import extract_pair_features_v0

    rows = extract_pair_features_v0(args.pair_windows, args.sample_index, args.controller_map, args.out_jsonl, args.out_npz, args.report)
    hand_rows = sum(1 for row in rows if row.get("feature_quality", {}).get("has_hand_to_partner_features"))
    print(f"Pair features written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; rows with hand-to-partner features: {hand_rows}")
    return 0


def cmd_write_manual_label_schema_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.manual_labels import write_manual_label_schema_v2

    paths = write_manual_label_schema_v2(args.out, args.template, args.guide)
    print(f"Manual label schema written: {paths['schema']}")
    print(f"Manual label template written: {paths['template']}")
    print(f"Manual label guide written: {paths['guide']}")
    return 0


def cmd_validate_manual_labels_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.manual_label_validation import validate_manual_labels_v2

    result = validate_manual_labels_v2(args.labels, args.schema, args.windows, args.pair_windows, args.out)
    print(f"Manual label validation report written: {args.out}")
    print(f"Status: {result['status']}; errors={len(result['errors'])}; warnings={len(result['warnings'])}")
    return 1 if result["status"] == "error" else 0


def cmd_build_review_batch_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.review_batch import build_review_batch_v2

    rows = build_review_batch_v2(
        args.windows,
        args.features,
        args.weak_labels,
        args.pair_windows,
        args.pair_features,
        args.clusters,
        args.out_dir,
        batch_size=args.batch_size,
        max_per_scene=args.max_per_scene,
        max_per_sample=args.max_per_sample,
        prefer_pair_context=_arg_bool(args.prefer_pair_context),
    )
    print(f"Review batch written: {args.out_dir}")
    print(f"Review items: {len(rows)}")
    return 0


def cmd_render_review_previews_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.review_previews import render_review_previews_v1

    summary = render_review_previews_v1(args.review_batch, args.sample_index, args.controller_map, args.out_dir)
    print(f"Review previews written: {args.out_dir}")
    print(f"Items: {summary['items']}; matplotlib_available={summary['matplotlib_available']}")
    return 0


def cmd_merge_manual_label_batch(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.manual_label_merge import merge_manual_label_batch

    summary = merge_manual_label_batch(args.base, args.batch, args.out, backup=_arg_bool(args.backup), report=args.report)
    print(f"Manual label merge status: {summary.get('status')}")
    print(f"Manual labels output: {summary['out']}")
    print(f"Merged windows: {summary['merged_windows']}; pair windows: {summary['merged_pair_windows']}")
    return 1 if summary.get("status") == "error" else 0


def cmd_inspect_edited_label_batch(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.edited_label_batch import inspect_edited_label_batch

    result = inspect_edited_label_batch(args.stub, args.edited, args.windows, args.pair_windows, args.out)
    print(f"Edited label batch inspection written: {args.out}")
    print(f"Safe to merge: {result['safe_to_merge']}; usable entries: {result['usable_edited_entries']}; errors={len(result['errors'])}")
    return 0 if result["safe_to_merge"] else 1


def cmd_summarize_manual_labels(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.manual_label_summary import summarize_manual_labels

    summary = summarize_manual_labels(args.labels, args.windows, args.pair_windows, args.out)
    print(f"Manual label summary written: {args.out}")
    print(f"Windows: {summary['total_labeled_windows']}; pair windows: {summary['total_labeled_pair_windows']}")
    return 0


def cmd_plan_ml_splits_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.splits import plan_ml_splits_v1

    plan = plan_ml_splits_v1(args.dataset, args.labels, args.out, args.report)
    print(f"Split plan written: {args.out}")
    print(f"Random window split allowed: {plan['random_window_split_allowed']}; supervised split ready={plan['can_plan_supervised_split']}")
    return 0


def cmd_build_ml_dataset_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.dataset_v2 import build_ml_dataset_v2

    summary = build_ml_dataset_v2(args.features, args.windows, args.weak_labels, args.manual_labels, args.out, args.report)
    print(f"ML dataset v2 written: {args.out}")
    print(f"Shape: {summary['shape']}; manual labels: {summary['manual_label_count']}; weak labels: {summary['weak_label_count']}")
    return 0


def cmd_analyze_supervised_readiness(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.supervised_readiness import analyze_supervised_readiness

    summary = analyze_supervised_readiness(args.dataset, args.labels, args.split_plan, args.out)
    print(f"Supervised readiness report written: {args.out}")
    print(f"Eligible labels: {summary['eligible_labels'] or 'None'}")
    return 0


def cmd_train_supervised_baseline_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.supervised_baseline import train_supervised_baseline_v0

    summary = train_supervised_baseline_v0(args.dataset, args.split_plan, args.out_dir, args.report)
    print(f"Supervised baseline report written: {args.report}")
    print(f"Trained: {summary['trained']}; reason: {summary['reason']}")
    return 0


def cmd_build_active_review_batch_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.active_review_batch import build_active_review_batch_v3

    rows = build_active_review_batch_v3(
        args.windows,
        args.features,
        args.weak_labels,
        args.pair_windows,
        args.pair_features,
        args.manual_labels,
        args.supervised_readiness,
        args.out_dir,
        batch_size=args.batch_size,
        max_per_scene=args.max_per_scene,
        max_per_sample=args.max_per_sample,
        prefer_coverage_gaps=_arg_bool(args.prefer_coverage_gaps),
    )
    print(f"Active review batch written: {args.out_dir}")
    print(f"Review items: {len(rows)}")
    return 0


def cmd_find_latest_review_batch(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.review_batch_discovery import find_latest_review_batch

    result = find_latest_review_batch(args.run_dir, args.out)
    latest = result.get("latest_batch") or {}
    print(f"Latest review batch report written: {args.out}")
    print(f"Status: {result['status']}; latest={latest.get('batch_name')}")
    return 0


def cmd_write_labeling_next_step(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.labeling_next_step import write_labeling_next_step

    result = write_labeling_next_step(args.run_dir, args.out)
    latest = result.get("latest_batch") or {}
    print(f"Human labeling next-step report written: {args.out}")
    print(f"Status: {result['status']}; latest={latest.get('batch_name')}")
    return 0


def cmd_ingest_latest_edited_batch(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.ingest_latest import ingest_latest_edited_batch

    result = ingest_latest_edited_batch(args.run_dir, args.schema, stop_if_missing=_arg_bool(args.stop_if_missing))
    print(f"Ingest latest edited batch status: {result['status']}")
    print(result.get("message", ""))
    return 0 if result["status"] in {"waiting_for_human_labels", "ingested", "no_valid_batch"} else 1


def cmd_generate_machine_label_proposals_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.machine_label_proposals import generate_machine_label_proposals_v1

    rows = generate_machine_label_proposals_v1(
        args.run_dir,
        args.features,
        args.pair_features,
        args.weak_labels,
        args.windows,
        args.pair_windows,
        args.out_jsonl,
        args.out_yaml,
        args.report,
    )
    print(f"Machine label proposals written: {args.out_jsonl}")
    print(f"Proposals: {len(rows)}")
    return 0


def cmd_build_silver_labels_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.silver_labels import build_silver_labels_v1

    rows = build_silver_labels_v1(args.proposals, args.out_jsonl, args.out_yaml, args.report, min_confidence=args.min_confidence)
    print(f"Silver labels written: {args.out_jsonl}")
    print(f"Silver records: {len(rows)}")
    return 0


def cmd_compare_machine_labels_to_manual(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.label_comparison import compare_machine_labels_to_manual

    summary = compare_machine_labels_to_manual(args.manual_labels, args.silver_labels, args.out)
    print(f"Machine/manual comparison written: {args.out}")
    print(f"Status: {summary['status']}; overlap={summary['overlap_windows']}")
    return 0


def cmd_build_ml_dataset_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.dataset_v3 import build_ml_dataset_v3

    summary = build_ml_dataset_v3(args.features, args.windows, args.weak_labels, args.manual_labels, args.silver_labels, args.out, args.report)
    print(f"ML dataset v3 written: {args.out}")
    print(f"Shape: {summary['shape']}; silver labels: {summary['silver_label_count']}")
    return 0


def cmd_analyze_silver_readiness(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.silver_readiness import analyze_silver_readiness

    summary = analyze_silver_readiness(args.dataset, args.silver_labels, args.out)
    print(f"Silver readiness report written: {args.out}")
    print(f"Eligible proxy labels: {summary['eligible_labels'] or 'None'}")
    return 0


def cmd_train_silver_baseline_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.silver_baseline import train_silver_baseline_v0

    summary = train_silver_baseline_v0(args.dataset, args.out_dir, args.report)
    print(f"Silver baseline report written: {args.report}")
    print(f"Trained: {summary['trained']}; reason: {summary['reason']}")
    return 0


def cmd_build_machine_proposal_review_batch(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.machine_proposal_review_batch import build_machine_proposal_review_batch

    rows = build_machine_proposal_review_batch(
        args.run_dir,
        args.proposals,
        args.silver_labels,
        args.out_dir,
        batch_size=args.batch_size,
        max_per_scene=args.max_per_scene,
        max_per_sample=args.max_per_sample,
    )
    print(f"Machine proposal review batch written: {args.out_dir}")
    print(f"Review items: {len(rows)}")
    return 0


def cmd_run_machine_labeling_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.machine_labeling_v1 import run_machine_labeling_v1

    summary = run_machine_labeling_v1(args.run_dir, args.min_silver_confidence, train_silver_baseline=_arg_bool(args.train_silver_baseline))
    print(f"Machine labeling v1 status: {summary['status']}")
    print(f"Proposals: {summary.get('proposal_count', 0)}; silver records: {summary.get('silver_record_count', 0)}")
    print(f"Manual labels modified: {summary.get('manual_labels_modified', False)}")
    return 0 if summary["status"] == "ok" else 1


def cmd_audit_machine_labels_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.machine_label_audit import audit_machine_labels_v1

    summary = audit_machine_labels_v1(args.run_dir, args.proposals, args.silver_labels, args.windows, args.pair_windows, args.out, args.out_json)
    print(f"Machine label audit written: {args.out}")
    print(f"Proposals: {summary['total_proposals']}; conflicts={summary['conflict_counts']}")
    return 0


def cmd_aggregate_machine_labels_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.machine_label_aggregation import aggregate_machine_labels_v2

    summary = aggregate_machine_labels_v2(args.proposals, args.out_window_jsonl, args.out_pair_jsonl, args.report)
    print(f"Aggregated machine labels written: {args.out_window_jsonl}, {args.out_pair_jsonl}")
    print(f"Window scores: {summary['window_score_rows']}; pair scores: {summary['pair_score_rows']}")
    return 0


def cmd_build_silver_labels_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.silver_labels_v2 import build_silver_labels_v2

    summary = build_silver_labels_v2(args.window_scores, args.pair_scores, args.out_window_jsonl, args.out_pair_jsonl, args.out_yaml, args.report, min_score=args.min_score)
    print(f"Silver v2 labels written: {args.out_window_jsonl}, {args.out_pair_jsonl}")
    print(f"Window records: {summary['v2_silver_window_records']}; pair records: {summary['v2_silver_pair_records']}")
    return 0


def cmd_build_ml_dataset_v4(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.dataset_v4 import build_ml_dataset_v4

    summary = build_ml_dataset_v4(args.features, args.windows, args.weak_labels, args.manual_labels, args.silver_window_labels, args.silver_pair_labels, args.out, args.report)
    print(f"ML dataset v4 written: {args.out}")
    print(f"Shape: {summary['shape']}; default trainable silver labels={summary['default_trainable_silver_labels']}")
    return 0


def cmd_analyze_silver_readiness_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.silver_readiness_v2 import analyze_silver_readiness_v2

    summary = analyze_silver_readiness_v2(args.dataset, args.silver_window_labels, args.silver_pair_labels, args.out)
    print(f"Silver readiness v2 report written: {args.out}")
    print(f"Ready: {summary['silver_proxy_training_ready']}; labels={summary['labels_trainable_by_default']}")
    return 0


def cmd_train_silver_baseline_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.silver_baseline_v1 import train_silver_baseline_v1

    summary = train_silver_baseline_v1(args.dataset, args.readiness, args.out_dir, args.report, allow_numpy_fallback=_arg_bool(args.allow_numpy_fallback))
    print(f"Silver baseline v1 report written: {args.report}")
    print(f"Trained: {summary['trained']}; sklearn={summary['sklearn_used']}; numpy={summary['numpy_fallback_used']}")
    return 0


def cmd_build_machine_proposal_review_batch_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.machine_proposal_review_batch_v2 import build_machine_proposal_review_batch_v2

    rows = build_machine_proposal_review_batch_v2(
        args.run_dir,
        args.window_scores,
        args.pair_scores,
        args.silver_window_labels,
        args.silver_pair_labels,
        args.out_dir,
        batch_size=args.batch_size,
        max_per_scene=args.max_per_scene,
        max_per_sample=args.max_per_sample,
    )
    print(f"Machine proposal review batch v2 written: {args.out_dir}")
    print(f"Review items: {len(rows)}")
    return 0


def cmd_run_machine_labeling_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.machine_labeling_v2 import run_machine_labeling_v2

    summary = run_machine_labeling_v2(
        args.run_dir,
        min_silver_score=args.min_silver_score,
        train_silver_baseline=_arg_bool(args.train_silver_baseline),
        allow_numpy_fallback=_arg_bool(args.allow_numpy_fallback),
    )
    print(f"Machine labeling v2 status: {summary['status']}")
    print(f"Silver v2 window records: {summary.get('silver_v2_window_records')}; pair records: {summary.get('silver_v2_pair_records')}")
    print(f"Baseline trained: {summary.get('baseline_trained')}; manual labels modified: {summary.get('manual_labels_modified', False)}")
    return 0 if summary["status"] == "ok" else 1


def cmd_export_reality_audit_100(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.reality_audit import export_reality_audit_100

    summary = export_reality_audit_100(args.run_dir, args.out_dir, count=args.count)
    print(f"Reality audit export written: {args.out_dir}")
    print(f"Audit items: {summary['audit_items']}; categories={summary['category_distribution']}")
    print(f"Preview images: {summary['preview_images']}; manual labels modified={summary['manual_labels_modified']}")
    return 0


def cmd_summarize_reality_audit(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.reality_audit import summarize_reality_audit

    summary = summarize_reality_audit(args.annotations, args.audit_batch, args.out)
    print(f"Reality audit summary written: {args.out}")
    print(f"Status: {summary['status']}; audit items={summary['audit_items']}")
    return 0


def cmd_export_semantic_review_010(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.semantic_review import export_semantic_review_010

    summary = export_semantic_review_010(
        args.run_dir,
        args.out_dir,
        count=args.count,
        attempt_timeline_export=_arg_bool(args.attempt_timeline_export),
        export_mode=args.export_mode,
        candidate_db=args.candidate_db,
        use_cowgirl_candidate_db=_arg_bool(args.use_cowgirl_candidate_db),
        use_body_motion_quality=_arg_bool(args.use_body_motion_quality),
        prefer_clean_body_motion=_arg_bool(args.prefer_clean_body_motion),
        use_handmade_reference_matches=_arg_bool(args.use_handmade_reference_matches),
        prefer_longer_cowgirl_windows=_arg_bool(args.prefer_longer_cowgirl_windows),
        min_cowgirl_window_seconds=args.min_cowgirl_window_seconds,
        use_cowgirl_candidate_score_v2=_arg_bool(args.use_cowgirl_candidate_score_v2),
        use_cowgirl_candidate_score_v3=_arg_bool(args.use_cowgirl_candidate_score_v3),
        use_cowgirl_candidate_score_v4=_arg_bool(args.use_cowgirl_candidate_score_v4),
        use_cowgirl_candidate_score_v5=_arg_bool(args.use_cowgirl_candidate_score_v5),
        use_cowgirl_candidate_score_v6=_arg_bool(args.use_cowgirl_candidate_score_v6),
        use_cowgirl_candidate_score_v7=_arg_bool(args.use_cowgirl_candidate_score_v7),
        use_cowgirl_candidate_score_v8=_arg_bool(args.use_cowgirl_candidate_score_v8),
        use_cowgirl_candidate_score_v9=_arg_bool(args.use_cowgirl_candidate_score_v9),
        use_rider_receiver_discrimination=_arg_bool(args.use_rider_receiver_discrimination),
        use_relative_motion_features=_arg_bool(args.use_relative_motion_features),
        use_trajectory_shape_features=_arg_bool(args.use_trajectory_shape_features),
        use_relative_reference_matches=_arg_bool(args.use_relative_reference_matches),
        use_pose_export_validity=_arg_bool(args.use_pose_export_validity),
        use_controller_validity=_arg_bool(args.use_controller_validity),
        use_pose_anchor_completeness=_arg_bool(args.use_pose_anchor_completeness),
        use_controller_orientation_validity=_arg_bool(args.use_controller_orientation_validity),
    )
    print(f"Semantic review 010 written: {args.out_dir}")
    print(f"Review items: {summary['review_items']}; categories={summary['category_distribution']}")
    print(f"Timeline exports: {summary['timeline_exports_successful']} successful / {summary['timeline_exports_attempted']} attempted")
    return 0


def cmd_summarize_semantic_review_010(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.semantic_review import summarize_semantic_review_010

    summary = summarize_semantic_review_010(args.answers, args.review, args.out)
    print(f"Semantic review 010 result written: {args.out}")
    print(f"Status: {summary['status']}; review items={summary['review_items']}")
    return 0


def _arg_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _iter_json_files(raw_dir: Path, recursive: bool) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not recursive:
        return sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    results: list[Path] = []
    for path in raw_dir.rglob("*.json"):
        rel_parts = path.relative_to(raw_dir).parts[:-1]
        if any(part in IGNORED_SCAN_DIR_NAMES for part in rel_parts):
            continue
        results.append(path)
    return sorted(results)


def _scan_totals(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_json_files": len(items),
        "parsed_json_files": sum(1 for item in items if item.get("parse_status") == "ok"),
        "parse_failures": sum(1 for item in items if item.get("parse_status") == "error"),
        "vam_scenes": sum(1 for item in items if item.get("is_vam_scene")),
        "external_timeline_exports": sum(1 for item in items if item.get("is_external_timeline_export")),
        "with_native_motion_tracks": sum(1 for item in items if item.get("has_native_motion_tracks")),
        "with_motion_animation_master": sum(1 for item in items if item.get("has_motion_animation_master")),
        "with_timeline": sum(1 for item in items if item.get("has_timeline")),
        "with_person_atoms": sum(1 for item in items if item.get("person_atoms_count", 0) > 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "scan-raw-folder":
        return cmd_scan_raw_folder(args)
    if args.command == "audit-project-state":
        return cmd_audit_project_state(args)
    if args.command == "audit-repo-safety":
        return cmd_audit_repo_safety(args)
    if args.command == "local-status":
        return cmd_local_status(args)
    if args.command == "export-reality-audit-100":
        return cmd_export_reality_audit_100(args)
    if args.command == "summarize-reality-audit":
        return cmd_summarize_reality_audit(args)
    if args.command == "export-semantic-review-010":
        return cmd_export_semantic_review_010(args)
    if args.command == "summarize-semantic-review-010":
        return cmd_summarize_semantic_review_010(args)
    if args.command == "prepare-clean-run":
        return cmd_prepare_clean_run(args)
    if args.command == "build-motion-source-index":
        return cmd_build_motion_source_index(args)
    if args.command == "extract-motion-samples":
        return cmd_extract_motion_samples(args)
    if args.command == "build-movement-windows":
        return cmd_build_movement_windows(args)
    if args.command == "extract-cowgirl-features-v0":
        return cmd_extract_cowgirl_features(args)
    if args.command == "apply-manual-labels":
        return cmd_apply_manual_labels(args)
    if args.command == "build-ml-dataset-v0":
        return cmd_build_ml_dataset(args)
    if args.command == "analyze-ml-v0":
        return cmd_analyze_ml(args)
    if args.command == "audit-baked-samples":
        return cmd_audit_baked(args)
    if args.command == "audit-body-motion-quality":
        return cmd_audit_body_motion_quality(args)
    if args.command == "score-cowgirl-candidates-v2":
        return cmd_score_cowgirl_candidates_v2(args)
    if args.command == "score-rider-receiver-v1":
        return cmd_score_rider_receiver_v1(args)
    if args.command == "score-cowgirl-candidates-v3":
        return cmd_score_cowgirl_candidates_v3(args)
    if args.command == "build-relative-motion-windows":
        return cmd_build_relative_motion_windows(args)
    if args.command == "extract-relative-motion-features":
        return cmd_extract_relative_motion_features(args)
    if args.command == "analyze-trajectory-shapes":
        return cmd_analyze_trajectory_shapes(args)
    if args.command == "score-cowgirl-candidates-v4":
        return cmd_score_cowgirl_candidates_v4(args)
    if args.command == "audit-pose-export-validity":
        return cmd_audit_pose_export_validity(args)
    if args.command == "audit-pose-anchor-completeness":
        return cmd_audit_pose_anchor_completeness(args)
    if args.command == "audit-controller-validity":
        return cmd_audit_controller_validity(args)
    if args.command == "audit-controller-orientation-validity":
        return cmd_audit_controller_orientation_validity(args)
    if args.command == "audit-controller-distance-validity":
        return cmd_audit_controller_distance_validity(args)
    if args.command == "audit-cowgirl-core-controllers":
        return cmd_audit_cowgirl_core_controllers(args)
    if args.command == "audit-bj-oral-trap-guard":
        return cmd_audit_bj_oral_trap_guard(args)
    if args.command == "score-cowgirl-candidates-v5":
        return cmd_score_cowgirl_candidates_v5(args)
    if args.command == "score-cowgirl-candidates-v6":
        return cmd_score_cowgirl_candidates_v6(args)
    if args.command == "score-cowgirl-candidates-v7":
        return cmd_score_cowgirl_candidates_v7(args)
    if args.command == "score-cowgirl-candidates-v8":
        return cmd_score_cowgirl_candidates_v8(args)
    if args.command == "score-cowgirl-candidates-v9":
        return cmd_score_cowgirl_candidates_v9(args)
    if args.command == "score-cowgirl-candidates-v10":
        return cmd_score_cowgirl_candidates_v10(args)
    if args.command == "build-cowgirl-candidate-db-v1":
        return cmd_build_cowgirl_candidate_db_v1(args)
    if args.command == "build-cowgirl-candidate-db-v2":
        return cmd_build_cowgirl_candidate_db_v2(args)
    if args.command == "discover-controller-map":
        return cmd_discover_controller_map(args)
    if args.command == "extract-cowgirl-features-v1":
        return cmd_extract_cowgirl_features_v1(args)
    if args.command == "import-handmade-reference-animations":
        return cmd_import_handmade_reference_animations(args)
    if args.command == "extract-handmade-reference-features":
        return cmd_extract_handmade_reference_features(args)
    if args.command == "build-handmade-relative-reference-features":
        return cmd_build_handmade_relative_reference_features(args)
    if args.command == "build-handmade-reference-signatures":
        return cmd_build_handmade_reference_signatures(args)
    if args.command == "compare-wild-to-handmade-references":
        return cmd_compare_wild_to_handmade_references(args)
    if args.command == "compare-relative-wild-to-handmade":
        return cmd_compare_relative_wild_to_handmade(args)
    if args.command == "build-context-pair-candidates":
        return cmd_build_context_pairs(args)
    if args.command == "generate-weak-labels-v1":
        return cmd_generate_weak_labels(args)
    if args.command == "build-review-queue-v1":
        return cmd_build_review_queue(args)
    if args.command == "build-ml-dataset-v1":
        return cmd_build_ml_dataset_v1(args)
    if args.command == "analyze-ml-v1":
        return cmd_analyze_ml_v1(args)
    if args.command == "cluster-ml-v1":
        return cmd_cluster_ml_v1(args)
    if args.command == "audit-data-integrity":
        return cmd_audit_data_integrity(args)
    if args.command == "calibrate-weak-labels-v2":
        return cmd_calibrate_weak_labels_v2(args)
    if args.command == "build-pair-windows-v1":
        return cmd_build_pair_windows_v1(args)
    if args.command == "extract-pair-features-v0":
        return cmd_extract_pair_features_v0(args)
    if args.command == "write-manual-label-schema-v2":
        return cmd_write_manual_label_schema_v2(args)
    if args.command == "validate-manual-labels-v2":
        return cmd_validate_manual_labels_v2(args)
    if args.command == "build-review-batch-v2":
        return cmd_build_review_batch_v2(args)
    if args.command == "render-review-previews-v1":
        return cmd_render_review_previews_v1(args)
    if args.command == "merge-manual-label-batch":
        return cmd_merge_manual_label_batch(args)
    if args.command == "inspect-edited-label-batch":
        return cmd_inspect_edited_label_batch(args)
    if args.command == "summarize-manual-labels":
        return cmd_summarize_manual_labels(args)
    if args.command == "plan-ml-splits-v1":
        return cmd_plan_ml_splits_v1(args)
    if args.command == "build-ml-dataset-v2":
        return cmd_build_ml_dataset_v2(args)
    if args.command == "analyze-supervised-readiness":
        return cmd_analyze_supervised_readiness(args)
    if args.command == "train-supervised-baseline-v0":
        return cmd_train_supervised_baseline_v0(args)
    if args.command == "build-active-review-batch-v3":
        return cmd_build_active_review_batch_v3(args)
    if args.command == "find-latest-review-batch":
        return cmd_find_latest_review_batch(args)
    if args.command == "write-labeling-next-step":
        return cmd_write_labeling_next_step(args)
    if args.command == "ingest-latest-edited-batch":
        return cmd_ingest_latest_edited_batch(args)
    if args.command == "generate-machine-label-proposals-v1":
        return cmd_generate_machine_label_proposals_v1(args)
    if args.command == "build-silver-labels-v1":
        return cmd_build_silver_labels_v1(args)
    if args.command == "compare-machine-labels-to-manual":
        return cmd_compare_machine_labels_to_manual(args)
    if args.command == "build-ml-dataset-v3":
        return cmd_build_ml_dataset_v3(args)
    if args.command == "analyze-silver-readiness":
        return cmd_analyze_silver_readiness(args)
    if args.command == "train-silver-baseline-v0":
        return cmd_train_silver_baseline_v0(args)
    if args.command == "build-machine-proposal-review-batch":
        return cmd_build_machine_proposal_review_batch(args)
    if args.command == "run-machine-labeling-v1":
        return cmd_run_machine_labeling_v1(args)
    if args.command == "audit-machine-labels-v1":
        return cmd_audit_machine_labels_v1(args)
    if args.command == "aggregate-machine-labels-v2":
        return cmd_aggregate_machine_labels_v2(args)
    if args.command == "build-silver-labels-v2":
        return cmd_build_silver_labels_v2(args)
    if args.command == "build-ml-dataset-v4":
        return cmd_build_ml_dataset_v4(args)
    if args.command == "analyze-silver-readiness-v2":
        return cmd_analyze_silver_readiness_v2(args)
    if args.command == "train-silver-baseline-v1":
        return cmd_train_silver_baseline_v1(args)
    if args.command == "build-machine-proposal-review-batch-v2":
        return cmd_build_machine_proposal_review_batch_v2(args)
    if args.command == "run-machine-labeling-v2":
        return cmd_run_machine_labeling_v2(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    semantic_review.add_argument("--use-cowgirl-candidate-score-v10", default="false")
    semantic_review.add_argument("--use-cowgirl-candidate-score-v11", default="false")
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

    vam_review_package = subparsers.add_parser("build-vam-review-package", help="Build a practical VaM scene/time review package for a semantic review JSONL.")
    vam_review_package.add_argument("--review", required=True)
    vam_review_package.add_argument("--run-dir", required=True)
    vam_review_package.add_argument("--source-run", required=True)
    vam_review_package.add_argument("--out-dir", required=True)
    vam_review_package.add_argument("--attempt-timeline-segments", default="true")

    launch_review_ui = subparsers.add_parser("launch-review-ui", help="Launch local semantic review UI using only stdlib HTTP serving.")
    launch_review_ui.add_argument("--run-dir", required=True)
    launch_review_ui.add_argument("--review-dir", required=True)
    launch_review_ui.add_argument("--host", default="127.0.0.1")
    launch_review_ui.add_argument("--port", type=int, default=8765)

    static_review_ui = subparsers.add_parser("build-static-review-ui", help="Build static semantic review UI files for a review folder.")
    static_review_ui.add_argument("--run-dir", required=True)
    static_review_ui.add_argument("--review-dir", required=True)
    static_review_ui.add_argument("--out-dir", required=True)

    digital_twin_previews = subparsers.add_parser("render-digital-twin-review-previews-v0", help="Render audit-only skeleton/contact-sheet previews for a semantic review batch.")
    digital_twin_previews.add_argument("--run-dir", required=True)
    digital_twin_previews.add_argument("--review-dir", required=True)
    digital_twin_previews.add_argument("--out-dir", required=True)

    digital_twin_previews_v1 = subparsers.add_parser("render-digital-twin-previews-v1", help="Render animated audit-only digital-twin GIF/MP4/contact-sheet previews.")
    digital_twin_previews_v1.add_argument("--run-dir", required=True)
    digital_twin_previews_v1.add_argument("--review-dir", required=True)
    digital_twin_previews_v1.add_argument("--out-dir", required=True)
    digital_twin_previews_v1.add_argument("--fps", type=int, default=12)
    digital_twin_previews_v1.add_argument("--width", type=int, default=960)
    digital_twin_previews_v1.add_argument("--height", type=int, default=720)
    digital_twin_previews_v1.add_argument("--frames", type=int, default=32)
    digital_twin_previews_v1.add_argument("--make-gif", default="true")
    digital_twin_previews_v1.add_argument("--make-mp4", default="auto")
    digital_twin_previews_v1.add_argument("--view", default="side", choices=["side", "front", "top", "three_quarter"])

    visual_judge_requests = subparsers.add_parser("build-visual-judge-requests-v0", help="Build local visual-judge request manifests from digital-twin previews.")
    visual_judge_requests.add_argument("--review-dir", required=True)
    visual_judge_requests.add_argument("--preview-dir", required=True)
    visual_judge_requests.add_argument("--out-jsonl", required=True)
    visual_judge_requests.add_argument("--mode", default="blind")

    capture_requests = subparsers.add_parser("build-vam-capture-requests-v0", help="Build manual VaM viewport capture request manifest.")
    capture_requests.add_argument("--review-dir", required=True)
    capture_requests.add_argument("--out-jsonl", required=True)
    capture_requests.add_argument("--output-root", required=True)
    capture_requests.add_argument("--frame-count", type=int, default=16)
    capture_requests.add_argument("--duration-seconds", type=float, default=4.0)

    reality_capture = subparsers.add_parser("run-vam-reality-capture-v0", help="Call optional local VaM BepInEx capture bridge.")
    reality_capture.add_argument("--requests", required=True)
    reality_capture.add_argument("--bridge-url", required=True)
    reality_capture.add_argument("--mode", default="status_only", choices=["status_only", "manual_step", "batch_current"])
    reality_capture.add_argument("--out", required=True)

    capture_sheets = subparsers.add_parser("build-vam-capture-contact-sheets-v0", help="Build contact sheets from real VaM capture frames.")
    capture_sheets.add_argument("--capture-results", required=True)
    capture_sheets.add_argument("--out-dir", required=True)

    manual_pose_import = subparsers.add_parser("import-manual-pose-captures-v1", help="Import VaM SkeletonPoseCaptureTool JSON snapshots.")
    manual_pose_import.add_argument("--input-dir", required=True)
    manual_pose_import.add_argument("--out-jsonl", required=True)
    manual_pose_import.add_argument("--report", required=True)

    manual_pose_report = subparsers.add_parser("report-manual-pose-captures-v1", help="Summarize imported manual pose captures.")
    manual_pose_report.add_argument("--captures", required=True)
    manual_pose_report.add_argument("--out", required=True)

    manual_pose_extract = subparsers.add_parser("extract-manual-pose-captures-v1", help="Extract manual VaM pose capture ZIP into an ignored run folder.")
    manual_pose_extract.add_argument("--zip", required=True)
    manual_pose_extract.add_argument("--out-dir", required=True)

    manual_pose_explanations = subparsers.add_parser("parse-manual-pose-explanations-v1", help="Parse human explanation notes for manual pose captures.")
    manual_pose_explanations.add_argument("--explanations", required=True)
    manual_pose_explanations.add_argument("--out-json", required=True)
    manual_pose_explanations.add_argument("--out-yaml", required=True)
    manual_pose_explanations.add_argument("--report", required=True)

    manual_pose_gt = subparsers.add_parser("build-manual-pose-ground-truth-v1", help="Merge manual pose captures, screenshots, and human labels.")
    manual_pose_gt.add_argument("--capture-dir", required=True)
    manual_pose_gt.add_argument("--human-labels", required=True)
    manual_pose_gt.add_argument("--out-jsonl", required=True)
    manual_pose_gt.add_argument("--out-csv", required=True)
    manual_pose_gt.add_argument("--report", required=True)

    manual_pose_gt_report = subparsers.add_parser("report-manual-pose-ground-truth-v1", help="Write family-specific manual pose ground-truth reports.")
    manual_pose_gt_report.add_argument("--ground-truth", required=True)
    manual_pose_gt_report.add_argument("--out-dir", required=True)

    manual_pose_gt_gallery = subparsers.add_parser("build-manual-pose-ground-truth-gallery-v1", help="Build HTML gallery for manual pose ground-truth captures.")
    manual_pose_gt_gallery.add_argument("--ground-truth", required=True)
    manual_pose_gt_gallery.add_argument("--out-html", required=True)

    visual_judge_requests_v1 = subparsers.add_parser("build-visual-judge-requests-v1", help="Build local visual judge requests preferring real VaM captures.")
    visual_judge_requests_v1.add_argument("--review-dir", required=True)
    visual_judge_requests_v1.add_argument("--vam-capture-sheets", required=True)
    visual_judge_requests_v1.add_argument("--digital-twin-previews", required=True)
    visual_judge_requests_v1.add_argument("--out-jsonl", required=True)
    visual_judge_requests_v1.add_argument("--mode", default="blind")

    lmstudio_judge = subparsers.add_parser("run-lmstudio-vlm-judge-v0", help="Run or dry-run local LM Studio VLM judge.")
    lmstudio_judge.add_argument("--requests", required=True)
    lmstudio_judge.add_argument("--base-url", required=True)
    lmstudio_judge.add_argument("--model", default="nsfwvision-v4-qwen3.5-9b")
    lmstudio_judge.add_argument("--out-jsonl", required=True)
    lmstudio_judge.add_argument("--out-raw-dir", required=True)
    lmstudio_judge.add_argument("--dry-run", default="true")

    visual_calibration = subparsers.add_parser("build-visual-judge-calibration-set-v1", help="Build a local visual judge calibration set.")
    visual_calibration.add_argument("--run-dir", required=True)
    visual_calibration.add_argument("--out-dir", required=True)

    visual_trust = subparsers.add_parser("evaluate-vlm-visual-judge-v1", help="Evaluate VLM visual judge calibration and write trust gate.")
    visual_trust.add_argument("--calibration-set", required=True)
    visual_trust.add_argument("--base-url", required=True)
    visual_trust.add_argument("--model", default="nsfwvision-v4-qwen3.5-9b")
    visual_trust.add_argument("--out-dir", required=True)
    visual_trust.add_argument("--dry-run", default="true")

    multisignal = subparsers.add_parser("build-multisignal-review-priorities-v0", help="Combine heuristic, ML, and VLM signals into review priorities.")
    multisignal.add_argument("--run-dir", required=True)
    multisignal.add_argument("--review-dir", required=True)
    multisignal.add_argument("--model-scores", required=True)
    multisignal.add_argument("--visual-results", required=True)
    multisignal.add_argument("--out-jsonl", required=True)
    multisignal.add_argument("--report", required=True)

    translate_intent = subparsers.add_parser("translate-motion-intent-v1", help="Translate prompt into top-down motion intent plan.")
    translate_intent.add_argument("--prompt", required=True)
    translate_intent.add_argument("--ontology", required=True)
    translate_intent.add_argument("--phrases", required=True)
    translate_intent.add_argument("--out", required=True)

    sourcebook = subparsers.add_parser("ingest-semantik-sourcebook-v2", help="Extract and register the Semantik master DOCX as ontology sourcebook.")
    sourcebook.add_argument("--source-docx", required=True)
    sourcebook.add_argument("--out-dir", required=True)
    sourcebook.add_argument("--report", required=True)

    stickman_library = subparsers.add_parser("build-semantic-stickman-pose-library-v1", help="Build schematic semantic stickman pose library from ontology.")
    stickman_library.add_argument("--ontology", required=True)
    stickman_library.add_argument("--out-json", required=True)
    stickman_library.add_argument("--report", required=True)

    stickman_examples = subparsers.add_parser("build-semantic-motion-examples-v1", help="Build semantic stickman motion examples from pose library.")
    stickman_examples.add_argument("--pose-library", required=True)
    stickman_examples.add_argument("--ontology", required=True)
    stickman_examples.add_argument("--out-json", required=True)
    stickman_examples.add_argument("--report", required=True)

    stickman_render = subparsers.add_parser("render-semantic-stickman-previews-v1", help="Render semantic stickman GIF/contact sheet previews.")
    stickman_render.add_argument("--motion-examples", required=True)
    stickman_render.add_argument("--out-dir", required=True)
    stickman_render.add_argument("--width", type=int, default=1280)
    stickman_render.add_argument("--height", type=int, default=720)
    stickman_render.add_argument("--fps", type=int, default=12)
    stickman_render.add_argument("--make-gif", default="true")
    stickman_render.add_argument("--make-contact-sheet", default="true")

    stickman_validate = subparsers.add_parser("validate-semantic-stickman-examples-v1", help="Validate semantic stickman examples against ontology grammar.")
    stickman_validate.add_argument("--motion-examples", required=True)
    stickman_validate.add_argument("--ontology", required=True)
    stickman_validate.add_argument("--out", required=True)

    stickman_gallery = subparsers.add_parser("build-semantic-stickman-gallery-v1", help="Build semantic stickman HTML/Markdown gallery.")
    stickman_gallery.add_argument("--preview-dir", required=True)
    stickman_gallery.add_argument("--out-html", required=True)
    stickman_gallery.add_argument("--out-md", required=True)

    stickman_render_v2 = subparsers.add_parser("render-semantic-stickman-previews-v2", help="Render semantic stickman previews with labels, partner targets, alignment, and support context.")
    stickman_render_v2.add_argument("--motion-examples", required=True)
    stickman_render_v2.add_argument("--out-dir", required=True)
    stickman_render_v2.add_argument("--width", type=int, default=1600)
    stickman_render_v2.add_argument("--height", type=int, default=900)
    stickman_render_v2.add_argument("--fps", type=int, default=12)
    stickman_render_v2.add_argument("--make-gif", default="true")
    stickman_render_v2.add_argument("--make-contact-sheet", default="true")
    stickman_render_v2.add_argument("--show-labels", default="true")
    stickman_render_v2.add_argument("--show-partner", default="true")
    stickman_render_v2.add_argument("--show-alignment", default="true")
    stickman_render_v2.add_argument("--show-support-targets", default="true")

    stickman_validate_v2 = subparsers.add_parser("validate-semantic-stickman-examples-v2", help="Validate semantic stickman v2 previews for partner/alignment/support context.")
    stickman_validate_v2.add_argument("--motion-examples", required=True)
    stickman_validate_v2.add_argument("--preview-dir", required=True)
    stickman_validate_v2.add_argument("--ontology", required=True)
    stickman_validate_v2.add_argument("--out", required=True)

    stickman_gallery_v2 = subparsers.add_parser("build-semantic-stickman-gallery-v2", help="Build semantic stickman v2 HTML/Markdown gallery with legends and warnings.")
    stickman_gallery_v2.add_argument("--preview-dir", required=True)
    stickman_gallery_v2.add_argument("--out-html", required=True)
    stickman_gallery_v2.add_argument("--out-md", required=True)

    stickman_examples_v2 = subparsers.add_parser("build-semantic-motion-examples-v2-contact-aware", help="Build contact-aware semantic motion examples around partner interaction targets.")
    stickman_examples_v2.add_argument("--pose-library", required=True)
    stickman_examples_v2.add_argument("--ontology", required=True)
    stickman_examples_v2.add_argument("--out-json", required=True)
    stickman_examples_v2.add_argument("--report", required=True)

    stickman_render_v3 = subparsers.add_parser("render-semantic-stickman-previews-v3", help="Render contact-aware semantic stickman previews with validity overlays.")
    stickman_render_v3.add_argument("--motion-examples", required=True)
    stickman_render_v3.add_argument("--out-dir", required=True)
    stickman_render_v3.add_argument("--width", type=int, default=1600)
    stickman_render_v3.add_argument("--height", type=int, default=900)
    stickman_render_v3.add_argument("--fps", type=int, default=12)
    stickman_render_v3.add_argument("--make-gif", default="true")
    stickman_render_v3.add_argument("--make-contact-sheet", default="true")
    stickman_render_v3.add_argument("--show-labels", default="true")
    stickman_render_v3.add_argument("--show-partner", default="true")
    stickman_render_v3.add_argument("--show-alignment", default="true")
    stickman_render_v3.add_argument("--show-support-targets", default="true")
    stickman_render_v3.add_argument("--show-contact-zone", default="true")
    stickman_render_v3.add_argument("--show-alignment-tolerance", default="true")
    stickman_render_v3.add_argument("--show-validity-overlay", default="true")
    stickman_render_v3.add_argument("--contact-aware", default="true")

    stickman_validate_v3 = subparsers.add_parser("validate-semantic-stickman-examples-v3", help="Validate contact-aware semantic stickman previews.")
    stickman_validate_v3.add_argument("--motion-examples", required=True)
    stickman_validate_v3.add_argument("--preview-dir", required=True)
    stickman_validate_v3.add_argument("--ontology", required=True)
    stickman_validate_v3.add_argument("--out", required=True)

    stickman_gallery_v3 = subparsers.add_parser("build-semantic-stickman-gallery-v3", help="Build semantic stickman v3 HTML/Markdown gallery with contact-validity summaries.")
    stickman_gallery_v3.add_argument("--preview-dir", required=True)
    stickman_gallery_v3.add_argument("--out-html", required=True)
    stickman_gallery_v3.add_argument("--out-md", required=True)

    vam_semantic_preview = subparsers.add_parser("export-vam-semantic-preview-v0", help="Export review-only native Timeline clips from contact-aware semantic motion examples.")
    vam_semantic_preview.add_argument("--motion-examples", required=True)
    vam_semantic_preview.add_argument("--out-dir", required=True)
    vam_semantic_preview.add_argument("--duration", type=float, default=4.0)
    vam_semantic_preview.add_argument("--fps", type=int, default=60)

    vam_semantic_preview_validate = subparsers.add_parser("validate-vam-semantic-preview-v0", help="Validate review-only VaM semantic preview package.")
    vam_semantic_preview_validate.add_argument("--preview-dir", required=True)
    vam_semantic_preview_validate.add_argument("--out", required=True)

    manual_gt_timeline = subparsers.add_parser("export-manual-gt-timeline-examples-v1", help="Export review-only Timeline examples from real manual pose ground truth.")
    manual_gt_timeline.add_argument("--ground-truth", required=True)
    manual_gt_timeline.add_argument("--out-dir", required=True)
    manual_gt_timeline.add_argument("--duration", type=float, default=4.0)
    manual_gt_timeline.add_argument("--fps", type=int, default=60)
    manual_gt_timeline.add_argument("--copy-to-vam", default="false")

    manual_gt_timeline_validate = subparsers.add_parser("validate-manual-gt-timeline-examples-v1", help="Validate manual-ground-truth Timeline examples.")
    manual_gt_timeline_validate.add_argument("--preview-dir", required=True)
    manual_gt_timeline_validate.add_argument("--out", required=True)

    manual_gt_timeline_v2 = subparsers.add_parser("export-manual-gt-timeline-examples-v2", help="Export sparse review-only Timeline examples with captured rotations.")
    manual_gt_timeline_v2.add_argument("--ground-truth", required=True)
    manual_gt_timeline_v2.add_argument("--out-dir", required=True)
    manual_gt_timeline_v2.add_argument("--duration", type=float, default=4.0)
    manual_gt_timeline_v2.add_argument("--keyframe-rate", type=float, default=2.0)
    manual_gt_timeline_v2.add_argument("--copy-to-vam", default="false")
    manual_gt_timeline_v2.add_argument("--include-rotations", default="true")
    manual_gt_timeline_v2.add_argument("--allow-dense-export", default="false")

    manual_gt_timeline_validate_v2 = subparsers.add_parser("validate-manual-gt-timeline-examples-v2", help="Validate sparse manual-ground-truth Timeline examples with rotations.")
    manual_gt_timeline_validate_v2.add_argument("--preview-dir", required=True)
    manual_gt_timeline_validate_v2.add_argument("--out", required=True)
    manual_gt_timeline_validate_v2.add_argument("--allow-dense-export", default="false")

    manual_gt_timeline_v3 = subparsers.add_parser("export-manual-gt-timeline-examples-v3", help="Export sparse manual GT Timeline examples with hipControl primary mapping.")
    manual_gt_timeline_v3.add_argument("--ground-truth", required=True)
    manual_gt_timeline_v3.add_argument("--out-dir", required=True)
    manual_gt_timeline_v3.add_argument("--duration", type=float, default=4.0)
    manual_gt_timeline_v3.add_argument("--keyframe-rate", type=float, default=1.0)
    manual_gt_timeline_v3.add_argument("--copy-to-vam", default="false")
    manual_gt_timeline_v3.add_argument("--include-rotations", default="true")
    manual_gt_timeline_v3.add_argument("--require-hip-control", default="true")
    manual_gt_timeline_v3.add_argument("--allow-high-key-density", default="false")
    manual_gt_timeline_v3.add_argument("--allow-dense-export", default="false")

    manual_gt_timeline_validate_v3 = subparsers.add_parser("validate-manual-gt-timeline-examples-v3", help="Validate manual GT Timeline examples with hipControl v3 mapping.")
    manual_gt_timeline_validate_v3.add_argument("--preview-dir", required=True)
    manual_gt_timeline_validate_v3.add_argument("--out", required=True)
    manual_gt_timeline_validate_v3.add_argument("--allow-high-key-density", default="false")
    manual_gt_timeline_validate_v3.add_argument("--allow-dense-export", default="false")

    manual_gt_timeline_v4 = subparsers.add_parser("export-manual-gt-timeline-examples-v4", help="Export manual GT Timeline examples with v4 amplitude profiles.")
    manual_gt_timeline_v4.add_argument("--ground-truth", required=True)
    manual_gt_timeline_v4.add_argument("--out-dir", required=True)
    manual_gt_timeline_v4.add_argument("--duration", type=float, default=4.0)
    manual_gt_timeline_v4.add_argument("--keyframe-rate", type=float, default=1.0)
    manual_gt_timeline_v4.add_argument("--copy-to-vam", default="false")
    manual_gt_timeline_v4.add_argument("--include-rotations", default="true")
    manual_gt_timeline_v4.add_argument("--require-hip-control", default="true")
    manual_gt_timeline_v4.add_argument("--amplitude-profile", required=True)
    manual_gt_timeline_v4.add_argument("--allow-high-key-density", default="false")
    manual_gt_timeline_v4.add_argument("--allow-dense-export", default="false")

    manual_gt_timeline_validate_v4 = subparsers.add_parser("validate-manual-gt-timeline-examples-v4", help="Validate manual GT Timeline examples with v4 amplitude profiles.")
    manual_gt_timeline_validate_v4.add_argument("--preview-dir", required=True)
    manual_gt_timeline_validate_v4.add_argument("--out", required=True)
    manual_gt_timeline_validate_v4.add_argument("--allow-high-key-density", default="false")
    manual_gt_timeline_validate_v4.add_argument("--allow-dense-export", default="false")

    pose_first = subparsers.add_parser("resolve-pose-first-semantics-v1", help="Resolve candidates with top-down pose-first motion ontology rules.")
    pose_first.add_argument("--run-dir", required=True)
    pose_first.add_argument("--pose-semantics", required=True)
    pose_first.add_argument("--relative-features", required=True)
    pose_first.add_argument("--interaction-semantics", required=True)
    pose_first.add_argument("--candidate-db", required=True)
    pose_first.add_argument("--rules", required=True)
    pose_first.add_argument("--out-jsonl", required=True)
    pose_first.add_argument("--report", required=True)

    ontology_align = subparsers.add_parser("align-candidates-to-motion-ontology-v1", help="Align candidate DBs with top-down motion ontology.")
    ontology_align.add_argument("--run-dir", required=True)
    ontology_align.add_argument("--ontology", required=True)
    ontology_align.add_argument("--semantic-db", required=True)
    ontology_align.add_argument("--cowgirl-db", required=True)
    ontology_align.add_argument("--resolved", required=True)
    ontology_align.add_argument("--out-jsonl", required=True)
    ontology_align.add_argument("--report", required=True)

    parameter_calibration = subparsers.add_parser("calibrate-motion-parameters-v1", help="Calibrate numeric motion parameter ranges from ontology-consistent human-reviewed candidates.")
    parameter_calibration.add_argument("--run-dir", required=True)
    parameter_calibration.add_argument("--ontology", required=True)
    parameter_calibration.add_argument("--resolved", required=True)
    parameter_calibration.add_argument("--relative-features", required=True)
    parameter_calibration.add_argument("--trajectory-features", required=True)
    parameter_calibration.add_argument("--human-ledger", required=True)
    parameter_calibration.add_argument("--out-json", required=True)
    parameter_calibration.add_argument("--report", required=True)

    ingest_review_ui = subparsers.add_parser("ingest-review-ui-answers", help="Ingest audit-only answers exported from the local review UI.")
    ingest_review_ui.add_argument("--answers", required=True)
    ingest_review_ui.add_argument("--review-dir", required=True)
    ingest_review_ui.add_argument("--out-ledger", required=True)
    ingest_review_ui.add_argument("--report", required=True)
    ingest_review_ui.add_argument("--overwrite", default="false")

    human_ml_labels = subparsers.add_parser("build-human-reviewed-ml-labels-v1", help="Build Cowgirl ML labels from human review artifacts only.")
    human_ml_labels.add_argument("--run-dir", required=True)
    human_ml_labels.add_argument("--human-ledger", required=True)
    human_ml_labels.add_argument("--out-jsonl", required=True)
    human_ml_labels.add_argument("--report", required=True)

    cowgirl_ml_features = subparsers.add_parser("build-cowgirl-ml-feature-table-v1", help="Build supervised Cowgirl ML feature table from human-reviewed labels.")
    cowgirl_ml_features.add_argument("--run-dir", required=True)
    cowgirl_ml_features.add_argument("--labels", required=True)
    cowgirl_ml_features.add_argument("--relative-features", required=True)
    cowgirl_ml_features.add_argument("--trajectory-features", required=True)
    cowgirl_ml_features.add_argument("--pose-features", required=True)
    cowgirl_ml_features.add_argument("--pose-semantics", required=True)
    cowgirl_ml_features.add_argument("--partner-relative-features", required=True)
    cowgirl_ml_features.add_argument("--interaction-semantics", required=True)
    cowgirl_ml_features.add_argument("--semantic-actions", required=True)
    cowgirl_ml_features.add_argument("--candidate-db", required=True)
    cowgirl_ml_features.add_argument("--out-npz", required=True)
    cowgirl_ml_features.add_argument("--out-meta", required=True)
    cowgirl_ml_features.add_argument("--report", required=True)

    cowgirl_ml_split = subparsers.add_parser("split-cowgirl-ml-dataset-v1", help="Create grouped leakage-safe Cowgirl ML splits.")
    cowgirl_ml_split.add_argument("--feature-table", required=True)
    cowgirl_ml_split.add_argument("--metadata", required=True)
    cowgirl_ml_split.add_argument("--out-dir", required=True)
    cowgirl_ml_split.add_argument("--group-by", default="source_scene_file")
    cowgirl_ml_split.add_argument("--seed", type=int, default=42)

    cowgirl_ml_train = subparsers.add_parser("train-cowgirl-ml-baseline-v1", help="Train small Cowgirl review-assist baseline models.")
    cowgirl_ml_train.add_argument("--feature-table", required=True)
    cowgirl_ml_train.add_argument("--metadata", required=True)
    cowgirl_ml_train.add_argument("--splits", required=True)
    cowgirl_ml_train.add_argument("--out-dir", required=True)

    cowgirl_ml_score = subparsers.add_parser("score-clean-v3-with-cowgirl-model-v1", help="Score clean_v3 candidates with Cowgirl review-assist model.")
    cowgirl_ml_score.add_argument("--run-dir", required=True)
    cowgirl_ml_score.add_argument("--model-dir", required=True)
    cowgirl_ml_score.add_argument("--feature-source", default="all_candidates")
    cowgirl_ml_score.add_argument("--out-jsonl", required=True)
    cowgirl_ml_score.add_argument("--report", required=True)

    cowgirl_ml_review = subparsers.add_parser("export-ml-assisted-cowgirl-review-v1", help="Export strict novelty ML-assisted Cowgirl review batch.")
    cowgirl_ml_review.add_argument("--run-dir", required=True)
    cowgirl_ml_review.add_argument("--model-scores", required=True)
    cowgirl_ml_review.add_argument("--reviewed-index", required=True)
    cowgirl_ml_review.add_argument("--out-dir", required=True)
    cowgirl_ml_review.add_argument("--count", type=int, default=20)
    cowgirl_ml_review.add_argument("--max-per-scene", type=int, default=2)
    cowgirl_ml_review.add_argument("--max-per-sample", type=int, default=1)
    cowgirl_ml_review.add_argument("--build-vam-package", default="true")
    cowgirl_ml_review.add_argument("--build-static-ui", default="true")

    cowgirl_ml_labels_v2 = subparsers.add_parser("build-cowgirl-ml-labels-v2", help="Build Cowgirl ML v2 labels from human review and manual GT only.")
    cowgirl_ml_labels_v2.add_argument("--base-run", required=True)
    cowgirl_ml_labels_v2.add_argument("--new-run", required=True)
    cowgirl_ml_labels_v2.add_argument("--human-ledger", required=True)
    cowgirl_ml_labels_v2.add_argument("--manual-gt", required=True)
    cowgirl_ml_labels_v2.add_argument("--out-jsonl", required=True)
    cowgirl_ml_labels_v2.add_argument("--report", required=True)

    cowgirl_ml_labels_v3 = subparsers.add_parser("build-cowgirl-ml-labels-v3", help="Build Cowgirl ML v3 item-level labels from human review.")
    cowgirl_ml_labels_v3.add_argument("--new-run", required=True)
    cowgirl_ml_labels_v3.add_argument("--out-jsonl", required=True)
    cowgirl_ml_labels_v3.add_argument("--report", required=True)

    cowgirl_ml_features_v2 = subparsers.add_parser("build-cowgirl-ml-feature-table-v2", help="Build Cowgirl ML v2 feature table with cycle/gate/controller features.")
    cowgirl_ml_features_v2.add_argument("--labels", required=True)
    cowgirl_ml_features_v2.add_argument("--pose-resolved", required=True)
    cowgirl_ml_features_v2.add_argument("--cycle-features", required=True)
    cowgirl_ml_features_v2.add_argument("--motion-resolved", required=True)
    cowgirl_ml_features_v2.add_argument("--candidates", required=True)
    cowgirl_ml_features_v2.add_argument("--manual-gt", required=True)
    cowgirl_ml_features_v2.add_argument("--out-npz", required=True)
    cowgirl_ml_features_v2.add_argument("--out-meta", required=True)
    cowgirl_ml_features_v2.add_argument("--report", required=True)

    cowgirl_ml_train_v2 = subparsers.add_parser("train-cowgirl-ml-v2", help="Train Cowgirl ML v2 review-ranker models.")
    cowgirl_ml_train_v2.add_argument("--feature-table", required=True)
    cowgirl_ml_train_v2.add_argument("--metadata", required=True)
    cowgirl_ml_train_v2.add_argument("--out-dir", required=True)

    cowgirl_ml_score_v2 = subparsers.add_parser("score-new-scenes-cowgirl-ml-v2", help="Score new scenes with Cowgirl ML v2 review-ranker.")
    cowgirl_ml_score_v2.add_argument("--model-dir", required=True)
    cowgirl_ml_score_v2.add_argument("--feature-table", required=True)
    cowgirl_ml_score_v2.add_argument("--metadata", required=True)
    cowgirl_ml_score_v2.add_argument("--out-jsonl", required=True)
    cowgirl_ml_score_v2.add_argument("--report", required=True)

    cowgirl_ml_review_v2 = subparsers.add_parser("export-ml-assisted-cowgirl-review-v2", help="Export ML-assisted Cowgirl review v2 batch.")
    cowgirl_ml_review_v2.add_argument("--new-run", required=True)
    cowgirl_ml_review_v2.add_argument("--scores", required=True)
    cowgirl_ml_review_v2.add_argument("--candidates", required=True)
    cowgirl_ml_review_v2.add_argument("--out-dir", required=True)
    cowgirl_ml_review_v2.add_argument("--count", type=int, default=30)
    cowgirl_ml_review_v2.add_argument("--build-static-ui", default="true")
    cowgirl_ml_review_v2.add_argument("--build-vam-package", default="true")

    relational_features_v1 = subparsers.add_parser("extract-relational-semantic-features-v1", help="Extract actor/partner contact, target, alignment, and axis features.")
    relational_features_v1.add_argument("--run-dir", required=True)
    relational_features_v1.add_argument("--pair-windows", required=True)
    relational_features_v1.add_argument("--pair-features", required=True)
    relational_features_v1.add_argument("--sample-index", required=True)
    relational_features_v1.add_argument("--controller-map", required=True)
    relational_features_v1.add_argument("--out-jsonl", required=True)
    relational_features_v1.add_argument("--report", required=True)

    rig_anatomy_features = subparsers.add_parser("extract-rig-anatomy-features-v1", help="Map controller cycle features into semantic rig-anatomy region features.")
    rig_anatomy_features.add_argument("--run-dir", required=True)
    rig_anatomy_features.add_argument("--anatomy", required=True)
    rig_anatomy_features.add_argument("--roles", required=True)
    rig_anatomy_features.add_argument("--out-jsonl", required=True)
    rig_anatomy_features.add_argument("--report", required=True)

    nlp_lexicon = subparsers.add_parser("build-nlp-lexicon-v1", help="Build active manual NLP lexicon plus inactive external candidates.")
    nlp_lexicon.add_argument("--manual", required=True)
    nlp_lexicon.add_argument("--sources", required=True)
    nlp_lexicon.add_argument("--out", required=True)
    nlp_lexicon.add_argument("--report", required=True)
    nlp_lexicon.add_argument("--allow-web", default="false")

    nlp_tokens = subparsers.add_parser("resolve-nlp-tokens-v1", help="Resolve prompt terms into semantic/anatomy/action/style tokens.")
    nlp_tokens.add_argument("--prompt", required=True)
    nlp_tokens.add_argument("--lexicon", required=True)
    nlp_tokens.add_argument("--component-ontology", required=True)
    nlp_tokens.add_argument("--out", required=True)

    nlp_intent = subparsers.add_parser("build-motion-intent-from-prompt-v1", help="Build component MotionIntentPlan from prompt without Timeline export.")
    nlp_intent.add_argument("--prompt", required=True)
    nlp_intent.add_argument("--lexicon", required=True)
    nlp_intent.add_argument("--component-ontology", required=True)
    nlp_intent.add_argument("--out", required=True)

    web_context = subparsers.add_parser("collect-web-motion-context-v1", help="Collect conservative web-context research cards.")
    web_context.add_argument("--topics", required=True)
    web_context.add_argument("--out-dir", required=True)
    web_context.add_argument("--allow-web", default="false")
    web_context.add_argument("--max-sources-per-category", type=int, default=10)

    web_patches = subparsers.add_parser("build-web-context-ontology-patches-v1", help="Build inactive ontology patch candidates from web context cards.")
    web_patches.add_argument("--research-dir", required=True)
    web_patches.add_argument("--current-ontology", required=True)
    web_patches.add_argument("--current-anatomy", required=True)
    web_patches.add_argument("--out-yaml", required=True)
    web_patches.add_argument("--report", required=True)

    research_client = subparsers.add_parser("build-research-client-v0", help="Build read-only local research client skeleton.")
    research_client.add_argument("--run-dir", required=True)
    research_client.add_argument("--new-run", required=True)
    research_client.add_argument("--out-dir", required=True)

    eval_ml_review = subparsers.add_parser("evaluate-ml-assisted-review-v1", help="Evaluate ML-assisted review v1 against human answers.")
    eval_ml_review.add_argument("--review-dir", required=True)
    eval_ml_review.add_argument("--model-scores", required=True)
    eval_ml_review.add_argument("--answers", required=True)
    eval_ml_review.add_argument("--out", required=True)

    active_learning = subparsers.add_parser("run-cowgirl-ml-active-learning-v2", help="Run Cowgirl ML active-learning loop after v1 answers exist.")
    active_learning.add_argument("--run-dir", required=True)
    active_learning.add_argument("--review-dir", required=True)
    active_learning.add_argument("--out-dir", required=True)

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

    bj_oral_domain = subparsers.add_parser("classify-bj-oral-domain", help="Classify BJ/oral semantic-family candidates and preserve them outside Cowgirl.")
    bj_oral_domain.add_argument("--run-dir", required=True)
    bj_oral_domain.add_argument("--relative-features", required=True)
    bj_oral_domain.add_argument("--trajectory-features", required=True)
    bj_oral_domain.add_argument("--relative-reference-matches", required=True)
    bj_oral_domain.add_argument("--cowgirl-core-controllers", required=True)
    bj_oral_domain.add_argument("--out-jsonl", required=True)
    bj_oral_domain.add_argument("--report", required=True)

    bj_oral_guard = subparsers.add_parser("audit-bj-oral-trap-guard", help="Compatibility wrapper for classify-bj-oral-domain.")
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

    cowgirl_score_v11 = subparsers.add_parser("score-cowgirl-candidates-v11", help="Score Cowgirl candidates with calibrated core gates and BJ/oral semantic-family preservation.")
    cowgirl_score_v11.add_argument("--run-dir", required=True)
    cowgirl_score_v11.add_argument("--relative-reference-matches", required=True)
    cowgirl_score_v11.add_argument("--relative-features", required=True)
    cowgirl_score_v11.add_argument("--trajectory-features", required=True)
    cowgirl_score_v11.add_argument("--body-quality", required=True)
    cowgirl_score_v11.add_argument("--rider-receiver-scores", required=True)
    cowgirl_score_v11.add_argument("--pose-export-validity", required=True)
    cowgirl_score_v11.add_argument("--controller-validity", required=True)
    cowgirl_score_v11.add_argument("--pose-anchor-completeness", required=True)
    cowgirl_score_v11.add_argument("--controller-orientation-validity", required=True)
    cowgirl_score_v11.add_argument("--controller-distance-validity", required=True)
    cowgirl_score_v11.add_argument("--cowgirl-core-controllers", required=True)
    cowgirl_score_v11.add_argument("--bj-oral-domain", default=None)
    cowgirl_score_v11.add_argument("--bj-oral-trap-guard", default=None)
    cowgirl_score_v11.add_argument("--features", required=True)
    cowgirl_score_v11.add_argument("--out-jsonl", required=True)
    cowgirl_score_v11.add_argument("--report", required=True)

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

    candidate_db_v3 = subparsers.add_parser("build-cowgirl-candidate-db-v3", help="Build Cowgirl candidate DB v3 with semantic-family fields.")
    candidate_db_v3.add_argument("--run-dir", required=True)
    candidate_db_v3.add_argument("--candidate-scores", required=True)
    candidate_db_v3.add_argument("--relative-features", required=True)
    candidate_db_v3.add_argument("--trajectory-features", required=True)
    candidate_db_v3.add_argument("--body-quality", required=True)
    candidate_db_v3.add_argument("--pose-anchor-completeness", required=True)
    candidate_db_v3.add_argument("--controller-validity", required=True)
    candidate_db_v3.add_argument("--controller-orientation-validity", required=True)
    candidate_db_v3.add_argument("--controller-distance-validity", required=True)
    candidate_db_v3.add_argument("--cowgirl-core-controllers", required=True)
    candidate_db_v3.add_argument("--bj-oral-domain", required=True)
    candidate_db_v3.add_argument("--out-jsonl", required=True)
    candidate_db_v3.add_argument("--out-csv", required=True)
    candidate_db_v3.add_argument("--report", required=True)

    semantic_db_v0 = subparsers.add_parser("build-semantic-candidate-db-v0", help="Build global multi-family semantic candidate inventory v0.")
    semantic_db_v0.add_argument("--run-dir", required=True)
    semantic_db_v0.add_argument("--cowgirl-db", required=True)
    semantic_db_v0.add_argument("--bj-oral-domain", required=True)
    semantic_db_v0.add_argument("--relative-features", required=True)
    semantic_db_v0.add_argument("--trajectory-features", required=True)
    semantic_db_v0.add_argument("--out-jsonl", required=True)
    semantic_db_v0.add_argument("--out-csv", required=True)
    semantic_db_v0.add_argument("--report", required=True)

    primitives = subparsers.add_parser("extract-cowgirl-motion-primitives-v0", help="Extract abstract Cowgirl relative motion primitives from generation-safe candidates.")
    primitives.add_argument("--candidate-db", required=True)
    primitives.add_argument("--relative-features", required=True)
    primitives.add_argument("--trajectory-features", required=True)
    primitives.add_argument("--relative-index", required=True)
    primitives.add_argument("--out-jsonl", required=True)
    primitives.add_argument("--out-report", required=True)

    primitive_groups = subparsers.add_parser("group-cowgirl-motion-primitives-v0", help="Group abstract Cowgirl motion primitives by subtype, trajectory, rhythm, and amplitude.")
    primitive_groups.add_argument("--primitives", required=True)
    primitive_groups.add_argument("--out-json", required=True)
    primitive_groups.add_argument("--report", required=True)

    draft_plan = subparsers.add_parser("draft-motion-plan-v0", help="Draft a rule-based semantic motion plan. This is not final text-to-animation.")
    draft_plan.add_argument("--prompt", required=True)
    draft_plan.add_argument("--out", required=True)

    retrieve = subparsers.add_parser("retrieve-primitives-for-plan-v0", help="Retrieve abstract primitives matching a semantic motion plan without exporting Timeline.")
    retrieve.add_argument("--plan", required=True)
    retrieve.add_argument("--primitive-groups", required=True)
    retrieve.add_argument("--primitives", required=True)
    retrieve.add_argument("--out", required=True)
    retrieve.add_argument("--report", required=True)

    flow = subparsers.add_parser("generate-motion-flow-skeleton-v0", help="Generate a placeholder relative motion flow skeleton, not a Timeline export.")
    flow.add_argument("--plan", required=True)
    flow.add_argument("--retrieved-primitives", required=True)
    flow.add_argument("--out", required=True)
    flow.add_argument("--report", required=True)

    synth_flow = subparsers.add_parser("synthesize-motion-flow-v0", help="Synthesize actual relative controller curves from primitive group statistics without Timeline export.")
    synth_flow.add_argument("--plan", required=True)
    synth_flow.add_argument("--primitive-groups", required=True)
    synth_flow.add_argument("--primitives", required=True)
    synth_flow.add_argument("--out-json", required=True)
    synth_flow.add_argument("--out-npz", required=True)
    synth_flow.add_argument("--report", required=True)
    synth_flow.add_argument("--duration", type=float, default=4.0)
    synth_flow.add_argument("--fps", type=float, default=60.0)
    synth_flow.add_argument("--seed", type=int, default=42)

    synth_flow_v1 = subparsers.add_parser("synthesize-motion-flow-v1", help="Synthesize coordinated Cowgirl relative controller curves v1.")
    synth_flow_v1.add_argument("--plan", required=True)
    synth_flow_v1.add_argument("--primitive-groups", required=True)
    synth_flow_v1.add_argument("--primitives", required=True)
    synth_flow_v1.add_argument("--coordination-profile", default="cowgirl_oval_grind_v1")
    synth_flow_v1.add_argument("--out-json", required=True)
    synth_flow_v1.add_argument("--out-npz", required=True)
    synth_flow_v1.add_argument("--report", required=True)
    synth_flow_v1.add_argument("--duration", type=float, default=4.0)
    synth_flow_v1.add_argument("--fps", type=float, default=60.0)
    synth_flow_v1.add_argument("--seed", type=int, default=42)
    synth_flow_v1.add_argument("--tempo", default="slow")
    synth_flow_v1.add_argument("--vertical-scale", type=float, default=1.25)
    synth_flow_v1.add_argument("--lateral-scale", type=float, default=0.70)
    synth_flow_v1.add_argument("--forward-back-scale", type=float, default=1.0)
    synth_flow_v1.add_argument("--chest-follower-scale", type=float, default=0.35)

    validate_flow = subparsers.add_parser("validate-generated-motion-flow-v0", help="Validate synthesized relative generated motion flow safety.")
    validate_flow.add_argument("--flow", required=True)
    validate_flow.add_argument("--out", required=True)

    preview_flow = subparsers.add_parser("render-generated-motion-preview-v0", help="Render technical preview plots for generated relative motion flow.")
    preview_flow.add_argument("--flow", required=True)
    preview_flow.add_argument("--out-dir", required=True)

    baseline_pose = subparsers.add_parser("create-synthetic-baseline-pose-v0", help="Create a synthetic neutral body-controller baseline pose for review retargeting.")
    baseline_pose.add_argument("--out", required=True)

    cowgirl_baseline = subparsers.add_parser("create-cowgirl-review-baseline-pose-v1", help="Create a synthetic Cowgirl kneeling/forward review baseline pose.")
    cowgirl_baseline.add_argument("--out", required=True)
    cowgirl_baseline.add_argument("--style", default="kneeling_forward")

    retarget_flow = subparsers.add_parser("retarget-motion-flow-v0", help="Retarget generated relative motion flow onto a baseline pose without source world coordinates.")
    retarget_flow.add_argument("--flow", required=True)
    retarget_flow.add_argument("--baseline-pose", required=True)
    retarget_flow.add_argument("--out-json", required=True)
    retarget_flow.add_argument("--out-npz", required=True)
    retarget_flow.add_argument("--report", required=True)

    retarget_flow_v1 = subparsers.add_parser("retarget-motion-flow-v1", help="Retarget generated Cowgirl motion flow v1 onto Cowgirl review baseline.")
    retarget_flow_v1.add_argument("--flow", required=True)
    retarget_flow_v1.add_argument("--baseline-pose", required=True)
    retarget_flow_v1.add_argument("--out-json", required=True)
    retarget_flow_v1.add_argument("--out-npz", required=True)
    retarget_flow_v1.add_argument("--report", required=True)

    validate_retarget = subparsers.add_parser("validate-retargeted-motion-flow-v0", help="Validate baseline-retargeted generated motion flow.")
    validate_retarget.add_argument("--retargeted-flow", required=True)
    validate_retarget.add_argument("--out", required=True)

    validate_retarget_v1 = subparsers.add_parser("validate-retargeted-motion-flow-v1", help="Validate Cowgirl retargeted generated motion flow v1.")
    validate_retarget_v1.add_argument("--retargeted-flow", required=True)
    validate_retarget_v1.add_argument("--out", required=True)

    preview_retarget = subparsers.add_parser("render-retargeted-motion-preview-v0", help="Render preview plots for a retargeted generated motion flow.")
    preview_retarget.add_argument("--retargeted-flow", required=True)
    preview_retarget.add_argument("--out-dir", required=True)

    preview_retarget_v1 = subparsers.add_parser("render-retargeted-motion-preview-v1", help="Render enhanced preview plots for retargeted Cowgirl motion flow v1.")
    preview_retarget_v1.add_argument("--retargeted-flow", required=True)
    preview_retarget_v1.add_argument("--out-dir", required=True)

    export_retarget = subparsers.add_parser("export-retargeted-flow-timeline-v0", help="Export review-flow JSON from retargeted generated motion. This is not native Timeline JSON.")
    export_retarget.add_argument("--retargeted-flow", required=True)
    export_retarget.add_argument("--validation", required=True)
    export_retarget.add_argument("--out-dir", required=True)

    review_player = subparsers.add_parser("export-generated-flow-for-vam-review", help="Export retargeted flow as JSON for the VaM Generated Motion Review Player.")
    review_player.add_argument("--retargeted-flow", required=True)
    review_player.add_argument("--out", required=True)
    review_player.add_argument("--report", required=True)

    review_player_v1 = subparsers.add_parser("export-generated-flow-for-vam-review-v1", help="Export retargeted Cowgirl v1 flow as JSON for VaM review player.")
    review_player_v1.add_argument("--retargeted-flow", required=True)
    review_player_v1.add_argument("--out", required=True)
    review_player_v1.add_argument("--report", required=True)

    prepare_review_player = subparsers.add_parser("prepare-vam-review-player-v0", help="Prepare VaM review player JSON, C# script, and user instructions.")
    prepare_review_player.add_argument("--retargeted-flow", required=True)
    prepare_review_player.add_argument("--out-dir", required=True)

    first_review = subparsers.add_parser("run-first-generated-motion-review-v0", help="Run the first generated motion review prototype pipeline.")
    first_review.add_argument("--plan", required=True)
    first_review.add_argument("--primitive-groups", required=True)
    first_review.add_argument("--primitives", required=True)
    first_review.add_argument("--out-dir", required=True)
    first_review.add_argument("--duration", type=float, default=4.0)
    first_review.add_argument("--fps", type=float, default=60.0)
    first_review.add_argument("--seed", type=int, default=42)

    cowgirl_v1_review = subparsers.add_parser("run-cowgirl-motion-flow-v1-review", help="Run the Cowgirl motion flow v1 review pipeline.")
    cowgirl_v1_review.add_argument("--plan", required=True)
    cowgirl_v1_review.add_argument("--primitive-groups", required=True)
    cowgirl_v1_review.add_argument("--primitives", required=True)
    cowgirl_v1_review.add_argument("--out-dir", required=True)
    cowgirl_v1_review.add_argument("--duration", type=float, default=4.0)
    cowgirl_v1_review.add_argument("--fps", type=float, default=60.0)
    cowgirl_v1_review.add_argument("--seed", type=int, default=42)

    native_timeline = subparsers.add_parser("export-generated-flow-native-timeline-v0", help="Export retargeted generated flow as native AcidBubbles Timeline JSON.")
    native_timeline.add_argument("--retargeted-flow", required=True)
    native_timeline.add_argument("--out", required=True)
    native_timeline.add_argument("--report", required=True)

    native_timeline_v1 = subparsers.add_parser("export-generated-flow-native-timeline-v1", help="Export baseline-baked generated flow as native AcidBubbles Timeline JSON v1.")
    native_timeline_v1.add_argument("--retargeted-flow", required=True)
    native_timeline_v1.add_argument("--baseline-pose", required=True)
    native_timeline_v1.add_argument("--out", required=True)
    native_timeline_v1.add_argument("--report", required=True)
    native_timeline_v1.add_argument("--include-baseline-keyframe", default="true")
    native_timeline_v1.add_argument("--include-rotation-tracks", default="true")

    native_timeline_validate = subparsers.add_parser("validate-native-timeline-export-v0", help="Validate generated native Timeline JSON structure.")
    native_timeline_validate.add_argument("--timeline", required=True)
    native_timeline_validate.add_argument("--report", required=True)

    native_timeline_validate_v1 = subparsers.add_parser("validate-native-timeline-export-v1", help="Validate generated native Timeline JSON v1 baseline baking.")
    native_timeline_validate_v1.add_argument("--timeline", required=True)
    native_timeline_validate_v1.add_argument("--baseline-pose", required=True)
    native_timeline_validate_v1.add_argument("--report", required=True)

    native_timeline_review = subparsers.add_parser("run-native-timeline-export-review-v0", help="Export, validate, and write instructions for native Timeline import review.")
    native_timeline_review.add_argument("--retargeted-flow", required=True)
    native_timeline_review.add_argument("--out-dir", required=True)

    native_timeline_review_v1 = subparsers.add_parser("run-native-timeline-export-review-v1", help="Export, validate, and write instructions for baseline-baked native Timeline import review v1.")
    native_timeline_review_v1.add_argument("--retargeted-flow", required=True)
    native_timeline_review_v1.add_argument("--baseline-pose", required=True)
    native_timeline_review_v1.add_argument("--out-dir", required=True)

    pose_features_v0 = subparsers.add_parser("extract-pose-features-v0", help="Extract pose feature proxies for clean_v3 semantic rescan.")
    pose_features_v0.add_argument("--relative-index", required=True)
    pose_features_v0.add_argument("--body-quality", required=True)
    pose_features_v0.add_argument("--pose-anchor-completeness", required=True)
    pose_features_v0.add_argument("--controller-validity", required=True)
    pose_features_v0.add_argument("--out-jsonl", required=True)
    pose_features_v0.add_argument("--report", required=True)

    classify_poses_v0 = subparsers.add_parser("classify-poses-v0", help="Classify pose semantics separately from motion semantics.")
    classify_poses_v0.add_argument("--pose-features", required=True)
    classify_poses_v0.add_argument("--relative-reference-matches", required=False, default=None)
    classify_poses_v0.add_argument("--handmade-features", required=False, default=None)
    classify_poses_v0.add_argument("--out-jsonl", required=True)
    classify_poses_v0.add_argument("--report", required=True)

    partner_features_v0 = subparsers.add_parser("extract-partner-relative-features-v0", help="Extract partner-relative interaction/contact feature proxies.")
    partner_features_v0.add_argument("--pair-windows", required=True)
    partner_features_v0.add_argument("--pair-features", required=True)
    partner_features_v0.add_argument("--relative-index", required=True)
    partner_features_v0.add_argument("--pose-semantics", required=True)
    partner_features_v0.add_argument("--out-jsonl", required=True)
    partner_features_v0.add_argument("--report", required=True)

    classify_interactions_v0 = subparsers.add_parser("classify-interactions-v0", help="Classify partner-relative interaction semantics.")
    classify_interactions_v0.add_argument("--partner-relative-features", required=True)
    classify_interactions_v0.add_argument("--pose-semantics", required=True)
    classify_interactions_v0.add_argument("--semantic-actions", required=False, default=None)
    classify_interactions_v0.add_argument("--out-jsonl", required=True)
    classify_interactions_v0.add_argument("--report", required=True)

    semantic_actions_v0 = subparsers.add_parser("build-semantic-actions-v0", help="Build Semantic Action candidates from pose, motion, interaction, and contact evidence.")
    semantic_actions_v0.add_argument("--candidate-db", required=True)
    semantic_actions_v0.add_argument("--pose-semantics", required=True)
    semantic_actions_v0.add_argument("--relative-reference-matches", required=False, default=None)
    semantic_actions_v0.add_argument("--interaction-semantics", required=True)
    semantic_actions_v0.add_argument("--out-jsonl", required=True)
    semantic_actions_v0.add_argument("--report", required=True)

    cowgirl_v5 = subparsers.add_parser("build-cowgirl-candidate-db-v5", help="Build Cowgirl DB v5 from clean_v3 Semantic Candidate DB.")
    cowgirl_v5.add_argument("--semantic-candidate-db", required=True)
    cowgirl_v5.add_argument("--out-jsonl", required=True)
    cowgirl_v5.add_argument("--out-csv", required=True)
    cowgirl_v5.add_argument("--report", required=True)

    primitives_v1 = subparsers.add_parser("extract-cowgirl-motion-primitives-v1", help="Extract Cowgirl primitives with pose/partner/contact requirements.")
    primitives_v1.add_argument("--candidate-db", required=True)
    primitives_v1.add_argument("--relative-features", required=True)
    primitives_v1.add_argument("--trajectory-features", required=True)
    primitives_v1.add_argument("--pose-semantics", required=True)
    primitives_v1.add_argument("--interaction-semantics", required=True)
    primitives_v1.add_argument("--out-jsonl", required=True)
    primitives_v1.add_argument("--out-report", required=True)

    draft_plan_v1 = subparsers.add_parser("draft-motion-plan-v1", help="Draft a pose/partner/contact-aware semantic motion plan.")
    draft_plan_v1.add_argument("--prompt", required=True)
    draft_plan_v1.add_argument("--out", required=True)

    interaction_baseline = subparsers.add_parser("select-interaction-baseline-for-plan-v0", help="Create a synthetic partner-relative baseline for a motion plan.")
    interaction_baseline.add_argument("--plan", required=True)
    interaction_baseline.add_argument("--out", required=True)

    partner_flow = subparsers.add_parser("synthesize-partner-relative-flow-v0", help="Synthesize partner-relative flow with contact/support constraints.")
    partner_flow.add_argument("--plan", required=True)
    partner_flow.add_argument("--primitive-groups", required=True)
    partner_flow.add_argument("--baseline", required=True)
    partner_flow.add_argument("--out-json", required=True)
    partner_flow.add_argument("--report", required=True)

    validate_partner_flow = subparsers.add_parser("validate-partner-relative-flow-v0", help="Validate partner-relative contact/support constraints.")
    validate_partner_flow.add_argument("--flow", required=True)
    validate_partner_flow.add_argument("--out", required=True)

    semantic_rescan = subparsers.add_parser("run-semantic-rescan-v1", help="Run clean_v3 semantic rescan: pose + motion + partner interaction + contact.")
    semantic_rescan.add_argument("--source-run", required=True)
    semantic_rescan.add_argument("--out-run", required=True)

    ingest_v15 = subparsers.add_parser("ingest-v15-human-findings", help="Store semantic_review_010_v15 human audit findings without touching manual labels.")
    ingest_v15.add_argument("--review-dir", required=True)

    rebuild_v3_calibration = subparsers.add_parser("rebuild-clean-v3-semantic-actions-v1", help="Rebuild clean_v3 semantic actions and DBs with v15 calibration rules.")
    rebuild_v3_calibration.add_argument("--run-dir", required=True)
    rebuild_v3_calibration.add_argument("--previous-review", default=None)

    export_v16 = subparsers.add_parser("export-semantic-review-v16", help="Export calibrated clean_v3 semantic review v16 and optional VaM package.")
    export_v16.add_argument("--run-dir", required=True)
    export_v16.add_argument("--out-dir", required=True)
    export_v16.add_argument("--count", type=int, default=10)
    export_v16.add_argument("--build-vam-package", default="true")
    export_v16.add_argument("--previous-review", default=None)

    calibration = subparsers.add_parser("run-clean-v3-calibration-v1", help="Ingest v15 findings, rebuild calibrated DBs, and export v16 review.")
    calibration.add_argument("--run-dir", required=True)
    calibration.add_argument("--previous-review", required=True)
    calibration.add_argument("--out-review", required=True)

    calibration_v16 = subparsers.add_parser("run-clean-v3-v16-calibration", help="Ingest v16 findings, rebuild clean-motion gated DBs, and export v17 review.")
    calibration_v16.add_argument("--run-dir", required=True)
    calibration_v16.add_argument("--previous-review", required=True)
    calibration_v16.add_argument("--out-review", required=True)

    pose_support_rescan = subparsers.add_parser("run-clean-v3-pose-support-rescan", help="Add frontal Cowgirl lean-back supported pose/contact support as a focused clean_v3 audit layer.")
    pose_support_rescan.add_argument("--run-dir", required=True)
    pose_support_rescan.add_argument("--out-suffix", default="lean_back_support_v1")

    new_scene_delta = subparsers.add_parser("compare-new-scenes-to-clean-v3", help="Compare a separate new-scene delta run against clean_v3.")
    new_scene_delta.add_argument("--base-run", required=True)
    new_scene_delta.add_argument("--new-run", required=True)
    new_scene_delta.add_argument("--out", required=True)

    new_scene_import = subparsers.add_parser("run-new-scenes-delta-import", help="Scan, bake, analyze, and review only a new VaM scene folder as a separate delta run.")
    new_scene_import.add_argument("--raw-dir", required=True)
    new_scene_import.add_argument("--base-run", required=True)
    new_scene_import.add_argument("--out-run", required=True)

    focused_new_scene_review = subparsers.add_parser("build-focused-new-scenes-review", help="Build a balanced calibration review batch from the clean_v3 new-scene delta run.")
    focused_new_scene_review.add_argument("--run-dir", required=True)
    focused_new_scene_review.add_argument("--previous-review", required=True)
    focused_new_scene_review.add_argument("--out-dir", required=True)

    strict_new_scene_cowgirl_review = subparsers.add_parser("build-strict-new-scenes-cowgirl-review", help="Build a stricter Cowgirl-only review batch after noisy broad new-scene review.")
    strict_new_scene_cowgirl_review.add_argument("--run-dir", required=True)
    strict_new_scene_cowgirl_review.add_argument("--out-dir", required=True)
    strict_new_scene_cowgirl_review.add_argument("--previous-review", default=None)
    strict_new_scene_cowgirl_review.add_argument("--human-answers", default=None)
    strict_new_scene_cowgirl_review.add_argument("--batch-size", type=int, default=None)
    strict_new_scene_cowgirl_review.add_argument("--batch-index", type=int, default=1)

    new_scenes_pose_first_v2 = subparsers.add_parser("resolve-new-scenes-pose-first-semantics-v2", help="Rescan clean_v3_new_scenes with ontology v2 and manual-GT pose-first rules.")
    new_scenes_pose_first_v2.add_argument("--new-run", required=True)
    new_scenes_pose_first_v2.add_argument("--base-run", required=True)
    new_scenes_pose_first_v2.add_argument("--ontology", required=True)
    new_scenes_pose_first_v2.add_argument("--rules", required=True)
    new_scenes_pose_first_v2.add_argument("--manual-gt", required=True)
    new_scenes_pose_first_v2.add_argument("--out-jsonl", required=True)
    new_scenes_pose_first_v2.add_argument("--report", required=True)

    new_scenes_candidate_db_v2 = subparsers.add_parser("build-new-scenes-ontology-candidate-db-v2", help="Build ontology-aligned v2 candidate DB for clean_v3_new_scenes.")
    new_scenes_candidate_db_v2.add_argument("--new-run", required=True)
    new_scenes_candidate_db_v2.add_argument("--resolved", required=True)
    new_scenes_candidate_db_v2.add_argument("--ontology", required=True)
    new_scenes_candidate_db_v2.add_argument("--manual-gt", required=True)
    new_scenes_candidate_db_v2.add_argument("--out-jsonl", required=True)
    new_scenes_candidate_db_v2.add_argument("--out-csv", required=True)
    new_scenes_candidate_db_v2.add_argument("--report", required=True)

    new_scenes_family_reports_v2 = subparsers.add_parser("write-new-scenes-family-reports-v2", help="Write family-specific reports for new-scenes semantic rescan v2.")
    new_scenes_family_reports_v2.add_argument("--new-run", required=True)
    new_scenes_family_reports_v2.add_argument("--candidates", required=True)
    new_scenes_family_reports_v2.add_argument("--out-dir", required=True)

    new_scenes_review_v2 = subparsers.add_parser("export-new-scenes-semantic-review-v2", help="Export a strict ontology-v2 review batch from new scenes.")
    new_scenes_review_v2.add_argument("--new-run", required=True)
    new_scenes_review_v2.add_argument("--candidates", required=True)
    new_scenes_review_v2.add_argument("--out-dir", required=True)
    new_scenes_review_v2.add_argument("--count", type=int, default=20)
    new_scenes_review_v2.add_argument("--build-vam-package", default="true")
    new_scenes_review_v2.add_argument("--build-static-ui", default="true")

    cycle_features_v1 = subparsers.add_parser("extract-motion-cycle-features-v1", help="Extract cycle-aware controller motion features from relative window tracks.")
    cycle_features_v1.add_argument("--run-dir", required=True)
    cycle_features_v1.add_argument("--out-jsonl", required=True)
    cycle_features_v1.add_argument("--report", required=True)

    motion_semantics_v1 = subparsers.add_parser("resolve-new-scenes-motion-semantics-v1", help="Resolve new scenes with cycle-aware Motion Semantics v1.")
    motion_semantics_v1.add_argument("--new-run", required=True)
    motion_semantics_v1.add_argument("--pose-resolved", required=True)
    motion_semantics_v1.add_argument("--cycle-features", required=True)
    motion_semantics_v1.add_argument("--relational-features", default=None)
    motion_semantics_v1.add_argument("--ontology", required=True)
    motion_semantics_v1.add_argument("--cycle-rules", required=True)
    motion_semantics_v1.add_argument("--manual-gt", required=True)
    motion_semantics_v1.add_argument("--out-jsonl", required=True)
    motion_semantics_v1.add_argument("--report", required=True)

    motion_candidates_v1 = subparsers.add_parser("build-new-scenes-motion-candidate-db-v1", help="Build motion-state candidate DB from Motion Semantics v1.")
    motion_candidates_v1.add_argument("--new-run", required=True)
    motion_candidates_v1.add_argument("--motion-resolved", required=True)
    motion_candidates_v1.add_argument("--out-jsonl", required=True)
    motion_candidates_v1.add_argument("--out-csv", required=True)
    motion_candidates_v1.add_argument("--report", required=True)

    motion_review_v1 = subparsers.add_parser("export-motion-semantics-review-v1", help="Export a motion-cycle semantic review batch.")
    motion_review_v1.add_argument("--new-run", required=True)
    motion_review_v1.add_argument("--candidates", required=True)
    motion_review_v1.add_argument("--out-dir", required=True)
    motion_review_v1.add_argument("--count", type=int, default=20)
    motion_review_v1.add_argument("--build-static-ui", default="true")
    motion_review_v1.add_argument("--build-vam-package", default="true")

    reviewed_index = subparsers.add_parser("build-reviewed-window-index", help="Build an audit-only index of all reviewed windows/items across runs.")
    reviewed_index.add_argument("--run-dir", required=True)
    reviewed_index.add_argument("--include-runs", required=True)
    reviewed_index.add_argument("--out-jsonl", required=True)
    reviewed_index.add_argument("--out-csv", required=True)
    reviewed_index.add_argument("--report", required=True)

    duplicate_audit = subparsers.add_parser("audit-review-duplicates", help="Audit exact and near duplicate review selections.")
    duplicate_audit.add_argument("--reviewed-index", required=True)
    duplicate_audit.add_argument("--out", required=True)

    strict_novel_review = subparsers.add_parser("export-strict-novel-review", help="Export a deduplicated novel semantic review batch.")
    strict_novel_review.add_argument("--run-dir", required=True)
    strict_novel_review.add_argument("--candidate-db", required=True)
    strict_novel_review.add_argument("--reviewed-index", required=True)
    strict_novel_review.add_argument("--out-dir", required=True)
    strict_novel_review.add_argument("--count", type=int, default=10)
    strict_novel_review.add_argument("--max-per-scene", type=int, default=2)
    strict_novel_review.add_argument("--max-per-sample", type=int, default=1)
    strict_novel_review.add_argument("--allow-reviewed-overlap", default="false")
    strict_novel_review.add_argument("--allow-near-duplicates", default="false")
    strict_novel_review.add_argument("--diversity-mode", default="strict")
    strict_novel_review.add_argument("--build-vam-package", default="true")
    strict_novel_review.add_argument("--build-static-ui", default="true")

    export_review_segments = subparsers.add_parser("export-review-timeline-segments-to-vam", help="Copy review-only Timeline segments into VaM PluginData animations for direct manual import.")
    export_review_segments.add_argument("--review-dir", required=True)
    export_review_segments.add_argument("--vam-animations-dir", required=True)
    export_review_segments.add_argument("--run-dir", default=None)
    export_review_segments.add_argument("--subdir", default=None)

    sanitize_scenes = subparsers.add_parser("sanitize-run-scene-identifiers", help="Replace local scene names in run artifacts with stable neutral aliases.")
    sanitize_scenes.add_argument("--run-dir", required=True)
    sanitize_scenes.add_argument("--alias-map-out", required=True)
    sanitize_scenes.add_argument("--report", required=True)
    sanitize_scenes.add_argument("--dry-run", default="false")

    ledger = subparsers.add_parser("build-human-review-ledger", help="Build audit-only human review memory ledger.")
    ledger.add_argument("--run-dir", required=True)
    ledger.add_argument("--include-runs", required=True)
    ledger.add_argument("--out-jsonl", required=True)
    ledger.add_argument("--out-csv", required=True)
    ledger.add_argument("--report", required=True)

    taxonomy = subparsers.add_parser("build-error-taxonomy-report", help="Build error taxonomy report from human review ledger.")
    taxonomy.add_argument("--human-review-ledger", required=True)
    taxonomy.add_argument("--out", required=True)

    db_validate = subparsers.add_parser("validate-semantic-dbs", help="Validate semantic and Cowgirl DB invariants.")
    db_validate.add_argument("--run-dir", required=True)
    db_validate.add_argument("--semantic-db", required=True)
    db_validate.add_argument("--cowgirl-db", required=True)
    db_validate.add_argument("--out", required=True)

    dashboard = subparsers.add_parser("write-clean-v3-dashboard", help="Write clean_v3 semantic QA dashboard.")
    dashboard.add_argument("--run-dir", required=True)
    dashboard.add_argument("--out-md", required=True)
    dashboard.add_argument("--out-html", required=True)

    drift = subparsers.add_parser("compare-clean-v2-clean-v3", help="Write clean_v2 to clean_v3 semantic drift report.")
    drift.add_argument("--clean-v2", required=True)
    drift.add_argument("--clean-v3", required=True)
    drift.add_argument("--out", required=True)

    review_plan = subparsers.add_parser("plan-larger-review-batch-v1", help="Plan a larger review batch without exporting it.")
    review_plan.add_argument("--run-dir", required=True)
    review_plan.add_argument("--semantic-db", required=True)
    review_plan.add_argument("--cowgirl-db", required=True)
    review_plan.add_argument("--out", required=True)

    prompt_matrix = subparsers.add_parser("write-prompt-capability-matrix", help="Write honest prompt capability matrix.")
    prompt_matrix.add_argument("--run-dir", required=True)
    prompt_matrix.add_argument("--out", required=True)

    status = subparsers.add_parser("clean-v3-status", help="Print and write clean_v3 operator status.")
    status.add_argument("--run-dir", required=True)

    overnight = subparsers.add_parser("run-clean-v3-overnight-qa", help="Run resilient clean_v3 overnight QA reports.")
    overnight.add_argument("--run-dir", required=True)
    overnight.add_argument("--include-runs", required=True)

    lineage = subparsers.add_parser("write-candidate-lineage-report", help="Write clean_v3 candidate lineage report.")
    lineage.add_argument("--run-dir", required=True)
    lineage.add_argument("--out", required=True)

    reproducibility = subparsers.add_parser("run-clean-v3-reproducibility-audit", help="Write schema, manifest, lineage, reproducibility, and operator reports.")
    reproducibility.add_argument("--run-dir", required=True)

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
    candidates = sum(1 for row in rows if row.get("bj_oral_motion_candidate"))
    preserved = sum(1 for row in rows if row.get("preserve_for_future_dataset"))
    print(f"BJ/oral domain classifier written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; bj_oral_candidates: {candidates}; preserved_for_future_dataset: {preserved}")
    return 0


def cmd_classify_bj_oral_domain(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.bj_oral_domain_classifier import classify_bj_oral_domain

    rows = classify_bj_oral_domain(
        args.run_dir,
        args.relative_features,
        args.trajectory_features,
        args.relative_reference_matches,
        args.cowgirl_core_controllers,
        args.out_jsonl,
        args.report,
    )
    candidates = sum(1 for row in rows if row.get("bj_oral_motion_candidate"))
    preserved = sum(1 for row in rows if row.get("preserve_for_future_dataset"))
    print(f"BJ/oral domain classifier written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; bj_oral_candidates: {candidates}; preserved_for_future_dataset: {preserved}")
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


def cmd_score_cowgirl_candidates_v11(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.cowgirl_candidate_scoring import score_cowgirl_candidates_v11

    bj_oral_domain = args.bj_oral_domain or args.bj_oral_trap_guard
    if not bj_oral_domain:
        raise ValueError("score-cowgirl-candidates-v11 requires --bj-oral-domain or compatibility --bj-oral-trap-guard")
    rows = score_cowgirl_candidates_v11(
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
        bj_oral_domain,
        args.features,
        args.out_jsonl,
        args.report,
    )
    generation = sum(1 for row in rows if row.get("cowgirl_v11_category") in {"semantic_cowgirl_generation_safe", "semantic_cowgirl_core_soft_fail_generation_safe"})
    soft = sum(1 for row in rows if row.get("semantic_cowgirl_core_soft_fail_generation_safe"))
    bj = sum(1 for row in rows if row.get("semantic_family") == "bj_oral")
    print(f"Cowgirl candidate scores v11 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; generation_safe: {generation}; core_soft_fail_accepted: {soft}; bj_oral_preserved: {bj}")
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


def cmd_build_cowgirl_candidate_db_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.cowgirl_candidate_database import build_cowgirl_candidate_db_v3

    rows = build_cowgirl_candidate_db_v3(
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
        args.bj_oral_domain,
        args.out_jsonl,
        args.out_csv,
        args.report,
    )
    generation = sum(1 for row in rows if row.get("generation_safe"))
    soft = sum(1 for row in rows if row.get("category") == "semantic_cowgirl_core_soft_fail_generation_safe")
    bj = sum(1 for row in rows if row.get("semantic_family") == "bj_oral")
    print(f"Cowgirl candidate DB v3 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; generation_safe: {generation}; core_soft_fail: {soft}; bj_oral_preserved: {bj}")
    return 0


def cmd_build_semantic_candidate_db_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.semantic_candidate_database import build_semantic_candidate_db_v0

    rows = build_semantic_candidate_db_v0(
        args.run_dir,
        args.cowgirl_db,
        args.bj_oral_domain,
        args.relative_features,
        args.trajectory_features,
        args.out_jsonl,
        args.out_csv,
        args.report,
    )
    families: dict[str, int] = {}
    for row in rows:
        family = str(row.get("semantic_family") or "unknown")
        families[family] = families.get(family, 0) + 1
    print(f"Semantic candidate DB v0 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; families: {families}")
    return 0


def cmd_extract_cowgirl_motion_primitives_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.primitive_extractor import extract_cowgirl_motion_primitives_v0

    rows = extract_cowgirl_motion_primitives_v0(
        args.candidate_db,
        args.relative_features,
        args.trajectory_features,
        args.relative_index,
        args.out_jsonl,
        args.out_report,
    )
    print(f"Cowgirl motion primitives v0 written: {args.out_jsonl}")
    print(f"Primitives: {len(rows)}")
    return 0


def cmd_group_cowgirl_motion_primitives_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.primitive_groups import group_cowgirl_motion_primitives_v0

    data = group_cowgirl_motion_primitives_v0(args.primitives, args.out_json, args.report)
    groups = data.get("groups", [])
    print(f"Cowgirl motion primitive groups v0 written: {args.out_json}")
    print(f"Groups: {len(groups)}")
    return 0


def cmd_draft_motion_plan_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.prompt_to_plan import draft_motion_plan_v0

    plan = draft_motion_plan_v0(args.prompt, args.out)
    print(f"Draft motion plan v0 written: {args.out}")
    print(f"Plan: {plan.get('plan_id')}; family={plan.get('family')}; subtypes={plan.get('requested_subtypes')}")
    return 0


def cmd_retrieve_primitives_for_plan_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.primitive_retrieval import retrieve_primitives_for_plan_v0

    result = retrieve_primitives_for_plan_v0(args.plan, args.primitive_groups, args.primitives, args.out, args.report)
    total = sum(int(match.get("candidate_count") or 0) for match in result.get("matches", []))
    print(f"Retrieved primitives v0 written: {args.out}")
    print(f"Candidate primitive matches: {total}; timeline_export_performed={result.get('timeline_export_performed')}")
    return 0


def cmd_generate_motion_flow_skeleton_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.motion_flow_generator import generate_motion_flow_skeleton_v0

    flow = generate_motion_flow_skeleton_v0(args.plan, args.retrieved_primitives, args.out, args.report)
    print(f"Motion flow skeleton v0 written: {args.out}")
    print(f"Flow: {flow.get('flow_id')}; export_ready={flow.get('export_ready')}; coordinate_space={flow.get('coordinate_space')}")
    return 0


def cmd_synthesize_motion_flow_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.motion_flow_synthesis import synthesize_motion_flow_v0

    flow = synthesize_motion_flow_v0(
        args.plan,
        args.primitive_groups,
        args.primitives,
        args.out_json,
        args.out_npz,
        args.report,
        duration=args.duration,
        fps=args.fps,
        seed=args.seed,
    )
    print(f"Generated motion flow v0 written: {args.out_json}")
    print(f"Flow: {flow.get('flow_id')}; group={flow.get('selected_primitive_group')}; export_ready={flow.get('export_ready')}")
    return 0


def cmd_synthesize_motion_flow_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.motion_flow_synthesis import synthesize_motion_flow_v1

    flow = synthesize_motion_flow_v1(
        args.plan,
        args.primitive_groups,
        args.primitives,
        args.coordination_profile,
        args.out_json,
        args.out_npz,
        args.report,
        duration=args.duration,
        fps=args.fps,
        seed=args.seed,
        tempo=args.tempo,
        vertical_scale=args.vertical_scale,
        lateral_scale=args.lateral_scale,
        forward_back_scale=args.forward_back_scale,
        chest_follower_scale=args.chest_follower_scale,
    )
    print(f"Generated motion flow v1 written: {args.out_json}")
    print(f"Flow: {flow.get('flow_id')}; controllers={len(flow.get('controller_tracks', []) or [])}; profile={args.coordination_profile}")
    return 0


def cmd_validate_generated_motion_flow_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.generated_motion_validation import validate_generated_motion_flow_v0

    result = validate_generated_motion_flow_v0(args.flow, args.out)
    print(f"Generated motion flow validation written: {args.out}")
    print(f"Passed: {result.get('passed')}; safe_for_timeline_export={result.get('safe_for_timeline_export')}")
    return 0 if result.get("passed") else 1


def cmd_render_generated_motion_preview_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.generated_motion_preview import render_generated_motion_preview_v0

    manifest = render_generated_motion_preview_v0(args.flow, args.out_dir)
    print(f"Generated motion preview written: {args.out_dir}")
    print(f"Status: {manifest.get('status')}; files={len(manifest.get('files', []))}")
    return 0


def cmd_create_synthetic_baseline_pose_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.baseline_pose import create_synthetic_baseline_pose_v0

    data = create_synthetic_baseline_pose_v0(args.out)
    print(f"Synthetic baseline pose v0 written: {args.out}")
    print(f"Baseline: {data.get('baseline_id')}; controllers={len(data.get('controller_poses', []) or [])}")
    return 0


def cmd_create_cowgirl_review_baseline_pose_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.baseline_pose import create_cowgirl_review_baseline_pose_v1

    data = create_cowgirl_review_baseline_pose_v1(args.out, args.style)
    print(f"Cowgirl review baseline pose v1 written: {args.out}")
    print(f"Baseline: {data.get('baseline_id')}; style={data.get('style')}; controllers={len(data.get('controller_poses', []) or [])}")
    return 0


def cmd_retarget_motion_flow_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.relative_flow_retargeter import retarget_motion_flow_v0

    data = retarget_motion_flow_v0(args.flow, args.baseline_pose, args.out_json, args.out_npz, args.report)
    print(f"Retargeted motion flow v0 written: {args.out_json}")
    print(f"Flow: {data.get('flow_id')}; controllers={len(data.get('controller_tracks', []) or [])}; review_candidate={data.get('safe_for_review_export_candidate')}")
    return 0


def cmd_retarget_motion_flow_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.relative_flow_retargeter import retarget_motion_flow_v1

    data = retarget_motion_flow_v1(args.flow, args.baseline_pose, args.out_json, args.out_npz, args.report)
    print(f"Retargeted motion flow v1 written: {args.out_json}")
    print(f"Flow: {data.get('flow_id')}; baseline_style={data.get('baseline_style')}; controllers={len(data.get('controller_tracks', []) or [])}")
    return 0


def cmd_validate_retargeted_motion_flow_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.retarget_validation import validate_retargeted_motion_flow_v0

    result = validate_retargeted_motion_flow_v0(args.retargeted_flow, args.out)
    print(f"Retargeted motion flow validation written: {args.out}")
    print(f"Passed: {result.get('passed')}; review_candidate={result.get('export_review_safe_candidate')}; generation_candidate={result.get('generation_template_candidate')}")
    return 0 if result.get("passed") else 1


def cmd_validate_retargeted_motion_flow_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.retarget_validation import validate_retargeted_motion_flow_v1

    result = validate_retargeted_motion_flow_v1(args.retargeted_flow, args.out)
    print(f"Retargeted motion flow v1 validation written: {args.out}")
    print(f"Passed: {result.get('passed')}; review_candidate={result.get('export_review_safe_candidate')}")
    return 0 if result.get("passed") else 1


def cmd_render_retargeted_motion_preview_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.retargeted_motion_preview import render_retargeted_motion_preview_v0

    manifest = render_retargeted_motion_preview_v0(args.retargeted_flow, args.out_dir)
    print(f"Retargeted motion preview written: {args.out_dir}")
    print(f"Status: {manifest.get('status')}; files={len(manifest.get('files', []))}")
    return 0


def cmd_render_retargeted_motion_preview_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.retargeted_motion_preview import render_retargeted_motion_preview_v1

    manifest = render_retargeted_motion_preview_v1(args.retargeted_flow, args.out_dir)
    print(f"Retargeted motion preview v1 written: {args.out_dir}")
    print(f"Status: {manifest.get('status')}; files={len(manifest.get('files', []))}")
    return 0


def cmd_export_retargeted_flow_timeline_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.timeline_from_retargeted_flow import export_retargeted_flow_timeline_v0

    result = export_retargeted_flow_timeline_v0(args.retargeted_flow, args.validation, args.out_dir)
    print(f"Retargeted review-flow JSON status: {result.get('status')}")
    if result.get("timeline_json"):
        print(f"Review-flow JSON (not native Timeline): {result.get('timeline_json')}")
    return 0


def cmd_export_generated_flow_for_vam_review(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.review_player_export import export_generated_flow_for_vam_review

    data = export_generated_flow_for_vam_review(args.retargeted_flow, args.out, args.report)
    print(f"VaM review player JSON written: {args.out}")
    print(f"Controllers: {len(data.get('controllers', []) or [])}; native_timeline_importable={data.get('native_timeline_importable')}")
    return 0


def cmd_export_generated_flow_for_vam_review_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.review_player_export import export_generated_flow_for_vam_review_v1

    data = export_generated_flow_for_vam_review_v1(args.retargeted_flow, args.out, args.report)
    print(f"VaM review player JSON v1 written: {args.out}")
    print(f"Controllers: {len(data.get('controllers', []) or [])}; schema={data.get('schema')}")
    return 0


def cmd_prepare_vam_review_player_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.review_player_export import prepare_vam_review_player_v0

    summary = prepare_vam_review_player_v0(args.retargeted_flow, args.out_dir)
    print(f"VaM review player package written: {args.out_dir}")
    print(f"JSON: {summary.get('review_player_json')}")
    print(f"Script source: {summary.get('script_source')}")
    if summary.get("script_copied_to"):
        print(f"Script copied to: {summary.get('script_copied_to')}")
    return 0


def cmd_run_first_generated_motion_review_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.first_generated_motion_review import run_first_generated_motion_review_v0

    summary = run_first_generated_motion_review_v0(args.plan, args.primitive_groups, args.primitives, args.out_dir, args.duration, args.fps, args.seed)
    print(f"First generated motion review v0 written: {args.out_dir}")
    print(f"Generated validation={summary.get('generated_flow_validation_passed')}; retarget validation={summary.get('retarget_validation_passed')}; export={(summary.get('timeline_export') or {}).get('status')}")
    return 0


def cmd_run_cowgirl_motion_flow_v1_review(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.first_generated_motion_review import run_cowgirl_motion_flow_v1_review

    summary = run_cowgirl_motion_flow_v1_review(args.plan, args.primitive_groups, args.primitives, args.out_dir, args.duration, args.fps, args.seed)
    print(f"Cowgirl motion flow v1 review written: {args.out_dir}")
    print(f"Validation={summary.get('validation_passed')}; JSON={summary.get('review_player_json')}; secure_path={summary.get('review_player_secure_path')}")
    return 0


def cmd_export_generated_flow_native_timeline_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.native_timeline_exporter import export_generated_flow_native_timeline_v0

    data = export_generated_flow_native_timeline_v0(args.retargeted_flow, args.out, args.report)
    clip = (data.get("Clips") or [{}])[0]
    print(f"Native Timeline JSON written: {args.out}")
    print(f"Animation={clip.get('AnimationName')}; controllers={len(clip.get('Controllers', []) or [])}")
    return 0


def cmd_export_generated_flow_native_timeline_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.native_timeline_exporter import export_generated_flow_native_timeline_v1

    data = export_generated_flow_native_timeline_v1(
        args.retargeted_flow,
        args.baseline_pose,
        args.out,
        args.report,
        include_baseline_keyframe=_arg_bool(args.include_baseline_keyframe),
        include_rotation_tracks=_arg_bool(args.include_rotation_tracks),
    )
    clip = (data.get("Clips") or [{}])[0]
    meta = data.get("VAMTimelineAIGeneratedMetadata") or {}
    print(f"Native Timeline JSON v1 written: {args.out}")
    print(f"Animation={clip.get('AnimationName')}; controllers={len(clip.get('Controllers', []) or [])}; baseline_keyframe={meta.get('includes_baseline_keyframe')}; rotations={meta.get('includes_rotation_tracks')}")
    return 0


def cmd_validate_native_timeline_export_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.native_timeline_validation import validate_native_timeline_export_v0

    result = validate_native_timeline_export_v0(args.timeline, args.report)
    print(f"Native Timeline validation written: {args.report}")
    print(f"Passed={result.get('passed')}; expected_importable={result.get('expected_importable')}")
    return 0 if result.get("passed") else 1


def cmd_validate_native_timeline_export_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.native_timeline_validation import validate_native_timeline_export_v1

    result = validate_native_timeline_export_v1(args.timeline, args.baseline_pose, args.report)
    print(f"Native Timeline validation v1 written: {args.report}")
    print(f"Passed={result.get('passed')}; expected_importable={result.get('expected_importable')}; pose_context={result.get('expected_pose_context')}")
    return 0 if result.get("passed") else 1


def cmd_run_native_timeline_export_review_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.native_timeline_exporter import run_native_timeline_export_review_v0

    summary = run_native_timeline_export_review_v0(args.retargeted_flow, args.out_dir)
    print(f"Native Timeline export review written: {args.out_dir}")
    print(f"Timeline={summary.get('timeline')}; validation={summary.get('validation_passed')}; expected_importable={summary.get('expected_importable')}")
    return 0 if summary.get("validation_passed") else 1


def cmd_run_native_timeline_export_review_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.native_timeline_exporter import run_native_timeline_export_review_v1

    summary = run_native_timeline_export_review_v1(args.retargeted_flow, args.baseline_pose, args.out_dir)
    print(f"Native Timeline export review v1 written: {args.out_dir}")
    print(f"Timeline={summary.get('timeline')}; validation={summary.get('validation_passed')}; expected_importable={summary.get('expected_importable')}; pose_context={summary.get('expected_pose_context')}")
    return 0 if summary.get("validation_passed") else 1


def cmd_extract_pose_features_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.features.pose_features import extract_pose_features_v0

    rows = extract_pose_features_v0(args.relative_index, args.body_quality, args.pose_anchor_completeness, args.controller_validity, args.out_jsonl, args.report)
    print(f"Pose features v0 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}")
    return 0


def cmd_classify_poses_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.pose_classifier import classify_poses_v0

    rows = classify_poses_v0(args.pose_features, args.relative_reference_matches, args.handmade_features, args.out_jsonl, args.report)
    families: dict[str, int] = {}
    for row in rows:
        family = str(row.get("pose_family") or "unknown")
        families[family] = families.get(family, 0) + 1
    print(f"Pose semantics v0 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; families: {families}")
    return 0


def cmd_extract_partner_relative_features_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.features.partner_relative_features import extract_partner_relative_features_v0

    rows = extract_partner_relative_features_v0(args.pair_windows, args.pair_features, args.relative_index, args.pose_semantics, args.out_jsonl, args.report)
    print(f"Partner-relative features v0 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}")
    return 0


def cmd_classify_interactions_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.interaction_classifier import classify_interactions_v0

    rows = classify_interactions_v0(args.partner_relative_features, args.pose_semantics, args.semantic_actions, args.out_jsonl, args.report)
    families: dict[str, int] = {}
    for row in rows:
        family = str(row.get("interaction_family") or "unknown")
        families[family] = families.get(family, 0) + 1
    print(f"Interaction semantics v0 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; families: {families}")
    return 0


def cmd_build_semantic_actions_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.semantic_action import build_semantic_actions_v0

    rows = build_semantic_actions_v0(args.candidate_db, args.pose_semantics, args.relative_reference_matches, args.interaction_semantics, args.out_jsonl, args.report)
    families: dict[str, int] = {}
    for row in rows:
        family = str(row.get("semantic_family") or "unknown")
        families[family] = families.get(family, 0) + 1
    print(f"Semantic actions v0 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; families: {families}")
    return 0


def cmd_build_cowgirl_candidate_db_v5(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.cowgirl_candidate_database import build_cowgirl_candidate_db_v5

    rows = build_cowgirl_candidate_db_v5(args.semantic_candidate_db, args.out_jsonl, args.out_csv, args.report)
    safe = sum(1 for row in rows if row.get("generation_safe"))
    print(f"Cowgirl candidate DB v5 written: {args.out_jsonl}")
    print(f"Rows: {len(rows)}; generation_safe: {safe}")
    return 0


def cmd_extract_cowgirl_motion_primitives_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.primitive_extractor import extract_cowgirl_motion_primitives_v1

    rows = extract_cowgirl_motion_primitives_v1(args.candidate_db, args.relative_features, args.trajectory_features, args.pose_semantics, args.interaction_semantics, args.out_jsonl, args.out_report)
    print(f"Cowgirl motion primitives v1 written: {args.out_jsonl}")
    print(f"Primitives: {len(rows)}")
    return 0


def cmd_draft_motion_plan_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.prompt_to_plan import draft_motion_plan_v1

    plan = draft_motion_plan_v1(args.prompt, args.out)
    phase = (plan.get("sequence") or [{}])[0]
    print(f"Draft motion plan v1 written: {args.out}")
    print(f"Family={plan.get('family')}; pose={plan.get('requested_pose_subtype')}; support={(phase.get('interaction') or {}).get('support_mode')}")
    return 0


def cmd_select_interaction_baseline_for_plan_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.baseline_pose import select_interaction_baseline_for_plan_v0

    baseline = select_interaction_baseline_for_plan_v0(args.plan, args.out)
    print(f"Interaction baseline written: {args.out}")
    print(f"Baseline={baseline.get('baseline_id')}; support={baseline.get('support_mode')}; partner_refs={len(baseline.get('partner_references') or {})}")
    return 0


def cmd_synthesize_partner_relative_flow_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.partner_relative_flow import synthesize_partner_relative_flow_v0

    flow = synthesize_partner_relative_flow_v0(args.plan, args.primitive_groups, args.baseline, args.out_json, args.report)
    print(f"Partner-relative flow v0 written: {args.out_json}")
    print(f"Flow={flow.get('flow_id')}; support={flow.get('support_mode')}; controllers={len(flow.get('controller_tracks') or [])}")
    return 0


def cmd_validate_partner_relative_flow_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.interaction_validation import validate_partner_relative_flow_v0

    result = validate_partner_relative_flow_v0(args.flow, args.out)
    print(f"Partner-relative validation written: {args.out}")
    print(f"Passed={result.get('passed')}; contact_constraints_valid={result.get('contact_constraints_valid')}")
    return 0 if result.get("passed") else 1


def cmd_run_semantic_rescan_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.runs.semantic_rescan_v1 import run_semantic_rescan_v1

    summary = run_semantic_rescan_v1(args.source_run, args.out_run)
    print(f"Semantic rescan v1 written: {args.out_run}")
    print(f"Motion primitives v1: {summary.get('motion_primitive_v1_count')}; review_v15={(summary.get('review_v15') or {}).get('count')}; validation={summary.get('partner_relative_flow_validation_passed')}")
    return 0


def cmd_ingest_v15_human_findings(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.clean_v3_calibration_v1 import ingest_v15_human_findings

    summary = ingest_v15_human_findings(args.review_dir)
    print(f"v15 human findings stored: {args.review_dir}")
    print(f"Review items: {summary['review_items']}")
    return 0


def cmd_rebuild_clean_v3_semantic_actions_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.clean_v3_calibration_v1 import rebuild_clean_v3_semantic_actions_v1

    summary = rebuild_clean_v3_semantic_actions_v1(args.run_dir, args.previous_review)
    print(f"clean_v3 semantic actions v1 rebuilt: {args.run_dir}")
    print(f"Semantic actions: {summary['semantic_actions']}")
    print(f"Cowgirl DB v6 categories: {summary['cowgirl_db_counts']}")
    return 0


def cmd_export_semantic_review_v16(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.clean_v3_calibration_v1 import export_semantic_review_v16

    summary = export_semantic_review_v16(
        args.run_dir,
        args.out_dir,
        count=args.count,
        build_vam_package=_arg_bool(args.build_vam_package),
        previous_review=args.previous_review,
    )
    print(f"Semantic review v16 written: {args.out_dir}")
    print(f"Review items: {summary['review_items']}; categories={summary['category_counts']}")
    return 0


def cmd_run_clean_v3_calibration_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.clean_v3_calibration_v1 import run_clean_v3_calibration_v1

    summary = run_clean_v3_calibration_v1(args.run_dir, args.previous_review, args.out_review)
    print(f"clean_v3 calibration v1 complete: {args.run_dir}")
    print(f"v16 review: {args.out_review}; items={summary['v16_review']['review_items']}")
    print(f"Cowgirl DB v6 categories: {summary['rebuild']['cowgirl_db_counts']}")
    return 0


def cmd_run_clean_v3_v16_calibration(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.clean_v3_v16_calibration import run_clean_v3_v16_calibration

    summary = run_clean_v3_v16_calibration(args.run_dir, args.previous_review, args.out_review)
    print(f"clean_v3 v16 calibration complete: {args.run_dir}")
    print(f"v17 review: {args.out_review}; items={summary['v17_review']['review_items']}")
    print(f"Clean motion gate counts: {summary['rebuild']['clean_motion_gate_counts']}")
    print(f"Cowgirl DB v7 categories: {summary['rebuild']['cowgirl_db_counts']}")
    return 0


def cmd_run_clean_v3_pose_support_rescan(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.pose_support_rescan import run_clean_v3_pose_support_rescan

    summary = run_clean_v3_pose_support_rescan(args.run_dir, args.out_suffix)
    print(f"clean_v3 pose/support rescan complete: {args.run_dir}")
    print(f"Cowgirl DB v8 categories: {summary['cowgirl_candidate_db_v8_counts']}")
    print(f"Focused review: {summary['focused_review']['review_dir']}")
    return 0


def cmd_compare_new_scenes_to_clean_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.new_scene_delta_report import compare_new_scenes_to_clean_v3

    summary = compare_new_scenes_to_clean_v3(args.base_run, args.new_run, args.out)
    print(f"New scene delta report written: {args.out}")
    print(f"New scenes: {summary.get('new_scenes')}; motion sources: {summary.get('new_sources')}; windows: {summary.get('new_windows')}")
    return 0


def cmd_run_new_scenes_delta_import(args: argparse.Namespace) -> int:
    from vam_timeline_ai.runs.new_scenes_delta_import import run_new_scenes_delta_import

    summary = run_new_scenes_delta_import(args.raw_dir, args.base_run, args.out_run)
    print(f"New scenes delta import complete: {args.out_run}")
    print(f"Scenes: {summary.get('scene_count_found')}; sources: {summary.get('motion_sources')}; windows: {summary.get('movement_windows')}")
    print(f"Review items: {((summary.get('review') or {}).get('review_items'))}")
    return 0


def cmd_build_focused_new_scenes_review(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.new_scene_review_planner import build_focused_new_scenes_review

    summary = build_focused_new_scenes_review(args.run_dir, args.previous_review, args.out_dir)
    print(f"Focused new-scenes review written: {args.out_dir}")
    print(f"Selected: {summary['selected']}; counts={summary['selected_counts']}")
    print(f"Static UI: {(summary['static_review_ui'] or {}).get('index')}")
    return 0


def cmd_build_strict_new_scenes_cowgirl_review(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.new_scene_review_planner import build_strict_new_scenes_cowgirl_review

    summary = build_strict_new_scenes_cowgirl_review(
        args.run_dir,
        args.out_dir,
        args.previous_review,
        args.human_answers,
        args.batch_size,
        args.batch_index,
    )
    print(f"Strict new-scenes Cowgirl review written: {args.out_dir}")
    print(f"Selected: {summary['selected']}; counts={summary['selected_counts']}")
    if summary.get("batch_size"):
        print(f"Batch: {summary['batch_index']} of size {summary['batch_size']} from {summary['full_selected_count']} selected candidates")
    print(f"Human answers used: {summary['human_answer_count']}; excluded windows={summary['human_exclusion_windows']}")
    print(f"Static UI: {(summary['static_review_ui'] or {}).get('index')}")
    return 0


def cmd_resolve_new_scenes_pose_first_semantics_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.new_scenes_semantic_rescan_v2 import resolve_new_scenes_pose_first_semantics_v2

    summary = resolve_new_scenes_pose_first_semantics_v2(
        args.new_run,
        args.base_run,
        args.ontology,
        args.rules,
        args.manual_gt,
        args.out_jsonl,
        args.report,
    )
    if summary.get("status") == "blocked":
        print(f"New-scenes semantic rescan blocked: {summary.get('blocked_report')}")
        print(f"Missing inputs: {summary.get('missing')}")
        return 0
    print(f"New-scenes pose-first v2 resolved: {summary['out_jsonl']}")
    print(f"Records: {summary['records']}; families={summary['family_counts']}")
    return 0


def cmd_build_new_scenes_ontology_candidate_db_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.new_scenes_semantic_rescan_v2 import build_new_scenes_ontology_candidate_db_v2

    summary = build_new_scenes_ontology_candidate_db_v2(
        args.new_run,
        args.resolved,
        args.ontology,
        args.manual_gt,
        args.out_jsonl,
        args.out_csv,
        args.report,
    )
    print(f"New-scenes ontology candidate DB v2 written: {summary['out_jsonl']}")
    print(f"Records: {summary['records']}; categories={summary['category_counts']}")
    return 0


def cmd_write_new_scenes_family_reports_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.new_scenes_semantic_rescan_v2 import write_new_scenes_family_reports_v2

    summary = write_new_scenes_family_reports_v2(args.new_run, args.candidates, args.out_dir)
    print(f"New-scenes family reports v2 written: {summary['reports_dir']}")
    print(f"Drift report: {summary['drift_report']}")
    return 0


def cmd_export_new_scenes_semantic_review_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.new_scenes_semantic_rescan_v2 import export_new_scenes_semantic_review_v2

    summary = export_new_scenes_semantic_review_v2(
        args.new_run,
        args.candidates,
        args.out_dir,
        count=args.count,
        build_vam_package=_as_bool(args.build_vam_package),
        build_static_ui=_as_bool(args.build_static_ui),
    )
    print(f"New-scenes semantic review v2 written: {summary['out_dir']}")
    print(f"Items: {summary['review_items']}; static UI={(summary['static_ui'] or {}).get('index')}")
    return 0


def cmd_extract_motion_cycle_features_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.features.motion_cycle_features import extract_motion_cycle_features_v1

    summary = extract_motion_cycle_features_v1(args.run_dir, args.out_jsonl, args.report)
    print(f"Motion cycle features written: {summary['out_jsonl']}")
    print(f"Records: {summary['records']}; loaded_npz={summary['loaded_npz']}; missing_npz={summary['missing_npz']}")
    return 0


def cmd_resolve_new_scenes_motion_semantics_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.motion_semantic_resolver import resolve_new_scenes_motion_semantics_v1

    summary = resolve_new_scenes_motion_semantics_v1(
        args.new_run,
        args.pose_resolved,
        args.cycle_features,
        args.ontology,
        args.cycle_rules,
        args.manual_gt,
        args.out_jsonl,
        args.report,
        relational_features=args.relational_features,
    )
    print(f"Motion semantics v1 written: {summary['out_jsonl']}")
    print(f"Records: {summary['records']}; states={summary['motion_state_counts']}")
    return 0


def cmd_build_new_scenes_motion_candidate_db_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.motion_semantic_resolver import build_new_scenes_motion_candidate_db_v1

    summary = build_new_scenes_motion_candidate_db_v1(args.new_run, args.motion_resolved, args.out_jsonl, args.out_csv, args.report)
    print(f"Motion candidate DB v1 written: {summary['out_jsonl']}")
    print(f"Records: {summary['records']}; categories={summary['category_counts']}")
    return 0


def cmd_export_motion_semantics_review_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.motion_semantic_resolver import export_motion_semantics_review_v1

    summary = export_motion_semantics_review_v1(
        args.new_run,
        args.candidates,
        args.out_dir,
        count=args.count,
        build_static_ui=_as_bool(args.build_static_ui),
        build_vam_package=_as_bool(args.build_vam_package),
    )
    print(f"Motion semantics review v1 written: {summary['out_dir']}")
    print(f"Items: {summary['review_items']}; static UI={(summary['static_ui'] or {}).get('index')}")
    return 0


def cmd_export_review_timeline_segments_to_vam(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.vam_timeline_segment_export import export_review_timeline_segments_to_vam

    summary = export_review_timeline_segments_to_vam(args.review_dir, args.vam_animations_dir, args.run_dir, args.subdir)
    print(f"Review Timeline segments copied to: {summary['target_dir']}")
    print(f"Copied: {summary['copied']}; unavailable: {summary['unavailable']}")
    print(f"Index: {summary['index_jsonl']}")
    return 0


def cmd_sanitize_run_scene_identifiers(args: argparse.Namespace) -> int:
    from vam_timeline_ai.runs.sanitize_scene_identifiers import sanitize_run_scene_identifiers

    summary = sanitize_run_scene_identifiers(
        args.run_dir,
        args.alias_map_out,
        args.report,
        dry_run=_as_bool(args.dry_run),
    )
    print(f"Scene identifiers sanitized: {summary['run_dir']}")
    print(f"Changed files: {summary['changed_files']}; aliases: {summary['scene_aliases']}; dry_run={summary['dry_run']}")
    print(f"Report: {summary['report']}")
    return 0


def cmd_build_reviewed_window_index(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.review_deduplication import build_reviewed_window_index

    summary = build_reviewed_window_index(args.run_dir, args.include_runs, args.out_jsonl, args.out_csv, args.report)
    print(f"Reviewed window index written: {args.out_jsonl}")
    print(f"Records: {summary['records']}; exact={summary['exact_duplicate_count']}; near={summary['near_duplicate_count']}")
    return 0


def cmd_audit_review_duplicates(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.review_deduplication import audit_review_duplicates

    summary = audit_review_duplicates(args.reviewed_index, args.out)
    print(f"Review duplicate audit written: {args.out}")
    print(f"Total: {summary['total']}; exact={summary['exact_duplicate_count']}; near={summary['near_duplicate_count']}")
    return 0


def cmd_export_strict_novel_review(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.review_deduplication import export_strict_novel_review

    summary = export_strict_novel_review(
        args.run_dir,
        args.candidate_db,
        args.reviewed_index,
        args.out_dir,
        count=args.count,
        max_per_scene=args.max_per_scene,
        max_per_sample=args.max_per_sample,
        allow_reviewed_overlap=_as_bool(args.allow_reviewed_overlap),
        allow_near_duplicates=_as_bool(args.allow_near_duplicates),
        build_vam_package=_as_bool(args.build_vam_package),
        build_static_ui=_as_bool(args.build_static_ui),
        diversity_mode=args.diversity_mode,
    )
    print(f"Strict novel review written: {args.out_dir}")
    print(f"Exported: {summary['exported_count']}/{summary['requested_count']}; trust={summary['quality']['trust_level']}")
    return 0


def cmd_build_human_reviewed_ml_labels_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.human_label_dataset import build_human_reviewed_ml_labels_v1

    summary = build_human_reviewed_ml_labels_v1(args.run_dir, args.human_ledger, args.out_jsonl, args.report)
    print(f"Human-reviewed ML labels written: {args.out_jsonl}")
    print(f"Rows={summary.get('total_human_reviewed')}; cowgirl={summary.get('cowgirl_label_counts')}; clean={summary.get('clean_motion_label_counts')}")
    return 0


def cmd_build_cowgirl_ml_feature_table_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.supervised_feature_table import build_cowgirl_ml_feature_table_v1

    summary = build_cowgirl_ml_feature_table_v1(
        args.run_dir,
        args.labels,
        args.relative_features,
        args.trajectory_features,
        args.pose_features,
        args.pose_semantics,
        args.partner_relative_features,
        args.interaction_semantics,
        args.semantic_actions,
        args.candidate_db,
        args.out_npz,
        args.out_meta,
        args.report,
    )
    print(f"Cowgirl ML feature table written: {args.out_npz}")
    print(f"Rows={summary.get('rows')}; features={summary.get('features')}; shape={summary.get('shape')}")
    return 0


def cmd_build_cowgirl_ml_labels_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_ml_label_dataset_v2 import build_cowgirl_ml_labels_v2

    summary = build_cowgirl_ml_labels_v2(args.base_run, args.new_run, args.human_ledger, args.manual_gt, args.out_jsonl, args.report)
    print(f"Cowgirl ML labels v2 written: {args.out_jsonl}")
    print(f"Rows={summary.get('rows')}; counts={summary.get('label_counts', {}).get('label_cowgirl_semantic_family')}")
    return 0


def cmd_build_cowgirl_ml_labels_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_ml_label_dataset_v3 import build_cowgirl_ml_labels_v3

    summary = build_cowgirl_ml_labels_v3(args.new_run, args.out_jsonl, args.report)
    print(f"Cowgirl ML labels v3 written: {args.out_jsonl}")
    print(f"Rows={summary.get('rows')}; counts={summary.get('label_counts', {}).get('label_cowgirl_clean_motion')}")
    return 0


def cmd_build_cowgirl_ml_feature_table_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_ml_feature_table_v2 import build_cowgirl_ml_feature_table_v2

    summary = build_cowgirl_ml_feature_table_v2(
        args.labels,
        args.pose_resolved,
        args.cycle_features,
        args.motion_resolved,
        args.candidates,
        args.manual_gt,
        args.out_npz,
        args.out_meta,
        args.report,
    )
    print(f"Cowgirl ML feature table v2 written: {args.out_npz}")
    print(f"Rows={summary.get('rows')}; features={summary.get('features')}; shape={summary.get('shape')}; labeled={summary.get('labeled_rows')}")
    return 0


def cmd_train_cowgirl_ml_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_ml_model_v2 import train_cowgirl_ml_v2

    summary = train_cowgirl_ml_v2(args.feature_table, args.metadata, args.out_dir)
    print(f"Cowgirl ML v2 output: {args.out_dir}")
    print(f"Trained={summary.get('trained')}; verdict={summary.get('trust_verdict')}")
    return 0


def cmd_score_new_scenes_cowgirl_ml_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_ml_model_v2 import score_new_scenes_cowgirl_ml_v2

    summary = score_new_scenes_cowgirl_ml_v2(args.model_dir, args.feature_table, args.metadata, args.out_jsonl, args.report)
    print(f"Cowgirl ML v2 scores written: {args.out_jsonl}")
    print(f"Status={summary.get('status')}; rows={summary.get('rows')}; buckets={summary.get('bucket_counts')}")
    return 0


def cmd_export_ml_assisted_cowgirl_review_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_ml_review_v2 import export_ml_assisted_cowgirl_review_v2

    summary = export_ml_assisted_cowgirl_review_v2(
        args.new_run,
        args.scores,
        args.candidates,
        args.out_dir,
        args.count,
        build_static_ui=_as_bool(args.build_static_ui),
        build_vam_package=_as_bool(args.build_vam_package),
    )
    print(f"ML-assisted Cowgirl review v2 written: {args.out_dir}")
    print(f"Selected={summary.get('selected')}; buckets={summary.get('bucket_counts')}")
    return 0


def cmd_extract_relational_semantic_features_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.features.relational_semantic_features import extract_relational_semantic_features_v1

    summary = extract_relational_semantic_features_v1(
        args.run_dir,
        args.pair_windows,
        args.pair_features,
        args.sample_index,
        args.controller_map,
        args.out_jsonl,
        args.report,
    )
    print(f"Relational semantic features written: {args.out_jsonl}")
    print(f"Rows={summary.get('rows')}; pair_errors={summary.get('pair_errors')}")
    return 0


def cmd_extract_rig_anatomy_features_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.features.rig_anatomy_features import extract_rig_anatomy_features_v1

    summary = extract_rig_anatomy_features_v1(
        args.run_dir,
        args.anatomy,
        args.roles,
        args.out_jsonl,
        args.report,
    )
    print(f"Rig anatomy features written: {args.out_jsonl}")
    print(f"Rows={summary.get('records')}; dominant_regions={summary.get('dominant_regions')}")
    return 0


def cmd_build_nlp_lexicon_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.nlp.external_lexicon_import import build_nlp_lexicon_v1

    summary = build_nlp_lexicon_v1(
        args.manual,
        args.sources,
        args.out,
        args.report,
        allow_web=_as_bool(args.allow_web),
    )
    print(f"NLP lexicon written: {args.out}")
    print(f"Active={summary.get('active_entries')}; candidate={summary.get('candidate_entries')}")
    return 0


def cmd_resolve_nlp_tokens_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.nlp.nlp_token_resolver import resolve_nlp_tokens_v1

    result = resolve_nlp_tokens_v1(args.prompt, args.lexicon, args.component_ontology, args.out)
    print(f"NLP token resolution written: {args.out}")
    print(f"Matches={len(result.get('matches') or [])}; unresolved={result.get('unresolved_requirements')}")
    return 0


def cmd_build_motion_intent_from_prompt_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.nlp.nlp_token_resolver import build_motion_intent_from_prompt_v1

    result = build_motion_intent_from_prompt_v1(args.prompt, args.lexicon, args.component_ontology, args.out)
    plan = result.get("intent_plan") or {}
    print(f"Motion intent plan written: {args.out}")
    phases = plan.get("phases") or plan.get("sequence_phases") or []
    family = ((phases[0] or {}).get("base_state") or {}).get("family") if phases else None
    print(f"Family={family}; phases={len(phases)}")
    return 0


def cmd_collect_web_motion_context_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.research.web_context_research import collect_web_motion_context_v1

    summary = collect_web_motion_context_v1(
        args.topics,
        args.out_dir,
        allow_web=_as_bool(args.allow_web),
        max_sources_per_category=args.max_sources_per_category,
    )
    print(f"Web-context research written: {args.out_dir}")
    print(f"Cards={summary.get('cards')}; blocked={summary.get('blocked_fetches')}")
    return 0


def cmd_build_web_context_ontology_patches_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.research.web_context_ontology_patches import build_web_context_ontology_patches_v1

    summary = build_web_context_ontology_patches_v1(
        args.research_dir,
        args.current_ontology,
        args.current_anatomy,
        args.out_yaml,
        args.report,
    )
    print(f"Inactive web-context patch candidates written: {args.out_yaml}")
    print(f"Candidates={summary.get('patch_candidates')}; accepted={summary.get('accepted_candidates', 0)}")
    return 0


def cmd_build_research_client_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ui.research_client import build_research_client_v0

    summary = build_research_client_v0(args.run_dir, args.new_run, args.out_dir)
    print(f"Research client written: {summary.get('index')}")
    return 0


def cmd_split_cowgirl_ml_dataset_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.grouped_splits import split_cowgirl_ml_dataset_v1

    summary = split_cowgirl_ml_dataset_v1(args.feature_table, args.metadata, args.out_dir, args.group_by, args.seed)
    print(f"Cowgirl ML split written: {args.out_dir}")
    print(f"Status={summary.get('status')}; split_counts={summary.get('split_counts')}; leakage={summary.get('leakage_warnings')}")
    return 0


def cmd_train_cowgirl_ml_baseline_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_baseline_model import train_cowgirl_ml_baseline_v1

    summary = train_cowgirl_ml_baseline_v1(args.feature_table, args.metadata, args.splits, args.out_dir)
    print(f"Cowgirl ML baseline output: {args.out_dir}")
    print(f"Trained={summary.get('trained')}; model_type={summary.get('model_type', summary.get('reason'))}")
    return 0


def cmd_score_clean_v3_with_cowgirl_model_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.cowgirl_model_scoring import score_clean_v3_with_cowgirl_model_v1

    summary = score_clean_v3_with_cowgirl_model_v1(args.run_dir, args.model_dir, args.feature_source, args.out_jsonl, args.report)
    print(f"Cowgirl model scores written: {args.out_jsonl}")
    print(f"Status={summary.get('status')}; rows={summary.get('rows')}; priorities={summary.get('priority_counts')}")
    return 0


def cmd_export_ml_assisted_cowgirl_review_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.ml_assisted_review_batch import export_ml_assisted_cowgirl_review_v1

    summary = export_ml_assisted_cowgirl_review_v1(
        args.run_dir,
        args.model_scores,
        args.reviewed_index,
        args.out_dir,
        count=args.count,
        max_per_scene=args.max_per_scene,
        max_per_sample=args.max_per_sample,
        build_vam_package=_as_bool(args.build_vam_package),
        build_static_ui=_as_bool(args.build_static_ui),
    )
    print(f"ML-assisted Cowgirl review written: {args.out_dir}")
    print(f"Exported={summary.get('exported_count')}/{summary.get('requested_count')}; buckets={summary.get('bucket_counts')}")
    return 0


def cmd_evaluate_ml_assisted_review_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.ml_review_evaluation import evaluate_ml_assisted_review_v1

    summary = evaluate_ml_assisted_review_v1(args.review_dir, args.model_scores, args.answers, args.out)
    print(f"ML-assisted review evaluation written: {args.out}")
    print(f"Answers={summary.get('answers')}; helped={summary.get('ml_v1_helped')}; buckets={list((summary.get('bucket_stats') or {}).keys())}")
    return 0


def cmd_run_cowgirl_ml_active_learning_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ml.active_learning_loop import run_cowgirl_ml_active_learning_v2

    summary = run_cowgirl_ml_active_learning_v2(args.run_dir, args.review_dir, args.out_dir)
    print(f"Cowgirl ML active learning v2 output: {args.out_dir}")
    print(f"Status={summary.get('status')}; answers_found={summary.get('answers_found')}")
    return 0


def cmd_build_human_review_ledger(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.human_review_memory import build_human_review_ledger

    summary = build_human_review_ledger(args.run_dir, args.include_runs, args.out_jsonl, args.out_csv, args.report)
    print(f"Human review ledger written: {args.out_jsonl}")
    print(f"Records: {summary['records']}; known verdicts={summary['known_human_verdicts']}")
    return 0


def cmd_build_error_taxonomy_report(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.error_taxonomy import build_error_taxonomy_report

    summary = build_error_taxonomy_report(args.human_review_ledger, args.out)
    print(f"Error taxonomy report written: {args.out}")
    print(f"Top items: {summary['top_items']}")
    return 0


def cmd_validate_semantic_dbs(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.db_invariant_validator import validate_semantic_dbs

    summary = validate_semantic_dbs(args.run_dir, args.semantic_db, args.cowgirl_db, args.out)
    print(f"Semantic DB invariant report written: {args.out}")
    print(f"Errors={summary['errors']}; warnings={summary['warnings']}")
    return 0


def cmd_write_clean_v3_dashboard(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.semantic_qa_dashboard import write_clean_v3_dashboard

    summary = write_clean_v3_dashboard(args.run_dir, args.out_md, args.out_html)
    print(f"clean_v3 dashboard written: {args.out_md}")
    print(f"Semantic records={summary['semantic_records']}; Cowgirl records={summary['cowgirl_records']}")
    return 0


def cmd_compare_clean_v2_clean_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.run_drift_report import compare_clean_v2_clean_v3

    summary = compare_clean_v2_clean_v3(args.clean_v2, args.clean_v3, args.out)
    print(f"Drift report written: {args.out}")
    print(f"v2 Cowgirl={summary['v2_cowgirl_records']}; v3 Cowgirl={summary['v3_cowgirl_records']}")
    return 0


def cmd_plan_larger_review_batch_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.review_batch_planner import plan_larger_review_batch_v1

    summary = plan_larger_review_batch_v1(args.run_dir, args.semantic_db, args.cowgirl_db, args.out)
    print(f"Larger review batch plan written: {args.out}")
    print(f"Planned={summary['planned_total']} / target={summary['target_total']}")
    return 0


def cmd_write_prompt_capability_matrix(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.prompt_capability_matrix import write_prompt_capability_matrix

    write_prompt_capability_matrix(args.run_dir, args.out)
    print(f"Prompt capability matrix written: {args.out}")
    return 0


def cmd_clean_v3_status(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.clean_v3_status import clean_v3_status

    summary = clean_v3_status(args.run_dir)
    print(f"clean_v3 status written: {summary['out']}")
    print(f"Run exists={summary['run_exists']}; blockers={len(summary['blockers'])}")
    return 0


def cmd_run_clean_v3_overnight_qa(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.clean_v3_overnight_qa import run_clean_v3_overnight_qa

    summary = run_clean_v3_overnight_qa(args.run_dir, args.include_runs)
    print(f"clean_v3 overnight QA summary written: {summary['summary']}")
    return 0


def cmd_write_candidate_lineage_report(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.candidate_lineage import write_candidate_lineage_report

    summary = write_candidate_lineage_report(args.run_dir, args.out)
    print(f"Candidate lineage report written: {summary['out']}")
    return 0


def cmd_run_clean_v3_reproducibility_audit(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.reproducibility_audit import run_clean_v3_reproducibility_audit

    summary = run_clean_v3_reproducibility_audit(args.run_dir)
    print(f"clean_v3 reproducibility audit summary written: {summary['summary']}")
    return 0


def cmd_launch_review_ui(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ui.review_ui import launch_review_ui

    launch_review_ui(args.run_dir, args.review_dir, args.host, args.port)
    return 0


def cmd_build_static_review_ui(args: argparse.Namespace) -> int:
    from vam_timeline_ai.ui.review_ui import build_static_review_ui

    summary = build_static_review_ui(args.run_dir, args.review_dir, args.out_dir)
    print(f"Static review UI written: {summary['index']}")
    print(f"Review items: {summary['review_items']}; candidate rows: {summary['candidate_rows']}")
    return 0


def cmd_render_digital_twin_review_previews_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.digital_twin_preview import render_digital_twin_review_previews_v0

    summary = render_digital_twin_review_previews_v0(args.run_dir, args.review_dir, args.out_dir)
    print(f"Digital twin previews written: {summary['out_dir']}")
    print(f"Rendered: {summary['previews_rendered']}; unavailable: {summary['could_not_visualize']}")
    print(f"Report written: {summary['report']}")
    if summary.get("static_review_ui"):
        print(f"Static UI refreshed: {summary['static_review_ui'].get('index')}")
    return 0


def cmd_render_digital_twin_previews_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.digital_twin_preview import render_digital_twin_previews_v1

    summary = render_digital_twin_previews_v1(
        args.run_dir,
        args.review_dir,
        args.out_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        frames=args.frames,
        make_gif=_as_bool(args.make_gif),
        make_mp4=args.make_mp4,
        view=args.view,
    )
    print(f"Digital twin animated previews written: {summary['out_dir']}")
    print(
        "Rendered: "
        f"{summary['previews_rendered']}; GIFs: {summary['gif_created']}; "
        f"MP4s: {summary['mp4_created']}; sheets: {summary['contact_sheets_created']}"
    )
    print(f"Report written: {summary['report']}")
    if summary.get("static_review_ui"):
        print(f"Static UI refreshed: {summary['static_review_ui'].get('index')}")
    return 0


def cmd_build_visual_judge_requests_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.visual_judge_requests import build_visual_judge_requests_v0

    summary = build_visual_judge_requests_v0(args.review_dir, args.preview_dir, args.out_jsonl, mode=args.mode)
    print(f"Visual judge requests written: {summary['out_jsonl']}")
    print(f"Requests: {summary['requests']}; primary types: {summary['by_primary_visual_type']}")
    print(f"Report written: {summary['report']}")
    return 0


def cmd_build_vam_capture_requests_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.vision.vam_capture_requests import build_vam_capture_requests_v0

    summary = build_vam_capture_requests_v0(args.review_dir, args.out_jsonl, args.output_root, args.frame_count, args.duration_seconds)
    print(f"VaM capture requests written: {summary['out_jsonl']}")
    print(f"Requests: {summary['requests']}")
    return 0


def cmd_run_vam_reality_capture_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.vision.vam_capture_bridge_client import run_vam_reality_capture_v0

    summary = run_vam_reality_capture_v0(args.requests, args.bridge_url, args.mode, args.out)
    print(f"VaM reality capture status: {summary['status']}")
    print(f"Bridge available: {summary.get('bridge_available')}; out: {summary['out']}")
    if summary.get("blocked_report"):
        print(f"Blocked report: {summary['blocked_report']}")
    return 0


def cmd_build_vam_capture_contact_sheets_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.vision.vam_capture_contact_sheet import build_vam_capture_contact_sheets_v0

    summary = build_vam_capture_contact_sheets_v0(args.capture_results, args.out_dir)
    print(f"VaM capture contact sheets written: {summary['out_dir']}")
    print(f"Sheets: {summary['sheets']}; GIFs: {summary['gifs']}")
    return 0


def cmd_import_manual_pose_captures_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.manual_pose_capture_importer import import_manual_pose_captures_v1

    summary = import_manual_pose_captures_v1(args.input_dir, args.out_jsonl, args.report)
    print(f"Manual pose captures imported: {summary['captures']}")
    print(f"Invalid files: {summary['invalid_files']}; report: {summary['report']}")
    return 0


def cmd_report_manual_pose_captures_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.manual_pose_capture_report import report_manual_pose_captures_v1

    summary = report_manual_pose_captures_v1(args.captures, args.out)
    print(f"Manual pose capture report written: {summary['out']}")
    print(f"Captures: {summary['captures']}; families: {summary['family_counts']}")
    return 0


def cmd_extract_manual_pose_captures_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.manual_pose_extraction import extract_manual_pose_captures_v1

    summary = extract_manual_pose_captures_v1(args.zip, args.out_dir)
    print(f"Manual pose captures extracted: {summary['files_extracted']}")
    print(f"JSON: {summary['json_count']}; PNG: {summary['png_count']}; report: {summary['report']}")
    return 0


def cmd_parse_manual_pose_explanations_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.manual_pose_explanation_parser import parse_manual_pose_explanations_v1

    summary = parse_manual_pose_explanations_v1(args.explanations, args.out_json, args.out_yaml, args.report)
    print(f"Manual pose labels parsed: {summary['labels']}")
    print(f"Families: {summary['family_counts']}; report: {summary['report']}")
    return 0


def cmd_build_manual_pose_ground_truth_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.datasets.manual_pose_ground_truth import build_manual_pose_ground_truth_v1

    summary = build_manual_pose_ground_truth_v1(args.capture_dir, args.human_labels, args.out_jsonl, args.out_csv, args.report)
    print(f"Manual pose ground truth rows: {summary['captures']}")
    print(f"Matched labels: {summary['matched_labels']}; ontology patch: {summary['ontology_patch']}")
    return 0


def cmd_report_manual_pose_ground_truth_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.manual_pose_ground_truth_report import report_manual_pose_ground_truth_v1

    summary = report_manual_pose_ground_truth_v1(args.ground_truth, args.out_dir)
    print(f"Manual pose ground-truth reports written: {summary['out_dir']}")
    print(f"Captures: {summary['captures']}; families: {summary['family_counts']}")
    return 0


def cmd_build_manual_pose_ground_truth_gallery_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.manual_pose_ground_truth_report import build_manual_pose_ground_truth_gallery_v1

    summary = build_manual_pose_ground_truth_gallery_v1(args.ground_truth, args.out_html)
    print(f"Manual pose ground-truth gallery written: {summary['out_html']}")
    print(f"Captures: {summary['captures']}")
    return 0


def cmd_build_visual_judge_requests_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.vision.visual_judge_requests import build_visual_judge_requests_v1

    summary = build_visual_judge_requests_v1(args.review_dir, args.vam_capture_sheets, args.digital_twin_previews, args.out_jsonl, mode=args.mode)
    print(f"Visual judge requests v1 written: {summary['out_jsonl']}")
    print(f"Requests: {summary['requests']}; primary types: {summary['by_primary_visual_type']}")
    return 0


def cmd_run_lmstudio_vlm_judge_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.vision.lmstudio_vlm_judge import run_lmstudio_vlm_judge_v0

    summary = run_lmstudio_vlm_judge_v0(args.requests, args.base_url, args.model, args.out_jsonl, args.out_raw_dir, dry_run=_as_bool(args.dry_run))
    print(f"LM Studio VLM judge status: {summary['status']}")
    print(f"Model: {summary.get('model', args.model)}; requests: {summary.get('requests')}; out: {summary.get('out_jsonl')}")
    if summary.get("blocked_report"):
        print(f"Blocked report: {summary['blocked_report']}")
    return 0


def cmd_build_visual_judge_calibration_set_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.vision.visual_judge_calibration import build_visual_judge_calibration_set_v1

    summary = build_visual_judge_calibration_set_v1(args.run_dir, args.out_dir)
    print(f"Visual judge calibration set written: {summary['out_dir']}")
    print(f"Items: {summary['items']}; with visual path: {summary['with_visual_path']}")
    return 0


def cmd_evaluate_vlm_visual_judge_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.vision.visual_judge_trust_gate import evaluate_vlm_visual_judge_v1

    summary = evaluate_vlm_visual_judge_v1(args.calibration_set, args.base_url, args.model, args.out_dir, dry_run=_as_bool(args.dry_run))
    print(f"VLM trust gate: {summary['trust_gate']}")
    print(f"Out dir: {args.out_dir}")
    return 0


def cmd_build_multisignal_review_priorities_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.multisignal_triage import build_multisignal_review_priorities_v0

    summary = build_multisignal_review_priorities_v0(args.run_dir, args.review_dir, args.model_scores, args.visual_results, args.out_jsonl, args.report)
    print(f"Multisignal priorities written: {summary['out_jsonl']}")
    print(f"Priority counts: {summary['priority_counts']}")
    return 0


def cmd_translate_motion_intent_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.motion_intent_translator import translate_motion_intent_v1

    summary = translate_motion_intent_v1(args.prompt, args.ontology, args.phrases, args.out)
    print(f"Motion intent plan written: {summary['out']}")
    print(f"Family: {summary['family']}; motion: {summary['motion_subtype']}; pose: {summary['pose_subtype']}; contact: {summary['contact_support']}")
    if summary.get("invalid_mappings_prevented"):
        print(f"Invalid mappings prevented: {summary['invalid_mappings_prevented']}")
    return 0


def cmd_ingest_semantik_sourcebook_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.ontology_sourcebook import ingest_semantik_sourcebook_v2

    summary = ingest_semantik_sourcebook_v2(args.source_docx, args.out_dir, args.report)
    print(f"Semantik sourcebook ingestion status: {summary['status']}")
    print(f"Report written: {args.report}")
    if summary.get("manifest"):
        print(f"Manifest: {summary['manifest']}")
    return 0


def cmd_build_semantic_stickman_pose_library_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.semantic_pose_library import build_semantic_stickman_pose_library_v1

    summary = build_semantic_stickman_pose_library_v1(args.ontology, args.out_json, args.report)
    print(f"Semantic stickman pose library written: {summary['out_json']}")
    print(f"Poses: {summary['pose_count']}; families: {summary['family_counts']}")
    return 0


def cmd_build_semantic_motion_examples_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.semantic_motion_examples import build_semantic_motion_examples_v1

    summary = build_semantic_motion_examples_v1(args.pose_library, args.ontology, args.out_json, args.report)
    print(f"Semantic motion examples written: {summary['out_json']}")
    print(f"Examples: {summary['example_count']}; families: {summary['family_counts']}")
    return 0


def cmd_render_semantic_stickman_previews_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.semantic_stickman_renderer import render_semantic_stickman_previews_v1

    summary = render_semantic_stickman_previews_v1(
        args.motion_examples,
        args.out_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        make_gif=_as_bool(args.make_gif),
        make_contact_sheet=_as_bool(args.make_contact_sheet),
    )
    print(f"Semantic stickman previews written: {summary['out_dir']}")
    print(f"Rendered: {summary['rendered']}; GIFs: {summary['gif_count']}; contact sheets: {summary['contact_sheet_count']}")
    return 0


def cmd_validate_semantic_stickman_examples_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.semantic_stickman_validation import validate_semantic_stickman_examples_v1

    summary = validate_semantic_stickman_examples_v1(args.motion_examples, args.ontology, args.out)
    print(f"Semantic stickman validation written: {summary['out']}")
    print(f"Status: {summary['status']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_build_semantic_stickman_gallery_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.semantic_stickman_gallery import build_semantic_stickman_gallery_v1

    summary = build_semantic_stickman_gallery_v1(args.preview_dir, args.out_html, args.out_md)
    print(f"Semantic stickman gallery written: {summary['out_html']}")
    print(f"Items: {summary['items']}; families: {summary['families']}")
    return 0


def cmd_render_semantic_stickman_previews_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.semantic_stickman_renderer import render_semantic_stickman_previews_v2

    summary = render_semantic_stickman_previews_v2(
        args.motion_examples,
        args.out_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        make_gif=_as_bool(args.make_gif),
        make_contact_sheet=_as_bool(args.make_contact_sheet),
        show_labels=_as_bool(args.show_labels),
        show_partner=_as_bool(args.show_partner),
        show_alignment=_as_bool(args.show_alignment),
        show_support_targets=_as_bool(args.show_support_targets),
    )
    print(f"Semantic stickman v2 previews written: {summary['out_dir']}")
    print(f"Rendered: {summary['rendered']}; GIFs: {summary['gif_count']}; contact sheets: {summary['contact_sheet_count']}")
    return 0


def cmd_validate_semantic_stickman_examples_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.semantic_stickman_validation import validate_semantic_stickman_examples_v2

    summary = validate_semantic_stickman_examples_v2(args.motion_examples, args.preview_dir, args.ontology, args.out)
    print(f"Semantic stickman v2 validation written: {summary['out']}")
    print(f"Status: {summary['status']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_build_semantic_stickman_gallery_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.semantic_stickman_gallery import build_semantic_stickman_gallery_v2

    summary = build_semantic_stickman_gallery_v2(args.preview_dir, args.out_html, args.out_md)
    print(f"Semantic stickman v2 gallery written: {summary['out_html']}")
    print(f"Items: {summary['items']}; families: {summary['families']}")
    return 0


def cmd_build_semantic_motion_examples_v2_contact_aware(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.semantic_motion_examples import build_semantic_motion_examples_v2_contact_aware

    summary = build_semantic_motion_examples_v2_contact_aware(args.pose_library, args.ontology, args.out_json, args.report)
    print(f"Contact-aware semantic motion examples written: {summary['out_json']}")
    print(f"Examples: {summary['example_count']}; valid alignments: {summary['alignment_valid']}; invalid: {summary['alignment_invalid']}")
    return 0


def cmd_render_semantic_stickman_previews_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.visualization.semantic_stickman_renderer import render_semantic_stickman_previews_v3

    summary = render_semantic_stickman_previews_v3(
        args.motion_examples,
        args.out_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        make_gif=_as_bool(args.make_gif),
        make_contact_sheet=_as_bool(args.make_contact_sheet),
        show_labels=_as_bool(args.show_labels),
        show_partner=_as_bool(args.show_partner),
        show_alignment=_as_bool(args.show_alignment),
        show_support_targets=_as_bool(args.show_support_targets),
        show_contact_zone=_as_bool(args.show_contact_zone),
        show_alignment_tolerance=_as_bool(args.show_alignment_tolerance),
        show_validity_overlay=_as_bool(args.show_validity_overlay),
        contact_aware=_as_bool(args.contact_aware),
    )
    print(f"Semantic stickman v3 previews written: {summary['out_dir']}")
    print(f"Rendered: {summary['rendered']}; GIFs: {summary['gif_count']}; contact sheets: {summary['contact_sheet_count']}")
    return 0


def cmd_validate_semantic_stickman_examples_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.semantic_stickman_validation import validate_semantic_stickman_examples_v3

    summary = validate_semantic_stickman_examples_v3(args.motion_examples, args.preview_dir, args.ontology, args.out)
    print(f"Semantic stickman v3 validation written: {summary['out']}")
    print(f"Status: {summary['status']}; valid: {summary['valid']}; invalid: {summary['invalid']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_build_semantic_stickman_gallery_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.reports.semantic_stickman_gallery import build_semantic_stickman_gallery_v3

    summary = build_semantic_stickman_gallery_v3(args.preview_dir, args.out_html, args.out_md)
    print(f"Semantic stickman v3 gallery written: {summary['out_html']}")
    print(f"Items: {summary['items']}; families: {summary['families']}")
    return 0


def cmd_export_vam_semantic_preview_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.vam_semantic_preview_exporter import export_vam_semantic_preview_v0

    summary = export_vam_semantic_preview_v0(args.motion_examples, args.out_dir, duration=args.duration, fps=args.fps)
    print(f"VaM semantic preview package written: {summary['out_dir']}")
    print(f"Timeline clips exported: {summary['exported_clips']}; blocked: {summary['blocked_clips']}")
    print(f"Import instructions: {summary['import_instructions']}")
    return 0


def cmd_validate_vam_semantic_preview_v0(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.vam_semantic_preview_validation import validate_vam_semantic_preview_v0

    summary = validate_vam_semantic_preview_v0(args.preview_dir, args.out)
    print(f"VaM semantic preview validation written: {summary['out']}")
    print(f"Status: {summary['status']}; clips: {summary['clips']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_export_manual_gt_timeline_examples_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_exporter import export_manual_gt_timeline_examples_v1

    summary = export_manual_gt_timeline_examples_v1(args.ground_truth, args.out_dir, duration=args.duration, fps=args.fps, copy_to_vam=_as_bool(args.copy_to_vam))
    print(f"Manual GT Timeline clips exported: {summary['clips_exported']}")
    print(f"Out dir: {summary['out_dir']}")
    if summary.get("copied_to_vam"):
        print(f"Copied to VaM: {summary['copied_to_vam']}")
    if summary.get("skipped"):
        print(f"Skipped: {summary['skipped']}")
    return 0


def cmd_validate_manual_gt_timeline_examples_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_validation import validate_manual_gt_timeline_examples_v1

    summary = validate_manual_gt_timeline_examples_v1(args.preview_dir, args.out)
    print(f"Manual GT Timeline validation written: {summary['out']}")
    print(f"Status: {summary['status']}; clips: {summary['clips']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_export_manual_gt_timeline_examples_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_exporter import export_manual_gt_timeline_examples_v2

    summary = export_manual_gt_timeline_examples_v2(
        args.ground_truth,
        args.out_dir,
        duration=args.duration,
        keyframe_rate=args.keyframe_rate,
        copy_to_vam=_as_bool(args.copy_to_vam),
        include_rotations=_as_bool(args.include_rotations),
        allow_dense_export=_as_bool(args.allow_dense_export),
    )
    print(f"Manual GT Timeline v2 clips exported: {summary['clips_exported']}")
    print(f"Out dir: {summary['out_dir']}")
    print(f"Rotation source report: {summary['rotation_source_report']}")
    if summary.get("copied_to_vam"):
        print(f"Copied to VaM: {summary['copied_to_vam']}")
    if summary.get("skipped"):
        print(f"Skipped: {summary['skipped']}")
    return 0


def cmd_validate_manual_gt_timeline_examples_v2(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_validation import validate_manual_gt_timeline_examples_v2

    summary = validate_manual_gt_timeline_examples_v2(args.preview_dir, args.out, allow_dense_export=_as_bool(args.allow_dense_export))
    print(f"Manual GT Timeline v2 validation written: {summary['out']}")
    print(f"Status: {summary['status']}; clips: {summary['clips']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_export_manual_gt_timeline_examples_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_exporter import export_manual_gt_timeline_examples_v3

    summary = export_manual_gt_timeline_examples_v3(
        args.ground_truth,
        args.out_dir,
        duration=args.duration,
        keyframe_rate=args.keyframe_rate,
        copy_to_vam=_as_bool(args.copy_to_vam),
        include_rotations=_as_bool(args.include_rotations),
        require_hip_control=_as_bool(args.require_hip_control),
        allow_high_key_density=_as_bool(args.allow_high_key_density),
        allow_dense_export=_as_bool(args.allow_dense_export),
    )
    print(f"Manual GT Timeline v3 clips exported: {summary['clips_exported']}")
    print(f"Out dir: {summary['out_dir']}")
    print(f"Rotation source report: {summary['rotation_source_report']}")
    print(f"Keyframe rate: {summary['keyframe_rate']}")
    if summary.get("copied_to_vam"):
        print(f"Copied to VaM: {summary['copied_to_vam']}")
    if summary.get("skipped"):
        print(f"Skipped: {summary['skipped']}")
    return 0


def cmd_validate_manual_gt_timeline_examples_v3(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_validation import validate_manual_gt_timeline_examples_v3

    summary = validate_manual_gt_timeline_examples_v3(
        args.preview_dir,
        args.out,
        allow_high_key_density=_as_bool(args.allow_high_key_density),
        allow_dense_export=_as_bool(args.allow_dense_export),
    )
    print(f"Manual GT Timeline v3 validation written: {summary['out']}")
    print(f"Status: {summary['status']}; clips: {summary['clips']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_export_manual_gt_timeline_examples_v4(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_exporter import export_manual_gt_timeline_examples_v4

    summary = export_manual_gt_timeline_examples_v4(
        args.ground_truth,
        args.out_dir,
        duration=args.duration,
        keyframe_rate=args.keyframe_rate,
        copy_to_vam=_as_bool(args.copy_to_vam),
        include_rotations=_as_bool(args.include_rotations),
        require_hip_control=_as_bool(args.require_hip_control),
        amplitude_profile=args.amplitude_profile,
        allow_high_key_density=_as_bool(args.allow_high_key_density),
        allow_dense_export=_as_bool(args.allow_dense_export),
    )
    print(f"Manual GT Timeline v4 clips exported: {summary['clips_exported']}")
    print(f"Out dir: {summary['out_dir']}")
    print(f"Rotation source report: {summary['rotation_source_report']}")
    print(f"Motion amplitude profile report: {summary['motion_amplitude_profile_report']}")
    print(f"Keyframe rate: {summary['keyframe_rate']}")
    if summary.get("copied_to_vam"):
        print(f"Copied to VaM: {summary['copied_to_vam']}")
    if summary.get("skipped"):
        print(f"Skipped: {summary['skipped']}")
    return 0


def cmd_validate_manual_gt_timeline_examples_v4(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.manual_gt_timeline_validation import validate_manual_gt_timeline_examples_v4

    summary = validate_manual_gt_timeline_examples_v4(
        args.preview_dir,
        args.out,
        allow_high_key_density=_as_bool(args.allow_high_key_density),
        allow_dense_export=_as_bool(args.allow_dense_export),
    )
    print(f"Manual GT Timeline v4 validation written: {summary['out']}")
    print(f"Status: {summary['status']}; clips: {summary['clips']}; errors: {summary['errors']}; warnings: {summary['warnings']}")
    return 0 if summary["status"] == "ok" else 1


def cmd_resolve_pose_first_semantics_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.pose_first_resolver import resolve_pose_first_semantics_v1

    summary = resolve_pose_first_semantics_v1(
        args.run_dir,
        args.pose_semantics,
        args.relative_features,
        args.interaction_semantics,
        args.candidate_db,
        args.rules,
        args.out_jsonl,
        args.report,
    )
    print(f"Pose-first resolved semantics written: {summary['out_jsonl']}")
    print(f"Records: {summary['records']}; families: {summary['family_counts']}; conflicts: {summary['conflicts']}")
    print(f"Report written: {summary['report']}")
    return 0


def cmd_align_candidates_to_motion_ontology_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.semantics.ontology_alignment import align_candidates_to_motion_ontology_v1

    summary = align_candidates_to_motion_ontology_v1(
        args.run_dir,
        args.ontology,
        args.semantic_db,
        args.cowgirl_db,
        args.resolved,
        args.out_jsonl,
        args.report,
    )
    print(f"Ontology-aligned candidates written: {summary['out_jsonl']}")
    print(f"Records: {summary['records']}; matches: {summary['match_counts']}; priorities: {summary['priority_counts']}")
    print(f"Report written: {summary['report']}")
    return 0


def cmd_calibrate_motion_parameters_v1(args: argparse.Namespace) -> int:
    from vam_timeline_ai.generation.motion_parameter_calibration import calibrate_motion_parameters_v1

    summary = calibrate_motion_parameters_v1(
        args.run_dir,
        args.ontology,
        args.resolved,
        args.relative_features,
        args.trajectory_features,
        args.human_ledger,
        args.out_json,
        args.report,
    )
    print(f"Motion parameter profiles written: {summary['out_json']}")
    print(f"Profiles: {summary['profiles']}; insufficient: {summary['insufficient']}; skipped: {summary['skipped']}")
    print(f"Report written: {summary['report']}")
    return 0


def cmd_ingest_review_ui_answers(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.review_answer_ingestion import ingest_review_ui_answers

    summary = ingest_review_ui_answers(args.answers, args.review_dir, args.out_ledger, args.report, overwrite=_as_bool(args.overwrite))
    print(f"Review UI answers ingested: {summary['answers']} answer(s), {summary['new_ledger_records']} new ledger record(s), skipped={summary.get('duplicates_skipped')}")
    print(f"Report written: {summary['report']}")
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


def cmd_build_vam_review_package(args: argparse.Namespace) -> int:
    from vam_timeline_ai.audits.vam_review_package import build_vam_review_package

    summary = build_vam_review_package(
        args.review,
        args.run_dir,
        args.source_run,
        args.out_dir,
        attempt_timeline_segments=_arg_bool(args.attempt_timeline_segments),
    )
    print(f"VaM review package written: {args.out_dir}")
    print(
        "Items: "
        f"{summary['review_items']}; scenes={summary['scene_count']}; "
        f"Timeline segments={summary['timeline_segments_successful']} successful / "
        f"{summary['timeline_segments_attempted']} attempted"
    )
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
    if args.command == "build-vam-review-package":
        return cmd_build_vam_review_package(args)
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
    if args.command == "classify-bj-oral-domain":
        return cmd_classify_bj_oral_domain(args)
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
    if args.command == "score-cowgirl-candidates-v11":
        return cmd_score_cowgirl_candidates_v11(args)
    if args.command == "build-cowgirl-candidate-db-v1":
        return cmd_build_cowgirl_candidate_db_v1(args)
    if args.command == "build-cowgirl-candidate-db-v2":
        return cmd_build_cowgirl_candidate_db_v2(args)
    if args.command == "build-cowgirl-candidate-db-v3":
        return cmd_build_cowgirl_candidate_db_v3(args)
    if args.command == "build-semantic-candidate-db-v0":
        return cmd_build_semantic_candidate_db_v0(args)
    if args.command == "extract-cowgirl-motion-primitives-v0":
        return cmd_extract_cowgirl_motion_primitives_v0(args)
    if args.command == "group-cowgirl-motion-primitives-v0":
        return cmd_group_cowgirl_motion_primitives_v0(args)
    if args.command == "draft-motion-plan-v0":
        return cmd_draft_motion_plan_v0(args)
    if args.command == "retrieve-primitives-for-plan-v0":
        return cmd_retrieve_primitives_for_plan_v0(args)
    if args.command == "generate-motion-flow-skeleton-v0":
        return cmd_generate_motion_flow_skeleton_v0(args)
    if args.command == "synthesize-motion-flow-v0":
        return cmd_synthesize_motion_flow_v0(args)
    if args.command == "synthesize-motion-flow-v1":
        return cmd_synthesize_motion_flow_v1(args)
    if args.command == "validate-generated-motion-flow-v0":
        return cmd_validate_generated_motion_flow_v0(args)
    if args.command == "render-generated-motion-preview-v0":
        return cmd_render_generated_motion_preview_v0(args)
    if args.command == "create-synthetic-baseline-pose-v0":
        return cmd_create_synthetic_baseline_pose_v0(args)
    if args.command == "create-cowgirl-review-baseline-pose-v1":
        return cmd_create_cowgirl_review_baseline_pose_v1(args)
    if args.command == "retarget-motion-flow-v0":
        return cmd_retarget_motion_flow_v0(args)
    if args.command == "retarget-motion-flow-v1":
        return cmd_retarget_motion_flow_v1(args)
    if args.command == "validate-retargeted-motion-flow-v0":
        return cmd_validate_retargeted_motion_flow_v0(args)
    if args.command == "validate-retargeted-motion-flow-v1":
        return cmd_validate_retargeted_motion_flow_v1(args)
    if args.command == "render-retargeted-motion-preview-v0":
        return cmd_render_retargeted_motion_preview_v0(args)
    if args.command == "render-retargeted-motion-preview-v1":
        return cmd_render_retargeted_motion_preview_v1(args)
    if args.command == "export-retargeted-flow-timeline-v0":
        return cmd_export_retargeted_flow_timeline_v0(args)
    if args.command == "export-generated-flow-for-vam-review":
        return cmd_export_generated_flow_for_vam_review(args)
    if args.command == "export-generated-flow-for-vam-review-v1":
        return cmd_export_generated_flow_for_vam_review_v1(args)
    if args.command == "prepare-vam-review-player-v0":
        return cmd_prepare_vam_review_player_v0(args)
    if args.command == "run-first-generated-motion-review-v0":
        return cmd_run_first_generated_motion_review_v0(args)
    if args.command == "run-cowgirl-motion-flow-v1-review":
        return cmd_run_cowgirl_motion_flow_v1_review(args)
    if args.command == "export-generated-flow-native-timeline-v0":
        return cmd_export_generated_flow_native_timeline_v0(args)
    if args.command == "export-generated-flow-native-timeline-v1":
        return cmd_export_generated_flow_native_timeline_v1(args)
    if args.command == "validate-native-timeline-export-v0":
        return cmd_validate_native_timeline_export_v0(args)
    if args.command == "validate-native-timeline-export-v1":
        return cmd_validate_native_timeline_export_v1(args)
    if args.command == "run-native-timeline-export-review-v0":
        return cmd_run_native_timeline_export_review_v0(args)
    if args.command == "run-native-timeline-export-review-v1":
        return cmd_run_native_timeline_export_review_v1(args)
    if args.command == "extract-pose-features-v0":
        return cmd_extract_pose_features_v0(args)
    if args.command == "classify-poses-v0":
        return cmd_classify_poses_v0(args)
    if args.command == "extract-partner-relative-features-v0":
        return cmd_extract_partner_relative_features_v0(args)
    if args.command == "classify-interactions-v0":
        return cmd_classify_interactions_v0(args)
    if args.command == "build-semantic-actions-v0":
        return cmd_build_semantic_actions_v0(args)
    if args.command == "build-cowgirl-candidate-db-v5":
        return cmd_build_cowgirl_candidate_db_v5(args)
    if args.command == "extract-cowgirl-motion-primitives-v1":
        return cmd_extract_cowgirl_motion_primitives_v1(args)
    if args.command == "draft-motion-plan-v1":
        return cmd_draft_motion_plan_v1(args)
    if args.command == "select-interaction-baseline-for-plan-v0":
        return cmd_select_interaction_baseline_for_plan_v0(args)
    if args.command == "synthesize-partner-relative-flow-v0":
        return cmd_synthesize_partner_relative_flow_v0(args)
    if args.command == "validate-partner-relative-flow-v0":
        return cmd_validate_partner_relative_flow_v0(args)
    if args.command == "run-semantic-rescan-v1":
        return cmd_run_semantic_rescan_v1(args)
    if args.command == "ingest-v15-human-findings":
        return cmd_ingest_v15_human_findings(args)
    if args.command == "rebuild-clean-v3-semantic-actions-v1":
        return cmd_rebuild_clean_v3_semantic_actions_v1(args)
    if args.command == "export-semantic-review-v16":
        return cmd_export_semantic_review_v16(args)
    if args.command == "run-clean-v3-calibration-v1":
        return cmd_run_clean_v3_calibration_v1(args)
    if args.command == "run-clean-v3-v16-calibration":
        return cmd_run_clean_v3_v16_calibration(args)
    if args.command == "run-clean-v3-pose-support-rescan":
        return cmd_run_clean_v3_pose_support_rescan(args)
    if args.command == "compare-new-scenes-to-clean-v3":
        return cmd_compare_new_scenes_to_clean_v3(args)
    if args.command == "run-new-scenes-delta-import":
        return cmd_run_new_scenes_delta_import(args)
    if args.command == "build-focused-new-scenes-review":
        return cmd_build_focused_new_scenes_review(args)
    if args.command == "build-strict-new-scenes-cowgirl-review":
        return cmd_build_strict_new_scenes_cowgirl_review(args)
    if args.command == "resolve-new-scenes-pose-first-semantics-v2":
        return cmd_resolve_new_scenes_pose_first_semantics_v2(args)
    if args.command == "build-new-scenes-ontology-candidate-db-v2":
        return cmd_build_new_scenes_ontology_candidate_db_v2(args)
    if args.command == "write-new-scenes-family-reports-v2":
        return cmd_write_new_scenes_family_reports_v2(args)
    if args.command == "export-new-scenes-semantic-review-v2":
        return cmd_export_new_scenes_semantic_review_v2(args)
    if args.command == "extract-motion-cycle-features-v1":
        return cmd_extract_motion_cycle_features_v1(args)
    if args.command == "extract-relational-semantic-features-v1":
        return cmd_extract_relational_semantic_features_v1(args)
    if args.command == "extract-rig-anatomy-features-v1":
        return cmd_extract_rig_anatomy_features_v1(args)
    if args.command == "build-nlp-lexicon-v1":
        return cmd_build_nlp_lexicon_v1(args)
    if args.command == "resolve-nlp-tokens-v1":
        return cmd_resolve_nlp_tokens_v1(args)
    if args.command == "build-motion-intent-from-prompt-v1":
        return cmd_build_motion_intent_from_prompt_v1(args)
    if args.command == "collect-web-motion-context-v1":
        return cmd_collect_web_motion_context_v1(args)
    if args.command == "build-web-context-ontology-patches-v1":
        return cmd_build_web_context_ontology_patches_v1(args)
    if args.command == "build-research-client-v0":
        return cmd_build_research_client_v0(args)
    if args.command == "resolve-new-scenes-motion-semantics-v1":
        return cmd_resolve_new_scenes_motion_semantics_v1(args)
    if args.command == "build-new-scenes-motion-candidate-db-v1":
        return cmd_build_new_scenes_motion_candidate_db_v1(args)
    if args.command == "export-motion-semantics-review-v1":
        return cmd_export_motion_semantics_review_v1(args)
    if args.command == "export-review-timeline-segments-to-vam":
        return cmd_export_review_timeline_segments_to_vam(args)
    if args.command == "sanitize-run-scene-identifiers":
        return cmd_sanitize_run_scene_identifiers(args)
    if args.command == "build-reviewed-window-index":
        return cmd_build_reviewed_window_index(args)
    if args.command == "audit-review-duplicates":
        return cmd_audit_review_duplicates(args)
    if args.command == "export-strict-novel-review":
        return cmd_export_strict_novel_review(args)
    if args.command == "build-human-reviewed-ml-labels-v1":
        return cmd_build_human_reviewed_ml_labels_v1(args)
    if args.command == "build-cowgirl-ml-feature-table-v1":
        return cmd_build_cowgirl_ml_feature_table_v1(args)
    if args.command == "build-cowgirl-ml-labels-v2":
        return cmd_build_cowgirl_ml_labels_v2(args)
    if args.command == "build-cowgirl-ml-labels-v3":
        return cmd_build_cowgirl_ml_labels_v3(args)
    if args.command == "build-cowgirl-ml-feature-table-v2":
        return cmd_build_cowgirl_ml_feature_table_v2(args)
    if args.command == "train-cowgirl-ml-v2":
        return cmd_train_cowgirl_ml_v2(args)
    if args.command == "score-new-scenes-cowgirl-ml-v2":
        return cmd_score_new_scenes_cowgirl_ml_v2(args)
    if args.command == "export-ml-assisted-cowgirl-review-v2":
        return cmd_export_ml_assisted_cowgirl_review_v2(args)
    if args.command == "split-cowgirl-ml-dataset-v1":
        return cmd_split_cowgirl_ml_dataset_v1(args)
    if args.command == "train-cowgirl-ml-baseline-v1":
        return cmd_train_cowgirl_ml_baseline_v1(args)
    if args.command == "score-clean-v3-with-cowgirl-model-v1":
        return cmd_score_clean_v3_with_cowgirl_model_v1(args)
    if args.command == "export-ml-assisted-cowgirl-review-v1":
        return cmd_export_ml_assisted_cowgirl_review_v1(args)
    if args.command == "evaluate-ml-assisted-review-v1":
        return cmd_evaluate_ml_assisted_review_v1(args)
    if args.command == "run-cowgirl-ml-active-learning-v2":
        return cmd_run_cowgirl_ml_active_learning_v2(args)
    if args.command == "build-human-review-ledger":
        return cmd_build_human_review_ledger(args)
    if args.command == "build-error-taxonomy-report":
        return cmd_build_error_taxonomy_report(args)
    if args.command == "validate-semantic-dbs":
        return cmd_validate_semantic_dbs(args)
    if args.command == "write-clean-v3-dashboard":
        return cmd_write_clean_v3_dashboard(args)
    if args.command == "compare-clean-v2-clean-v3":
        return cmd_compare_clean_v2_clean_v3(args)
    if args.command == "plan-larger-review-batch-v1":
        return cmd_plan_larger_review_batch_v1(args)
    if args.command == "write-prompt-capability-matrix":
        return cmd_write_prompt_capability_matrix(args)
    if args.command == "clean-v3-status":
        return cmd_clean_v3_status(args)
    if args.command == "run-clean-v3-overnight-qa":
        return cmd_run_clean_v3_overnight_qa(args)
    if args.command == "write-candidate-lineage-report":
        return cmd_write_candidate_lineage_report(args)
    if args.command == "run-clean-v3-reproducibility-audit":
        return cmd_run_clean_v3_reproducibility_audit(args)
    if args.command == "launch-review-ui":
        return cmd_launch_review_ui(args)
    if args.command == "build-static-review-ui":
        return cmd_build_static_review_ui(args)
    if args.command == "render-digital-twin-review-previews-v0":
        return cmd_render_digital_twin_review_previews_v0(args)
    if args.command == "render-digital-twin-previews-v1":
        return cmd_render_digital_twin_previews_v1(args)
    if args.command == "build-visual-judge-requests-v0":
        return cmd_build_visual_judge_requests_v0(args)
    if args.command == "build-vam-capture-requests-v0":
        return cmd_build_vam_capture_requests_v0(args)
    if args.command == "run-vam-reality-capture-v0":
        return cmd_run_vam_reality_capture_v0(args)
    if args.command == "build-vam-capture-contact-sheets-v0":
        return cmd_build_vam_capture_contact_sheets_v0(args)
    if args.command == "import-manual-pose-captures-v1":
        return cmd_import_manual_pose_captures_v1(args)
    if args.command == "report-manual-pose-captures-v1":
        return cmd_report_manual_pose_captures_v1(args)
    if args.command == "extract-manual-pose-captures-v1":
        return cmd_extract_manual_pose_captures_v1(args)
    if args.command == "parse-manual-pose-explanations-v1":
        return cmd_parse_manual_pose_explanations_v1(args)
    if args.command == "build-manual-pose-ground-truth-v1":
        return cmd_build_manual_pose_ground_truth_v1(args)
    if args.command == "report-manual-pose-ground-truth-v1":
        return cmd_report_manual_pose_ground_truth_v1(args)
    if args.command == "build-manual-pose-ground-truth-gallery-v1":
        return cmd_build_manual_pose_ground_truth_gallery_v1(args)
    if args.command == "build-visual-judge-requests-v1":
        return cmd_build_visual_judge_requests_v1(args)
    if args.command == "run-lmstudio-vlm-judge-v0":
        return cmd_run_lmstudio_vlm_judge_v0(args)
    if args.command == "build-visual-judge-calibration-set-v1":
        return cmd_build_visual_judge_calibration_set_v1(args)
    if args.command == "evaluate-vlm-visual-judge-v1":
        return cmd_evaluate_vlm_visual_judge_v1(args)
    if args.command == "build-multisignal-review-priorities-v0":
        return cmd_build_multisignal_review_priorities_v0(args)
    if args.command == "translate-motion-intent-v1":
        return cmd_translate_motion_intent_v1(args)
    if args.command == "ingest-semantik-sourcebook-v2":
        return cmd_ingest_semantik_sourcebook_v2(args)
    if args.command == "build-semantic-stickman-pose-library-v1":
        return cmd_build_semantic_stickman_pose_library_v1(args)
    if args.command == "build-semantic-motion-examples-v1":
        return cmd_build_semantic_motion_examples_v1(args)
    if args.command == "render-semantic-stickman-previews-v1":
        return cmd_render_semantic_stickman_previews_v1(args)
    if args.command == "validate-semantic-stickman-examples-v1":
        return cmd_validate_semantic_stickman_examples_v1(args)
    if args.command == "build-semantic-stickman-gallery-v1":
        return cmd_build_semantic_stickman_gallery_v1(args)
    if args.command == "render-semantic-stickman-previews-v2":
        return cmd_render_semantic_stickman_previews_v2(args)
    if args.command == "validate-semantic-stickman-examples-v2":
        return cmd_validate_semantic_stickman_examples_v2(args)
    if args.command == "build-semantic-stickman-gallery-v2":
        return cmd_build_semantic_stickman_gallery_v2(args)
    if args.command == "build-semantic-motion-examples-v2-contact-aware":
        return cmd_build_semantic_motion_examples_v2_contact_aware(args)
    if args.command == "render-semantic-stickman-previews-v3":
        return cmd_render_semantic_stickman_previews_v3(args)
    if args.command == "validate-semantic-stickman-examples-v3":
        return cmd_validate_semantic_stickman_examples_v3(args)
    if args.command == "build-semantic-stickman-gallery-v3":
        return cmd_build_semantic_stickman_gallery_v3(args)
    if args.command == "export-vam-semantic-preview-v0":
        return cmd_export_vam_semantic_preview_v0(args)
    if args.command == "validate-vam-semantic-preview-v0":
        return cmd_validate_vam_semantic_preview_v0(args)
    if args.command == "export-manual-gt-timeline-examples-v1":
        return cmd_export_manual_gt_timeline_examples_v1(args)
    if args.command == "validate-manual-gt-timeline-examples-v1":
        return cmd_validate_manual_gt_timeline_examples_v1(args)
    if args.command == "export-manual-gt-timeline-examples-v2":
        return cmd_export_manual_gt_timeline_examples_v2(args)
    if args.command == "validate-manual-gt-timeline-examples-v2":
        return cmd_validate_manual_gt_timeline_examples_v2(args)
    if args.command == "export-manual-gt-timeline-examples-v3":
        return cmd_export_manual_gt_timeline_examples_v3(args)
    if args.command == "validate-manual-gt-timeline-examples-v3":
        return cmd_validate_manual_gt_timeline_examples_v3(args)
    if args.command == "export-manual-gt-timeline-examples-v4":
        return cmd_export_manual_gt_timeline_examples_v4(args)
    if args.command == "validate-manual-gt-timeline-examples-v4":
        return cmd_validate_manual_gt_timeline_examples_v4(args)
    if args.command == "resolve-pose-first-semantics-v1":
        return cmd_resolve_pose_first_semantics_v1(args)
    if args.command == "align-candidates-to-motion-ontology-v1":
        return cmd_align_candidates_to_motion_ontology_v1(args)
    if args.command == "calibrate-motion-parameters-v1":
        return cmd_calibrate_motion_parameters_v1(args)
    if args.command == "ingest-review-ui-answers":
        return cmd_ingest_review_ui_answers(args)
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

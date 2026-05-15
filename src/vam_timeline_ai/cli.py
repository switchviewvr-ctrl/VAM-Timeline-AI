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
    if args.command == "discover-controller-map":
        return cmd_discover_controller_map(args)
    if args.command == "extract-cowgirl-features-v1":
        return cmd_extract_cowgirl_features_v1(args)
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
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

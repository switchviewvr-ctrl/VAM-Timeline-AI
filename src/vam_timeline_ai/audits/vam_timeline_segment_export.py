"""Copy review-only Timeline segments into VaM's PluginData animations folder."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv
import shutil

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui import build_static_review_ui


def export_review_timeline_segments_to_vam(
    review_dir: str | Path,
    vam_animations_dir: str | Path,
    run_dir: str | Path | None = None,
    subdir: str | Path | None = None,
) -> dict[str, Any]:
    review = Path(review_dir).resolve()
    animations_root = Path(vam_animations_dir).resolve()
    target = animations_root / (Path(subdir) if subdir else Path("VAMTimelineAI") / review.name)
    target.mkdir(parents=True, exist_ok=True)

    package = review / "vam_review_package"
    manifest_path = package / "vam_review_manifest.jsonl"
    review_path = review / "semantic_review_010.jsonl"
    manifest = load_jsonl(manifest_path)
    review_rows = load_jsonl(review_path)
    copied: list[dict[str, Any]] = []

    review_by_id = {row.get("review_id"): row for row in review_rows if row.get("review_id")}
    for item in manifest:
        rid = str(item.get("review_id") or "")
        src_text = str(item.get("timeline_export_path") or "")
        if not rid or not src_text:
            copied.append(_status(item, None, "unavailable", "timeline segment missing"))
            continue
        src = Path(src_text)
        if src_text and not src.is_absolute():
            candidates = [
                (Path.cwd() / src).resolve(),
                (review / src).resolve(),
                (package / src).resolve(),
            ]
            src = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not src.exists() or src.is_dir():
            copied.append(_status(item, None, "unavailable", "timeline segment missing"))
            continue
        dest_name = _segment_name(item)
        dest = target / dest_name
        if dest.exists():
            dest = _dedupe_path(dest)
        shutil.copy2(src, dest)
        rel_dest = str(dest)
        item["vam_animation_path"] = rel_dest
        item["vam_animation_export_status"] = "copied"
        if rid in review_by_id:
            review_by_id[rid]["vam_animation_path"] = rel_dest
            review_by_id[rid]["vam_animation_export_status"] = "copied"
        copied.append(_status(item, dest, "copied", "ok"))

    write_jsonl(manifest_path, manifest)
    if review_rows:
        write_jsonl(review_path, list(review_by_id.values()))
    write_jsonl(target / "vam_timeline_segment_index.jsonl", copied)
    _write_csv(target / "vam_timeline_segment_index.csv", copied)
    _write_report(target / "README.md", review, target, copied)
    _write_report(review / "vam_timeline_segment_export_to_plugin_data.md", review, target, copied)
    if run_dir:
        build_static_review_ui(run_dir, review, review / "review_ui_static")

    counts = Counter(row.get("status") for row in copied)
    return {
        "status": "ok",
        "review_dir": str(review),
        "vam_animations_dir": str(animations_root),
        "target_dir": str(target),
        "total_items": len(copied),
        "copied": counts.get("copied", 0),
        "unavailable": counts.get("unavailable", 0),
        "index_jsonl": str(target / "vam_timeline_segment_index.jsonl"),
        "index_csv": str(target / "vam_timeline_segment_index.csv"),
        "report": str(target / "README.md"),
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }


def _segment_name(item: dict[str, Any]) -> str:
    label = item.get("review_label") or item.get("semantic_review_label")
    if label:
        return f"{_safe_stem(str(label))}.timeline.json"
    rid = str(item.get("review_id") or "review_000").replace("review_", "")
    bucket = _short_bucket(str(item.get("why_selected") or item.get("category") or "candidate"))
    return f"{rid}_{bucket}.timeline.json"


def _short_bucket(bucket: str) -> str:
    names = {
        "cowgirl_clean_motion_generation_safe": "cowgirl_clean",
        "cowgirl_clean_motion_low_confidence_short": "cowgirl_short",
        "cowgirl_pose_context_low_motion": "cowgirl_pose_low_motion",
        "cowgirl_transition_setup": "cowgirl_transition",
        "not_cowgirl_standing_hand_head": "not_cowgirl_standing",
        "not_cowgirl_bj_oral": "not_cowgirl_bj_oral",
        "unknown_or_unusable_high_movement": "unknown_high_motion",
        "contact_support_ambiguous": "contact_ambiguous",
    }
    return names.get(bucket, _safe_stem(bucket))


def _safe_stem(text: str) -> str:
    keep = []
    for ch in text.replace(".json", ""):
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out[:80] or "item"


def _dedupe_path(path: Path) -> Path:
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 1000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique file path for {path}")


def _status(item: dict[str, Any], dest: Path | None, status: str, reason: str) -> dict[str, Any]:
    return {
        "review_id": item.get("review_id"),
        "status": status,
        "reason": reason,
        "vam_animation_path": str(dest) if dest else "",
        "source_timeline_segment": item.get("timeline_export_path"),
        "source_scene_file": item.get("source_scene_file"),
        "technical_atom_id": item.get("technical_atom_id"),
        "start_seconds": item.get("start_seconds"),
        "end_seconds": item.get("end_seconds"),
        "why_selected": item.get("why_selected"),
        "semantic_family": item.get("semantic_family"),
        "contact_support": item.get("contact_support"),
        "review_only": True,
        "not_training_truth": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "review_id",
        "status",
        "vam_animation_path",
        "source_scene_file",
        "technical_atom_id",
        "start_seconds",
        "end_seconds",
        "why_selected",
        "semantic_family",
        "contact_support",
        "source_timeline_segment",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(path: Path, review: Path, target: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row.get("status") for row in rows)
    lines = [
        "# VaM Timeline Segment Copies",
        "",
        "These are review-only Timeline source segments copied for manual VaM inspection.",
        "They are not generated motion, not ML labels, and not training truth.",
        "",
        f"- Review folder: `{review}`",
        f"- VaM animations target: `{target}`",
        f"- Total review items: {len(rows)}",
        f"- Copied: {counts.get('copied', 0)}",
        f"- Unavailable: {counts.get('unavailable', 0)}",
        "",
        "## How To Test In VaM",
        "",
        "1. Open VaM.",
        "2. Add/select the Timeline plugin on the relevant Person atom.",
        "3. Import a `.timeline.json` file from the target folder above.",
        "4. Use the review UI card to match scene, actor, and time range.",
        "5. Treat the result as a source-segment review aid only.",
        "",
        "## Files",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row.get('review_id')}`: `{row.get('status')}` - `{row.get('vam_animation_path') or row.get('reason')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Reports and gallery for manual VaM pose ground-truth captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import html
import os

from vam_timeline_ai.io.json_utils import load_jsonl


FAMILY_REPORTS = {
    "cowgirl": "cowgirl_pose_ground_truth.md",
    "doggy": "doggy_pose_ground_truth.md",
    "bj_oral": "bj_oral_pose_ground_truth.md",
    "handjob": "handjob_pose_ground_truth.md",
    "missionary": "missionary_pose_ground_truth.md",
}


def report_manual_pose_ground_truth_v1(ground_truth: str | Path, out_dir: str | Path) -> dict[str, Any]:
    rows = load_jsonl(ground_truth)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary = _summarize(rows, ground_truth, target)
    _write_overview(summary, rows, target / "overview.md")
    for family, filename in FAMILY_REPORTS.items():
        _write_family_report(family, [row for row in rows if _family(row) == family], target / filename)
    _write_ontology_suggestions(rows, target / "ontology_corrections_suggested.md")
    return summary


def build_manual_pose_ground_truth_gallery_v1(ground_truth: str | Path, out_html: str | Path) -> dict[str, Any]:
    rows = load_jsonl(ground_truth)
    target = Path(out_html)
    target.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        labels = row.get("human_labels") or {}
        partner = ((row.get("measurements") or {}).get("partner_relative") or {})
        hands = ((row.get("measurements") or {}).get("hand_target_candidates") or {})
        img = _relative_or_abs(row.get("screenshot_path") or "", target.parent)
        raw_json = _relative_or_abs(row.get("raw_capture_path") or "", target.parent)
        notes = html.escape(str(labels.get("raw_notes") or ""))
        cards.append(
            f"""
      <article class="card">
        <h2>{html.escape(str(row.get('capture_id')))}</h2>
        <div class="meta">{html.escape(str(labels.get('family')))} / {html.escape(str(labels.get('pose_subtype')))}</div>
        {'<img src="' + html.escape(img) + '" alt="capture screenshot">' if img else '<div class="missing">No screenshot</div>'}
        <dl>
          <dt>Motion intent</dt><dd>{html.escape(str(labels.get('motion_intent')))}</dd>
          <dt>Primary driver</dt><dd>{html.escape(str(labels.get('primary_driver')))}</dd>
          <dt>Anchors</dt><dd>{html.escape(', '.join(labels.get('anchors') or []))}</dd>
          <dt>Pelvis distance</dt><dd>{html.escape(str(partner.get('rider_pelvis_to_partner_pelvis_distance')))}</dd>
          <dt>L hand nearest</dt><dd>{html.escape(str((hands.get('lHandControl') or {}).get('nearest_target')))}</dd>
          <dt>R hand nearest</dt><dd>{html.escape(str((hands.get('rHandControl') or {}).get('nearest_target')))}</dd>
        </dl>
        <pre>{notes}</pre>
        <p><a href="{html.escape(raw_json)}">raw JSON</a></p>
      </article>"""
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Manual Pose Ground Truth V1</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f6f3; color: #1d1d1b; }}
    h1 {{ margin-bottom: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8d8d0; border-radius: 8px; padding: 14px; }}
    img {{ width: 100%; max-height: 320px; object-fit: contain; background: #111; border-radius: 4px; }}
    .meta {{ color: #555; margin-bottom: 10px; font-weight: 600; }}
    dl {{ display: grid; grid-template-columns: 125px 1fr; gap: 4px 10px; font-size: 14px; }}
    dt {{ font-weight: 700; }}
    pre {{ white-space: pre-wrap; background: #f1f1ea; padding: 8px; border-radius: 4px; font-size: 13px; }}
    .notice {{ margin: 10px 0 20px; color: #555; }}
  </style>
</head>
<body>
  <h1>Manual Pose Ground Truth V1</h1>
  <p class="notice">Human pose captures only. No ML training, no auto-labeling, no Timeline generation.</p>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    target.write_text(page, encoding="utf-8")
    return {"status": "ok", "captures": len(rows), "out_html": str(target)}


def _summarize(rows: list[dict[str, Any]], ground_truth: str | Path, out_dir: Path) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    subtype_counts: dict[str, int] = {}
    screenshots = 0
    matched = 0
    completeness_counts: dict[str, int] = {}
    distances_by_family: dict[str, list[float]] = {}
    for row in rows:
        family = _family(row)
        subtype = str((row.get("human_labels") or {}).get("pose_subtype") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        if "label_missing" not in (row.get("warnings") or []):
            matched += 1
        if row.get("screenshot_path"):
            screenshots += 1
        completeness = ((row.get("measurements") or {}).get("controller_completeness") or {})
        for key, value in completeness.items():
            if value:
                completeness_counts[key] = completeness_counts.get(key, 0) + 1
        distance = (((row.get("measurements") or {}).get("partner_relative") or {}).get("rider_pelvis_to_partner_pelvis_distance"))
        if isinstance(distance, (int, float)):
            distances_by_family.setdefault(family, []).append(float(distance))
    distance_summary = {
        family: {
            "count": len(values),
            "mean": round(sum(values) / len(values), 5),
            "min": round(min(values), 5),
            "max": round(max(values), 5),
        }
        for family, values in sorted(distances_by_family.items())
    }
    return {
        "status": "ok",
        "ground_truth": str(ground_truth),
        "out_dir": str(out_dir),
        "captures": len(rows),
        "matched_labels": matched,
        "missing_labels": len(rows) - matched,
        "screenshots": screenshots,
        "family_counts": family_counts,
        "subtype_counts": subtype_counts,
        "controller_completeness_counts": completeness_counts,
        "pelvis_distance_by_family": distance_summary,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
        "auto_labeling_run": False,
    }


def _write_overview(summary: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> None:
    lines = [
        "# Manual Pose Ground Truth Overview V1",
        "",
        f"- Captures: `{summary['captures']}`",
        f"- Matched human labels: `{summary['matched_labels']}`",
        f"- Missing labels: `{summary['missing_labels']}`",
        f"- Screenshots available: `{summary['screenshots']}`",
        f"- Families: `{summary['family_counts']}`",
        f"- Subtypes: `{summary['subtype_counts']}`",
        f"- Pelvis distance by family: `{summary['pelvis_distance_by_family']}`",
        "- ML training performed: `false`",
        "- Auto-labeling performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Controller Completeness",
        "",
    ]
    for key, count in sorted(summary["controller_completeness_counts"].items()):
        lines.append(f"- `{key}`: `{count}` / `{len(rows)}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_family_report(family: str, rows: list[dict[str, Any]], out: Path) -> None:
    title = family.replace("_", " ").title()
    lines = [
        f"# {title} Pose Ground Truth V1",
        "",
        f"- Captures: `{len(rows)}`",
        "- Source: human-created VaM poses plus captured controller transforms.",
        "- Interpretation: pose-ground-truth analysis only, not generation output.",
        "",
    ]
    if not rows:
        lines.append("No captures for this family.")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for row in rows:
        labels = row.get("human_labels") or {}
        partner = ((row.get("measurements") or {}).get("partner_relative") or {})
        geometry = ((row.get("measurements") or {}).get("pose_geometry") or {})
        hands = ((row.get("measurements") or {}).get("hand_target_candidates") or {})
        lines.extend(
            [
                f"## {row.get('capture_id')}",
                "",
                f"- Subtype: `{labels.get('pose_subtype')}`",
                f"- Motion intent: `{labels.get('motion_intent')}`",
                f"- Primary driver: `{labels.get('primary_driver')}`",
                f"- Anchors: `{labels.get('anchors')}`",
                f"- Hand support options: `{labels.get('hand_support_options')}`",
                f"- Rider pelvis to partner pelvis distance: `{partner.get('rider_pelvis_to_partner_pelvis_distance')}`",
                f"- Torso vector: `{geometry.get('torso_vector_chest_minus_pelvis')}`",
                f"- Foot spread: `{geometry.get('leg_spread_foot_distance')}`",
                f"- Knee spread: `{geometry.get('knee_spread_distance')}`",
                f"- L hand nearest target: `{(hands.get('lHandControl') or {}).get('nearest_target')}`",
                f"- R hand nearest target: `{(hands.get('rHandControl') or {}).get('nearest_target')}`",
                "",
                "Human notes:",
                "",
                str(labels.get("raw_notes") or "").strip(),
                "",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ontology_suggestions(rows: list[dict[str, Any]], out: Path) -> None:
    subtypes = sorted({str((row.get("human_labels") or {}).get("pose_subtype")) for row in rows if (row.get("human_labels") or {}).get("pose_subtype")})
    lines = [
        "# Suggested Ontology Corrections From Manual Pose Captures V1",
        "",
        "These are suggestions only. They are not applied automatically.",
        "",
        "## New Or Confirmed Pose Subtypes",
        "",
    ]
    lines.extend(f"- `{subtype}`" for subtype in subtypes)
    lines.extend(
        [
            "",
            "## Corrections",
            "",
            "- Cowgirl geometry should allow varied hand support while keeping pelvis/hip as the driver.",
            "- Cowgirl feet are usually static; knees can phase lightly with rhythm.",
            "- BJ/oral can reuse kneeling or Cowgirl-like base poses but remains head/chest driven with static pelvis.",
            "- HJ is hand-driven and must remain separate from Cowgirl even when the base pose is kneeling.",
            "- Doggy and Missionary examples often describe the woman as receiver/passive; body motion is response, not primary driver.",
            "- Future VaM preview mapping should be based on these captured controller relations, not invented stickman-to-controller guesses.",
            "",
            "## Safety",
            "",
            "- Do not use this as automatic labels for unrelated motion windows.",
            "- Do not train ML in this step.",
            "- Do not generate Timeline clips from this report.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _family(row: dict[str, Any]) -> str:
    return str((row.get("human_labels") or {}).get("family") or "unknown")


def _relative_or_abs(path: str, base: Path) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path

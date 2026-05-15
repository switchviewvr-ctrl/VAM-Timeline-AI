"""Manual YAML label override loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from vam_timeline_ai.io.json_utils import write_jsonl


ALLOWED_ROLES = {"rider", "receiver", "partner_context", "irrelevant", "unknown"}
ALLOWED_QUALITY = {"good", "usable", "questionable", "bad"}
ALLOWED_MOVEMENT_LABELS = {
    "cowgirl_vertical_bounce",
    "cowgirl_forward_back_rock",
    "cowgirl_lateral_sway",
    "cowgirl_circular_grind",
    "cowgirl_deep_slow",
    "cowgirl_fast_shallow",
    "cowgirl_upright",
    "cowgirl_lean_forward",
    "cowgirl_lean_back",
    "cowgirl_pause_hold",
    "cowgirl_adjustment_transition",
    "cowgirl_tempo_increase",
    "cowgirl_tempo_decrease",
    "cowgirl_depth_increase",
    "cowgirl_depth_decrease",
    "cowgirl_irregular_human_motion",
}
ALLOWED_CONTACT_LABELS = {
    "cowgirl_hand_supported_on_partner",
    "cowgirl_hand_supported_on_partner_chest",
    "cowgirl_hand_supported_on_partner_shoulders",
    "cowgirl_hand_supported_on_partner_hips",
    "cowgirl_hand_supported_on_floor_or_bed",
    "cowgirl_hands_on_own_thighs",
    "cowgirl_hands_on_own_body",
    "cowgirl_no_clear_hand_support",
    "contact_unknown",
}
ALLOWED_HEAD_LABELS = {
    "cowgirl_head_down",
    "cowgirl_head_up",
    "cowgirl_head_turn_left",
    "cowgirl_head_turn_right",
    "cowgirl_look_at_partner",
    "cowgirl_look_away",
    "head_attention_unknown",
}
ALLOWED_ROLE_CONTEXT_LABELS = {
    "rider_active",
    "receiver_passive",
    "partner_context_static",
    "partner_context_active",
    "role_unclear",
    "not_cowgirl",
}
ALLOWED_MANUAL_LABELS = ALLOWED_MOVEMENT_LABELS | ALLOWED_CONTACT_LABELS | ALLOWED_HEAD_LABELS | ALLOWED_ROLE_CONTEXT_LABELS


def apply_manual_labels(windows_path: str | Path, labels_path: str | Path, out: str | Path, report: str | Path) -> list[dict[str, Any]]:
    windows = _load_jsonl(windows_path)
    labels_file = Path(labels_path)
    using_real_labels = labels_file.exists() and labels_file.name != "manual_labels.template.yaml"
    labels_data = _load_yaml(labels_file) if labels_file.exists() else {}
    if not using_real_labels:
        labels_data = {}
    rows = [apply_labels_to_window(row, labels_data) for row in windows]
    write_jsonl(out, rows)
    _write_report(rows, report, labels_file, using_real_labels)
    return rows


def write_manual_label_schema_v2(out: str | Path, template: str | Path, guide: str | Path) -> dict[str, str]:
    schema_text = _schema_v2_text()
    template_text = _template_v2_text()
    guide_text = _guide_text()
    for path, text in [(out, schema_text), (template, template_text), (guide, guide_text)]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return {"schema": str(out), "template": str(template), "guide": str(guide)}


def apply_labels_to_window(window: dict[str, Any], labels_data: dict[str, Any]) -> dict[str, Any]:
    out = dict(window)
    out.setdefault("labels", [])
    out.setdefault("negative_labels", [])
    out.setdefault("uncertain_labels", [])
    out.setdefault("needs_manual_review", True)
    out.setdefault("manual_label_notes", [])
    sample_override = labels_data.get("samples", {}).get(str(out.get("sample_id")), {})
    if sample_override:
        if "include_for_cowgirl_db" in sample_override:
            out["include_for_ml"] = bool(sample_override["include_for_cowgirl_db"])
        if "semantic_role" in sample_override:
            out["semantic_role_guess"] = sample_override["semantic_role"]
        if sample_override.get("notes"):
            out["manual_label_notes"].append(sample_override["notes"])
    window_override = labels_data.get("windows", {}).get(str(out.get("window_id")), {})
    if window_override:
        out["labels"] = list(window_override.get("labels", out.get("labels", [])))
        out["negative_labels"] = list(window_override.get("negative_labels", out.get("negative_labels", [])))
        out["uncertain_labels"] = list(window_override.get("uncertain_labels", out.get("uncertain_labels", [])))
        out["needs_manual_review"] = bool(window_override.get("needs_manual_review", False))
        out["manual_label_confidence"] = window_override.get("confidence", "manual")
        out["manual_movement_quality"] = window_override.get("movement_quality", out.get("manual_movement_quality"))
        out["manual_focus_actor"] = window_override.get("focus_actor", out.get("manual_focus_actor", "unknown"))
        if "include_for_ml" in window_override:
            out["include_for_ml"] = bool(window_override.get("include_for_ml"))
        if "semantic_role" in window_override:
            out["semantic_role_guess"] = window_override["semantic_role"]
        if window_override.get("partner_context_atom"):
            out["partner_context_atom"] = window_override["partner_context_atom"]
        if window_override.get("notes"):
            out["manual_label_notes"].append(window_override["notes"])
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_report(rows: list[dict[str, Any]], report: str | Path, labels_file: Path, using_real_labels: bool) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    labeled = sum(1 for row in rows if row.get("labels"))
    negative = sum(1 for row in rows if row.get("negative_labels"))
    uncertain = sum(1 for row in rows if row.get("uncertain_labels"))
    lines = [
        "# Manual Label Report",
        "",
        f"- Labels path requested: `{labels_file}`",
        f"- Real manual labels loaded: {using_real_labels}",
        f"- Windows processed: {len(rows)}",
        f"- Windows with labels: {labeled}",
        f"- Windows with negative labels: {negative}",
        f"- Windows with uncertain labels: {uncertain}",
    ]
    if not using_real_labels:
        lines.append("- Template labels were not applied as real training labels.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _schema_v2_text() -> str:
    return f"""# Manual label schema v2 for VaM Timeline AI.
# This is a human-readable YAML schema. It documents allowed fields and labels.

allowed_roles: {sorted(ALLOWED_ROLES)}
allowed_quality: {sorted(ALLOWED_QUALITY)}
allowed_manual_labels:
{_yaml_list(sorted(ALLOWED_MANUAL_LABELS))}

sections:
  scenes:
    fields: [notes, exclude_scene, reason]
  actors:
    fields: [semantic_role, focus_actor, confidence, notes]
  samples:
    fields: [include_for_cowgirl_db, include_for_ml, semantic_role, quality, notes]
  windows:
    fields: [labels, negative_labels, uncertain_labels, semantic_role, focus_actor, partner_context_atom, movement_quality, include_for_ml, confidence, notes]
  pair_windows:
    fields: [rider_window_id, receiver_window_id, rider_atom_id, receiver_atom_id, pair_labels, contact_labels, confidence, notes]

notes:
  - Weak labels are not manual labels.
  - Atom names are technical identifiers, not semantic roles.
  - Filename hints can prioritize review but cannot label data.
  - Multi-label windows are expected.
"""


def _template_v2_text() -> str:
    return """# Manual labels template v2.
# Copy this file to data/labels/manual_labels.yaml before entering real labels.
# Do not use this template file as training data.

scenes:
  "example_scene.json":
    notes: ""
    exclude_scene: false
    reason: ""

actors:
  "example_scene.json":
    "technical_atom_id":
      semantic_role: "unknown"
      focus_actor: "unknown"
      confidence: 0.0
      notes: ""

samples:
  "example_sample_id":
    include_for_cowgirl_db: false
    include_for_ml: false
    semantic_role: "unknown"
    quality: "questionable"
    notes: ""

windows:
  "example_sample_id:0.000-4.000":
    labels: []
    negative_labels: []
    uncertain_labels: []
    semantic_role: "unknown"
    focus_actor: "unknown"
    partner_context_atom: ""
    movement_quality: "questionable"
    include_for_ml: false
    confidence: 0.0
    notes: ""

pair_windows:
  "example_pair_window_id":
    rider_window_id: ""
    receiver_window_id: ""
    rider_atom_id: ""
    receiver_atom_id: ""
    pair_labels: []
    contact_labels: []
    confidence: 0.0
    notes: ""
"""


def _guide_text() -> str:
    return """# Manual Labeling Guide: Cowgirl/Riding v1

Manual labels are ground-truth candidates created by visual/human review. Weak labels are only hints. Do not copy weak labels into manual labels unless the motion is actually confirmed.

## What To Label

Label only what is visible or strongly supported. It is fine to leave fields unknown. A useful review item may have one label, many labels, negative labels, or only notes.

## Roles

Use `rider`, `receiver`, `partner_context`, `irrelevant`, or `unknown`. Do not infer role from atom names such as `man`, `Person`, or character names.

## Movement Labels

Use multi-labels for movement: vertical bounce, forward/back rock, lateral sway, circular grind, slow/deep, fast/shallow, upright, lean forward/back, pause/hold, adjustment transition, tempo/depth changes, and irregular human motion.

## Contact And Hands

Only use partner contact labels when pair context or visual review supports them. Without partner evidence, prefer uncertainty or own-body/no-clear-support labels.

## Negative And Control Examples

Negative examples are important. Include `not_cowgirl`, unclear roles, static/passive context, and windows from non-riding scenes when useful.

## ML Safety

Do not train supervised classifiers until labels exist across multiple scenes and samples with negative/control examples. Random window train/test splits are invalid because windows overlap.
"""


def _yaml_list(items: list[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)

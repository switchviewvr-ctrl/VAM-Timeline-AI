"""Parse human explanation notes for manual VaM pose captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import unicodedata

from vam_timeline_ai.io.json_utils import dump_json


ENTRY_RE = re.compile(r"^(pose_capture_\d{8}_\d{6})\s*:\s*$", re.IGNORECASE)


def parse_manual_pose_explanations_v1(
    explanations: str | Path,
    out_json: str | Path,
    out_yaml: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    text = _read_text(explanations)
    labels = parse_explanation_text(text)
    payload = {
        "schema_version": "manual_pose_human_labels_v1",
        "source": str(explanations),
        "labels": labels,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }
    dump_json(out_json, payload)
    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(out_yaml).write_text(_to_yaml(payload) + "\n", encoding="utf-8")
    summary = _summary(labels, explanations, out_json, out_yaml, report)
    _write_report(summary, labels, report)
    return summary


def parse_explanation_text(text: str) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    current_id: str | None = None
    notes: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = ENTRY_RE.match(line)
        if match:
            if current_id:
                labels.append(_label_from_notes(current_id, notes))
            current_id = match.group(1)
            notes = []
            continue
        if current_id and line:
            notes.append(line.lstrip("-").strip())
    if current_id:
        labels.append(_label_from_notes(current_id, notes))
    return labels


def _label_from_notes(capture_id: str, notes: list[str]) -> dict[str, Any]:
    raw = "\n".join(notes).strip()
    text = _fold(raw)
    family = _family(text)
    subtype = _subtype(text, family)
    role = _role(text, family)
    primary = _primary_driver(family)
    secondary = _secondary_drivers(text, family)
    anchors = _anchors(text, family)
    hand_support = _hand_support(text)
    foot_behavior = _foot_behavior(text)
    knee_behavior = _knee_behavior(text)
    valid_motions = _valid_motions(text, family)
    warnings: list[str] = []
    if family == "unknown":
        warnings.append("family_uncertain")
    if subtype == "unknown":
        warnings.append("pose_subtype_uncertain")
    return {
        "capture_id": capture_id,
        "raw_notes": raw,
        "family": family,
        "pose_subtype": subtype,
        "role_active_passive": role,
        "motion_intent": ", ".join(valid_motions) if valid_motions else "unknown",
        "primary_driver": primary,
        "secondary_drivers": secondary,
        "anchors": anchors,
        "hand_support_options": hand_support,
        "foot_behavior": foot_behavior,
        "knee_behavior": knee_behavior,
        "generation_valid_motions": valid_motions,
        "warnings": warnings,
        "notes": notes,
    }


def _family(text: str) -> str:
    if "deepthroat" in text or re.search(r"\bbj\b", text):
        return "bj_oral"
    if re.search(r"\bhj\b", text) or "handjob" in text:
        return "handjob"
    if "doggy" in text:
        return "doggy"
    if "missionary" in text:
        return "missionary"
    if "cowgirl" in text:
        return "cowgirl"
    return "unknown"


def _subtype(text: str, family: str) -> str:
    if family == "bj_oral":
        if "deepthroat" in text or "bruckt" in text:
            return "bj_deepthroat_bridge"
        if "cowgirl stellung" in text:
            return "bj_kneeling_cowgirl_like"
        return "bj_kneeling"
    if family == "handjob":
        if "knien" in text and "steht" in text:
            return "hj_kneeling_to_standing_partner"
        return "hj_pose"
    if family == "doggy":
        if "tisch" in text and "standing" in text:
            return "standing_doggy_table"
        if "standing" in text:
            return "standing_doggy"
        if "ein bein" in text and "tisch" in text:
            return "doggy_table_one_leg"
        if "klassische" in text:
            return "doggy_classic"
        return "doggy_classic"
    if family == "missionary":
        if "standing" in text and "leg up" in text:
            return "standing_missionary_leg_up"
        if "tisch" in text:
            return "missionary_table"
        if "legs up" in text:
            return "missionary_legs_up"
        return "missionary_pose"
    if family == "cowgirl":
        if "hande uber dem kopf" in text or "eigenen kopf" in text:
            return "cowgirl_hands_over_head"
        if "sessel" in text:
            return "cowgirl_chair_supported"
        if "sitting" in text:
            return "sitting_cowgirl_intimate"
        if "lean back" in text or "hinter ihr" in text:
            return "cowgirl_lean_back_object_supported"
        if "lean forward" in text or "stark nach vorn" in text:
            return "lean_forward_cowgirl_table_or_partner_supported"
        if "leicht nach vorn" in text:
            return "cowgirl_classic_lean_forward_light"
        if "klassische" in text:
            return "cowgirl_classic"
        return "cowgirl_pose"
    return "unknown"


def _role(text: str, family: str) -> str:
    if "passiv oder aktiv" in text:
        return "female_active_or_passive"
    if "frau passiv" in text or "frau ist passiv" in text:
        return "female_passive_receiver"
    if "frau ist aktiv" in text or "frau aktiv" in text:
        return "female_active_driver"
    if family in {"cowgirl", "bj_oral", "handjob"}:
        return "female_active_or_context_dependent"
    if family in {"doggy", "missionary"}:
        return "female_passive_receiver"
    return "unknown"


def _primary_driver(family: str) -> str:
    if family == "bj_oral":
        return "head_neck"
    if family == "handjob":
        return "hand"
    if family == "cowgirl":
        return "pelvis_hip"
    if family == "doggy":
        return "partner_driver_body_response"
    if family == "missionary":
        return "partner_driver_receiver_response"
    return "unknown"


def _secondary_drivers(text: str, family: str) -> list[str]:
    drivers: list[str] = []
    if "hande" in text or "hand" in text:
        drivers.append("hands")
    if family in {"cowgirl", "doggy", "missionary"}:
        drivers.append("knees_or_legs_reactive")
    if family == "bj_oral":
        drivers.append("chest_abdomen_support")
    return sorted(set(drivers))


def _anchors(text: str, family: str) -> list[str]:
    anchors = []
    if "fusse" in text or "fuesse" in text:
        anchors.append("feet_static")
    if "hande" in text or "hand" in text:
        anchors.append("hands_contextual")
    if "knien" in text or "knie" in text:
        anchors.append("knees_contextual")
    if family in {"doggy", "missionary"}:
        anchors.append("body_support_contextual")
    return sorted(set(anchors))


def _hand_support(text: str) -> list[str]:
    options: list[str] = []
    if "brust des mannes" in text:
        options.append("hands_on_partner_chest")
    if "thighs des mannes" in text or "beine des mannes" in text:
        options.append("hands_on_partner_thighs_or_legs")
    if "eigenen thighs" in text or "eigene thighs" in text:
        options.append("hands_on_self_thighs")
    if "eigener brust" in text or "eigene brust" in text:
        options.append("hands_on_self_chest")
    if "objekt" in text or "boden" in text or "bett" in text or "tisch" in text:
        options.append("hands_on_object_floor_bed_or_table")
    if "kopf" in text:
        options.append("hands_on_head")
    if not options and ("hand" in text or "hande" in text):
        options.append("hands_contextual_unknown_target")
    return sorted(set(options))


def _foot_behavior(text: str) -> str:
    if "fusse meist komplett still" in text or "fuesse meist komplett still" in text or "fusse sind still" in text or "fuesse sind still" in text or "fusse meist still" in text or "fuesse meist still" in text:
        return "mostly_static"
    if "fusse bewegen sich" in text or "fuesse bewegen sich" in text:
        return "reactive_to_partner_motion"
    return "unknown"


def _knee_behavior(text: str) -> str:
    if "knie gehen evtl auseinander" in text:
        return "may_phase_out_in"
    if "beine sind meist still" in text:
        return "mostly_static"
    if "beine bewegen sich" in text or "bein ist still" in text:
        return "reactive_or_mixed"
    return "unknown"


def _valid_motions(text: str, family: str) -> list[str]:
    motions: list[str] = []
    for key, value in [
        ("bouncing", "bouncing"),
        ("grinding", "grinding"),
        ("riding", "riding"),
        ("kopfbewegung", "head_bob"),
        ("vor und ruckwartsbewegung", "hand_forward_back"),
        ("pose hold", "pose_hold"),
        ("stosse", "partner_thrust_response"),
    ]:
        if key in text:
            motions.append(value)
    if not motions:
        defaults = {
            "cowgirl": ["bouncing", "grinding", "riding"],
            "bj_oral": ["head_bob"],
            "handjob": ["hand_forward_back"],
            "doggy": ["partner_thrust_response"],
            "missionary": ["partner_thrust_response"],
        }
        motions.extend(defaults.get(family, []))
    return sorted(set(motions))


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return folded.encode("ascii", "ignore").decode("ascii")


def _read_text(path: str | Path) -> str:
    p = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return p.read_text(errors="replace")


def _summary(labels: list[dict[str, Any]], explanations: str | Path, out_json: str | Path, out_yaml: str | Path, report: str | Path) -> dict[str, Any]:
    families: dict[str, int] = {}
    subtypes: dict[str, int] = {}
    for label in labels:
        families[label["family"]] = families.get(label["family"], 0) + 1
        subtypes[label["pose_subtype"]] = subtypes.get(label["pose_subtype"], 0) + 1
    return {
        "status": "ok",
        "explanations": str(explanations),
        "out_json": str(out_json),
        "out_yaml": str(out_yaml),
        "report": str(report),
        "labels": len(labels),
        "family_counts": families,
        "subtype_counts": subtypes,
        "ml_training_run": False,
        "manual_labels_yaml_modified": False,
    }


def _write_report(summary: dict[str, Any], labels: list[dict[str, Any]], report: str | Path) -> None:
    target = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Human Pose Label Parse Report V1",
        "",
        f"- Explanations: `{summary['explanations']}`",
        f"- Labels parsed: `{summary['labels']}`",
        f"- Family counts: `{summary['family_counts']}`",
        f"- Subtype counts: `{summary['subtype_counts']}`",
        "- ML training performed: `false`",
        "- manual_labels.yaml modified: `false`",
        "",
        "## Parsed Labels",
        "",
    ]
    for label in labels:
        lines.append(f"- `{label['capture_id']}`: `{label['family']}` / `{label['pose_subtype']}` / driver `{label['primary_driver']}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_yaml_scalar(value)}"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(ch in text for ch in ":\n#{}[]"):
        return json.dumps(text, ensure_ascii=False)
    return text

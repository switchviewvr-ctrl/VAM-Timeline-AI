"""Validate semantic stickman examples against ontology-level invariants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_json
from vam_timeline_ai.io.json_utils import safe_id_for_path
from vam_timeline_ai.generation.semantic_interaction_constraints import constraint_for_example, evaluate_interaction_constraints
from vam_timeline_ai.semantics.ontology_loader import load_motion_families


def validate_semantic_stickman_examples_v1(motion_examples: str | Path, ontology: str | Path, out: str | Path) -> dict[str, Any]:
    data = load_json(motion_examples)
    load_motion_families(ontology)  # verifies ontology can be loaded
    errors: list[str] = []
    warnings: list[str] = []
    examples = data.get("examples", [])
    for ex in examples:
        cid = str(ex.get("concept_id"))
        labels = ex.get("labels") or {}
        drivers = set(labels.get("primary_driver") or [])
        anchors = set(labels.get("anchors") or [])
        not_labels = set(ex.get("not_labels") or [])
        pose = str(ex.get("pose_subtype") or "")
        family = str(ex.get("family") or "")
        if family == "cowgirl" and "pelvis_hip" not in drivers:
            errors.append(f"{cid}: Cowgirl example lacks pelvis_hip driver")
        if family == "cowgirl" and "reverse_cowgirl" in cid and "back_to_partner" not in str(labels):
            errors.append(f"{cid}: reverse/cowgirl ambiguity")
        if cid == "cowgirl_lean_back_supported" and "reverse_cowgirl" not in not_labels:
            errors.append(f"{cid}: lean-back must explicitly exclude reverse_cowgirl")
        if family == "bj_oral" and not ({"head_neck", "chest_abdomen"} & drivers):
            errors.append(f"{cid}: BJ/oral lacks head/chest driver")
        if family == "bj_oral" and "pelvis_hip" in drivers:
            errors.append(f"{cid}: BJ/oral incorrectly uses pelvis_hip as driver")
        if family == "doggy" and not ({"hands", "knees"} <= anchors or "hands" in anchors and "feet" in anchors):
            errors.append(f"{cid}: Doggy lacks hand/knee equivalent support")
        if family == "missionary" and "missionary" not in pose:
            errors.append(f"{cid}: Missionary example is not supine/missionary pose")
        if family == "reverse_cowgirl" and "back_to_partner" not in _frame_or_labels_text(ex):
            errors.append(f"{cid}: Reverse Cowgirl lacks back_to_partner/facing-away marker")
        if family == "standing_hand_head" and "pelvis_hip" in drivers:
            errors.append(f"{cid}: Standing hand/head negative uses pelvis driver")
        if family == "handjob" and "hands" not in drivers:
            errors.append(f"{cid}: Handjob lacks hand driver")
        if "Person/root" in str(ex) or "world transform" in str(ex):
            errors.append(f"{cid}: forbidden Person/root/world wording found")
    status = "ok" if not errors else "failed"
    lines = [
        "# Semantic Stickman Validation V1",
        "",
        f"- Status: {status}",
        f"- Examples checked: {len(examples)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "- ML training performed: false",
        "- Timeline animation generated: false",
        "- manual_labels.yaml modified: false",
        "",
        "## Errors",
        "",
        *(f"- {e}" for e in errors),
        "",
        "## Warnings",
        "",
        *(f"- {w}" for w in warnings),
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": status, "examples": len(examples), "errors": len(errors), "warnings": len(warnings), "out": str(out), "error_messages": errors}


def validate_semantic_stickman_examples_v2(motion_examples: str | Path, preview_dir: str | Path, ontology: str | Path, out: str | Path) -> dict[str, Any]:
    data = load_json(motion_examples)
    load_motion_families(ontology)
    root = Path(preview_dir)
    errors: list[str] = []
    warnings: list[str] = []
    examples = data.get("examples", [])
    for ex in examples:
        cid = str(ex.get("concept_id"))
        family = str(ex.get("family") or "")
        labels = ex.get("labels") or {}
        anchors = set(labels.get("anchors") or [])
        targets = ex.get("contact_targets") or {}
        meta_path = root / safe_id_for_path(cid) / "metadata.json"
        if not meta_path.exists():
            errors.append(f"{cid}: preview metadata missing")
            continue
        meta = load_json(meta_path)
        ctx = meta.get("semantic_context") or {}
        if not ctx.get("bodypart_labels_drawn"):
            errors.append(f"{cid}: bodypart labels were not rendered")
        if family in {"cowgirl", "reverse_cowgirl"}:
            if not ctx.get("has_partner_pelvis_target"):
                errors.append(f"{cid}: Cowgirl/reverse preview lacks partner_pelvis target")
            if not ctx.get("has_alignment_target"):
                errors.append(f"{cid}: Cowgirl/reverse preview lacks rider pelvis to partner pelvis alignment vector")
            if ctx.get("appears_floating_warning") and "hover" not in str(ex.get("pose_subtype")):
                errors.append(f"{cid}: Cowgirl/reverse appears floating without hover subtype")
        if cid == "cowgirl_lean_forward_supported" and not targets:
            errors.append(f"{cid}: lean-forward must include front support target")
        if cid == "cowgirl_lean_forward_supported" and not ctx.get("has_support_targets"):
            errors.append(f"{cid}: lean-forward support marker missing")
        if cid == "cowgirl_lean_back_supported" and not any("legs_or_thighs" in str(t) or "behind" in str(t) for t in targets.values()):
            errors.append(f"{cid}: lean-back must include behind/legs-or-thighs support target")
        if cid == "cowgirl_lean_back_supported" and not ctx.get("has_support_targets"):
            errors.append(f"{cid}: lean-back behind support marker missing")
        if family == "doggy":
            if "hands" not in anchors or not ({"knees", "feet"} & anchors):
                errors.append(f"{cid}: doggy lacks front support and knee/foot anchors")
            if not ctx.get("has_partner_reference"):
                errors.append(f"{cid}: doggy partner-behind reference missing")
        if family == "bj_oral" and "head_to_partner_pelvis" not in (ctx.get("target_vectors") or []):
            errors.append(f"{cid}: BJ/oral lacks head/chest to partner pelvis target relation")
        if family == "missionary" and not ctx.get("has_partner_reference"):
            errors.append(f"{cid}: missionary lacks partner/body context")
    status = "ok" if not errors else "failed"
    lines = [
        "# Semantic Stickman Validation V2",
        "",
        f"- Status: {status}",
        f"- Examples checked: {len(examples)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "- Bodypart labels required: true",
        "- Partner/alignment/support context required: true",
        "- ML training performed: false",
        "- Timeline animation generated: false",
        "- manual_labels.yaml modified: false",
        "",
        "## Errors",
        "",
        *(f"- {e}" for e in errors),
        "",
        "## Warnings",
        "",
        *(f"- {w}" for w in warnings),
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": status, "examples": len(examples), "errors": len(errors), "warnings": len(warnings), "out": str(out), "error_messages": errors}


def validate_semantic_stickman_examples_v3(motion_examples: str | Path, preview_dir: str | Path, ontology: str | Path, out: str | Path) -> dict[str, Any]:
    data = load_json(motion_examples)
    load_motion_families(ontology)
    root = Path(preview_dir)
    errors: list[str] = []
    warnings: list[str] = []
    examples = data.get("examples", [])
    valid_count = 0
    invalid_count = 0
    for ex in examples:
        cid = str(ex.get("concept_id"))
        family = str(ex.get("family") or "")
        labels = ex.get("labels") or {}
        anchors = set(labels.get("anchors") or [])
        meta_path = root / safe_id_for_path(cid) / "metadata.json"
        if not meta_path.exists():
            errors.append(f"{cid}: preview metadata missing")
            continue
        meta = load_json(meta_path)
        ctx = meta.get("semantic_context") or {}
        if not ctx.get("bodypart_labels_drawn"):
            errors.append(f"{cid}: bodypart labels were not rendered")
        constraint = constraint_for_example(ex)
        result = evaluate_interaction_constraints(ex)
        if constraint:
            if not result.get("valid"):
                invalid_count += 1
                errors.append(f"{cid}: interaction alignment failed: {result.get('failed_constraints')}")
            else:
                valid_count += 1
            if ctx.get("interaction_alignment_valid") is not True:
                errors.append(f"{cid}: renderer metadata does not mark interaction alignment valid")
            if ctx.get("alignment_distance_max") is None:
                errors.append(f"{cid}: alignment distance missing from renderer metadata")
        if family in {"cowgirl", "reverse_cowgirl"}:
            if not ctx.get("has_partner_pelvis_target"):
                errors.append(f"{cid}: rider family lacks partner pelvis target")
            if not ctx.get("has_alignment_target"):
                errors.append(f"{cid}: rider family lacks alignment axis")
            if "feet" not in anchors and "knees" not in anchors and "hands" not in anchors:
                errors.append(f"{cid}: rider family lacks support anchors")
            if cid == "cowgirl_lean_back_supported":
                if labels.get("facing_context") != "front_cowgirl":
                    errors.append(f"{cid}: lean-back Cowgirl must remain front_cowgirl")
                if "reverse_cowgirl" not in set(ex.get("not_labels") or []):
                    errors.append(f"{cid}: lean-back Cowgirl must explicitly exclude reverse_cowgirl")
                if not ctx.get("has_support_targets"):
                    errors.append(f"{cid}: lean-back Cowgirl missing behind support target")
            if family == "reverse_cowgirl" and labels.get("facing_context") != "back_to_partner":
                errors.append(f"{cid}: reverse Cowgirl must be back_to_partner")
        if family == "doggy":
            if "hands" not in anchors or not ({"knees", "feet"} & anchors):
                errors.append(f"{cid}: doggy must show front support and knee/foot support")
            if labels.get("facing_context") != "partner_behind":
                errors.append(f"{cid}: doggy must show partner_behind relation")
        if family == "bj_oral":
            if "head_to_partner_pelvis" not in (ctx.get("target_vectors") or []):
                errors.append(f"{cid}: BJ/oral must show head/chest path to partner pelvis target")
            if "pelvis_hip" in set(labels.get("primary_driver") or []):
                errors.append(f"{cid}: BJ/oral must not use pelvis riding driver")
        if family == "missionary":
            if "missionary" not in str(ex.get("pose_subtype") or ""):
                errors.append(f"{cid}: missionary must be supine/missionary pose")
    status = "ok" if not errors else "failed"
    lines = [
        "# Semantic Stickman Validation V3",
        "",
        f"- Status: {status}",
        f"- Examples checked: {len(examples)}",
        f"- Contact-constrained examples valid: {valid_count}",
        f"- Contact-constrained examples invalid: {invalid_count}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "- Interaction targets are constraints: true",
        "- ML training performed: false",
        "- Timeline animation generated: false",
        "- manual_labels.yaml modified: false",
        "",
        "## Errors",
        "",
        *(f"- {e}" for e in errors),
        "",
        "## Warnings",
        "",
        *(f"- {w}" for w in warnings),
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": status, "examples": len(examples), "valid": valid_count, "invalid": invalid_count, "errors": len(errors), "warnings": len(warnings), "out": str(out), "error_messages": errors}


def _frame_or_labels_text(example: dict[str, Any]) -> str:
    return str(example.get("labels") or {}) + str(example.get("frames", [])[:1])

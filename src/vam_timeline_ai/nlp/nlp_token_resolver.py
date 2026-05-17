"""Rule-based NLP token resolver for component motion intent.

The resolver is intentionally conservative. It returns unresolved requirements
instead of guessing unsupported semantics and never exports Timeline data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from vam_timeline_ai.generation.component_motion_intent import (
    ActionConstraint,
    BaseState,
    MotionIntentPlan,
    MotionProfile,
    SequencePhase,
)
from vam_timeline_ai.semantics.ontology_loader import load_yaml


def resolve_nlp_tokens_v1(
    prompt: str,
    lexicon: str | Path,
    component_ontology: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    lex = load_yaml(lexicon)
    ontology = load_yaml(component_ontology)
    matches = _match_entries(prompt, lex.get("entries", []))
    result = {
        "schema": "nlp_token_resolution_v1",
        "prompt": prompt,
        "subject": _subject(prompt),
        "target": _target(prompt),
        "matches": matches,
        "families": _values(matches, "family"),
        "actions": _values(matches, "action_id"),
        "anatomy_regions": _values(matches, "semantic_region"),
        "pose_modifiers": _values(matches, "pose_modifier"),
        "styles": _styles(matches),
        "targets": _values(matches, "target_node"),
        "sequence_markers": _values(matches, "sequence_marker"),
        "unresolved_requirements": _unresolved(prompt, matches),
        "component_ontology_schema": ontology.get("schema"),
        "generated_timeline": False,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    _write_json(out, result)
    return result


def build_motion_intent_from_prompt_v1(
    prompt: str,
    lexicon: str | Path,
    component_ontology: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    lex = load_yaml(lexicon)
    ontology = load_yaml(component_ontology)
    matches = _match_entries(prompt, lex.get("entries", []))
    durations = _durations(prompt)
    families = _values(matches, "family")
    family = _prefer_family(families)
    styles = _styles(matches)
    pose_modifiers = _values(matches, "pose_modifier")
    constraints = _constraints(matches, ontology)
    phases = _phases(prompt, ontology, family, styles, pose_modifiers, durations, constraints)
    plan = MotionIntentPlan(
        sequence_id="prompt_intent_v1",
        subject=_subject(prompt),
        target=_target(prompt),
        phases=phases,
        unresolved_requirements=_unresolved(prompt, matches),
        safety_rules=list(ontology.get("safety_rules") or []),
        generated_timeline=False,
    )
    result = {
        "schema": "motion_intent_from_prompt_v1",
        "prompt": prompt,
        "intent_plan": plan.to_dict(),
        "matched_terms": matches,
        "generated_timeline": False,
        "manual_labels_modified": False,
        "ml_training_performed": False,
    }
    _write_json(out, result)
    return result


def _match_entries(prompt: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = _norm(prompt)
    matches: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("active") is False:
            continue
        for term in entry.get("terms") or []:
            norm = _norm(str(term))
            if not norm:
                continue
            if _contains_term(text, norm):
                start = _term_start(text, norm)
                matches.append(
                    {
                        "entry_id": entry.get("id"),
                        "type": entry.get("type"),
                        "term": term,
                        "maps_to": entry.get("maps_to") or {},
                        "source": entry.get("source", "manual"),
                        "start": start,
                    }
                )
                break
    return sorted(matches, key=lambda m: (int(m.get("start", 10**9)), str(m.get("entry_id", ""))))


def _contains_term(text: str, term: str) -> bool:
    if " " in term:
        return term in text
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _term_start(text: str, term: str) -> int:
    if " " in term:
        pos = text.find(term)
        return pos if pos >= 0 else 10**9
    match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text)
    return match.start() if match else 10**9


def _norm(text: str) -> str:
    return str(text).casefold().replace("ue", "ü").replace("ae", "ä").replace("oe", "ö")


def _values(matches: list[dict[str, Any]], key: str) -> list[str]:
    out: list[str] = []
    for match in matches:
        value = (match.get("maps_to") or {}).get(key)
        if isinstance(value, list):
            out.extend(str(v) for v in value)
        elif value:
            out.append(str(value))
    return _dedupe(out)


def _styles(matches: list[dict[str, Any]]) -> list[str]:
    styles: list[str] = []
    for match in matches:
        maps = match.get("maps_to") or {}
        if maps.get("style"):
            styles.append(str(maps["style"]))
        if maps.get("style_modifier"):
            styles.append(str(maps["style_modifier"]))
    return _dedupe(styles)


def _subject(prompt: str) -> str:
    text = _norm(prompt)
    if any(t in text for t in ["frau", "woman", "female"]):
        return "female_or_primary_actor"
    return "primary_actor"


def _target(prompt: str) -> str:
    text = _norm(prompt)
    if any(t in text for t in ["mann", "man", "male", "ihm"]):
        return "male_or_partner_actor"
    return "partner_actor"


def _prefer_family(families: list[str]) -> str:
    for family in ["reverse_cowgirl", "cowgirl", "bj_oral", "handjob", "doggy", "missionary"]:
        if family in families:
            return family
    return families[0] if families else "unknown"


def _durations(prompt: str) -> list[float]:
    out: list[float] = []
    for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:seconds?|sekunden?|s)\b", prompt, flags=re.IGNORECASE):
        out.append(float(match.group(1).replace(",", ".")))
    return out


def _constraints(matches: list[dict[str, Any]], ontology: dict[str, Any]) -> list[ActionConstraint]:
    action_ids = _values(matches, "action_id")
    targets = _values(matches, "target_node")
    constraints: list[ActionConstraint] = []
    for action_id in action_ids:
        cfg = (ontology.get("actions") or {}).get(action_id, {})
        constraints.append(
            ActionConstraint(
                action_id=action_id,
                effectors=list(cfg.get("effectors_default") or []),
                target_node=targets[0] if targets else "unresolved_target",
                mode=str(cfg.get("mode") or "unknown"),
                weight=float(cfg.get("weight") or 1.0),
                relative_velocity=cfg.get("relative_velocity"),
            )
        )
    return constraints


def _phases(
    prompt: str,
    ontology: dict[str, Any],
    family: str,
    styles: list[str],
    pose_modifiers: list[str],
    durations: list[float],
    constraints: list[ActionConstraint],
) -> list[SequencePhase]:
    base_cfg = (ontology.get("base_states") or {}).get(family, {})
    if not styles:
        styles = ["default"]
    if "teasing" in styles and "slow_soft" in styles:
        styles = ["slow_teasing" if s == "slow_soft" else s for s in styles if s != "teasing"] or ["slow_teasing"]
    if _norm(prompt).find("into") >= 0 or "dann" in _norm(prompt) or "then" in _norm(prompt):
        phase_styles = styles[:2] if len(styles) >= 2 else styles + styles[:1]
    else:
        phase_styles = styles[:1]
    phases: list[SequencePhase] = []
    for idx, style in enumerate(phase_styles):
        pose_subtype = _pose_subtype_for_phase(ontology, family, pose_modifiers, idx)
        motion_profile = _profile(ontology, style)
        base = BaseState(
            family=family,
            pose_subtype=pose_subtype,
            actor_role=str(base_cfg.get("actor_role") or "unknown"),
            partner_role=str(base_cfg.get("partner_role") or "unknown"),
            driver_region=str(base_cfg.get("driver_region") or "unknown"),
            required_anchors=list(base_cfg.get("required_anchors") or []),
            partner_relation=list(base_cfg.get("partner_relation") or []),
        )
        phase = SequencePhase(
            phase_id=f"phase_{idx + 1}",
            duration_seconds=durations[idx] if idx < len(durations) else None,
            base_state=base,
            motion_profile=motion_profile,
            constraints=constraints if idx == 0 else [c for c in constraints],
        )
        if idx < len(phase_styles) - 1:
            phase.transition_to_next = {"type": "interpolate_pose_or_style", "duration_seconds": 2.0}
        phases.append(phase)
    return phases


def _pose_subtype_for_phase(ontology: dict[str, Any], family: str, modifiers: list[str], idx: int) -> str:
    base_cfg = (ontology.get("base_states") or {}).get(family, {})
    pose_cfg = ontology.get("pose_modifiers") or {}
    if modifiers:
        modifier = modifiers[min(idx, len(modifiers) - 1)]
        return str((pose_cfg.get(modifier) or {}).get(f"{family}_pose_subtype") or (pose_cfg.get(modifier) or {}).get("cowgirl_pose_subtype") or base_cfg.get("default_pose_subtype") or "unknown")
    return str(base_cfg.get("default_pose_subtype") or "unknown")


def _profile(ontology: dict[str, Any], style: str) -> MotionProfile:
    cfg = (ontology.get("motion_profiles") or {}).get(style, {})
    return MotionProfile(
        tempo_profile=style,
        frequency_hz=list(cfg.get("frequency_hz") or []),
        amplitude_multiplier=float(cfg.get("amplitude_multiplier") or 1.0),
        curve_type=str(cfg.get("curve_type") or "smooth_sine"),
        follower_lag=str(cfg.get("follower_lag") or "medium"),
        impact_profile="impact" if "impact" in style else "none",
    )


def _unresolved(prompt: str, matches: list[dict[str, Any]]) -> list[str]:
    unresolved: list[str] = []
    if not _values(matches, "family"):
        unresolved.append("base motion family not resolved")
    for match in matches:
        maps = match.get("maps_to") or {}
        if maps.get("target_required") and not _values(matches, "target_node"):
            unresolved.append(f"target required for action {maps.get('action_id')}")
    return _dedupe(unresolved)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

"""Collect conservative web-context research cards.

The collector writes candidate cards only. It does not scrape adult data, does
not create labels, and does not auto-apply ontology changes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import html
import re
import urllib.request

from vam_timeline_ai.io.json_utils import write_jsonl
from vam_timeline_ai.research.research_cards import ResearchCard
from vam_timeline_ai.semantics.ontology_loader import load_yaml, yaml


def collect_web_motion_context_v1(
    topics: str | Path,
    out_dir: str | Path,
    allow_web: bool,
    max_sources_per_category: int = 10,
) -> dict[str, Any]:
    topic_data = load_yaml(topics)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    blocked: list[str] = []
    if not allow_web:
        (out / "BLOCKED_NO_WEB_ACCESS.md").write_text("Web collection was disabled by --allow-web false.\n", encoding="utf-8")
    for category, cfg in (topic_data.get("categories") or {}).items():
        if not cfg.get("allowed", False):
            continue
        for url in list(cfg.get("seed_urls") or [])[:max_sources_per_category]:
            if allow_web:
                card = _fetch_card(str(url), str(category))
                if card is None:
                    blocked.append(str(url))
                    card = _seed_card(str(url), str(category), "Fetch failed; seed URL retained as review candidate.")
            else:
                card = _seed_card(str(url), str(category), "Web disabled; seed URL retained as review candidate.")
            cards.append(card.to_dict())
    if allow_web and cards and len(blocked) == len(cards):
        (out / "BLOCKED_NO_WEB_ACCESS.md").write_text(
            "All web seed URL fetches failed. Seed URLs were retained as inactive review candidates.\n",
            encoding="utf-8",
        )
    write_jsonl(out / "research_cards_v1.jsonl", cards)
    _write_review(out / "WEB_CONTEXT_REVIEW.md", cards, blocked, allow_web)
    return {
        "status": "ok" if cards else "blocked",
        "cards": len(cards),
        "blocked_fetches": len(blocked),
        "out_dir": str(out),
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }


def _fetch_card(url: str, category: str) -> ResearchCard | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VAMTimelineAIResearchBot/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:  # noqa: S310 - user-approved allowlist seed URLs
            raw = response.read(180000).decode("utf-8", errors="replace")
    except Exception:
        return None
    title = _title(raw) or url
    text = _text(raw)
    return ResearchCard(
        source_url=url,
        title=title,
        category=category,
        summary=_summary_for_category(category, text),
        extracted_concepts=_concepts_for_category(category, text),
        maps_to_project=_maps_for_category(category),
    )


def _seed_card(url: str, category: str, summary: str) -> ResearchCard:
    return ResearchCard(
        source_url=url,
        title=url,
        category=category,
        summary=summary,
        extracted_concepts=_concepts_for_category(category, ""),
        maps_to_project=_maps_for_category(category),
    )


def _title(raw: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
    return html.unescape(re.sub(r"\s+", " ", match.group(1)).strip()) if match else ""


def _text(raw: str) -> str:
    no_script = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
    no_tags = re.sub(r"<[^>]+>", " ", no_script)
    return html.unescape(re.sub(r"\s+", " ", no_tags)).strip()


def _summary_for_category(category: str, text: str) -> str:
    defaults = {
        "animation_principles": "Candidate context for follow-through, overlapping action, timing, spacing, and staging.",
        "ik_fk_rigging": "Candidate context for IK/FK controller concepts and end-effector constraints.",
        "signal_processing_cycle_detection": "Candidate context for zero crossings, peaks/troughs, and cycle detection terminology.",
        "lexical_resources": "Candidate source for synonym expansion and lexical relation ideas.",
    }
    return defaults.get(category, "Candidate external context.") + (" Source fetched." if text else "")


def _concepts_for_category(category: str, text: str) -> list[str]:
    concepts = {
        "animation_principles": ["timing", "spacing", "follow_through", "overlapping_action", "slow_in_slow_out"],
        "ik_fk_rigging": ["inverse_kinematics", "forward_kinematics", "end_effector", "constraint_weight"],
        "signal_processing_cycle_detection": ["zero_crossing", "peak", "trough", "cycle_count", "frequency"],
        "lexical_resources": ["synonym", "lemma", "lexical_relation"],
    }
    return concepts.get(category, [])


def _maps_for_category(category: str) -> list[str]:
    maps = {
        "animation_principles": ["motion_profiles", "follower_lag", "curve_type"],
        "ik_fk_rigging": ["action_constraints", "support_anchor", "IK_locked_dynamic"],
        "signal_processing_cycle_detection": ["motion_cycle_features", "biomechanical_motion_gates"],
        "lexical_resources": ["nlp_lexicon_candidate_terms"],
    }
    return maps.get(category, [])


def _write_review(path: Path, cards: list[dict[str, Any]], blocked: list[str], allow_web: bool) -> None:
    lines = [
        "# Web Context Review v1",
        "",
        "Web context is candidate context only. It is not VaM truth, not labels, and not training data.",
        "",
        f"- Web allowed: `{allow_web}`",
        f"- Cards: `{len(cards)}`",
        f"- Categories: `{dict(Counter(c.get('category') for c in cards))}`",
        f"- Blocked/fetch failed: `{blocked}`",
        "- Adult scraping performed: `false`",
        "- Auto-applied ontology changes: `false`",
        "- All generated patch candidates must default `accepted: false`.",
        "",
        "## Cards",
        "",
    ]
    for card in cards:
        lines.append(f"- `{card.get('category')}` [{card.get('title')}]({card.get('source_url')}) - {card.get('summary')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        import json

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

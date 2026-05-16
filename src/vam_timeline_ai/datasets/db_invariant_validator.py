"""Invariant checks for semantic candidate DBs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_jsonl


def validate_semantic_dbs(
    run_dir: str | Path,
    semantic_db: str | Path,
    cowgirl_db: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    semantic_path = _fallback(Path(semantic_db), [run / "datasets" / "semantic_candidate_db_v1.jsonl", run / "datasets" / "semantic_candidate_db_v0.jsonl"])
    cowgirl_path = _fallback(Path(cowgirl_db), [run / "datasets" / "cowgirl_candidate_db_v6.jsonl", run / "datasets" / "cowgirl_candidate_db_v5.jsonl"])
    semantic_rows = load_jsonl(semantic_path) if semantic_path else []
    cowgirl_rows = load_jsonl(cowgirl_path) if cowgirl_path else []
    errors: list[str] = []
    warnings: list[str] = []
    if not semantic_path:
        errors.append("semantic DB missing; checked no semantic records")
    if not cowgirl_path:
        errors.append("Cowgirl DB missing; checked no Cowgirl records")
    _check_semantic_rows(semantic_rows, errors, warnings)
    _check_cowgirl_rows(cowgirl_rows, errors, warnings)
    _write_report(semantic_path, cowgirl_path, semantic_rows, cowgirl_rows, errors, warnings, out)
    return {
        "status": "ok" if not errors else "errors",
        "semantic_db": str(semantic_path) if semantic_path else None,
        "cowgirl_db": str(cowgirl_path) if cowgirl_path else None,
        "errors": len(errors),
        "warnings": len(warnings),
        "error_examples": errors[:20],
        "warning_examples": warnings[:20],
    }


def _fallback(path: Path, candidates: list[Path]) -> Path | None:
    if path.exists():
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _check_semantic_rows(rows: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    for idx, row in enumerate(rows):
        ident = row.get("candidate_id") or row.get("window_id") or f"semantic[{idx}]"
        family = row.get("semantic_family")
        if not family:
            errors.append(f"{ident}: missing semantic_family")
        if "generation_safe" not in row:
            warnings.append(f"{ident}: generation_safe missing")
        elif not isinstance(row.get("generation_safe"), bool):
            errors.append(f"{ident}: generation_safe is not boolean")
        if family == "unknown" and row.get("generation_safe") is True:
            errors.append(f"{ident}: unknown record marked generation_safe")
        if family == "bj_oral" and row.get("preserve_for_future_dataset") is False:
            warnings.append(f"{ident}: bj_oral record is not marked preserve_for_future_dataset")
        if row.get("source_world_coords_used") is True and row.get("generation_safe") is True:
            errors.append(f"{ident}: generation_safe record claims source_world_coords_used")
        actor = str(row.get("technical_actor_id") or row.get("technical_atom_id") or "").lower()
        if row.get("generation_safe") is True and actor in {"person/root", "root", "world"}:
            errors.append(f"{ident}: root/world actor marked generation_safe")


def _check_cowgirl_rows(rows: list[dict[str, Any]], errors: list[str], warnings: list[str]) -> None:
    seen_windows: Counter[str] = Counter()
    seen_samples: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        ident = row.get("candidate_id") or row.get("window_id") or f"cowgirl[{idx}]"
        category = str(row.get("category") or "")
        family = row.get("semantic_family")
        seen_windows.update([str(row.get("window_id"))])
        if row.get("sample_id"):
            seen_samples.update([str(row.get("sample_id"))])
        if category == "cowgirl_clean_motion_generation_safe":
            if family != "cowgirl":
                errors.append(f"{ident}: clean generation-safe category is not semantic_family cowgirl")
            if row.get("phase") == "low_motion_hold":
                errors.append(f"{ident}: clean generation-safe category has low_motion_hold phase")
            if row.get("contact_support") in {"standing_hand_head_gesture"}:
                errors.append(f"{ident}: standing gesture marked clean Cowgirl")
            if row.get("anchor_motion_weird") or _num(row.get("lower_body_anchor_stability"), 1.0) < 0.3:
                errors.append(f"{ident}: anchor unsafe record marked clean generation-safe")
        if "bj_oral" in category and family == "cowgirl":
            errors.append(f"{ident}: BJ/oral category still marked Cowgirl")
        if category == "cowgirl_hands_on_partner_chest":
            if _num(row.get("contact_support_confidence")) < 0.5:
                warnings.append(f"{ident}: hands_on_partner_chest has weak/missing contact evidence")
            if row.get("contact_support") in {"unknown", "unknown_contact"}:
                errors.append(f"{ident}: hands_on_partner_chest claimed with missing partner context")
        if category == "cowgirl_missing_partner_context" and row.get("contact_support") == "hands_on_partner_chest":
            errors.append(f"{ident}: missing partner context claims hands_on_partner_chest")
        if row.get("generation_safe") is True and category not in {"cowgirl_clean_motion_generation_safe"}:
            errors.append(f"{ident}: non-clean category marked generation_safe")
    for wid, count in seen_windows.items():
        if wid and wid != "None" and count > 1:
            warnings.append(f"duplicate window_id in Cowgirl DB: {wid} ({count})")
    for sample, count in seen_samples.items():
        if count > 50:
            warnings.append(f"very high sample reuse in Cowgirl DB: {sample} ({count})")


def _write_report(
    semantic_path: Path | None,
    cowgirl_path: Path | None,
    semantic_rows: list[dict[str, Any]],
    cowgirl_rows: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    out: str | Path,
) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semantic DB Invariant Report",
        "",
        f"- Semantic DB: `{semantic_path}`",
        f"- Cowgirl DB: `{cowgirl_path}`",
        f"- Semantic records: {len(semantic_rows)}",
        f"- Cowgirl records: {len(cowgirl_rows)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Cowgirl Categories",
        "",
    ]
    cats = Counter(r.get("category") for r in cowgirl_rows)
    lines.extend(f"- `{k}`: {v}" for k, v in cats.most_common()) if cats else lines.append("- None")
    lines.extend(["", "## Invalid Generation-Safe Records", ""])
    invalid = [e for e in errors if "generation" in e.lower() or "clean generation-safe" in e.lower()]
    lines.extend(f"- {e}" for e in invalid[:50]) if invalid else lines.append("- None")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {e}" for e in errors[:100]) if errors else lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {w}" for w in warnings[:100]) if warnings else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value != value:
            return default
        return float(value)
    except Exception:
        return default

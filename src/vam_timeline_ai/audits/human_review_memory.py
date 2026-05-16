"""Human review memory ledger for audit-only semantic review findings."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import csv

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


LEDGER_FIELDS = [
    "review_id",
    "source_review_folder",
    "run_id",
    "source_scene_file",
    "technical_actor_id",
    "start_seconds",
    "end_seconds",
    "system_semantic_family",
    "human_semantic_family",
    "system_pose",
    "human_pose",
    "system_motion",
    "human_motion",
    "system_partner_relation",
    "human_partner_relation",
    "system_contact_support",
    "human_contact_support",
    "system_generation_safe",
    "human_generation_safe",
    "verdict",
    "error_tags",
    "notes",
]


def build_human_review_ledger(
    run_dir: str | Path,
    include_runs: str | Path,
    out_jsonl: str | Path,
    out_csv: str | Path,
    report: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir)
    roots = _parse_include_runs(include_runs)
    if run not in roots:
        roots.append(run)
    review_dirs = _find_review_dirs(roots)
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for review_dir in review_dirs:
        try:
            records.extend(_records_from_review_dir(review_dir))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{review_dir}: {exc}")
    records.sort(key=lambda r: (r.get("run_id") or "", r.get("source_review_folder") or "", r.get("review_id") or ""))
    write_jsonl(out_jsonl, records)
    _write_csv(records, out_csv)
    _write_report(records, warnings, report)
    return {
        "status": "ok",
        "records": len(records),
        "known_human_verdicts": sum(1 for r in records if r.get("verdict") not in {"unavailable", "unknown", ""}),
        "review_folders": len(review_dirs),
        "warnings": warnings,
        "out_jsonl": str(out_jsonl),
    }


def _parse_include_runs(value: str | Path) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def _find_review_dirs(roots: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        audits = root / "audits"
        if not audits.exists():
            continue
        for path in sorted(audits.glob("semantic_review*")):
            if path.is_dir() and path not in seen:
                seen.add(path)
                dirs.append(path)
    return dirs


def _records_from_review_dir(review_dir: Path) -> list[dict[str, Any]]:
    review_rows = {r.get("review_id"): r for r in load_jsonl(review_dir / "semantic_review_010.jsonl") if r.get("review_id")}
    manifest = {r.get("review_id"): r for r in load_jsonl(review_dir / "vam_review_package" / "vam_review_manifest.jsonl") if r.get("review_id")}
    notes = _load_human_notes(review_dir / "semantic_review_010_human_notes.yaml")
    answers = _load_answer_sheet(review_dir / "semantic_review_010_answer_sheet.yaml")
    package_answers = _load_answer_sheet(review_dir / "vam_review_package" / "vam_review_answer_sheet.yaml")
    ids = set(review_rows) | set(manifest) | set(notes) | set(answers) | set(package_answers)
    records: list[dict[str, Any]] = []
    for rid in sorted(ids):
        system = dict(review_rows.get(rid) or {})
        system.update({k: v for k, v in (manifest.get(rid) or {}).items() if _has_value(v)})
        human = {}
        human.update(answers.get(rid) or {})
        human.update(package_answers.get(rid) or {})
        human.update(notes.get(rid) or {})
        records.append(_ledger_record(rid, review_dir, system, human))
    return records


def _ledger_record(rid: str, review_dir: Path, system: dict[str, Any], human: dict[str, Any]) -> dict[str, Any]:
    pose = system.get("pose_semantics") if isinstance(system.get("pose_semantics"), dict) else {}
    motion = system.get("motion_semantics") if isinstance(system.get("motion_semantics"), dict) else {}
    verdict = _verdict(human)
    tags = list(human.get("actual_labels") or human.get("error_tags") or [])
    human_family = human.get("semantic_family") or human.get("actual_semantic_family") or ""
    return {
        "review_id": rid,
        "source_review_folder": str(review_dir),
        "run_id": _run_id(review_dir),
        "source_scene_file": system.get("source_scene_file") or _scene_file(system.get("source_scene_path")),
        "technical_actor_id": system.get("technical_actor_id") or system.get("technical_atom_id") or "",
        "start_seconds": system.get("start_seconds"),
        "end_seconds": system.get("end_seconds"),
        "system_semantic_family": system.get("semantic_family") or "",
        "human_semantic_family": human_family,
        "system_pose": _join_pose(pose.get("family") or system.get("pose_family"), pose.get("subtype") or system.get("pose_subtype")),
        "human_pose": human.get("actual_pose") or human.get("human_pose") or "",
        "system_motion": motion.get("subtype") or system.get("motion_subtype") or "",
        "human_motion": human.get("actual_motion") or human.get("human_motion") or "",
        "system_partner_relation": _join(system.get("partner_relation")),
        "human_partner_relation": human.get("actual_partner_relation") or human.get("human_partner_relation") or "",
        "system_contact_support": system.get("contact_support") or "",
        "human_contact_support": human.get("actual_contact_support") or human.get("human_contact_support") or "",
        "system_generation_safe": system.get("generation_safe"),
        "human_generation_safe": _human_generation_safe(system.get("generation_safe"), human),
        "verdict": verdict,
        "error_tags": tags,
        "notes": human.get("notes") or "",
    }


def _load_human_notes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("reviews") or {}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return False
    return True


def _load_answer_sheet(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("reviews") or {}


def _verdict(human: dict[str, Any]) -> str:
    value = str(human.get("user_verdict") or "").strip()
    if value:
        return value
    correctness = [
        human.get("semantic_family_correct"),
        human.get("pose_correct"),
        human.get("motion_correct"),
        human.get("partner_relation_correct"),
        human.get("contact_support_correct"),
        human.get("generation_safe_correct"),
    ]
    known = [str(v).lower() for v in correctness if str(v).lower() not in {"", "unknown", "none"}]
    if not known:
        return "unavailable"
    if all(v == "true" for v in known):
        return "correct"
    if any(v == "true" for v in known):
        return "partially_correct"
    if all(v == "false" for v in known):
        return "wrong"
    return "unclear"


def _human_generation_safe(system_value: Any, human: dict[str, Any]) -> Any:
    value = str(human.get("generation_safe_correct") or "").lower()
    if value == "true":
        return system_value
    if value == "false":
        return None
    return human.get("human_generation_safe") or ""


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["error_tags"] = _join(row.get("error_tags"))
            writer.writerow(out)


def _write_report(rows: list[dict[str, Any]], warnings: list[str], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    verdicts = Counter(r.get("verdict") for r in rows)
    tags = Counter(tag for r in rows for tag in (r.get("error_tags") or []))
    by_review = Counter(Path(r.get("source_review_folder") or "").name for r in rows)
    contact_known = [r for r in rows if r.get("human_contact_support")]
    lines = [
        "# Human Review Ledger Report",
        "",
        "This ledger collects audit review findings. It is not manual training ground truth.",
        "",
        f"- Ledger records: {len(rows)}",
        f"- Records with known human verdict: {sum(1 for r in rows if r.get('verdict') not in {'unavailable', 'unknown', ''})}",
        "",
        "## Verdict Counts",
        "",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in verdicts.most_common()) if verdicts else lines.append("- None")
    lines.extend(["", "## Reviews Covered", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in by_review.most_common()) if by_review else lines.append("- None")
    lines.extend(["", "## Common Error Tags", ""])
    lines.extend(f"- `{k}`: {v}" for k, v in tags.most_common(20)) if tags else lines.append("- None yet")
    lines.extend(["", "## Contact/Support Human Answers", "", f"- Known human contact/support fields: {len(contact_known)}"])
    lines.extend(["", "## Missing-Data Warnings", ""])
    lines.extend(f"- {w}" for w in warnings) if warnings else lines.append("- None")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_id(path: Path) -> str:
    parts = list(path.parts)
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def _scene_file(value: Any) -> str:
    return Path(str(value)).name if value else ""


def _join_pose(family: Any, subtype: Any) -> str:
    return " / ".join(str(v) for v in [family, subtype] if v)


def _join(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)

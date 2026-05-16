"""Local semantic review UI and static workbench builder."""

from __future__ import annotations

from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
import json
import shutil
import urllib.parse
import webbrowser

import yaml

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl
from vam_timeline_ai.ui.review_ui_assets import APP_JS, INDEX_HTML, STYLE_CSS


COMPACT_CANDIDATE_FIELDS = [
    "candidate_id",
    "window_id",
    "sample_id",
    "source_scene_file",
    "technical_actor_id",
    "technical_atom_id",
    "semantic_family",
    "category",
    "pose_family",
    "pose_subtype",
    "motion_subtype",
    "phase",
    "clean_motion_gate",
    "clean_motion_gate_reason",
    "hip_motion_strength",
    "pelvis_trajectory_strength",
    "pelvis_cycle_count",
    "motion_duration_confidence",
    "partner_relation",
    "contact_support",
    "generation_safe",
    "semantic_score",
    "pose_score",
    "motion_score",
    "interaction_score",
    "contact_support_confidence",
    "warnings",
]


def build_static_review_ui(run_dir: str | Path, review_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    review = Path(review_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_review_ui_data(run, review)
    (out / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (out / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    (out / "app.js").write_text(APP_JS, encoding="utf-8")
    (out / "review_data.js").write_text(
        "window.REVIEW_UI_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(_static_readme(out, review), encoding="utf-8")
    return {
        "status": "ok",
        "out_dir": str(out),
        "index": str(out / "index.html"),
        "review_items": len(data["review_items"]),
        "candidate_rows": len(data["candidates"]),
    }


def launch_review_ui(run_dir: str | Path, review_dir: str | Path, host: str, port: int) -> dict[str, Any]:
    review = Path(review_dir)
    out = review / "review_ui_static"
    summary = build_static_review_ui(run_dir, review, out)
    handler = _handler_factory(out, review)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Semantic Review UI running at {url}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    summary["url"] = url
    return summary


def build_review_ui_data(run: Path, review: Path) -> dict[str, Any]:
    review_rows = {r.get("review_id"): r for r in load_jsonl(review / "semantic_review_010.jsonl") if r.get("review_id")}
    manifest_rows = {
        r.get("review_id"): r
        for r in load_jsonl(review / "vam_review_package" / "vam_review_manifest.jsonl")
        if r.get("review_id")
    }
    review_items = []
    for rid in sorted(set(review_rows) | set(manifest_rows)):
        merged = {}
        merged.update(review_rows.get(rid) or {})
        merged.update(_non_empty(manifest_rows.get(rid) or {}))
        _add_item_paths(merged, review)
        review_items.append(_normalize_review_item(merged))
    candidates = _load_compact_candidates(run)
    ledger = load_jsonl(run / "audits" / "human_review_ledger.jsonl")
    return {
        "schema": "vam_timeline_ai_review_ui_v0",
        "review_name": review.name,
        "run_dir": str(run),
        "review_dir": str(review),
        "review_items": review_items,
        "candidates": candidates,
        "candidate_summary": _candidate_summary(candidates),
        "hypotheses": _hypotheses(candidates, ledger),
        "error_taxonomy": _error_taxonomy(ledger),
        "status": _status(run, review, review_items, candidates),
        "answer_schema": answer_schema(),
    }


def answer_schema() -> dict[str, Any]:
    return {
        "required": ["review_id"],
        "fields": [
            "semantic_family_correct",
            "actual_semantic_family",
            "pose_correct",
            "actual_pose",
            "motion_correct",
            "actual_motion",
            "partner_relation_correct",
            "actual_partner_relation",
            "contact_support_correct",
            "actual_contact_support",
            "generation_safe_correct",
            "actual_generation_safe",
            "verdict",
            "error_tags",
            "notes",
        ],
        "allowed_correctness": ["true", "false", "unknown", "not_applicable"],
        "audit_only": True,
        "writes_manual_labels": False,
    }


def save_ui_answers(review_dir: str | Path, answers: list[dict[str, Any]]) -> dict[str, Any]:
    review = Path(review_dir)
    cleaned = [validate_answer(row) for row in answers if row.get("review_id")]
    jsonl = review / "human_review_ui_answers.jsonl"
    yaml_path = review / "human_review_ui_answers.yaml"
    _backup(jsonl)
    _backup(yaml_path)
    write_jsonl(jsonl, cleaned)
    yaml_path.write_text(yaml.safe_dump({"reviews": {r["review_id"]: _without_review_id(r) for r in cleaned}}, sort_keys=False), encoding="utf-8")
    return {"status": "ok", "answers": len(cleaned), "jsonl": str(jsonl), "yaml": str(yaml_path)}


def validate_answer(row: dict[str, Any]) -> dict[str, Any]:
    rid = str(row.get("review_id") or "").strip()
    if not rid:
        raise ValueError("answer row is missing review_id")
    out = {"review_id": rid}
    allowed_correctness = {"true", "false", "unknown", "not_applicable", True, False}
    for field in answer_schema()["fields"]:
        value = row.get(field, "" if field != "error_tags" else [])
        if field.endswith("_correct") or field == "actual_generation_safe":
            if value in {True, False}:
                value = "true" if value else "false"
            value = str(value or "unknown")
            if value not in allowed_correctness:
                value = "unknown"
        if field == "error_tags":
            value = list(value) if isinstance(value, list) else [str(value)] if value else []
        out[field] = value
    return out


def _handler_factory(root: Path, review_dir: Path):
    class ReviewHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_POST(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/save-answers":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            try:
                body = json.loads(payload)
                summary = save_ui_answers(review_dir, body.get("answers") or [])
                text = f"Saved {summary['answers']} answers to {summary['jsonl']}"
                self.send_response(200)
                self.send_header("content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(text.encode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.send_response(400)
                self.send_header("content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(str(exc).encode("utf-8"))

    return ReviewHandler


def _non_empty(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if _has_value(v)}


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return False
    return True


def _add_item_paths(item: dict[str, Any], review: Path) -> None:
    rid = item.get("review_id")
    if not rid:
        return
    item_dir = review / "vam_review_package" / "items" / rid
    item_review = item_dir / "item_review.md"
    if item_review.exists():
        item["item_review_path"] = str(item_review)
    item["review_package_item_folder"] = str(item_dir)


def _normalize_review_item(item: dict[str, Any]) -> dict[str, Any]:
    pose = item.get("pose_semantics") if isinstance(item.get("pose_semantics"), dict) else {}
    motion = item.get("motion_semantics") if isinstance(item.get("motion_semantics"), dict) else {}
    out = dict(item)
    out.setdefault("pose_family", pose.get("family"))
    out.setdefault("pose_subtype", pose.get("subtype"))
    out.setdefault("motion_subtype", motion.get("subtype"))
    out.setdefault("phase", motion.get("phase"))
    scores = item.get("evidence_scores") if isinstance(item.get("evidence_scores"), dict) else {}
    for key in [
        "rider_above_partner_score",
        "pelvis_alignment_score",
        "hands_on_partner_chest_score",
        "hands_on_partner_hips_score",
        "partner_lying_score",
    ]:
        out.setdefault(key, scores.get(key))
    return out


def _load_compact_candidates(run: Path) -> list[dict[str, Any]]:
    cow = _load_first(run / "datasets" / "cowgirl_candidate_db_v7.jsonl", run / "datasets" / "cowgirl_candidate_db_v6.jsonl", run / "datasets" / "cowgirl_candidate_db_v5.jsonl")
    sem = _load_first(run / "datasets" / "semantic_candidate_db_v2.jsonl", run / "datasets" / "semantic_candidate_db_v1.jsonl", run / "datasets" / "semantic_candidate_db_v0.jsonl")
    rows = cow or sem
    compact = []
    for row in rows:
        compact.append(_compact_candidate(row))
    return compact


def _compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field) for field in COMPACT_CANDIDATE_FIELDS if field in row}
    warnings = row.get("warnings") or []
    if isinstance(warnings, list):
        out["warning_count"] = len(warnings)
        out["warnings"] = warnings[:3]
    elif warnings:
        out["warning_count"] = 1
        out["warnings"] = [str(warnings)]
    return out


def _load_first(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            return load_jsonl(path)
    return []


def _candidate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(candidates),
        "by_family": dict(Counter(str(r.get("semantic_family") or "unknown") for r in candidates).most_common()),
        "by_category": dict(Counter(str(r.get("category") or "unknown") for r in candidates).most_common(30)),
    }


def _hypotheses(candidates: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _hyp("Cowgirl generation-safe is reliable", "Generation-safe Cowgirl candidates should survive human review.", candidates, lambda r: r.get("category") == "cowgirl_clean_motion_generation_safe", "high"),
        _hyp("Cowgirl low-motion holds are separated from clean motion", "Low-motion Cowgirl pose/context should not appear as clean motion.", candidates, lambda r: "low_motion" in str(r.get("category")), "medium"),
        _hyp("BJ/oral is preserved as own family", "BJ/oral examples should be excluded from Cowgirl but retained.", candidates, lambda r: r.get("semantic_family") == "bj_oral" or "bj_oral" in str(r.get("category")), "medium"),
        _hyp("hands_on_partner_chest is reliable", "Partner chest contact requires strong target evidence.", candidates, lambda r: "chest" in str(r.get("contact_support")) or "chest" in str(r.get("category")), "high"),
        _hyp("hands_on_partner_hips is reliable", "Partner hip contact requires disambiguation from chest/head/floor.", candidates, lambda r: "hips" in str(r.get("contact_support")) or "hips" in str(r.get("category")), "medium"),
        _hyp("standing hand/head is excluded from Cowgirl", "Standing gesture candidates should stay negative for Cowgirl.", candidates, lambda r: "standing" in str(r.get("category")) or r.get("semantic_family") in {"hand_gesture", "head_gesture"}, "medium"),
        _hyp("receiver response is not active rider", "Receiver response should not be promoted to active Cowgirl.", candidates, lambda r: "receiver_response" in str(r.get("category")) or r.get("semantic_family") == "receiver_response", "medium"),
        _hyp("duplicate low-motion selection is fixed", "Review batches should avoid repeated near-duplicate holds.", candidates, lambda r: "low_motion" in str(r.get("category")), "medium"),
        _hyp("generation_safe excludes invalid pose/controller items", "Generation-safe should avoid invalid pose/controller warnings.", candidates, lambda r: r.get("generation_safe") is True, "high"),
        _hyp("partner relation is reliable", "Partner-relative claims should be validated against pair context.", candidates, lambda r: bool(r.get("partner_relation")), "high"),
    ]


def _hyp(name: str, description: str, rows: list[dict[str, Any]], pred, priority: str) -> dict[str, Any]:
    matched = [r for r in rows if pred(r)]
    confs = [float(r.get("contact_support_confidence") or r.get("interaction_score") or r.get("semantic_score") or 0.0) for r in matched]
    examples = [str(r.get("window_id") or r.get("candidate_id")) for r in matched[:20]]
    return {
        "name": name,
        "description": description,
        "count": len(matched),
        "average_confidence": sum(confs) / len(confs) if confs else None,
        "review_priority": priority,
        "recommended_examples": examples[:10],
    }


def _error_taxonomy(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    if not ledger:
        return {"available": False, "rows": []}
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for row in ledger:
        rid = row.get("review_id") or ""
        folder = Path(str(row.get("source_review_folder") or "")).name
        for tag in row.get("error_tags") or []:
            counts[str(tag)] += 1
            examples.setdefault(str(tag), []).append(f"{folder}/{rid}")
    return {
        "available": True,
        "rows": [
            {"error": key, "count": count, "examples": examples.get(key, [])[:8]}
            for key, count in counts.most_common(40)
        ],
    }


def _status(run: Path, review: Path, review_items: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_dir": str(run),
        "review_dir": str(review),
        "review_items": len(review_items),
        "candidate_rows": len(candidates),
        "manual_labels_yaml_touched": False,
        "ml_training": False,
        "internet_required": False,
        "server_optional": True,
    }


def _static_readme(out: Path, review: Path) -> str:
    return f"""# Static Semantic Review UI

Open `index.html` in a browser.

Answers are stored in browser localStorage. Use the Export Answers tab to
download `human_review_ui_answers.jsonl` or `human_review_ui_answers.yaml`.

This UI is audit-only. It does not modify `manual_labels.yaml`.

Review source: `{review}`
"""


def _without_review_id(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "review_id"}


def _backup(path: Path) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)

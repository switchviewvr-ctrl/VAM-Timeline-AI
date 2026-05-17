"""Static local research client skeleton."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import html
import json

from vam_timeline_ai.io.json_utils import load_jsonl
from vam_timeline_ai.semantics.ontology_loader import load_yaml


def build_research_client_v0(run_dir: str | Path, new_run: str | Path, out_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    new = Path(new_run)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "dashboard": _dashboard(run, new),
        "ontology": {
            "rig_anatomy": load_yaml("data/ontology/rig_anatomy_v1.yaml"),
            "component_ontology": load_yaml("data/ontology/component_ontology_v1.yaml"),
            "lexicon": load_yaml("data/ontology/nlp_lexicon_v1.yaml"),
        },
        "reports": _report_links(run, new),
        "read_only": True,
        "manual_labels_modified": False,
        "ml_training_performed": False,
        "timeline_generation_performed": False,
    }
    (out / "research_client_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "index.html").write_text(_html(data), encoding="utf-8")
    return {"status": "ok", "out_dir": str(out), "index": str(out / "index.html"), "read_only": True}


def _dashboard(run: Path, new: Path) -> dict[str, Any]:
    candidates = load_jsonl(new / "semantic_rescan_v2" / "ontology_motion_candidates_v1.jsonl")
    anatomy = load_jsonl(new / "features" / "rig_anatomy_features_v1.jsonl")
    manual_gt = load_jsonl(run / "manual_pose_ground_truth_v1" / "manual_pose_ground_truth_v1.jsonl")
    return {
        "base_run": str(run),
        "new_run": str(new),
        "candidate_count": len(candidates),
        "candidate_categories": dict(Counter(r.get("category") for r in candidates)),
        "rig_anatomy_feature_rows": len(anatomy),
        "manual_pose_ground_truth_rows": len(manual_gt),
    }


def _report_links(run: Path, new: Path) -> list[str]:
    paths = [
        new / "reports" / "nlp_lexicon_v1_report.md",
        new / "reports" / "rig_anatomy_features_v1_report.md",
        new / "nlp" / "token_resolution_example_v1.json",
        new / "nlp" / "motion_intent_example_v1.json",
        run / "ui" / "research_client_v0" / "research_client_data.json",
        Path("data/research/web_motion_context_v1/WEB_CONTEXT_REVIEW.md"),
    ]
    return [str(path) for path in paths if path.exists()]


def _html(data: dict[str, Any]) -> str:
    dashboard = data["dashboard"]
    tabs = [
        ("Dashboard", _pre(dashboard)),
        ("Review Queue Links", _links(data["reports"])),
        ("Manual Pose Ground Truth", f"<p>Rows: {dashboard.get('manual_pose_ground_truth_rows')}</p>"),
        ("Ontology Browser", _pre(data["ontology"].get("component_ontology", {}))),
        ("Rig Anatomy Browser", _pre(data["ontology"].get("rig_anatomy", {}))),
        ("NLP Lexicon Browser", _pre({"entries": len((data["ontology"].get("lexicon") or {}).get("entries") or [])})),
        ("Web Context Research", _links([p for p in data["reports"] if "web_motion_context" in p])),
        ("ML Ranker Status", "<p>Read-only placeholder. Train/score commands are intentionally not triggered by this client.</p>"),
        ("Prompt Intent Parser Demo", _links([p for p in data["reports"] if "motion_intent" in p or "nlp_token" in p])),
        ("Safety/Validation Reports", "<p>Use audit-repo-safety reports from the run folders.</p>"),
    ]
    nav = "".join(f"<button data-tab='{i}'>{html.escape(name)}</button>" for i, (name, _) in enumerate(tabs))
    sections = "".join(f"<section id='tab{i}'><h2>{html.escape(name)}</h2>{body}</section>" for i, (name, body) in enumerate(tabs))
    return (
        "<!doctype html><meta charset='utf-8'><title>VAM Timeline AI Research Client v0</title>"
        "<style>body{font-family:Arial;margin:24px;background:#f7f7f5;color:#202020}button{margin:4px;padding:8px 10px}"
        "section{display:none;background:white;border:1px solid #ccc;padding:16px;margin-top:12px}section:first-of-type{display:block}"
        "pre{white-space:pre-wrap;max-height:560px;overflow:auto;background:#f0f0f0;padding:12px}</style>"
        "<h1>VAM Timeline AI Research Client v0</h1><p>Read-only local research UI. No labels, ML, or Timeline exports are written here.</p>"
        f"<nav>{nav}</nav>{sections}"
        "<script>document.querySelectorAll('button[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('section').forEach(s=>s.style.display='none');document.getElementById('tab'+b.dataset.tab).style.display='block';});</script>"
    )


def _pre(value: Any) -> str:
    return "<pre>" + html.escape(json.dumps(value, indent=2, ensure_ascii=False)) + "</pre>"


def _links(paths: list[str]) -> str:
    if not paths:
        return "<p>No files found yet.</p>"
    return "<ul>" + "".join(f"<li><code>{html.escape(path)}</code></li>" for path in paths) + "</ul>"

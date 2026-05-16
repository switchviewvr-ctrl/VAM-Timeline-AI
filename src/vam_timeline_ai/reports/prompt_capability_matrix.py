"""Prompt capability matrix for honest operator status."""

from __future__ import annotations

from pathlib import Path


ROWS = [
    ("slow cowgirl grinding", "yes", "yes", "yes", "limited", "prototype relative flow", "experimental native export exists, not final", "medium", "needs more reviewed clean Cowgirl examples"),
    ("cowgirl riding", "yes", "partial", "partial", "limited", "prototype relative flow", "experimental native export exists, not final", "medium/low", "riding vs grinding needs more review"),
    ("vertical bounce", "yes", "partial", "yes", "limited", "not robust", "not ready", "low/medium", "needs subtype validation"),
    ("forward/back rock", "partial", "weak", "weak", "limited", "not robust", "not ready", "low", "primitive group currently sparse"),
    ("cowgirl hands on partner chest", "partial", "experimental", "no", "experimental", "not ready", "not ready", "low", "contact/support confidence needs v16 review"),
    ("cowgirl hands on floor/bed", "partial", "experimental", "no", "experimental", "not ready", "not ready", "low", "support target evidence incomplete"),
    ("cowgirl hands free", "partial", "weak", "no", "weak", "not ready", "not ready", "low", "needs explicit contact/support negatives"),
    ("BJ/oral motion", "yes as separate family", "candidate inventory", "no", "separate from Cowgirl", "not built", "not ready", "medium for preservation, low for generation", "family-specific DB needed"),
    ("standing hand/head gesture", "yes as negative/family", "candidate inventory", "no", "not applicable", "not built", "not ready", "medium as negative", "family semantics not developed"),
    ("doggy", "not yet", "no", "no", "no", "not built", "not ready", "not ready", "needs family-specific rescan"),
    ("transitions", "phase split started", "partial", "no", "limited", "not built", "not ready", "low", "transition semantics need human review"),
    ("prompt to native Timeline", "planning only", "indirect", "prototype only", "not reliable", "prototype only", "not final", "not ready", "needs reviewed semantic plan + retarget/export validation"),
]


def write_prompt_capability_matrix(run_dir: str | Path, out: str | Path) -> dict[str, str]:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Prompt Capability Matrix",
        "",
        "Honest status after clean_v3 calibration. This does not claim final text-to-animation readiness.",
        "",
        "| Prompt / Feature | Semantic recognition | Candidate DB support | Primitive support | Partner/contact support | Generation flow support | Native Timeline export support | Confidence | Missing pieces |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines = header + ["|" + "|".join(row) + "|" for row in ROWS]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "ok", "out": str(target)}

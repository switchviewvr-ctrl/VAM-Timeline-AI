"""Write human-friendly labeling next-step reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.semantics.review_batch_discovery import find_latest_review_batch


def write_labeling_next_step(run_dir: str | Path, out: str | Path) -> dict[str, Any]:
    discovery = find_latest_review_batch(run_dir)
    latest = discovery.get("latest_batch")
    lines = ["# Human Labeling Next Step", ""]
    if not latest:
        lines.extend([
            "No valid review batch was found.",
            "",
            "Build a review batch before labeling.",
        ])
    elif discovery["status"] == "ready_for_ingestion":
        batch_path = Path(latest["path"])
        lines.extend([
            f"Latest batch `{latest['batch_name']}` has usable edited labels.",
            "",
            f"- Usable edited entries: {latest['usable_edited_entries']}",
            "",
            "Run:",
            "",
            "```powershell",
            "python -m vam_timeline_ai.cli ingest-latest-edited-batch ^",
            f"  --run-dir {Path(run_dir)} ^",
            "  --schema data\\labels\\manual_labels.schema_v2.yaml ^",
            "  --stop-if-missing true",
            "```",
            "",
            f"Edited file: `{batch_path / 'manual_labels.edited.yaml'}`",
        ])
    else:
        batch_path = Path(latest["path"])
        lines.extend([
            f"Latest review batch is `{latest['batch_name']}` and is waiting for human labels.",
            "",
            "Open this preview index:",
            "",
            f"`{batch_path / 'previews' / 'index.html'}`",
            "",
            "Copy this stub:",
            "",
            f"`{batch_path / 'manual_labels.stub.yaml'}`",
            "",
            "Save the edited copy as:",
            "",
            f"`{batch_path / 'manual_labels.edited.yaml'}`",
            "",
            "Labeling rules:",
            "",
            "- Do not modify IDs.",
            "- Do not paste `weak_` labels as manual labels.",
            "- Include positive labels only when visually confirmed.",
            "- Add negative/control labels where useful.",
            "- Use uncertain labels instead of guessing.",
            "- Use `include_for_ml: true` only when confidence is useful.",
            "- Label across multiple scenes and samples.",
            "- Atom names and filenames are hints only, not semantic truth.",
        ])
    result = {"run_dir": str(run_dir), "status": discovery["status"], "latest_batch": latest}
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

"""Project and reference path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else fallback


DEFAULT_MOCAP_COMPILER = _env_path("VAM_TIMELINE_AI_MOCAP_COMPILER", DEFAULT_PROJECT_ROOT / "references" / "external" / "vam_mocap_dataset_compiler")
DEFAULT_TIMELINE_REPO = _env_path("VAM_TIMELINE_AI_TIMELINE_REPO", DEFAULT_PROJECT_ROOT / "references" / "external" / "vam-timeline")
DEFAULT_RAW_MOCAP_RESEARCH = _env_path("VAM_TIMELINE_AI_RAW_SCENES", DEFAULT_PROJECT_ROOT / "data" / "raw" / "vam_scenes")
DEFAULT_VIRTUAL_COMPANION = _env_path("VAM_TIMELINE_AI_VIRTUAL_COMPANION", DEFAULT_PROJECT_ROOT / "references" / "external" / "virtual_companion")


@dataclass(frozen=True)
class ReferencePaths:
    project_root: Path = DEFAULT_PROJECT_ROOT
    mocap_compiler: Path = DEFAULT_MOCAP_COMPILER
    timeline_repo: Path = DEFAULT_TIMELINE_REPO
    raw_mocap_research: Path = DEFAULT_RAW_MOCAP_RESEARCH
    virtual_companion: Path = DEFAULT_VIRTUAL_COMPANION

    def as_status_dict(self) -> dict[str, dict[str, object]]:
        return {
            "project_root": path_status(self.project_root),
            "mocap_compiler": path_status(self.mocap_compiler),
            "timeline_repo": path_status(self.timeline_repo),
            "raw_mocap_research": path_status(self.raw_mocap_research),
            "virtual_companion": path_status(self.virtual_companion),
        }


def path_status(path: str | Path) -> dict[str, object]:
    p = Path(path)
    return {"path": str(p), "exists": p.exists(), "is_dir": p.is_dir()}


def default_reference_paths() -> ReferencePaths:
    return ReferencePaths()

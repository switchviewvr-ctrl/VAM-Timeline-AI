"""Public GitHub repository safety checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


GENERATED_EXTENSIONS = {".npz", ".npy", ".pkl", ".joblib", ".model", ".onnx"}
PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".mp4", ".webm"}
BINARY_ATTR_EXTENSIONS = ["*.npz", "*.npy", "*.pkl", "*.joblib", "*.model", "*.onnx", "*.png", "*.jpg", "*.jpeg", "*.mp4", "*.webm", "*.var"]
REQUIRED_GITIGNORE_SNIPPETS = [
    "data/runs/",
    "data/**/samples/",
    "data/**/previews/",
    "data/**/*.npz",
    "data/**/*.npy",
    "data/**/*.pkl",
    "data/**/*.joblib",
    "data/**/*.model",
    "data/**/*.onnx",
    "data/**/*.png",
    "data/**/*.jpg",
    "data/**/*.jpeg",
    "data/**/*.mp4",
    "data/**/*.webm",
    "data/labels/manual_labels.yaml",
    "data/labels/**/*.edited.yaml",
    "data/labels/batches/",
    "outputs/",
    "*.var",
    "*.log",
]


def audit_repo_safety(project_root: str | Path, out: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    checks: list[dict[str, Any]] = []
    git_available = _git_available(root)
    is_repo = git_available and _git_ok(root, ["rev-parse", "--is-inside-work-tree"])
    branch = _git_text(root, ["branch", "--show-current"]) if is_repo else ""
    remotes = _git_text(root, ["remote", "-v"]).splitlines() if is_repo else []
    tracked = _git_text(root, ["ls-files"]).splitlines() if is_repo else []

    if git_available:
        checks.append(_check("git_available", "OK", "git command is available"))
    else:
        checks.append(_check("git_available", "WARNING", "git command is unavailable; tracked-file checks skipped"))
    checks.append(_check("is_git_repo", "OK" if is_repo else "WARNING", f"is git repo: {is_repo}"))
    checks.append(_check("remote_exists", "OK" if remotes else "WARNING", f"remotes: {remotes or 'none'}"))

    tracked_set = {p.replace("\\", "/") for p in tracked}
    raw_scenes = [p for p in tracked_set if _looks_like_raw_scene(root / p)]
    data_runs = [p for p in tracked_set if p.startswith("data/runs/")]
    manual_labels = [p for p in tracked_set if p == "data/labels/manual_labels.yaml" or p.endswith("/manual_labels.yaml")]
    edited_labels = [p for p in tracked_set if p.endswith("manual_labels.edited.yaml") or p.endswith(".edited.yaml")]
    generated = [p for p in tracked_set if Path(p).suffix.lower() in GENERATED_EXTENSIONS]
    previews = [p for p in tracked_set if Path(p).suffix.lower() in PREVIEW_EXTENSIONS and ("/previews/" in p or p.startswith("data/"))]
    var_files = [p for p in tracked_set if Path(p).suffix.lower() == ".var"]
    raw_json_candidates = [p for p in tracked_set if p.startswith("data/raw/") and p.lower().endswith(".json")]

    checks.extend([
        _list_check("raw_vam_scenes_tracked", raw_scenes, hard_error=True),
        _list_check("data_runs_tracked", data_runs, hard_error=True),
        _list_check("manual_labels_yaml_tracked", manual_labels, hard_error=True),
        _list_check("edited_manual_labels_tracked", edited_labels, hard_error=True),
        _list_check("generated_binary_arrays_or_models_tracked", generated, hard_error=True),
        _list_check("review_previews_tracked", previews, hard_error=True),
        _list_check("var_packages_tracked", var_files, hard_error=True),
        _list_check("raw_json_dumps_tracked", raw_json_candidates, hard_error=True),
    ])

    gitignore_text = _read(root / ".gitignore")
    missing_ignore = [snippet for snippet in REQUIRED_GITIGNORE_SNIPPETS if snippet not in gitignore_text]
    checks.append(_list_check("gitignore_missing_required_rules", missing_ignore, hard_error=False))

    gitattributes_text = _read(root / ".gitattributes")
    missing_attrs = [snippet for snippet in BINARY_ATTR_EXTENSIONS if snippet not in gitattributes_text]
    checks.append(_list_check("gitattributes_missing_binary_rules", missing_attrs, hard_error=False))

    checks.append(_check("github_workflow_exists", "OK" if (root / ".github" / "workflows").exists() else "WARNING", ".github/workflows present"))
    checks.append(_check("pyproject_exists", "OK" if (root / "pyproject.toml").exists() else "ERROR", "pyproject.toml present"))

    errors = [c for c in checks if c["status"] == "ERROR"]
    warnings = [c for c in checks if c["status"] == "WARNING"]
    result = {
        "project_root": str(root),
        "git_available": git_available,
        "is_git_repo": is_repo,
        "branch": branch,
        "remotes": remotes,
        "tracked_file_count": len(tracked),
        "status": "ERROR" if errors else ("WARNING" if warnings else "OK"),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    _write_report(result, out)
    return result


def _git_available(root: Path) -> bool:
    try:
        subprocess.run(["git", "--version"], cwd=root, capture_output=True, text=True, timeout=10)
        return True
    except Exception:
        return False


def _git_ok(root: Path, args: list[str]) -> bool:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20).returncode == 0
    except Exception:
        return False


def _git_text(root: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _looks_like_raw_scene(path: Path) -> bool:
    if path.suffix.lower() != ".json" or not path.exists() or path.stat().st_size > 10_000_000:
        return False
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return isinstance(data, dict) and isinstance(data.get("atoms"), list)
    except Exception:
        return False


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _check(name: str, status: str, message: str) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "items": []}


def _list_check(name: str, items: list[str], hard_error: bool) -> dict[str, Any]:
    if items:
        return {"name": name, "status": "ERROR" if hard_error else "WARNING", "message": f"{len(items)} item(s) found", "items": sorted(items)}
    return {"name": name, "status": "OK", "message": "none found", "items": []}


def _write_report(result: dict[str, Any], out: str | Path) -> None:
    lines = [
        "# Repository Safety Audit",
        "",
        f"- Project root: `{result['project_root']}`",
        f"- Status: {result['status']}",
        f"- Git available: {result['git_available']}",
        f"- Is git repo: {result['is_git_repo']}",
        f"- Branch: `{result['branch']}`",
        f"- Remotes: {result['remotes'] or 'none'}",
        f"- Tracked files: {result['tracked_file_count']}",
        "",
        "## Checks",
        "",
    ]
    for check in result["checks"]:
        lines.append(f"### {check['status']} - {check['name']}")
        lines.append("")
        lines.append(check["message"])
        if check["items"]:
            lines.append("")
            for item in check["items"][:100]:
                lines.append(f"- `{item}`")
            if len(check["items"]) > 100:
                lines.append(f"- ... {len(check['items']) - 100} more")
        lines.append("")
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")

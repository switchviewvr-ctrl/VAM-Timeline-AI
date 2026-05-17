"""Client for the optional local VaM Reality Capture Bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import urllib.error
import urllib.request

from vam_timeline_ai.io.json_utils import load_jsonl, write_jsonl


def run_vam_reality_capture_v0(requests: str | Path, bridge_url: str, mode: str, out: str | Path) -> dict[str, Any]:
    rows = load_jsonl(requests)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    status = _get_status(bridge_url)
    if not status.get("available"):
        blocked = target.parent / "BLOCKED_VAM_CAPTURE_BRIDGE_UNAVAILABLE.md"
        blocked.write_text(
            "# VaM Capture Bridge Unavailable\n\n"
            f"- Bridge URL: `{bridge_url}`\n"
            f"- Error: `{status.get('error')}`\n"
            "- Start VaM with the BepInEx capture bridge, then rerun.\n",
            encoding="utf-8",
        )
        write_jsonl(target, [{"status": "bridge_unavailable", "bridge_url": bridge_url, "error": status.get("error")}])
        return {"status": "blocked", "bridge_available": False, "requests": len(rows), "out": str(target), "blocked_report": str(blocked)}
    if mode == "status_only":
        write_jsonl(target, [{"status": "bridge_available", "bridge_status": status, "request_count": len(rows)}])
        return {"status": "ok", "bridge_available": True, "requests": len(rows), "captures": 0, "out": str(target)}
    results = []
    for row in rows:
        if mode not in {"manual_step", "batch_current"}:
            results.append({"review_id": row.get("review_id"), "status": "skipped", "reason": f"unsupported mode {mode}"})
            continue
        results.append(_capture_one(bridge_url, row))
    write_jsonl(target, results)
    return {"status": "ok", "bridge_available": True, "requests": len(rows), "captures": sum(1 for r in results if r.get("status") == "captured"), "out": str(target)}


def _get_status(bridge_url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(bridge_url.rstrip("/") + "/status", timeout=2.0) as response:
            return {"available": True, "response": json.loads(response.read().decode("utf-8"))}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def _capture_one(bridge_url: str, row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "review_id": row.get("review_id"),
        "output_dir": row.get("output_dir"),
        "frame_count": row.get("frame_count"),
        "duration_seconds": row.get("duration_seconds"),
        "capture_interval_seconds": row.get("capture_interval_seconds"),
        "hide_ui": True,
        "view_mode": "current_camera",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(bridge_url.rstrip("/") + "/capture_frames", data=data, method="POST", headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=max(5.0, float(row.get("duration_seconds") or 4.0) + 10.0)) as response:
            result = json.loads(response.read().decode("utf-8"))
        result.setdefault("review_id", row.get("review_id"))
        return result
    except (urllib.error.URLError, TimeoutError, Exception) as exc:  # noqa: BLE001
        return {"review_id": row.get("review_id"), "status": "failed", "error": str(exc)}

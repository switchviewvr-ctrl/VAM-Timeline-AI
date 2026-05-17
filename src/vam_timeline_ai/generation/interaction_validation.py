"""Validation for partner-relative generated flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vam_timeline_ai.io.json_utils import load_json


DISALLOWED = {"person", "root", "personcontrol", "control", "world"}


def validate_partner_relative_flow_v0(flow: str | Path | dict[str, Any], out: str | Path) -> dict[str, Any]:
    data = load_json(flow) if not isinstance(flow, dict) else flow
    checks: list[dict[str, Any]] = []
    _check(checks, "no_source_world_coords", data.get("source_world_coords_used") is False, data.get("source_world_coords_used"))
    _check(checks, "no_person_root_tracks", data.get("person_root_tracks_included") is False, data.get("person_root_tracks_included"))
    _check(checks, "no_clip_stitching", data.get("clip_stitching_used") is False, data.get("clip_stitching_used"))
    tracks = data.get("controller_tracks") or []
    names = {str(t.get("controller_name")) for t in tracks}
    _check(checks, "has_pelvis_track", "pelvisControl" in names or "hipControl" in names, sorted(names))
    for track in tracks:
        name = str(track.get("controller_name") or "")
        _check(checks, f"{name}:allowed_controller", name.lower() not in DISALLOWED, name)
        deltas = track.get("position_deltas") or []
        _check(checks, f"{name}:has_deltas", bool(deltas), len(deltas))
        _check(checks, f"{name}:finite", _finite(deltas), "finite")
        if track.get("role") == "anchor":
            _check(checks, f"{name}:anchor_stable", _max_abs(deltas) <= 1e-6, _max_abs(deltas))
    support = data.get("support_mode")
    refs = data.get("partner_references") or {}
    if support == "hands_on_partner_chest":
        _check(checks, "partner_chest_reference_present", bool(refs.get("partner_chest_reference")), refs.keys())
        hand_tracks = [t for t in tracks if str(t.get("controller_name")) in {"lHandControl", "rHandControl"}]
        _check(checks, "hand_support_tracks_present", len(hand_tracks) == 2, [t.get("controller_name") for t in hand_tracks])
        _check(checks, "hand_tracks_target_partner_chest", all(t.get("target") == "partner.chest" for t in hand_tracks), [t.get("target") for t in hand_tracks])
    passed = all(c["passed"] for c in checks)
    result = {
        "passed": passed,
        "export_ready": False,
        "contact_constraints_valid": passed and support == "hands_on_partner_chest",
        "checks": checks,
    }
    _write_report(result, out)
    return result


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _finite(values: list[Any]) -> bool:
    for row in values:
        for value in row:
            try:
                x = float(value)
            except Exception:
                return False
            if x != x or x in {float("inf"), float("-inf")}:
                return False
    return True


def _max_abs(values: list[Any]) -> float:
    max_value = 0.0
    for row in values:
        for value in row:
            try:
                max_value = max(max_value, abs(float(value)))
            except Exception:
                pass
    return max_value


def _write_report(result: dict[str, Any], out: str | Path) -> None:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Partner Relative Flow Validation V0",
        "",
        f"- Passed: {result.get('passed')}",
        f"- Export ready: {result.get('export_ready')}",
        f"- Contact constraints valid: {result.get('contact_constraints_valid')}",
        "",
        "## Checks",
        "",
    ]
    for check in result.get("checks") or []:
        lines.append(f"- {'PASS' if check.get('passed') else 'FAIL'} `{check.get('name')}`: `{check.get('detail')}`")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

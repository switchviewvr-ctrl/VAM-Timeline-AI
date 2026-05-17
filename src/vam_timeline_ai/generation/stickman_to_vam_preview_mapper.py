"""Map contact-aware semantic stickman examples to VaM controller previews."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np

from vam_timeline_ai.generation.generated_motion import is_allowed_generated_controller
from vam_timeline_ai.generation.semantic_stickman import as_point3
from vam_timeline_ai.generation.vam_semantic_preview import (
    STICKMAN_TO_VAM_CONTROLLER,
    VaMSemanticPreviewClip,
    is_disallowed_timeline_track,
)
from vam_timeline_ai.io.json_utils import dump_json, load_json


def map_stickman_to_vam_preview_v0(
    motion_examples: str | Path,
    out_json: str | Path,
    *,
    duration_seconds: float = 4.0,
    fps: int = 60,
) -> dict[str, Any]:
    data = load_json(motion_examples)
    clips = build_vam_semantic_preview_clips(data, duration_seconds=duration_seconds, fps=fps)
    payload = {
        "schema_version": "vam_semantic_preview_clips_v0",
        "source_motion_examples": str(motion_examples),
        "generated_from": "semantic_motion_examples_v2_contact_aware",
        "coordinate_space": "synthetic_review_local",
        "review_only": True,
        "source_world_coords_used": False,
        "person_root_world_tracks_included": False,
        "production_ready": False,
        "timeline_generation_final": False,
        "clips": [clip.to_dict() for clip in clips],
    }
    dump_json(out_json, payload)
    return {
        "status": "ok",
        "out_json": str(out_json),
        "clip_count": len(clips),
        "exportable_count": sum(1 for clip in clips if clip.export_status == "exported"),
        "blocked_count": sum(1 for clip in clips if clip.export_status != "exported"),
    }


def build_vam_semantic_preview_clips(data: dict[str, Any], *, duration_seconds: float = 4.0, fps: int = 60) -> list[VaMSemanticPreviewClip]:
    clips: list[VaMSemanticPreviewClip] = []
    for example in data.get("examples") or []:
        clips.append(_clip_from_example(example, duration_seconds=duration_seconds, fps=fps))
    return clips


def _clip_from_example(example: dict[str, Any], *, duration_seconds: float, fps: int) -> VaMSemanticPreviewClip:
    clip_id = str(example.get("concept_id") or "semantic_preview")
    tracks: list[dict[str, Any]] = []
    warnings = list(example.get("warnings") or [])
    frame_times = _target_times(duration_seconds, fps)
    source_frames = list(example.get("frames") or [])

    bodyparts = _available_bodyparts(source_frames)
    roles = _controller_roles(example)
    for bodypart in bodyparts:
        controller = STICKMAN_TO_VAM_CONTROLLER.get(bodypart)
        if not controller:
            continue
        if not is_allowed_generated_controller(controller) or is_disallowed_timeline_track(controller):
            warnings.append(f"skipped_disallowed_controller:{controller}")
            continue
        positions = _resample_bodypart(source_frames, bodypart, frame_times)
        if positions.size == 0:
            continue
        tracks.append(
            {
                "controller_name": controller,
                "source_bodypart": bodypart,
                "role": roles.get(bodypart, "follower_or_context"),
                "times": [round(float(t), 6) for t in frame_times],
                "positions": [[round(float(v), 6) for v in row] for row in positions.tolist()],
                "coordinate_space": "synthetic_review_local",
                "review_only": True,
            }
        )

    alignment = dict(example.get("alignment_validation") or {})
    export_status = "exported"
    family = str(example.get("family") or "")
    if family in {"cowgirl", "reverse_cowgirl"} and not _cowgirl_alignment_exportable(example):
        export_status = "blocked_invalid_alignment"
        warnings.append("blocked: rider pelvis is not within partner pelvis target tolerance")

    first_partner = {}
    if source_frames:
        first_partner = dict((source_frames[0].get("partner_reference_points") or {}))
    partner_reference = {
        "strategy": "metadata_reference_only_v0",
        "partner_reference_points": first_partner,
        "target_points": dict(example.get("target_points") or {}),
        "partner_setup_required_in_vam": True,
        "note": "Timeline imports on one Person atom do not create a partner atom; use these coordinates as review guides.",
    }

    review_notes = [
        "Synthetic review-only VaM controller sketch.",
        "Not production-ready and not final text-to-animation.",
        "Use only to see whether ontology meaning survives a rough VaM controller mapping.",
    ]
    if family in {"cowgirl", "reverse_cowgirl"}:
        review_notes.append("Check that pelvis remains visually aligned with the partner pelvis reference.")
    if family == "bj_oral":
        review_notes.append("Check that head/chest read as driver while pelvis remains mostly static.")
    if family == "doggy":
        review_notes.append("Check all-fours/bent-forward support and partner-behind relation.")

    return VaMSemanticPreviewClip(
        clip_id=clip_id,
        family=family,
        pose_subtype=str(example.get("pose_subtype") or ""),
        motion_subtype=str(example.get("motion_subtype") or ""),
        duration_seconds=float(duration_seconds),
        fps=int(fps),
        controllers=[track["controller_name"] for track in tracks],
        controller_tracks=tracks,
        partner_reference=partner_reference,
        interaction_constraints=list(example.get("interaction_constraints") or []),
        alignment_validation=alignment,
        labels=dict(example.get("labels") or {}),
        contact_targets=dict(example.get("contact_targets") or {}),
        support_targets=dict(example.get("support_targets") or {}),
        target_points=dict(example.get("target_points") or {}),
        contact_zone=dict(example.get("contact_zone") or {}),
        warnings=sorted(set(warnings)),
        review_notes=review_notes,
        export_status=export_status,
    )


def _available_bodyparts(frames: list[dict[str, Any]]) -> list[str]:
    if not frames:
        return []
    found: set[str] = set()
    for frame in frames:
        found.update((frame.get("controller_points") or {}).keys())
    return [bodypart for bodypart in STICKMAN_TO_VAM_CONTROLLER if bodypart in found]


def _target_times(duration_seconds: float, fps: int) -> np.ndarray:
    count = max(2, int(round(float(duration_seconds) * int(fps))) + 1)
    return np.linspace(0.0, float(duration_seconds), count, dtype=np.float32)


def _resample_bodypart(frames: list[dict[str, Any]], bodypart: str, target_times: np.ndarray) -> np.ndarray:
    if not frames:
        return np.zeros((0, 3), dtype=np.float32)
    frame_times = [float(frame.get("time_seconds") or 0.0) for frame in frames]
    period = _source_period(frames, frame_times)
    positions = []
    for t in target_times:
        source_t = float(t) % period if period > 0 else 0.0
        positions.append(_interpolate_point(frames, frame_times, bodypart, source_t, period))
    return np.asarray(positions, dtype=np.float32)


def _source_period(frames: list[dict[str, Any]], frame_times: list[float]) -> float:
    if len(frame_times) > 1:
        step = max(1e-6, frame_times[1] - frame_times[0])
    else:
        step = 1.0 / 12.0
    explicit = float(max((frame.get("time_seconds") or 0.0) for frame in frames) + step)
    return max(explicit, step)


def _interpolate_point(frames: list[dict[str, Any]], frame_times: list[float], bodypart: str, source_t: float, period: float) -> tuple[float, float, float]:
    if len(frames) == 1:
        return as_point3((frames[0].get("controller_points") or {}).get(bodypart))
    for idx, start_time in enumerate(frame_times):
        end_idx = (idx + 1) % len(frames)
        end_time = frame_times[end_idx] if end_idx > idx else period
        if start_time <= source_t <= end_time:
            start = as_point3((frames[idx].get("controller_points") or {}).get(bodypart))
            end = as_point3((frames[end_idx].get("controller_points") or {}).get(bodypart))
            span = max(1e-6, end_time - start_time)
            alpha = min(1.0, max(0.0, (source_t - start_time) / span))
            return (
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha,
                start[2] + (end[2] - start[2]) * alpha,
            )
    return as_point3((frames[-1].get("controller_points") or {}).get(bodypart))


def _controller_roles(example: dict[str, Any]) -> dict[str, str]:
    labels = example.get("labels") or {}
    drivers = set()
    for driver in labels.get("primary_driver") or []:
        if driver in {"pelvis_hip", "pelvis_counter_driver"}:
            drivers.update({"pelvis", "abdomen"})
        elif driver in {"head_neck"}:
            drivers.add("head")
        elif driver in {"chest_abdomen"}:
            drivers.update({"chest", "abdomen"})
        elif driver in {"hands"}:
            drivers.update({"lHand", "rHand"})
    anchors = set()
    for anchor in labels.get("anchors") or []:
        if anchor == "hands":
            anchors.update({"lHand", "rHand"})
        elif anchor == "knees":
            anchors.update({"lKnee", "rKnee"})
        elif anchor == "feet":
            anchors.update({"lFoot", "rFoot"})
        elif anchor == "chest":
            anchors.add("chest")
        elif anchor == "head":
            anchors.add("head")
    roles = {bodypart: "driver" for bodypart in drivers}
    for bodypart in anchors:
        roles.setdefault(bodypart, "anchor_support")
    for bodypart in ("chest", "head", "abdomen"):
        roles.setdefault(bodypart, "follower")
    return roles


def _cowgirl_alignment_exportable(example: dict[str, Any]) -> bool:
    validation = example.get("alignment_validation") or {}
    if validation.get("valid") is False:
        return False
    max_distance = float(validation.get("max_distance") or 0.0)
    allowed = float(validation.get("target_distance_max") or 0.0)
    if allowed <= 0:
        return False
    return max_distance <= allowed + 1e-6 and not math.isnan(max_distance)

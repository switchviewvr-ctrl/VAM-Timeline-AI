"""Render semantic stickman motion examples as GIFs and contact sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import html
import math

from vam_timeline_ai.generation.semantic_stickman import SKELETON_EDGES, as_point3
from vam_timeline_ai.io.json_utils import dump_json, load_json, safe_id_for_path


COLORS = {
    "actor": (20, 35, 55),
    "partner": (145, 145, 145),
    "driver": (220, 55, 45),
    "follower": (45, 105, 220),
    "anchor": (35, 145, 70),
    "trail": (245, 140, 30),
    "contact": (150, 65, 190),
    "text": (12, 20, 32),
    "muted": (90, 100, 115),
    "bg": (248, 250, 252),
}


def render_semantic_stickman_previews_v1(
    motion_examples: str | Path,
    out_dir: str | Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 12,
    make_gif: bool = True,
    make_contact_sheet: bool = True,
) -> dict[str, Any]:
    return _render_semantic_stickman_previews(
        motion_examples,
        out_dir,
        version="v1",
        width=width,
        height=height,
        fps=fps,
        make_gif=make_gif,
        make_contact_sheet=make_contact_sheet,
        show_labels=False,
        show_partner=True,
        show_alignment=False,
        show_support_targets=True,
        show_contact_zone=False,
        show_alignment_tolerance=False,
        show_validity_overlay=False,
        contact_aware=False,
    )


def render_semantic_stickman_previews_v2(
    motion_examples: str | Path,
    out_dir: str | Path,
    width: int = 1600,
    height: int = 900,
    fps: int = 12,
    make_gif: bool = True,
    make_contact_sheet: bool = True,
    show_labels: bool = True,
    show_partner: bool = True,
    show_alignment: bool = True,
    show_support_targets: bool = True,
) -> dict[str, Any]:
    return _render_semantic_stickman_previews(
        motion_examples,
        out_dir,
        version="v2",
        width=width,
        height=height,
        fps=fps,
        make_gif=make_gif,
        make_contact_sheet=make_contact_sheet,
        show_labels=show_labels,
        show_partner=show_partner,
        show_alignment=show_alignment,
        show_support_targets=show_support_targets,
        show_contact_zone=False,
        show_alignment_tolerance=False,
        show_validity_overlay=False,
        contact_aware=False,
    )


def render_semantic_stickman_previews_v3(
    motion_examples: str | Path,
    out_dir: str | Path,
    width: int = 1600,
    height: int = 900,
    fps: int = 12,
    make_gif: bool = True,
    make_contact_sheet: bool = True,
    show_labels: bool = True,
    show_partner: bool = True,
    show_alignment: bool = True,
    show_support_targets: bool = True,
    show_contact_zone: bool = True,
    show_alignment_tolerance: bool = True,
    show_validity_overlay: bool = True,
    contact_aware: bool = True,
) -> dict[str, Any]:
    return _render_semantic_stickman_previews(
        motion_examples,
        out_dir,
        version="v3",
        width=width,
        height=height,
        fps=fps,
        make_gif=make_gif,
        make_contact_sheet=make_contact_sheet,
        show_labels=show_labels,
        show_partner=show_partner,
        show_alignment=show_alignment,
        show_support_targets=show_support_targets,
        show_contact_zone=show_contact_zone,
        show_alignment_tolerance=show_alignment_tolerance,
        show_validity_overlay=show_validity_overlay,
        contact_aware=contact_aware,
    )


def _render_semantic_stickman_previews(
    motion_examples: str | Path,
    out_dir: str | Path,
    version: str,
    width: int,
    height: int,
    fps: int,
    make_gif: bool,
    make_contact_sheet: bool,
    show_labels: bool,
    show_partner: bool,
    show_alignment: bool,
    show_support_targets: bool,
    show_contact_zone: bool,
    show_alignment_tolerance: bool,
    show_validity_overlay: bool,
    contact_aware: bool,
) -> dict[str, Any]:
    data = load_json(motion_examples)
    examples = data.get("examples", [])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        pil_available = True
    except Exception as exc:  # noqa: BLE001
        pil_available = False
        Image = ImageDraw = ImageFont = None  # type: ignore
        pil_error = str(exc)
    else:
        pil_error = ""

    for example in examples:
        concept_id = str(example.get("concept_id") or "concept")
        item_dir = out / safe_id_for_path(concept_id)
        frames_dir = item_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "concept_id": concept_id,
            "family": example.get("family"),
            "motion_subtype": example.get("motion_subtype"),
            "pose_subtype": example.get("pose_subtype"),
            "status": "pending",
            "warnings": [],
            "render_options": {
                "show_labels": show_labels,
                "show_partner": show_partner,
                "show_alignment": show_alignment,
                "show_support_targets": show_support_targets,
                "show_contact_zone": show_contact_zone,
                "show_alignment_tolerance": show_alignment_tolerance,
                "show_validity_overlay": show_validity_overlay,
                "contact_aware": contact_aware,
            },
        }
        if not pil_available:
            entry.update({"status": "blocked", "warnings": [f"Pillow unavailable: {pil_error}"]})
            dump_json(item_dir / "metadata.json", {**entry, "example": example})
            manifest.append(entry)
            continue

        rendered_frames = []
        frame_paths = []
        frame_rows = list(example.get("frames") or [])
        bounds = _bounds_for_example(example)
        for index, frame in enumerate(frame_rows):
            image = Image.new("RGB", (int(width), int(height)), COLORS["bg"])
            draw = ImageDraw.Draw(image)
            font = _font(ImageFont, 15)
            font_bold = _font(ImageFont, 20)
            _draw_frame(
                draw,
                frame,
                example,
                bounds,
                int(width),
                int(height),
                font,
                font_bold,
                index,
                show_labels=show_labels,
                show_partner=show_partner,
                show_alignment=show_alignment,
                show_support_targets=show_support_targets,
                show_contact_zone=show_contact_zone,
                show_alignment_tolerance=show_alignment_tolerance,
                show_validity_overlay=show_validity_overlay,
                contact_aware=contact_aware,
            )
            frame_path = frames_dir / f"frame_{index:03d}.png"
            image.save(frame_path)
            rendered_frames.append(image)
            frame_paths.append(str(frame_path))
        gif_path = ""
        if make_gif and rendered_frames:
            gif = item_dir / "preview.gif"
            rendered_frames[0].save(gif, save_all=True, append_images=rendered_frames[1:], duration=max(20, int(1000 / max(1, fps))), loop=0)
            gif_path = str(gif)
        sheet_path = ""
        if make_contact_sheet and rendered_frames:
            sheet = _make_contact_sheet(rendered_frames, concept_id, Image, ImageDraw, ImageFont)
            contact = item_dir / "contact_sheet.png"
            sheet.save(contact)
            sheet_path = str(contact)
        context_flags = _semantic_context_flags(example, show_labels, show_partner, show_alignment, show_support_targets)
        item_warnings = sorted(set((entry.get("warnings") or []) + context_flags.get("warnings", [])))
        metadata = {
            **entry,
            "status": "rendered",
            "warnings": item_warnings,
            "frame_count": len(frame_paths),
            "frames_dir": str(frames_dir),
            "gif_path": gif_path,
            "contact_sheet_path": sheet_path,
            "metadata_path": str(item_dir / "metadata.json"),
            "semantic_context": context_flags,
            "example": example,
        }
        dump_json(item_dir / "metadata.json", metadata)
        manifest.append({k: v for k, v in metadata.items() if k != "example"})

    dump_json(out / f"semantic_stickman_preview_manifest_{version}.json", {"items": manifest})
    _write_report(out / f"semantic_stickman_preview_report_{version}.md", manifest, version)
    return {
        "status": "ok",
        "concepts": len(examples),
        "rendered": sum(1 for m in manifest if m.get("status") == "rendered"),
        "gif_count": sum(1 for m in manifest if m.get("gif_path")),
        "contact_sheet_count": sum(1 for m in manifest if m.get("contact_sheet_path")),
        "out_dir": str(out),
        "manifest": str(out / f"semantic_stickman_preview_manifest_{version}.json"),
    }


def _draw_frame(
    draw: Any,
    frame: dict[str, Any],
    example: dict[str, Any],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    font: Any,
    font_bold: Any,
    index: int,
    show_labels: bool,
    show_partner: bool,
    show_alignment: bool,
    show_support_targets: bool,
    show_contact_zone: bool,
    show_alignment_tolerance: bool,
    show_validity_overlay: bool,
    contact_aware: bool,
) -> None:
    pts = {k: as_point3(v) for k, v in (frame.get("controller_points") or {}).items()}
    partner = {k: as_point3(v) for k, v in (frame.get("partner_reference_points") or {}).items()}
    labels = example.get("labels") or {}
    drivers = set(labels.get("primary_driver") or [])
    anchors = set(labels.get("anchors") or [])
    followers = set(example.get("follower_curves") or [])
    trails = {k: [as_point3(p) for p in v] for k, v in (example.get("motion_trails") or {}).items()}
    map_pt = lambda p: _project(p, bounds, width, height)

    _draw_grid(draw, width, height)
    _draw_bed_plane(draw, bounds, width, height, map_pt, font)
    if show_contact_zone:
        _draw_contact_zone(draw, partner, example, map_pt, font, show_alignment_tolerance)
    if show_partner:
        _draw_partner(draw, partner, map_pt, font, show_labels=show_labels)
    if show_alignment:
        _draw_alignment(draw, pts, partner, example, map_pt, font)
    _draw_skeleton(draw, pts, map_pt, drivers, followers, anchors)
    _draw_trails(draw, trails, map_pt, index)
    if show_support_targets:
        _draw_contact_targets(draw, pts, partner, example.get("contact_targets") or {}, map_pt, font, show_labels=show_labels)
        _draw_anchor_targets(draw, pts, anchors, map_pt, font, show_labels=show_labels)
    if show_labels:
        _draw_bodypart_labels(draw, pts, map_pt, font)
    _draw_overlay(draw, example, labels, width, font, font_bold)
    if show_validity_overlay:
        _draw_validity_overlay(draw, example, width, font, font_bold)


def _draw_grid(draw: Any, width: int, height: int) -> None:
    for y in range(120, height - 50, 100):
        draw.line((40, y, width - 40, y), fill=(230, 235, 242), width=1)
    draw.line((40, height - 75, width - 40, height - 75), fill=(160, 170, 185), width=2)


def _draw_bed_plane(draw: Any, bounds: tuple[float, float, float, float], width: int, height: int, map_pt: Any, font: Any) -> None:
    min_x, max_x, _, _ = bounds
    y = 0.10
    a = map_pt((0.0, y, min_x))
    b = map_pt((0.0, y, max_x))
    draw.line((a[0], a[1], b[0], b[1]), fill=(118, 128, 145), width=4)
    draw.text((56, min(height - 55, a[1] + 6)), "bed/floor plane", fill=COLORS["muted"], font=font)


def _draw_partner(draw: Any, partner: dict[str, tuple[float, float, float]], map_pt: Any, font: Any, show_labels: bool) -> None:
    edges = [("partner_pelvis", "partner_chest"), ("partner_chest", "partner_head"), ("partner_pelvis", "partner_lThigh"), ("partner_pelvis", "partner_rThigh"), ("partner_lThigh", "partner_lLeg"), ("partner_rThigh", "partner_rLeg")]
    for a, b in edges:
        if a in partner and b in partner:
            draw.line((*map_pt(partner[a]), *map_pt(partner[b])), fill=COLORS["partner"], width=5)
    for key, point in partner.items():
        x, y = map_pt(point)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS["partner"])
        if show_labels and key in {"partner_head", "partner_chest", "partner_pelvis", "partner_lThigh", "partner_rThigh", "partner_lLeg", "partner_rLeg"}:
            _callout(draw, (x, y), _partner_label(key), font, COLORS["partner"], dx=10, dy=_label_dy(key))
    if partner:
        xs = [map_pt(p)[0] for p in partner.values()]
        ys = [map_pt(p)[1] for p in partner.values()]
        draw.rounded_rectangle((min(xs) - 18, min(ys) - 18, max(xs) + 18, max(ys) + 18), radius=8, outline=(170, 174, 182), width=2)
        _callout(draw, (min(xs), min(ys) - 24), "partner reference", font, COLORS["partner"], dx=0, dy=-8)


def _draw_alignment(draw: Any, pts: dict[str, tuple[float, float, float]], partner: dict[str, tuple[float, float, float]], example: dict[str, Any], map_pt: Any, font: Any) -> None:
    if "partner_pelvis" in partner:
        target_xy = map_pt(partner["partner_pelvis"])
        _draw_target(draw, target_xy, COLORS["contact"])
        _callout(draw, target_xy, "partner_pelvis_target / interaction_target", font, COLORS["contact"], dx=12, dy=-30)
        _draw_local_frame(draw, partner["partner_pelvis"], map_pt, font)
    if "pelvis" in pts and "partner_pelvis" in partner:
        a = map_pt(pts["pelvis"])
        b = map_pt(partner["partner_pelvis"])
        _arrow(draw, a, b, fill=(30, 120, 190), width=3)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        _callout(draw, mid, "rider pelvis -> partner pelvis alignment", font, (30, 120, 190), dx=12, dy=-14)
    if example.get("family") == "doggy" and "partner_pelvis" in partner and "pelvis" in pts:
        _callout(draw, map_pt(partner["partner_pelvis"]), "partner_behind driver reference", font, COLORS["contact"], dx=12, dy=10)
    if example.get("family") == "bj_oral" and "head" in pts and "partner_pelvis" in partner:
        _arrow(draw, map_pt(pts["head"]), map_pt(partner["partner_pelvis"]), fill=COLORS["contact"], width=3)
        _callout(draw, map_pt(pts["head"]), "head/chest target path", font, COLORS["contact"], dx=14, dy=-34)
    if str((example.get("labels") or {}).get("facing_context")) in {"back_to_partner", "facing_away"}:
        base = pts.get("chest") or pts.get("pelvis")
        if base:
            x, y = map_pt(base)
            _arrow(draw, (x, y), (x - 110, y), fill=(90, 80, 200), width=3)
            _callout(draw, (x - 110, y), "facing/back_to_partner", font, (90, 80, 200), dx=-5, dy=-28)


def _draw_contact_zone(draw: Any, partner: dict[str, tuple[float, float, float]], example: dict[str, Any], map_pt: Any, font: Any, show_tolerance: bool) -> None:
    zone = example.get("contact_zone") or {}
    center = zone.get("center")
    if center is None and "partner_pelvis" in partner:
        center = partner["partner_pelvis"]
    if center is None:
        return
    c = as_point3(center)
    radius = float(zone.get("radius") or (example.get("alignment_validation") or {}).get("target_distance_max") or 0.25)
    cx, cy = map_pt(c)
    rx = abs(map_pt((c[0], c[1], c[2] + radius))[0] - cx)
    ry = abs(map_pt((c[0], c[1] + radius, c[2]))[1] - cy)
    fill = (252, 235, 225)
    outline = (220, 100, 50)
    if show_tolerance:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=outline, width=3)
        _callout(draw, (cx + rx, cy - ry), "alignment tolerance/contact zone", font, outline, dx=8, dy=-10)
    _draw_target(draw, (cx, cy), outline)


def _draw_validity_overlay(draw: Any, example: dict[str, Any], width: int, font: Any, font_bold: Any) -> None:
    validation = example.get("alignment_validation") or {}
    if not validation:
        return
    valid = bool(validation.get("valid", True))
    max_distance = validation.get("max_distance", "-")
    limit = validation.get("target_distance_max", "-")
    failed = validation.get("failed_constraints") or []
    fill = (226, 248, 234) if valid else (255, 232, 230)
    outline = (35, 145, 70) if valid else (195, 55, 45)
    text = "valid alignment" if valid else "INVALID alignment"
    x0, y0 = width - 420, 116
    draw.rounded_rectangle((x0, y0, width - 28, y0 + 78), radius=10, fill=fill, outline=outline, width=2)
    draw.text((x0 + 14, y0 + 10), text, fill=outline, font=font_bold)
    draw.text((x0 + 14, y0 + 38), f"target distance max: {max_distance} / allowed {limit}", fill=COLORS["text"], font=font)
    if failed:
        draw.text((x0 + 14, y0 + 58), "failed: " + "; ".join(map(str, failed[:2])), fill=(165, 40, 40), font=font)


def _draw_local_frame(draw: Any, origin: tuple[float, float, float], map_pt: Any, font: Any) -> None:
    o = map_pt(origin)
    y_axis = map_pt((origin[0], origin[1] + 0.34, origin[2]))
    z_axis = map_pt((origin[0], origin[1], origin[2] + 0.42))
    _arrow(draw, o, y_axis, fill=(40, 155, 85), width=2)
    _arrow(draw, o, z_axis, fill=(210, 95, 45), width=2)
    _callout(draw, y_axis, "local Y/up", font, (40, 155, 85), dx=8, dy=-10)
    _callout(draw, z_axis, "local Z/interaction axis", font, (210, 95, 45), dx=8, dy=-10)


def _draw_skeleton(draw: Any, pts: dict[str, tuple[float, float, float]], map_pt: Any, drivers: set[str], followers: set[str], anchors: set[str]) -> None:
    for a, b in SKELETON_EDGES:
        if a in pts and b in pts:
            draw.line((*map_pt(pts[a]), *map_pt(pts[b])), fill=COLORS["actor"], width=5)
    driver_nodes = _driver_nodes(drivers)
    anchor_nodes = _anchor_nodes(anchors)
    follower_nodes = _follower_nodes(followers)
    for key, point in pts.items():
        x, y = map_pt(point)
        color = COLORS["actor"]
        radius = 7
        if key in follower_nodes:
            color, radius = COLORS["follower"], 8
        if key in anchor_nodes:
            color, radius = COLORS["anchor"], 9
        if key in driver_nodes:
            color, radius = COLORS["driver"], 11
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)


def _draw_trails(draw: Any, trails: dict[str, list[tuple[float, float, float]]], map_pt: Any, index: int) -> None:
    for key, points in trails.items():
        if key not in {"pelvis", "head", "lHand", "rHand"}:
            continue
        trail = points[: index + 1]
        if len(trail) < 2:
            continue
        color = COLORS["trail"] if key == "pelvis" else (80, 120, 230) if key == "head" else (165, 85, 190)
        xy = [map_pt(p) for p in trail]
        draw.line([coord for pt in xy for coord in pt], fill=color, width=3)


def _draw_contact_targets(draw: Any, pts: dict[str, tuple[float, float, float]], partner: dict[str, tuple[float, float, float]], targets: dict[str, str], map_pt: Any, font: Any, show_labels: bool) -> None:
    for source, target in targets.items():
        if source not in pts:
            continue
        target_point = _target_point_for(source, target, pts, partner)
        if not target_point:
            continue
        a = map_pt(pts[source])
        b = map_pt(target_point)
        _arrow(draw, a, b, fill=COLORS["contact"], width=3)
        _draw_target(draw, b, COLORS["contact"])
        if show_labels:
            _callout(draw, b, _target_label(target), font, COLORS["contact"], dx=12, dy=-18)


def _draw_anchor_targets(draw: Any, pts: dict[str, tuple[float, float, float]], anchors: set[str], map_pt: Any, font: Any, show_labels: bool) -> None:
    anchor_keys = _anchor_nodes(anchors)
    for key in sorted(anchor_keys):
        if key not in pts:
            continue
        point = pts[key]
        target = (point[0], 0.10, point[2])
        a = map_pt(point)
        b = map_pt(target)
        if abs(a[1] - b[1]) > 12:
            draw.line((a[0], a[1], b[0], b[1]), fill=(60, 155, 90), width=2)
        _draw_target(draw, b, COLORS["anchor"])
        if show_labels and key in {"lHand", "rHand", "lKnee", "rKnee", "lFoot", "rFoot", "chest", "head"}:
            _callout(draw, b, f"{key} anchor/support", font, COLORS["anchor"], dx=10, dy=10)


def _draw_bodypart_labels(draw: Any, pts: dict[str, tuple[float, float, float]], map_pt: Any, font: Any) -> None:
    wanted = ["pelvis", "abdomen", "chest", "head", "lHand", "rHand", "lElbow", "rElbow", "lThigh", "rThigh", "lKnee", "rKnee", "lFoot", "rFoot"]
    for key in wanted:
        if key not in pts:
            continue
        x, y = map_pt(pts[key])
        _callout(draw, (x, y), key, font, COLORS["text"], dx=_label_dx(key), dy=_label_dy(key))


def _draw_overlay(draw: Any, example: dict[str, Any], labels: dict[str, Any], width: int, font: Any, font_bold: Any) -> None:
    title = f"{example.get('concept_id')} | {example.get('family')} | {example.get('motion_subtype')}"
    draw.rounded_rectangle((24, 18, width - 24, 108), radius=12, fill=(255, 255, 255), outline=(210, 218, 230))
    draw.text((42, 30), title, fill=COLORS["text"], font=font_bold)
    detail = f"driver: {', '.join(labels.get('primary_driver') or [])}   anchors: {', '.join(labels.get('anchors') or [])}   support: {labels.get('contact_support')}"
    draw.text((42, 64), detail, fill=COLORS["text"], font=font)
    exclusions = ", ".join(labels.get("exclusions") or [])
    if exclusions:
        draw.text((42, 86), f"not: {exclusions}", fill=(180, 50, 45), font=font)
    legend_x = width - 370
    for i, (label, color) in enumerate([("driver", COLORS["driver"]), ("follower", COLORS["follower"]), ("anchor", COLORS["anchor"]), ("trail", COLORS["trail"])]):
        y = 34 + i * 18
        draw.ellipse((legend_x, y, legend_x + 11, y + 11), fill=color)
        draw.text((legend_x + 18, y - 2), label, fill=COLORS["muted"], font=font)


def _bounds_for_example(example: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for frame in example.get("frames") or []:
        for point in (frame.get("controller_points") or {}).values():
            p = as_point3(point)
            xs.append(p[2])
            ys.append(p[1])
        for point in (frame.get("partner_reference_points") or {}).values():
            p = as_point3(point)
            xs.append(p[2])
            ys.append(p[1])
    if not xs or not ys:
        return (-1.5, 1.5, 0.0, 2.2)
    return (min(xs) - 0.45, max(xs) + 0.45, min(ys) - 0.25, max(ys) + 0.35)


def _project(point: tuple[float, float, float], bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    z = point[2]
    y = point[1]
    left, right = 70, width - 70
    top, bottom = 130, height - 70
    sx = left + (z - min_x) / max(0.01, max_x - min_x) * (right - left)
    sy = bottom - (y - min_y) / max(0.01, max_y - min_y) * (bottom - top)
    return (int(sx), int(sy))


def _make_contact_sheet(frames: list[Any], concept_id: str, Image: Any, ImageDraw: Any, ImageFont: Any) -> Any:
    selected = _evenly_select(frames, min(12, len(frames)))
    thumb_w, thumb_h = 800, 450
    cols = 2
    rows = math.ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h + 46), COLORS["bg"])
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 14), f"{concept_id} contact sheet", fill=COLORS["text"], font=_font(ImageFont, 20))
    for i, image in enumerate(selected):
        thumb = image.copy()
        thumb.thumbnail((thumb_w, thumb_h - 24))
        x = (i % cols) * thumb_w
        y = 46 + (i // cols) * thumb_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 8, y + 8), f"frame {i+1}", fill=COLORS["muted"], font=_font(ImageFont, 14))
    return sheet


def _evenly_select(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / max(1, count - 1))] for i in range(count)]


def _font(ImageFont: Any, size: int) -> Any:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _driver_nodes(drivers: set[str]) -> set[str]:
    nodes: set[str] = set()
    if "pelvis_hip" in drivers or "pelvis_counter_driver" in drivers:
        nodes.update(["pelvis", "abdomen"])
    if "head_neck" in drivers:
        nodes.add("head")
    if "chest_abdomen" in drivers:
        nodes.add("chest")
    if "hands" in drivers or "left_hand" in drivers or "right_hand" in drivers:
        nodes.update(["lHand", "rHand"])
    return nodes


def _follower_nodes(followers: set[str]) -> set[str]:
    nodes: set[str] = set()
    if "chest" in followers:
        nodes.add("chest")
    if "head" in followers:
        nodes.add("head")
    if "pelvis" in followers:
        nodes.add("pelvis")
    if "legs" in followers:
        nodes.update(["lKnee", "rKnee", "lFoot", "rFoot"])
    return nodes


def _anchor_nodes(anchors: set[str]) -> set[str]:
    nodes: set[str] = set()
    if "feet" in anchors:
        nodes.update(["lFoot", "rFoot"])
    if "knees" in anchors:
        nodes.update(["lKnee", "rKnee"])
    if "hands" in anchors:
        nodes.update(["lHand", "rHand"])
    if "chest" in anchors or "head" in anchors or "back" in anchors:
        nodes.update(["chest", "head"])
    return nodes


def _target_point_for(source: str, target: str, pts: dict[str, tuple[float, float, float]], partner: dict[str, tuple[float, float, float]]) -> tuple[float, float, float] | None:
    target_key = _target_to_partner_key(target)
    if target_key and target_key in partner:
        return partner[target_key]
    if "legs_or_thighs" in target or "thigh" in target or "leg" in target:
        candidates = [partner[k] for k in ("partner_lThigh", "partner_rThigh", "partner_lLeg", "partner_rLeg") if k in partner]
        if candidates:
            return _avg(candidates)
    if "floor" in target or "bed" in target:
        p = pts.get(source)
        if p:
            return (p[0], 0.10, p[2])
    if "elevated_support" in target:
        p = pts.get(source)
        if p:
            return (p[0], 0.62, p[2])
    if "back" in target and "partner_chest" in partner:
        return partner["partner_chest"]
    return None


def _target_to_partner_key(target: str) -> str | None:
    if "chest" in target:
        return "partner_chest"
    if "pelvis" in target or "hip" in target:
        return "partner_pelvis"
    if "leg" in target or "thigh" in target:
        return "partner_lThigh"
    return None


def _avg(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
        sum(p[2] for p in points) / len(points),
    )


def _target_label(target: str) -> str:
    if "legs_or_thighs" in target:
        return "partner_legs/thighs support"
    return target.replace("partner.", "partner_")


def _partner_label(key: str) -> str:
    return key.replace("partner_l", "partner_left_").replace("partner_r", "partner_right_")


def _label_dx(key: str) -> int:
    if key.startswith("l"):
        return -92
    if key.startswith("r"):
        return 14
    return 12


def _label_dy(key: str) -> int:
    if "head" in key:
        return -28
    if "chest" in key:
        return -18
    if "pelvis" in key:
        return 10
    if "Knee" in key or "Thigh" in key or "Leg" in key:
        return 16
    if "Foot" in key or "Hand" in key:
        return 12
    return 0


def _draw_target(draw: Any, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=color, width=3)
    draw.line((x - 16, y, x + 16, y), fill=color, width=2)
    draw.line((x, y - 16, x, y + 16), fill=color, width=2)


def _arrow(draw: Any, a: tuple[int, int], b: tuple[int, int], fill: tuple[int, int, int], width: int = 2) -> None:
    draw.line((a[0], a[1], b[0], b[1]), fill=fill, width=width)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 10 + width
    for offset in (math.pi * 0.82, -math.pi * 0.82):
        end = (int(b[0] + math.cos(angle + offset) * size), int(b[1] + math.sin(angle + offset) * size))
        draw.line((b[0], b[1], end[0], end[1]), fill=fill, width=width)


def _callout(draw: Any, xy: tuple[int, int], text: str, font: Any, color: tuple[int, int, int], dx: int = 10, dy: int = -10) -> None:
    x, y = xy
    tx, ty = x + dx, y + dy
    bbox = draw.textbbox((tx, ty), text, font=font)
    pad = 3
    draw.rounded_rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), radius=4, fill=(255, 255, 255), outline=(220, 226, 235))
    draw.text((tx, ty), text, fill=color, font=font)


def _semantic_context_flags(example: dict[str, Any], show_labels: bool, show_partner: bool, show_alignment: bool, show_support_targets: bool) -> dict[str, Any]:
    first = (example.get("frames") or [{}])[0]
    pts = {k: as_point3(v) for k, v in (first.get("controller_points") or {}).items()}
    partner = {k: as_point3(v) for k, v in (first.get("partner_reference_points") or {}).items()}
    labels = example.get("labels") or {}
    anchors = set(labels.get("anchors") or [])
    targets = dict(example.get("contact_targets") or {})
    family = str(example.get("family") or "")
    cid = str(example.get("concept_id") or "")
    warnings: list[str] = []
    support_vectors = []
    for source, target in targets.items():
        if source in pts and _target_point_for(source, target, pts, partner):
            support_vectors.append(f"{source}->{target}")
    target_vectors = []
    if "pelvis" in pts and "partner_pelvis" in partner:
        target_vectors.append("rider_pelvis_to_partner_pelvis")
    if "head" in pts and "partner_pelvis" in partner and family == "bj_oral":
        target_vectors.append("head_to_partner_pelvis")
    has_support_targets = bool(support_vectors) or bool({"feet", "knees", "hands", "chest", "head", "back"} & anchors)
    floating = False
    if family in {"cowgirl", "reverse_cowgirl"}:
        if "partner_pelvis" not in partner:
            warnings.append("partner pelvis reference missing")
        if not {"feet", "knees", "hands"} & anchors and "hover" not in str(example.get("pose_subtype")):
            floating = True
            warnings.append("appears floating: no feet/knees/hands support anchors")
        if "lean_forward" in cid and not support_vectors:
            warnings.append("lean-forward support target missing")
        if "lean_back" in cid and not support_vectors:
            warnings.append("lean-back behind support target missing")
    if family == "doggy" and "partner_pelvis" not in partner:
        warnings.append("doggy partner-behind reference missing")
    if family == "bj_oral" and "head_to_partner_pelvis" not in target_vectors:
        warnings.append("BJ/oral head-to-partner-pelvis target missing")
    alignment_validation = example.get("alignment_validation") or {}
    return {
        "bodypart_labels_drawn": show_labels,
        "partner_reference_drawn": show_partner and bool(partner),
        "has_partner_reference": bool(partner),
        "has_partner_pelvis_target": "partner_pelvis" in partner,
        "has_alignment_target": show_alignment and "pelvis" in pts and "partner_pelvis" in partner,
        "has_support_targets": show_support_targets and has_support_targets,
        "support_vectors": support_vectors,
        "target_vectors": target_vectors,
        "appears_floating_warning": floating,
        "interaction_alignment_valid": alignment_validation.get("valid"),
        "alignment_distance_max": alignment_validation.get("max_distance"),
        "target_distance_max": alignment_validation.get("target_distance_max"),
        "failed_constraints": alignment_validation.get("failed_constraints", []),
        "warnings": warnings,
    }


def _write_report(path: Path, manifest: list[dict[str, Any]], version: str) -> None:
    rendered = sum(1 for item in manifest if item.get("status") == "rendered")
    gifs = sum(1 for item in manifest if item.get("gif_path"))
    sheets = sum(1 for item in manifest if item.get("contact_sheet_path"))
    floating = sum(1 for item in manifest if (item.get("semantic_context") or {}).get("appears_floating_warning"))
    missing_alignment = sum(1 for item in manifest if item.get("family") in {"cowgirl", "reverse_cowgirl"} and not (item.get("semantic_context") or {}).get("has_alignment_target"))
    lines = [
        f"# Semantic Stickman Preview Render Report {version.upper()}",
        "",
        f"- Concepts: {len(manifest)}",
        f"- Rendered: {rendered}",
        f"- GIFs: {gifs}",
        f"- Contact sheets: {sheets}",
        f"- Floating warnings: {floating}",
        f"- Cowgirl/reverse alignment missing: {missing_alignment}",
        "- Timeline animation generated: false",
        "- ML training performed: false",
        "- Person/root/world transforms used: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

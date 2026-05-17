# Digital Twin Animated Previews

Digital Twin Preview v1 renders audit-only skeleton previews from controller
tracks for human review and future visual-judge experiments.

Static Matplotlib contact sheets from v0 are not enough for visual judging. v1
prefers:

1. `preview.mp4` when ffmpeg is available.
2. `preview.gif` when Pillow is available.
3. `contact_sheet_large.png` as a readable image fallback.
4. Old static plots only as low-quality debug fallback.

The previews are not VaM rendering and not generated adult videos. They are
simple controller skeletons with pelvis/hand trails, contact/reference markers,
and system labels overlaid.

Visual model output remains review-assist only. It is never ground truth unless
a human explicitly verifies and promotes it through a separate label workflow.

If partner tracks are missing, partner pelvis/chest/head/leg markers are proxy
or unavailable. In that case, partner contact cannot be confidently judged from
the preview alone.

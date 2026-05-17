# Relative Flow Retargeting V0

Relative Flow Retargeting V0 maps a generated relative motion flow onto a baseline body-controller pose.

It does not use source scene world coordinates, Person/root transforms, or clip stitching.

## Baseline Pose

V0 can create a synthetic neutral baseline with body controller positions for:

- pelvis/hip
- chest/head
- feet/knees
- optional hands

The baseline is a review prototype. It is not imported from a VaM scene and is not a production retargeting target.

## Retargeting Rule

For each generated controller track:

```text
retargeted_position = baseline_position + generated_relative_delta
```

Anchor controllers such as feet and knees remain stable when their generated deltas are zero.

## Safety Boundary

The retargeted flow records:

- `source_world_coords_used: false`
- `clip_stitching_used: false`
- `person_root_included: false`
- `coordinate_space: retargeted_to_baseline_pose`

Validation checks controller presence, anchor stability, distance plausibility, jumps, and missing anchors.

## Export Status

Timeline export is review-only. If validation fails, the exporter writes `export_unavailable.md` instead of faking output.

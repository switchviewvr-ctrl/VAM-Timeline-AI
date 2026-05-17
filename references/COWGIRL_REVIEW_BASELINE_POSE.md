# Cowgirl Review Baseline Pose

The Cowgirl review baseline pose is a synthetic test pose for generated relative motion. It is not copied from a source scene and does not include Person/root/world transforms.

## Purpose

V0 was tested from a standing pose, which made the generated pelvis loop look like hula-hoop motion. V1 adds a synthetic Cowgirl-oriented baseline so the same relative motion can be reviewed in a more appropriate kneeling-forward context.

## Baseline Properties

- `source`: `synthetic_neutral`
- `style`: `kneeling_forward`
- `intended_family`: `cowgirl`
- `anchor_profile`: `kneeling_cowgirl`
- `generation_use`: `review_baseline_only`
- `world_coords_allowed`: `false`
- `person_root_included`: `false`

## Controllers

The baseline includes pelvis/hip, abdomen/chest/head, knee/foot anchors, and optional hand support anchors. Feet and knees are deliberately stable so the review can focus on pelvis and torso coordination.

## Limitations

This is not final retargeting to an arbitrary scene or body. It is a safe review scaffold for testing generated relative controller deltas.

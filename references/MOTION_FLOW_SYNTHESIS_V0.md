# Motion Flow Synthesis V0

Motion Flow Synthesis V0 is the first step that turns abstract motion primitive statistics into actual generated relative controller curves.

It is not final text-to-animation, not VaM Timeline export, and not clip stitching.

## Inputs

- A semantic motion plan, such as `slow cowgirl grinding, leaning forward`
- Cowgirl motion primitive groups
- Cowgirl motion primitives

The synthesizer uses primitive statistics such as subtype, trajectory shape, amplitude ranges, rhythm profile, and controller roles. It does not copy source Timeline keyframes.

## Output

The output is a `GeneratedMotionFlow` with:

- `coordinate_space: relative_body_motion`
- generated relative `position_deltas`
- no Person/root/world tracks
- `clip_stitching_used: false`
- `export_ready: false`

For v0, the generated Cowgirl grind flow contains:

- a hip or pelvis driver curve
- an optional chest follower offset for forward lean
- stable zero-delta knee and foot anchor tracks

## Current Limits

- No final Timeline export
- No retargeting to a VaM character pose
- No contact solver
- No rotation synthesis
- No ML training

The next stage is retargeting and safety validation, not raw-coordinate reuse.

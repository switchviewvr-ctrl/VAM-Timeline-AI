# Native Timeline Export V1 Baseline Baking

V0 proved the generated Timeline JSON was structurally importable, but VaM Timeline applied the position curves to the current/standing context instead of the intended Cowgirl pose. Timeline does not know how to apply "relative deltas from captured baseline" the way the debug review player did.

V1 fixes that by baking:

```text
generated relative motion + generated Cowgirl baseline pose = concrete Timeline controller targets
```

## What Changed

- `t=0` is forced to the generated Cowgirl/kneeling baseline.
- Subsequent frames are generated relative motion rebased around that baseline.
- Position tracks are exported for generated and anchor controllers.
- Rotation tracks are exported from synthetic approximate baseline rotations where available.
- Metadata records `generated_from_relative_flow: true`, `includes_baseline_keyframe: true`, and `generated_baseline_pose: true`.

## What Did Not Change

- No source scene world coordinates are used.
- No Person/root/world tracks are exported.
- No clip stitching is used.
- No ML training is performed.

## Import Status

Static validation checks the Timeline schema, keyframes, baseline t=0 values, anchors, and metadata. VaM import still needs manual confirmation, so `expected_importable` remains `unknown` until tested.

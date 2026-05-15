# Baked Sample Format v0

`extract-motion-samples` writes one `.npz` file per successfully baked technical motion source.

## NPZ Fields

- `times`: `float32 [frames]`
  - Sample times in seconds at the requested FPS.
- `positions`: `float32 [frames, controllers, 3]`
  - Controller position channels as baked from Timeline/native source data.
- `rotations`: `float32 [frames, controllers, 4]`
  - Quaternion channels in `x, y, z, w` order.
- `velocities`: `float32 [frames, controllers, 3]`
  - Finite-difference position velocities.
- `angular_deltas`: `float32 [frames, controllers, 4]`
  - Per-frame quaternion deltas.
- `controller_names`: object/string array `[controllers]`
  - Controller names matching axis 1 of positions/rotations.
- `metadata_json`: JSON string
  - Source metadata and technical bake warnings.

## Important Semantics

These baked arrays are technical motion data. They are not semantic labels.

`technical_atom_id` identifies the source atom only. It must not be treated as rider/receiver/female/male truth.

Loop quality, validation success, and source names are technical properties. They are not Cowgirl/Riding behavior labels.

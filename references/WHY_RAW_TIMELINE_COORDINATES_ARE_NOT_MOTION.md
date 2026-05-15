# Why Raw Timeline Coordinates Are Not Motion

Raw Timeline coordinates often describe where a controller was placed in one specific VaM scene. The same movement can appear at different world positions, rotations, scales, and actor placements.

Learning raw coordinates would teach the system source-scene placement rather than motion style.

The system should learn:

- local bodypart deltas
- relative trajectories
- rhythm and repetition
- trajectory shape
- partner-relative context where available
- safe body-controller motion signatures

It should not learn:

- absolute hip/world positions
- Person atom/root movement
- full source-scene poses
- filename-derived labels for wild data

This is why the relative motion layer exists before any more semantic ML work.


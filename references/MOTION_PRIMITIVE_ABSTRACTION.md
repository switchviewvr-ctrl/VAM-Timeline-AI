# Motion Primitive Abstraction

The final system should learn motion principles, not stitch source Timeline clips.

A `MotionPrimitive` is an abstract relative motion pattern learned from one or more valid candidate windows. It summarizes:

- semantic family and subtype
- relative pelvis/hip trajectory shape
- rhythm and tempo profile
- normalized amplitude profile
- controller role map
- anchor and safety requirements

It does not store raw source-scene coordinates as generation targets. It is not a Timeline clip and must not be concatenated as final output.

Candidate DB records are learning material for abstraction. They remain audit inventories, not human ground truth.

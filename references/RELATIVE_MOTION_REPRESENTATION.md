# Relative Motion Representation

Raw VaM Timeline coordinates are source-scene coordinates. They are useful for inspection, but they are not reusable motion knowledge and they are not safe final generation targets.

The relative motion layer converts each movement window into body-controller deltas:

- Person/root/world-like tracks are stripped.
- Only mapped body controllers are kept.
- Position and rotation deltas are measured from the start of the window.
- Position deltas are normalized by an estimated body scale when possible.
- Windows are marked `safe_for_learning=false` if they are root-only, controller-only, static/micro-motion, high teleport risk, or have no allowed bodypart controllers.

This is an intermediate representation for audit and feature extraction. It is not final retargeting.

Manual labels remain separate. Machine, weak, silver, relative, and trajectory outputs are hints only.


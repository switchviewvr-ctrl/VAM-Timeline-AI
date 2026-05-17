# Semantic Actions: Pose Plus Motion

The clean_v3 semantic unit is a Semantic Action:

- actor pose
- actor motion
- partner pose/relation
- contact/support
- phase
- generation safety

This replaces movement-window-only reasoning. A Cowgirl candidate must have
Cowgirl-compatible motion and pose, must not be BJ/oral motion, and must keep
contact/support separate unless partner target evidence exists.

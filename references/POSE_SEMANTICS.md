# Pose Semantics

Pose semantics describe body context separately from motion. A kneeling pose can
support Cowgirl, BJ/oral, transition, or unknown motion depending on movement
and partner relation.

clean_v3 writes `pose_semantics/pose_features_v0.jsonl` and
`pose_semantics/pose_semantics_v0.jsonl`. These records are audit candidates,
not manual labels and not ML training truth.

Key rule: pose alone must not classify motion.

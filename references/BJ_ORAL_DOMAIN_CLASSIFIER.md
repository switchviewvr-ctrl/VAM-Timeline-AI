# BJ/Oral Domain Classifier

BJ/oral motion is a valid semantic animation family. It is not a trap to discard.

The classifier detects BJ/oral candidates using relative motion features, trajectory features, handmade reference matches, and Cowgirl core-controller evidence. When BJ/oral evidence is high and pelvis/hip Cowgirl motion is weak or absent, the candidate is excluded from Cowgirl generation-safe sets and preserved for future BJ/oral dataset work.

Preferred terminology:

- `bj_oral_motion_candidate`
- `bj_oral_generation_candidate`
- `bj_head_dominant_motion`
- `bj_deep_candidate`
- `bj_shallow_candidate`
- `bj_twist_candidate`
- `not_cowgirl_bj_oral` for Cowgirl-specific filtering

Reports should say:

```text
BJ/oral candidate detected; excluded from Cowgirl generation-safe set, preserved for BJ/oral dataset.
```

They should not describe valid BJ/oral candidates as bad data. The old `audit-bj-oral-trap-guard` command remains as a compatibility wrapper around `classify-bj-oral-domain`.

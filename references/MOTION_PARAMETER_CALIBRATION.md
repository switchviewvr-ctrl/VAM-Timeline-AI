# Motion Parameter Calibration

Parameter calibration estimates numeric ranges after meaning has been defined by the ontology.

It may estimate pelvis amplitude, vertical/forward/lateral ratios, tempo, follower damping, and anchor stability from human-reviewed ontology-consistent examples.

If there is not enough human-reviewed data, calibration must report insufficient samples instead of inventing ranges.

## Accepted Readability Defaults

Manual GT Timeline Examples v4 is the accepted review-only baseline reference:

`data/runs/clean_v3/generation/manual_gt_timeline_examples_v4`

The current default amplitude profiles are:

`data/config/manual_gt_motion_amplitude_profiles_v1.yaml`

These profiles are defaults for visual readability, not final production animation values. They should be treated as guardrails for future generation experiments:

- strengthen only the real semantic driver,
- keep anchors static,
- preserve rotations,
- keep sparse keyframes,
- do not globally scale all controllers,
- do not continue tuning v4 without a concrete VaM test failure.

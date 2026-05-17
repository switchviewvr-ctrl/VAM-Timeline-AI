# Visual Judge Model Calibration

The VLM must pass calibration before its family guesses are used for triage.

Calibration examples should include clear Doggy, elevated Doggy, kneeling
Cowgirl, reverse Cowgirl, lean-back supported Cowgirl, standing hand/head,
BJ/oral, low-motion pose context, broken pose, and single-frame/no-partner
examples.

Trust levels:

- `disabled`: default until live calibration passes.
- `coarse_pose_only`: model can describe pose but not family.
- `review_assist_low_trust`: useful for finding disagreement, not agreement.
- `review_assist_medium`: can help prioritize manual review.
- `review_assist_high`: reserved for strong multi-family calibration.

No visual model output is ground truth without human confirmation.

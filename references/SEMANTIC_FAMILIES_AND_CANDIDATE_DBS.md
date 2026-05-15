# Semantic Families and Candidate DBs

Candidate DBs are audit inventories, not manual ground truth and not ML training data.

The current semantic family field supports:

- `cowgirl`
- `bj_oral`
- `doggy`
- `hand_gesture`
- `head_gesture`
- `transition`
- `receiver_response`
- `unknown`

Cowgirl-specific filters may exclude a candidate from Cowgirl while still preserving it for another family. For example, BJ/oral motion in a kneeling pose is a valid BJ/oral candidate, not bad data. It should be marked:

- `semantic_family: bj_oral`
- `excluded_from_cowgirl: true`
- `preserve_for_future_dataset: true`

The Cowgirl DB keeps Cowgirl generation-safe candidates separate from soft-fail, pose-invalid, BJ/oral, standing/gesture, receiver-response, and unknown examples. The global semantic candidate DB v0 begins merging these families into one inventory for future family-specific review.

Core-controller gates are generation-safety checks. A missing expected controller can be a `soft_fail` when other evidence is strong: visible pose is valid, body motion is meaningful, trajectory/reference evidence matches Cowgirl, and available controllers preserve the reviewed pose. Hard failures remain for hand/head-only motion or missing core motion with poor pose/controller evidence.

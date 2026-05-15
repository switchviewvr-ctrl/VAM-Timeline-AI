# Cowgirl/Riding Feature Extraction v1

Feature extraction v1 expands the project from root-only technical measurements toward semantic motion understanding. The output is still not semantic truth. It is a set of numeric measurements and uncertain proxies that help build review queues, weak labels, clustering, and later supervised datasets.

## Real Numeric Measurements

The extractor reads baked motion arrays and computes measurements from available VaM controller tracks:

- pelvis/hip/root position ranges, speeds, movement energy, pause ratio, and rhythm proxies
- chest/head/pelvis relative distances and torso motion energy
- hand motion energy and hand distance to own chest, pelvis/root, and head controllers
- knee/foot motion energy, stance-width proxy, and foot-stability proxy
- head motion energy, vertical range, rotation-change proxy, and head-to-chest relative motion

These values come from controller positions and rotations. They are useful for search, clustering, and review triage.

## Uncertain Proxies

Some fields are intentionally named as proxies because the current data does not prove the final behavior:

- `torso_lean_forward_proxy` and `torso_lean_back_proxy` use relative controller displacement, not a validated anatomical forward axis.
- `head_down_proxy_uncertain` and `head_up_proxy_uncertain` are head-relative-to-chest height proxies, not true gaze.
- `kneeling_or_squat_proxy_uncertain` estimates relative root/knee height and is not a reliable pose classifier by itself.
- rhythm/style features such as `slow_motion_score_proxy`, `fast_motion_score_proxy`, and `adjustment_transition_score_proxy` are numeric hints, not labels.

Missing evidence stays missing. If a controller group is unavailable, the related features are `NaN` and the row records missing controller groups.

## Required Controller Groups

The conservative body-part mapping looks for these controller groups:

- pelvis/root group: `pelvis`, `hip`, `root`, or `abdomen`
- torso group: `chest`, optionally `abdomen` and `head`
- hands group: `left_hand`, `right_hand`
- legs group: knees and/or feet
- head group: `head`

Controller names are technical identifiers only. They do not imply actor role, gender, rider/receiver status, or semantic behavior.

## Weak Labels Are Not Ground Truth

`generate-weak-labels-v1` creates labels prefixed with `weak_`, such as:

- `weak_high_vertical_bounce`
- `weak_forward_back_dominant`
- `weak_pause_or_hold`
- `weak_high_hand_motion`
- `weak_torso_active`

These labels are numeric threshold hints for review and clustering. They are stored separately from manual labels in the v1 ML dataset.

## Manual Labels Are Required

Supervised semantic ML should not begin until real manual labels exist in `data/labels/manual_labels.yaml`. Template labels are examples only and must not be treated as training data.

Manual review should focus on:

- windows selected from different clusters
- feature extremes such as high pelvis energy, pauses, irregular rhythm, high hand motion, and high torso motion
- scenes with Cowgirl/Riding filename hints for positive candidates
- scenes without such hints for negative/control examples

## Random Window Splits Are Invalid

Movement windows overlap by design. Randomly splitting windows leaks near-identical motion from the same source into train and test sets. Future evaluation must group by scene, sample, or source.

## Current Limitations

- No actor role inference from names.
- No partner contact inference until context pairs and partner controllers are validated.
- No eye gaze inference from controller motion alone.
- No text prompt system.
- No final generative model.
- No bridge playback.

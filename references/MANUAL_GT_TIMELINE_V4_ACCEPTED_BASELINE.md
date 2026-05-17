# Manual GT Timeline V4 Accepted Baseline

`data/runs/clean_v3/generation/manual_gt_timeline_examples_v4` is the accepted review-only baseline reference.

This package should be used for future controller mapping, Timeline export rules, pose-first generation expectations, amplitude defaults, and guardrails. Do not keep iterating on v4 unless a specific VaM test problem is reported.

## Accepted Rules

- Use real manual VaM pose captures as baseline frames.
- Export both Position and Rotation quaternion tracks for every exported controller.
- Do not export Person/root/world tracks.
- Use sparse semantic keyframes by default. The accepted review default is 1 FPS.
- Require `hipControl` when present in the source capture.
- For Cowgirl and Reverse Cowgirl, semantic `pelvis_hip` maps to VaM `hipControl` as the primary visible driver.
- `pelvisControl` is a secondary/light follower or static support controller. It must not be the sole Cowgirl driver.
- `lThighControl` and `rThighControl` may be conservative synchronized amplifiers.
- Cowgirl feet remain static in position and rotation.
- Cowgirl hands remain static/support unless the semantic family is hand-driven.
- BJ uses `headControl` plus chest support/follow. `hipControl` and `pelvisControl` remain static.
- HJ uses the active hand as driver. `hipControl` and `pelvisControl` remain static.
- Doggy receiver-response uses subtle hip/pelvis/chest response with hands, feet, and knees static.
- Missionary uses subtle counter/leg response, not Cowgirl riding motion.

## Accepted Defaults

Amplitude defaults live in:

`data/config/manual_gt_motion_amplitude_profiles_v1.yaml`

These profiles are not production animation polish. They are review-only readability defaults derived from manual VaM testing.

## Guardrail

Future generation should treat v4 as the reference for safe controller roles:

`semantic intent -> manual-grounded baseline/pose expectations -> driver/follower/anchor plan -> sparse Timeline controller tracks`

Controller movement is still the output layer, not the source of semantic meaning.

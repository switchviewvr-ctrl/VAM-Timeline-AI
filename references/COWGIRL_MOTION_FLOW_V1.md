# Cowgirl Motion Flow V1

Cowgirl Motion Flow V1 improves the first generated review motion after the VaM player test showed that V0 was safe but too isolated. V0 moved the pelvis in a mostly horizontal loop; V1 adds a Cowgirl review baseline and coordinated body followers.

## What V1 Generates

- Pelvis or hip driver motion in `relative_body_motion`.
- Damped abdomen, chest, and optional head follower deltas.
- Stable knee and foot anchor tracks.
- Optional hand support anchors.
- Reduced lateral-only hula-hoop behavior.
- Meaningful forward/back and vertical components for Cowgirl grinding/riding review.

## What V1 Does Not Do

- It does not copy source Timeline keyframes.
- It does not use source-world coordinates.
- It does not move Person/root/world transforms.
- It does not export native Timeline JSON.
- It does not train ML.
- It is not final text-to-animation.

## Main Command

```powershell
python -m vam_timeline_ai.cli run-cowgirl-motion-flow-v1-review `
  --plan data\runs\clean_v2\generation\draft_motion_plan_v0.json `
  --primitive-groups data\runs\clean_v2\generation\cowgirl_motion_primitive_groups_v0.json `
  --primitives data\runs\clean_v2\generation\cowgirl_motion_primitives_v0.jsonl `
  --out-dir data\runs\clean_v2\generation\cowgirl_motion_flow_v1_review `
  --duration 4.0 `
  --fps 60 `
  --seed 42
```

## Review Expectations

The first VaM review should check whether the pelvis no longer looks like a detached hula-hoop loop, whether chest/abdomen motion reads as coordinated body motion, and whether knee/foot anchors stay stable in a Cowgirl-like kneeling baseline.

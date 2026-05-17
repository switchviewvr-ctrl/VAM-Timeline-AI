# Cowgirl Pose And Motion Variation Guardrails

This note captures the current architecture boundary after manual VaM review.

Cowgirl recognition and future generation must not collapse into a single "hip moved" test. The system needs separate concepts:

- pose geometry,
- interaction alignment,
- driver motion,
- anchor stability,
- controller completeness,
- generation safety.

## Pose Geometry

A valid Cowgirl pose is a rider-over-partner relation with the rider hip/pelvis region aligned to the partner pelvis target. The body can vary:

- upright, slight forward lean, lean-back supported, sitting/intimate, squat/kneel variants,
- feet slightly shifted or asymmetrical within support limits,
- knees and thighs abducted enough to preserve rider support,
- hands free or supporting on partner/body/floor/object,
- torso and head orientation can vary as long as the pose remains rider-over-partner.

Pose geometry alone is not clean animation.

## Motion Semantics

Clean Cowgirl motion requires repeated hip-region motion, mapped in VaM primarily to `hipControl`.

Expected clean drivers:

- primary: `hipControl`,
- secondary/follower: `pelvisControl`, abdomen/chest/head,
- optional small amplifiers: thighs/knees,
- static anchors: feet and support hands unless explicitly described otherwise.

Reject as clean Cowgirl:

- missing `hipControl` with no valid hip/pelvis driver,
- no real transform-distance motion,
- a single rise, sit-down, turn, mount, dismount, or stand-up transition,
- crawling/locomotion,
- hand/head controller cycles that dominate the hip driver,
- only-feet or only-upper-body controller sets,
- empty keyframes.

## Speed And Readability

Speed is semantic:

- very slow partial motion can be a Cowgirl pose/context or short-cycle candidate,
- clean cyclic motion needs enough repeated return strokes,
- high-amplitude motion must not break anchors or pose relation,
- low-amplitude motion must still show real transform-distance motion.

Future prompt-to-motion work should use manual GT pose references and amplitude profiles, not synthesize pose baselines from stickmen alone.

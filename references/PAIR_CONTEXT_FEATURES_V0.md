# Pair Context Features v0

Pair/context features exist because single-actor motion cannot prove partner contact or active/passive relationships. They compare two technical actor/sample windows in the same scene and overlapping time range.

## What Is Measured

- activity contrast between actor A and actor B
- pelvis-to-pelvis distance and vertical offset proxies
- chest/head distance where available
- actor A hand distances to actor B chest/head/pelvis
- actor B hand distances to actor A chest/head/pelvis
- motion correlations between the two actors
- motion-based active actor candidate

## What Is Not Claimed

The `active_actor_candidate` field is a motion contrast candidate only. It is not a semantic role, not gender inference, and not a rider label. Manual review must decide rider/receiver/context roles.

## Coordinate Caveats

Forward/back axes are marked uncertain because baked VaM/Timeline/native controller data may be local/controller-space. Pair features are review aids, not final anatomical measurements.

## Why This Matters

Partner-contact labels such as `cowgirl_hand_supported_on_partner_chest` should not be created from single-actor hand positions alone. Pair features give reviewers better evidence by comparing hand positions to another actor's mapped body controllers.

# Generated Relative Motion Flow

A generated relative motion flow is an intermediate animation representation. It stores controller deltas in body-relative space, not source-scene world coordinates.

## Required Safety Properties

- All tracks use `relative_body_motion`
- Controller values are deltas, not absolute world placements
- Person/root/world tracks are absent
- Source Timeline keyframes are not copied
- Clip stitching is not used
- Timeline export is disabled until retargeting and export safety exist

## Track Roles

- `driver`: the main motion path, usually hip or pelvis for Cowgirl
- `follower`: body parts that respond to or support the driver motion
- `anchor`: stable pose-supporting controllers such as feet and knees
- `support`: optional contact/support controllers

## Why This Exists

Primitive retrieval tells us what kind of motion should be generated. A generated relative flow is the first synthesized curve representation of that intent.

It is still not a VaM-ready animation. Later stages must retarget the relative deltas to the current actor, solve anchors/contact, validate controller plausibility, and only then consider Timeline export.

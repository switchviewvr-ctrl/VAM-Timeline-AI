# First Generated Motion Review

The first generated motion review pipeline is a prototype path from a semantic motion plan to inspectable generated motion artifacts.

It runs:

1. synthesize a relative motion flow from primitive statistics
2. validate the generated relative flow
3. create a synthetic baseline pose
4. retarget the flow to that baseline
5. validate the retargeted flow
6. render generated and retargeted previews
7. attempt a review-only Timeline-style JSON export if validation passes

## What It Is

- A technical review pipeline
- A way to inspect generated relative curves on a neutral baseline
- A safety checkpoint before real retargeting/export work

## What It Is Not

- Final text-to-animation
- Production VaM Timeline export
- Clip stitching
- Source world-coordinate reuse
- ML training

The review export is marked review-only even when validation passes.

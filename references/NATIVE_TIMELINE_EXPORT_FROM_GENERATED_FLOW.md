# Native Timeline Export From Generated Flow

The main generated output path is now native AcidBubbles Timeline JSON:

```text
prompt -> semantic motion plan -> generated relative motion flow -> retarget/anchor safety -> native Timeline JSON
```

The Generated Motion Review Player remains useful for debugging relative deltas, but it is not the product target.

## Current Export V0

`export-generated-flow-native-timeline-v0` takes a retargeted generated flow and writes a Timeline-like JSON file with:

- `SerializeVersion: 283`
- `AtomType: Person`
- one clip named `Generated_Cowgirl_Grinding_V0`
- controller position curves for allowed body controllers
- identity rotation curves marked not targeted
- generated safety metadata

## Safety Rules

The exporter rejects Person/root/world/atom tracks and only exports allowed body controllers. It does not copy source Timeline coordinates or source keyframes. It encodes generated retargeted controller positions.

## Validation

`validate-native-timeline-export-v0` checks JSON structure, Timeline fields, controller safety, keyframe decoding, sorted times, finite values, and generated metadata. Static validation cannot guarantee VaM import success, so manual import testing remains required.

## V1 Baseline Baking

VaM testing showed that V0 imports structurally, but Timeline does not apply controller curves relative to a captured pose. V1 therefore bakes generated relative deltas onto a generated Cowgirl/kneeling baseline before export.

V1 adds:

- `t=0` baseline keyframes for all exported controllers.
- concrete position targets equal to `baseline_position + rebased_generated_delta`.
- synthetic approximate rotation tracks when requested.
- validation that `t=0` matches the baseline pose.

This keeps the data generated and scene-independent while giving Timeline concrete controller targets it can import and play.

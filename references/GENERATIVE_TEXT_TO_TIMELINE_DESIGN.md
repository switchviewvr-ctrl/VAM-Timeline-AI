# Generative Text-To-Timeline Design

## Final Goal

The final goal is:

```text
text prompt -> semantic motion plan -> VaM Timeline animation
```

The system should eventually understand prompts such as:

```text
slow deep cowgirl, leaning forward, hands on partner chest, looking down
```

and translate them into a motion plan made from semantic movement windows that can later become VaM Timeline controller animation.

## Why Loop Extraction Was The Wrong Main Target

Loop extraction is useful technical QA, but it is not behavior understanding.

A real rider does not move as a perfect unbroken loop. She may bounce, grind, rock, shift weight, pause, adjust posture, change tempo, change hand support, look down or away, and transition between behaviors.

Perfect loops are only one useful asset type. A generative system needs windows that describe movement behavior, including imperfect and transitional human motion.

## Why Movement Windows Are The Correct Unit

The primary unit should be the movement window:

- 2 seconds, stride 1 second
- 4 seconds, stride 2 seconds
- 8 seconds, stride 4 seconds

A single source animation can contain many different semantic states. Splitting it into overlapping windows lets the system preserve slow sections, fast sections, pauses, posture changes, contact changes, and transition moments.

Movement windows also match future prompt retrieval better than whole clips. A prompt may request a 4 second lean-forward hand-supported section even if the original source clip is 30 seconds long.

## Semantic Information Needed Before Generative AI

Before ML or generation, the database needs:

- actor role guesses that separate technical atom IDs from semantic roles
- partner context
- pelvis/hip movement features
- torso posture and counter-motion features
- hand/contact/support features
- leg/stance/stability features
- head/gaze/attention features
- rhythm/style/intensity features
- multi-label taxonomy per window
- manual overrides at scene, actor, sample, and window scope
- confidence and manual-review flags

Without those semantics, motion matching or ML would mostly learn technical controller similarity, not user-intended behavior.

## Future Prompt To Query Mapping

Prompt:

```text
slow deep cowgirl, leaning forward, hands on partner chest, looking down
```

Possible semantic query:

- domain: `cowgirl_riding`
- semantic role: `rider`
- labels include:
  - `cowgirl_deep_slow`
  - `cowgirl_lean_forward`
  - `cowgirl_hand_supported_on_partner_chest`
- `head_down_score` high
- duration preference: 4 or 8 second windows
- exclude windows with unresolved manual review

The result is not yet a final animation. It is a semantic motion plan candidate set that can later be arranged, blended, retimed, and exported to Timeline.

## Why Bridge, Playback, Motion Matching, And ML Should Wait

Bridge playback should wait because earlier audit work found coordinate-space risk: baked segment data is not automatically safe to stream as live world/control-space targets.

Motion matching should wait because nearest controller frames do not equal semantic intent.

ML should wait because the model needs meaningful labels and windows first. Training on technical splits such as `loop_good` or `training_good` would encode the wrong objective.

The safer order is:

1. build semantic window database v0
2. manually label and review Cowgirl/Riding windows
3. add semantic retrieval
4. then decide whether to export Timeline, prototype bridge playback, or train ML

## Current Scope

This design stage adds schema, taxonomy, feature definitions, movement-window utilities, and manual-label templates. It does not build a text parser, train a model, export Timeline clips, run VaM, or build bridge playback.

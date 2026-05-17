# Real Generation Input Requirements

The real final pipeline is:

```text
prompt -> semantic motion plan -> generated relative motion flow -> scene-aware retargeting -> native VaM Timeline animation
```

Synthetic review timelines are useful for debugging, but true scene-aware generation requires current-scene context from VaM.

## Required Current-Scene Inputs

- Target rider Person atom.
- Partner/receiver Person atom for interaction prompts.
- Current controller positions and rotations for the rider.
- Current controller positions and rotations for the partner.
- Partner pelvis reference.
- Partner chest reference.
- Partner head reference.
- Scene up/forward frame or partner-local frame.
- Body scale estimates for rider and partner.
- Controller availability and missing-controller policy.
- Controller physics/control state assumptions.
- Bed/floor/support reference if hands, knees, feet, or body support targets are relevant.

## Prompt Example

Prompt: `cowgirl grinding, hands on partner chest`

Required generation context:

- Rider atom.
- Partner atom.
- Partner chest target for both hands.
- Partner pelvis target for rider pelvis alignment.
- Rider baseline pose compatible with Cowgirl.
- Contact/support constraints that keep hands near partner chest.
- Knees/feet anchor constraints.
- No Person/root motion.
- No source-scene world-coordinate copying.

## Hard Boundary

Without current-scene partner references, the system can only create synthetic review timelines, not true scene-aware generation.
Candidate DBs and generated review flows are not enough to safely target a real VaM scene.

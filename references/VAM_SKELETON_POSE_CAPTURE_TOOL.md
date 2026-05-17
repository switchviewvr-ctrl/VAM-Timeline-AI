# VaM Skeleton Pose Capture Tool

`SkeletonPoseCaptureTool.cs` is a read-only VaM plugin for capturing real, manually posed VaM controller layouts.

It exists because semantic stickmen and synthetic controller guesses are not enough. Correct pose examples should come from VaM scenes the user sets up by hand.

## What It Does

- Lets the user select a Rider/Actor atom and a Partner/Receiver atom.
- Draws an ESP-style skeleton overlay for both selected Person atoms.
- Labels important controllers such as pelvis, chest, head, hands, knees, and feet.
- Draws neutral helper lines for partner pelvis alignment, head-to-partner-pelvis relation, and hand/contact distances.
- Saves the current controller transforms to JSON.
- Saves raw world transforms, atom-local transforms, and partner-relative measurements.

## What It Does Not Do

- It does not move controllers.
- It does not animate controllers.
- It does not save or modify the scene.
- It does not export Timeline clips.
- It does not train ML.
- It does not modify `manual_labels.yaml`.

## Output

Captures are saved to:

`<VaM>/Saves/PluginData/VAMTimelineAI/pose_captures`

The plugin also attempts an optional mirror to:

`<project-root>/data/runs/manual_pose_captures`

The VaM PluginData capture is the source of truth.

## Captured Data

Each JSON snapshot contains:

- Human pose labels and notes.
- Rider and partner atom IDs.
- Controller world position and rotation.
- Controller local position and rotation relative to the atom.
- Missing controller list.
- Rider pelvis to partner pelvis distance.
- Head/hands to partner pelvis/chest/thigh reference distances.
- Partner-pelvis-local deltas.
- Basic facing and pose hints.

These measurements are hints for ontology correction, not automatic labels.

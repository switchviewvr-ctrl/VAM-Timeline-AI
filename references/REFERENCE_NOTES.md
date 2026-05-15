# Reference Notes

This project is new, but it is not blind from scratch.

## Mocap Compiler Reference

Use `G:\VAM\Research\MocapResearch\vam_mocap_dataset_compiler` as the technical reference for scene parsing, Timeline v283 decoding, baking, quaternion handling, validation, preview generation, and offline Timeline export.

Do not treat its old curation names as semantic truth. `loop_good`, `training_good`, and related labels are technical dataset QA categories. The new project needs behavior-aware labels and movement windows.

## Timeline Repo Reference

Use `G:\VAM\Research\MocapResearch\vam-timeline-master` as the source of truth for AcidBubbles Timeline serialization, compressed keyframe decoding, curve evaluation behavior, and controller target semantics.

Timeline code should be reimplemented carefully when needed, with the repo as reference. This initial setup does not export or import Timeline clips.

## Virtual Companion Bridge Reference

Use `G:\Virtual Companion` only as a later playback architecture reference.

The bridge audit found that the existing bridge is reusable, but current C# apply logic expects world/control-space servo or additive targets and does not directly apply segment-local Timeline/native controller data. Direct streaming could create coordinate-space mismatch.

Bridge playback is therefore not the first priority. The safer order is semantic analysis first, offline Timeline-compatible tooling second, and bridge playback only after coordinate-space behavior is explicitly validated.

## Raw Mocap Scene Folder

Use `G:\VAM\Research\MocapResearch` as the raw scene reference folder. The first scan pass should be lightweight: list JSON files, identify likely VaM scenes, count Person atoms, and detect technical motion carriers. It should not bake, train, export, or deeply label semantics.

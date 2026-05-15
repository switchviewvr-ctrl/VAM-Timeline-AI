# Native Timeline Export Research

This note records the current understanding of AcidBubbles Timeline JSON import/export format for generated motion flows.

## References Inspected

- Old working exporter: `G:\VAM\Research\MocapResearch\vam_mocap_dataset_compiler\src\vam_mocap_dataset\timeline_export.py`
- Old v283 codec: `G:\VAM\Research\MocapResearch\vam_mocap_dataset_compiler\src\vam_mocap_dataset\timeline_codec.py`
- Current ported codec: `src/vam_timeline_ai/timeline/codec.py`
- Current parser: `src/vam_timeline_ai/timeline/parser.py`
- Handmade importable references: `data\runs\clean_v2\references\handmade_animations\raw\female_cowgirl_*.json`

## Expected Top-Level Structure

Timeline importable external animation JSON uses:

```json
{
  "SerializeVersion": "283",
  "AtomType": "Person",
  "Clips": []
}
```

The old exporter also writes `"SerializeMode": "2"`. Handmade references do not always include it, but it is present in prior generated exports from the old compiler.

## Animation / Clip Structure

Each clip in `Clips` contains fields such as:

- `AnimationName`
- `AnimationLength`
- `BlendDuration`
- `Loop`
- `PreserveLastFrame`
- `LoopSelfBlendDuration`
- `NextAnimationRandomizeWeight`
- `AutoTransitionPrevious`
- `AutoTransitionNext`
- `SyncTransitionTime`
- `SyncTransitionTimeNL`
- `EnsureQuaternionContinuity`
- `AnimationLayer`
- `Speed`
- `Weight`
- `Uninterruptible`
- `AnimationSegment`
- `NextAnimationName`
- `NextAnimationTime`
- `Controllers`

Handmade files can include a `Pose` block. Generated native export v0 intentionally does not include source-scene pose data.

## Controller Tracks

Controllers are represented as objects inside `clip.Controllers`:

```json
{
  "Controller": "hipControl",
  "TargetsPosition": 1,
  "TargetsRotation": 1,
  "ControlPosition": 1,
  "ControlRotation": 1,
  "X": [],
  "Y": [],
  "Z": [],
  "RotX": [],
  "RotY": [],
  "RotZ": [],
  "RotW": []
}
```

For generated v0 export:

- Position axes are exported from generated retargeted controller positions.
- Rotation axes are exported as identity quaternion curves and marked not targeted because generated rotations are not implemented safely yet.
- Only allowed body controllers are exported.
- Person/root/world/atom transforms are rejected.

## Keyframe Encoding

Timeline v283 uses compact string keyframes. The first character is a flag that records whether value and curve type are stored, followed by little-endian float hex for time and optionally value, then optionally curve type byte. The current ported codec supports this through:

- `TimelineKeyframe`
- `encode_keyframe_sequence`
- `decode_keyframe_sequence`

Generated export uses linear curve type `2`.

## Generated Flow Mapping

Retargeted flow fields map as:

- `controller_tracks[].controller_name` -> `Controller`
- `controller_tracks[].retargeted_positions[:,0]` -> `X`
- `controller_tracks[].retargeted_positions[:,1]` -> `Y`
- `controller_tracks[].retargeted_positions[:,2]` -> `Z`
- `controller_tracks[].times` -> keyframe times
- flow duration -> `AnimationLength`

The exporter does not use source Timeline keyframes. It encodes newly generated retargeted positions.

## Fields To Avoid

Generated native export must not include:

- Person atom transform tracks
- root/world/scene object tracks
- source-scene world coordinate targets
- custom review-player schemas as the root object
- clip stitching metadata that implies copied source clips

## Reused Code

The current exporter reuses the ported Timeline codec and follows the old `timeline_export.py` payload shape. It does not import old project modules directly.

## Importability Status

Static validation can confirm Timeline-like structure, decodable v283 keyframes, allowed controllers, sorted times, and generated safety metadata. It cannot guarantee VaM Timeline will accept the file. The first export must be manually tested in VaM and should be reported as:

```text
expected_importable: unknown
```

until confirmed by VaM.

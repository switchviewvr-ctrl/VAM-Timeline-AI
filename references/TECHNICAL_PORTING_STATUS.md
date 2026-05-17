# Technical Porting Status

This project inspected technical modules from:

`<VAM_TIMELINE_AI_MOCAP_COMPILER>/src/vam_mocap_dataset`

## Files Inspected

- `timeline_codec.py`
- `timeline_parser.py`
- `bezier.py`
- `native_motion.py`
- `quaternion_utils.py`
- `resample.py`
- `json_utils.py`

## Ported Technical Pieces

- Timeline compressed keyframe decode/encode basics into `src/vam_timeline_ai/timeline/codec.py`
- Timeline Bezier evaluation basics into `src/vam_timeline_ai/timeline/bezier.py`
- Timeline controller clip parsing/baking into `src/vam_timeline_ai/timeline/parser.py`
- Native `*Animation.steps[]` parsing/baking into `src/vam_timeline_ai/motion/native_motion.py`
- Quaternion normalization, continuity, slerp, and angular deltas into `src/vam_timeline_ai/motion/quaternion_utils.py`
- 60 Hz time grids, interpolation, velocity, and NPZ sample extraction into `src/vam_timeline_ai/motion/baker.py`
- Basic baked-array validation into `src/vam_timeline_ai/motion/validation.py`

## Not Ported

- Old curation logic such as `training_good`, `loop_good`, and stage labels
- Old motion matching as primary logic
- Old segment exporter or Timeline exporter as active pipeline steps
- Old bridge/playback logic
- Any semantic role inference based on atom names

## Current Limitations

- Timeline Bezier behavior is practical but not a full byte-for-byte reimplementation of every Timeline edge case.
- Cowgirl features v0 compute root/pelvis-like numeric features only.
- Hands, torso, legs, and head/gaze remain schema-level or missing-feature warnings until controller mapping and role detection are implemented.
- No supervised model is trained.
- Clustering uses a NumPy fallback because scikit-learn is not installed in the current environment.

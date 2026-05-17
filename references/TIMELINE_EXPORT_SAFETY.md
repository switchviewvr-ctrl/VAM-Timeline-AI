# Timeline Export Safety

Timeline segment export is a review convenience only. The safest verification path is still opening the original VaM scene at the reported time range.

Exports must never include Person/root/world transform tracks as final animation motion. Those tracks can teleport a character to source-scene coordinates or move the whole person instead of body controllers.

The guarded semantic review exporter strips or rejects:

- `Person` atom transform tracks
- root/world tracks
- whole-person `control` tracks
- unknown tracks that are not confidently mapped body controllers

Export metadata records:

- `exported_controller_count`
- `stripped_world_transform_count`
- `stripped_atom_root_count`
- `coordinate_space_assumption`
- `teleport_risk`
- `export_safe_for_import`
- `timeline_export_safe_for_animation`

If no allowed bodypart controllers remain, no fake segment is created. The item must be inspected in the original scene instead.

Full VaM poses should not be blindly loaded as animation data. Pose-like data may contain world-space source placement and must be stripped down to safe relative body-controller deltas before it can be considered for animation output.

## Accepted Manual GT V4 Rules

`data/runs/clean_v3/generation/manual_gt_timeline_examples_v4` is the accepted review-only baseline reference.

Future Timeline exports should preserve these rules:

- Include Position and Rotation quaternion tracks for every exported controller.
- Require `hipControl` when the source capture has it.
- For Cowgirl/Reverse Cowgirl, use `hipControl` as the primary visible driver.
- Do not use `pelvisControl` as the sole Cowgirl driver.
- Keep static anchors constant in position and rotation.
- Use sparse semantic keyframes by default, with 1 FPS as the accepted manual-GT review default.
- Use `data/config/manual_gt_motion_amplitude_profiles_v1.yaml` as the default readability profile set.
- Do not iterate on v4 unless a specific VaM test problem is reported.

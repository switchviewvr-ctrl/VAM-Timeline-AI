# Coordinate Space And Teleport Safety

VaM poses and Timeline tracks can contain world-space placement. Importing or learning those coordinates directly can teleport a Person atom back to the source scene location.

Final animation output must not use:

- Person atom transforms
- root/world transforms
- scene-level object transforms
- whole-person controller motion
- raw source-world controller coordinates as generation templates

Allowed semantic targets are real body controllers such as `hipControl`, `chestControl`, `headControl`, hands, elbows, knees, feet, and thighs when confidently mapped.

Review exports may still contain source controller coordinates for visual checking inside the original scene. These exports are explicitly marked:

- `exported_as_relative_motion: false`
- `safe_for_generation_template: false`
- `source_world_coords_stripped`
- `teleport_risk`

If no safe body-controller tracks remain after stripping Person/root/world tracks, no Timeline segment export is produced.


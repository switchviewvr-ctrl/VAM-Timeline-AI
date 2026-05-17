# Review Player V1

`GeneratedMotionReviewPlayer.cs` is a VaM review-only plugin for generated motion flow JSON. It is not a Timeline importer and does not create production animation.

## V1 Additions

- Shows loaded schema.
- Shows loaded, skipped, and missing controller counts.
- Shows active playback time.
- Adds separate scale sliders:
  - global scale
  - pelvis scale
  - vertical scale
  - lateral scale
  - forward/back scale
  - chest follower scale
- Adds toggles for pelvis, abdomen/chest followers, anchors, and loop playback.

## Safety Rules

The player only applies deltas to allowed body controllers. It skips disallowed controllers and never targets Person/root/world transforms. Motion is applied relative to the controller positions captured by `Capture Baseline`.

## VaM Test Steps

1. Add `GeneratedMotionReviewPlayer.cs` to the Person atom.
2. Load `Saves/PluginData/VAMTimelineAI/generated_motion_review_player_v1.json`.
3. Click `Load JSON`.
4. Put the Person in the intended review pose if needed.
5. Click `Capture Baseline`.
6. Click `Play`.
7. Adjust lateral, vertical, forward/back, pelvis, and chest follower scales during review.
8. Click `Reset To Baseline` to restore the captured pose.

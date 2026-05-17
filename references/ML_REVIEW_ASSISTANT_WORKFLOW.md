# ML Review Assistant Workflow

1. Build human-reviewed labels from audit artifacts.
2. Join labels with relative motion, trajectory, pose, interaction, and semantic action features.
3. Split by scene or another leakage-safe group.
4. Train a small CPU baseline if enough human labels exist.
5. Score candidates as review priorities.
6. Export a novelty-filtered review batch for the user.

The model only helps choose what to review next. It does not modify
`manual_labels.yaml`, does not train on weak/silver labels as truth, and does
not generate VaM Timeline animations.

# Review Batch Workflow

The review batch workflow turns numeric features and weak hints into a human labeling task.

## Steps

1. Calibrate weak labels.
2. Build pair windows and pair features.
3. Build a balanced review batch with limits per scene and sample.
4. Render static previews.
5. A human edits `manual_labels.stub.yaml`.
6. Validate the edited labels.
7. Merge them into `data/labels/manual_labels.yaml`.
8. Summarize label coverage.
9. Plan grouped train/validation/test splits.

## What The Batch Contains

- window ID and time range
- technical atom ID
- weak_v2 hints
- top numeric features
- optional pair context summary
- empty manual-label YAML entries

Weak labels are displayed as hints only. The stub file starts with empty `labels`, `negative_labels`, and `uncertain_labels`.

## Labeling Priorities

Start with a balanced set:

- obvious high-motion Cowgirl/Riding candidates
- pauses and adjustment transitions
- high hand-motion and hand-support candidates
- pair-context candidates where partner body proximity can be checked
- negative/control examples from non-riding-hint scenes
- static/passive partner context windows

## Safety

Do not infer roles from atom names. Do not use filenames as labels. Do not discard non-loop windows. Do not random-split overlapping windows.

# Supervised Baseline v0

The supervised baseline is optional and guarded by readiness checks. It is not the final generative AI.

## Allowed Inputs

The baseline may use:

- numeric Cowgirl/Riding feature vectors;
- real manual positive labels;
- real manual negative/control labels;
- include-for-ML flags;
- grouped split plans.

It must not use:

- weak labels as targets;
- filenames as targets;
- atom IDs as targets;
- random window splits;
- uncertain labels as positives.

## Blocking Conditions

Training is blocked when:

- no real manual labels exist;
- classes are too sparse;
- negative/control examples are missing;
- labels come from too few scenes or samples;
- grouped split planning fails;
- scikit-learn is unavailable.

When blocked, the command writes a report and no model is trained.

## Model Scope

If readiness passes and scikit-learn exists, the MVP baseline may fit simple per-label binary classifiers such as logistic regression. This tests whether the current features carry signal; it is not a text-to-animation model.

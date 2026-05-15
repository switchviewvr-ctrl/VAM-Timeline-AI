# Manual Labeling Guide: Cowgirl/Riding v1

Manual labels are ground-truth candidates created by visual/human review. Weak labels are only hints. Do not copy weak labels into manual labels unless the motion is actually confirmed.

## What To Label

Label only what is visible or strongly supported. It is fine to leave fields unknown. A useful review item may have one label, many labels, negative labels, or only notes.

## Roles

Use `rider`, `receiver`, `partner_context`, `irrelevant`, or `unknown`. Do not infer role from atom names such as `man`, `Person`, or character names.

## Movement Labels

Use multi-labels for movement: vertical bounce, forward/back rock, lateral sway, circular grind, slow/deep, fast/shallow, upright, lean forward/back, pause/hold, adjustment transition, tempo/depth changes, and irregular human motion.

## Contact And Hands

Only use partner contact labels when pair context or visual review supports them. Without partner evidence, prefer uncertainty or own-body/no-clear-support labels.

## Negative And Control Examples

Negative examples are important. Include `not_cowgirl`, unclear roles, static/passive context, and windows from non-riding scenes when useful.

## ML Safety

Do not train supervised classifiers until labels exist across multiple scenes and samples with negative/control examples. Random window train/test splits are invalid because windows overlap.

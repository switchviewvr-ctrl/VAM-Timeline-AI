# Silver Labels And Weak Supervision

Silver labels are high-confidence machine labels. They are useful, but they are still machine labels.

They are distinct from:

- weak labels, which are threshold hints such as `weak_v2_fast_motion_candidate`
- machine proposals, which are all rule-generated label candidates
- manual labels, which are human-edited ground-truth candidates

Silver label records include:

- `label_source: silver_machine_v1`
- `is_human_ground_truth: false`
- confidence by label
- rule IDs
- evidence summary

Silver labels must never be merged into `manual_labels.yaml`.

## Why Silver Metrics Are Not Semantic Accuracy

Silver labels are generated from the same numeric features used to train the optional silver baseline. A model can score well by learning the rules that produced the labels. This is feature sanity checking and weak-supervised proxy learning, not proof that the model understands Cowgirl/Riding semantics.

Use silver baselines to answer:

- Are the numeric features learnable?
- Do the proxy rules create separable groups?
- Which proposed classes need human review first?

Do not use silver baselines to claim:

- real semantic accuracy
- human-label performance
- readiness for final text-to-animation generation

## Review Batch

`build-machine-proposal-review-batch` creates a human review batch focused on checking machine proposals. It writes:

- `review_batch.jsonl`
- `review_batch.md`
- `machine_label_review.yaml`
- `manual_labels.stub.yaml`
- `batch_summary.md`

The manual stub remains empty. Machine suggestions appear as hints only.

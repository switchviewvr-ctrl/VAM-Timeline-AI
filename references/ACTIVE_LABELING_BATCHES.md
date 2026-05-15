# Active Labeling Batches

Active labeling batches choose the next windows to review after the project has some label coverage information.

## Batch Selection Goals

`build-active-review-batch-v3` prioritizes:

- sparse manual classes;
- missing negative/control examples;
- pair/contact context;
- unclear role cases;
- underrepresented scenes and samples;
- high-value weak_v2 hints;
- static/passive partner context;
- non-cowgirl/control examples.

Weak labels are shown as hints only. They are never copied into manual label fields.

## Batch 003

If batch_002 labels are missing or insufficient, batch_003 becomes another broad seed batch. If batch_002 labels are present, batch_003 should target the weakest coverage areas.

The human should open:

```text
data\runs\clean_v2\labels\batches\batch_003\previews\index.html
```

Then edit:

```text
data\runs\clean_v2\labels\batches\batch_003\manual_labels.stub.yaml
```

Save the edited result as `manual_labels.edited.yaml` before running the ingest workflow.

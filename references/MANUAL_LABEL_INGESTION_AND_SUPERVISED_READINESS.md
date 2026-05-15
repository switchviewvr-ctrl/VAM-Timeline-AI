# Manual Label Ingestion And Supervised Readiness

Manual labels are the first real semantic ground-truth candidates in this project. Weak labels, filenames, atom IDs, and cluster IDs are review aids only.

## Editing A Batch

For a review batch, copy or edit:

```text
manual_labels.stub.yaml
```

Save the human-edited result as:

```text
manual_labels.edited.yaml
```

Do not paste `weak_` labels into `labels`, `negative_labels`, `uncertain_labels`, `pair_labels`, or `contact_labels`. If a weak hint looks correct, translate it into an allowed manual label only after visual review.

## Safe Ingestion

Always inspect the edited batch before merging:

```powershell
python -m vam_timeline_ai.cli inspect-edited-label-batch ^
  --stub data\runs\clean_v2\labels\batches\batch_002\manual_labels.stub.yaml ^
  --edited data\runs\clean_v2\labels\batches\batch_002\manual_labels.edited.yaml ^
  --windows data\runs\clean_v2\semantic\movement_windows.jsonl ^
  --pair-windows data\runs\clean_v2\semantic\pair_windows_v1.jsonl ^
  --out data\runs\clean_v2\labels\batches\batch_002\edited_label_batch_inspection.md
```

The inspection rejects missing files, byte-identical stubs, empty/default entries, weak labels pasted as manual labels, and stale IDs.

Then merge into the run-local manual label file:

```powershell
python -m vam_timeline_ai.cli merge-manual-label-batch ^
  --base data\runs\clean_v2\labels\manual_labels.yaml ^
  --batch data\runs\clean_v2\labels\batches\batch_002\manual_labels.edited.yaml ^
  --out data\runs\clean_v2\labels\manual_labels.yaml ^
  --backup true ^
  --report data\runs\clean_v2\labels\manual_label_merge_report_batch_002.md
```

Empty stub entries are ignored. Existing labels are backed up and conflicts are reported.

## Validation

Validate merged labels against the clean run IDs:

```powershell
python -m vam_timeline_ai.cli validate-manual-labels-v2 ^
  --labels data\runs\clean_v2\labels\manual_labels.yaml ^
  --schema data\labels\manual_labels.schema_v2.yaml ^
  --windows data\runs\clean_v2\semantic\movement_windows.jsonl ^
  --pair-windows data\runs\clean_v2\semantic\pair_windows_v1.jsonl ^
  --out data\runs\clean_v2\labels\manual_label_validation_report.md
```

Unknown labels, invalid confidence values, weak labels in manual fields, and stale IDs are errors. Mixed or contradictory motion labels are usually warnings because real motion can be ambiguous.

## Dataset v2

`cowgirl_ml_dataset_v2.npz` stores:

- manual positive labels;
- manual negative labels;
- manual uncertain labels;
- weak labels;
- include-for-ML flags;
- confidence;
- movement quality;
- semantic role fields;
- grouped scene/sample/source IDs.

Uncertain labels are not positives. Weak labels are not manual labels.

## When Supervised ML Can Start

Supervised ML is allowed only when real manual labels have enough support:

- at least 20 positive examples for a class;
- at least 20 negative/control examples;
- at least 3 scenes;
- at least 5 samples;
- confidence preferably at least 0.6;
- grouped scene/sample/source evaluation is possible.

Random overlapping-window train/test splits are invalid.

# Latest Review Batch Workflow

Review batch numbers are local workflow artifacts. Do not hardcode `batch_002`, `batch_003`, or any later batch as final truth.

Use discovery:

```powershell
python -m vam_timeline_ai.cli find-latest-review-batch ^
  --run-dir data\runs\clean_v2 ^
  --out data\runs\clean_v2\labels\latest_review_batch_report.md
```

This scans:

```text
data\runs\clean_v2\labels\batches\batch_*
```

It selects the latest numeric batch that has a valid `review_batch.jsonl` and `manual_labels.stub.yaml`, preferring batches with `previews\index.html`.

## Human Next Step

Generate the exact local instruction file:

```powershell
python -m vam_timeline_ai.cli write-labeling-next-step ^
  --run-dir data\runs\clean_v2 ^
  --out data\runs\clean_v2\labels\human_labeling_next_step.md
```

If no edited labels exist, the report tells the human which preview index to open, which stub to copy, and where to save `manual_labels.edited.yaml`.

If usable edited labels exist, the report gives the ingestion command.

## Safe Ingestion

Run:

```powershell
python -m vam_timeline_ai.cli ingest-latest-edited-batch ^
  --run-dir data\runs\clean_v2 ^
  --schema data\labels\manual_labels.schema_v2.yaml ^
  --stop-if-missing true
```

If `manual_labels.edited.yaml` is missing, ingestion stops safely and writes the human next-step report. It does not merge labels, rebuild datasets, or train anything.

If edited labels exist, the command inspects, rejects empty stubs, rejects `weak_` labels in manual fields, validates IDs, merges labels, rebuilds dataset v2, plans grouped splits, and checks supervised readiness.

Weak labels remain hints only. Atom names and filenames are not semantic truth.

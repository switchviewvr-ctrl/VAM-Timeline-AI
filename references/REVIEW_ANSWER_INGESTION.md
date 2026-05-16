# Review Answer Ingestion

Answers exported from the Local Review UI are audit findings. They are useful
for calibration and future review planning, but they are not automatically
promoted to `manual_labels.yaml`.

## Ingest JSONL Answers

```powershell
python -m vam_timeline_ai.cli ingest-review-ui-answers ^
  --answers data\runs\clean_v3\audits\semantic_review_010_v16\human_review_ui_answers.jsonl ^
  --review-dir data\runs\clean_v3\audits\semantic_review_010_v16 ^
  --out-ledger data\runs\clean_v3\audits\human_review_ledger.jsonl ^
  --report data\runs\clean_v3\audits\review_ui_answer_ingestion_report.md
```

The ingestion command validates answer fields, appends review-specific ledger
records, and reports verdict counts, common error tags, semantic-family answer
coverage, and contact/support answer coverage.

## Boundaries

- Does not modify `manual_labels.yaml`.
- Does not train ML.
- Does not treat audit answers as final truth.
- Keeps `is_human_ground_truth: false` and `is_training_label: false`.

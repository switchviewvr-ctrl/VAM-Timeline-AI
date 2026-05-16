# clean_v3 Status and QA

clean_v3 is the interaction-aware semantic rescan. It separates motion, pose,
partner relation, contact/support, phase, and generation safety.

The QA workflow is deliberately conservative:

- v15 human findings are audit-only calibration feedback.
- v16 must be manually reviewed before any larger review batch.
- Candidate DBs are inventories, not manual ground truth.
- No ML training should start from these audit labels.

Primary status commands:

- `clean-v3-status`
- `run-clean-v3-overnight-qa`
- `write-clean-v3-dashboard`

The morning entry point is:

`data/runs/clean_v3/reports/overnight_qa_summary.md`


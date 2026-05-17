# Local Review UI

The local review UI is an audit-only Semantic Review Workbench for checking
clean_v3 hypotheses. It does not train ML, does not generate animations, does
not require internet access, and does not write `manual_labels.yaml`.

## Static Fallback

Build a static UI for the latest review folder:

```powershell
python -m vam_timeline_ai.cli build-static-review-ui ^
  --run-dir data\runs\clean_v3 ^
  --review-dir data\runs\clean_v3\audits\semantic_review_010_v16 ^
  --out-dir data\runs\clean_v3\audits\semantic_review_010_v16\review_ui_static
```

Open `index.html` from the output folder in a browser. Answers are stored in
browser localStorage and can be downloaded as JSONL or YAML.

## Local Server

The optional server uses only Python standard library modules:

```powershell
python -m vam_timeline_ai.cli launch-review-ui ^
  --run-dir data\runs\clean_v3 ^
  --review-dir data\runs\clean_v3\audits\semantic_review_010_v16 ^
  --host 127.0.0.1 ^
  --port 8765
```

When launched through the server, the UI can also save exported answers back
inside the selected review folder as `human_review_ui_answers.jsonl` and
`human_review_ui_answers.yaml`. Existing files are backed up first.

## What To Review

Use the Review Batch tab to confirm:

- semantic family
- pose family/subtype
- motion subtype and phase
- partner relation
- contact/support target
- generation safety
- system evidence/scores and warnings

The Candidate DB Explorer and Hypothesis Tester are triage helpers. Candidate
DB rows are still audit inventories, not training ground truth.

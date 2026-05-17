# GitHub Repo Data Safety

This repository is public. It must contain code, tests, documentation, schemas, templates, and lightweight project structure only.

## Local Data Stays Local

Generated clean-run artifacts such as:

```text
data\runs\clean_v2
data\baked
data\features
data\semantic
data\labels\batches
```

are local working data. They are required for analysis and labeling, but they are not public repository artifacts.

## Never Commit

Do not commit:

- raw VaM scene JSONs;
- `.var` packages;
- baked `.npz` or `.npy` arrays;
- feature matrices;
- generated JSONL window/feature/pair files;
- review preview images or videos;
- model files such as `.pkl`, `.joblib`, `.model`, `.onnx`;
- `data\runs\...`;
- `data\labels\manual_labels.yaml`;
- `manual_labels.edited.yaml`;
- any human-edited labels unless there is a later explicit anonymized export process.

Human labels are private/local by default.

## What Is Safe To Commit

Safe public artifacts include:

- source code under `src`;
- tests under `tests`;
- README and reference docs;
- manual label schemas/templates;
- `.gitkeep` placeholders;
- safe example config files.

## Weak Labels

Weak labels are generated hints. They are not semantic truth and should not be committed as a canonical dataset.

## Check Before Pushing

Run:

```powershell
python -m vam_timeline_ai.cli audit-repo-safety ^
  --project-root . ^
  --out data\runs\clean_v2\audits\repo_safety_report.md
```

The report should have no hard errors. Hard errors include tracked raw scenes, local run artifacts, human labels, previews, generated arrays, or models.

## GitHub Actions

CI is intentionally lightweight. It installs the package with dev dependencies, compiles sources, and runs unit tests. It does not run the local data pipeline and does not require private local VaM data.

The current workflow uses `actions/checkout@v4` and `actions/setup-python@v5`. If GitHub reports Node.js action deprecation warnings, update the action versions after verifying the official supported versions.

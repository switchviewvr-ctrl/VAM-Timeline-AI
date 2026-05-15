# Data Identity And Clean Runs

This project now treats data identity as part of the dataset contract. Manual labels must point to stable, unique IDs; otherwise a human label can silently attach to the wrong motion window after a rebuild.

## Why Duplicate IDs Are Dangerous

The early pipeline used readable names such as atom IDs, clip names, and short sample names as IDs. That was useful for debugging, but it caused collisions when different plugins, clips, scenes, or repeated Timeline animations shared names.

Duplicate IDs are especially dangerous for manual labeling:

- a `window_id` can refer to two different time ranges;
- a `sample_id` can refer to two different baked arrays;
- a feature row can overwrite or shadow another feature row;
- pair windows can become ambiguous;
- review batches can point to stale windows.

Manual labels should only be written against a strict-clean run with zero identity errors.

## ID Construction

IDs are built by `src/vam_timeline_ai/io/identity.py`.

All IDs are deterministic and include a short stable hash suffix. The readable prefix is for debugging; the hash prevents collisions when human-readable parts repeat.

`source_id` uses stable source metadata:

- source scene relative path;
- source type;
- technical atom ID;
- storable ID;
- plugin ID;
- clip name;
- clip index;
- optional controller or track identifier.

`sample_id` uses:

- source ID;
- FPS;
- extraction version;
- technical atom ID;
- clip name;
- clip index.

`window_id` uses:

- sample ID;
- start and end time formatted as fixed decimals;
- window size;
- stride;
- FPS.

`pair_id` uses:

- source scene relative path;
- both sample IDs;
- both clip names;
- pair extraction version.

`pair_window_id` uses:

- pair ID;
- both window IDs;
- overlap start/end time.

`feature_record_id` and `review_id` are derived from the target window or pair window plus the feature/review version.

Technical atom IDs remain technical identifiers only. They are never semantic roles.

## Clean Run Layout

Use:

```text
data\runs\clean_v2
```

The clean run keeps artifacts isolated from older top-level outputs:

```text
data\runs\clean_v2\audits
data\runs\clean_v2\semantic
data\runs\clean_v2\baked
data\runs\clean_v2\baked\samples
data\runs\clean_v2\features
data\runs\clean_v2\ml
data\runs\clean_v2\ml\datasets
data\runs\clean_v2\ml\reports
data\runs\clean_v2\labels
data\runs\clean_v2\labels\batches
```

`prepare-clean-run` creates these folders and writes:

```text
data\runs\clean_v2\run_manifest.json
data\runs\clean_v2\prepare_clean_run_report.md
```

The manifest records old artifact locations rather than deleting them.

## Rebuild Workflow

Run the full pipeline into `data\runs\clean_v2`, not top-level `data\semantic`, `data\features`, or `data\labels`.

The important stages are:

1. build motion source index;
2. extract motion samples;
3. audit baked samples;
4. discover controller map;
5. build movement windows;
6. extract Cowgirl/Riding features v1;
7. generate and calibrate weak labels;
8. build context pairs, pair windows, and pair features;
9. apply real manual labels if present;
10. build ML dataset v1;
11. run strict integrity audit;
12. only then create a review batch.

Strict integrity should be run with `--strict true` and should produce zero errors before human labeling starts.

## Stale Artifact Rules

Top-level `data\...` outputs may be from earlier experiments. Do not mix them with `data\runs\clean_v2` artifacts in the same command.

Avoid:

- using top-level windows with clean-run features;
- using clean-run review batches with old pair windows;
- merging labels from an old batch without validating IDs;
- treating v0 and v1 outputs as one dataset;
- using `batch_001` as final label input.

The recommended manual labeling batch after the identity cleanup is:

```text
data\runs\clean_v2\labels\batches\batch_002
```

Start review from:

```text
data\runs\clean_v2\labels\batches\batch_002\previews\index.html
```

Edit labels in:

```text
data\runs\clean_v2\labels\batches\batch_002\manual_labels.stub.yaml
```

Save the edited copy as `manual_labels.edited.yaml` before merging.

## Window Split Warning

Movement windows overlap heavily. A random window-level train/test split leaks adjacent frames, sample style, scene setup, and source-specific motion into both train and test.

Future supervised ML splits must be grouped by scene, sample, or source. Random overlapping-window splits remain invalid, even when IDs are clean.

## What Clean IDs Do Not Solve

Clean IDs make labels traceable. They do not make weak labels true, infer rider/receiver roles, prove contact semantics, or make a source semantically useful.

Weak labels remain review hints. Manual labels remain required for supervised semantic learning.

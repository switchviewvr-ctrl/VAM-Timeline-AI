# New Scenes Delta Import

New VaM scenes should first be analyzed as a separate delta run, not mixed directly into `clean_v3`.

The delta workflow scans only the requested folder, writes all artifacts under a new run directory, compares the results against the existing `clean_v3` reference, and exports a focused review batch for human validation.

## Command

```powershell
python -m vam_timeline_ai.cli run-new-scenes-delta-import ^
  --raw-dir "<path-to-local-vam-scenes>" ^
  --base-run data\runs\clean_v3 ^
  --out-run data\runs\clean_v3_new_scenes
```

This creates:

- `data\runs\clean_v3_new_scenes\semantic\motion_source_index.jsonl`
- `data\runs\clean_v3_new_scenes\baked\motion_sample_index.jsonl`
- `data\runs\clean_v3_new_scenes\semantic\movement_windows.jsonl`
- `data\runs\clean_v3_new_scenes\relative_motion\relative_motion_features.jsonl`
- `data\runs\clean_v3_new_scenes\relative_motion\trajectory_shape_features.jsonl`
- `data\runs\clean_v3_new_scenes\pose_semantics\pose_semantics_v0.jsonl`
- `data\runs\clean_v3_new_scenes\interaction_semantics\interaction_semantics_v0.jsonl`
- `data\runs\clean_v3_new_scenes\semantic_actions\semantic_actions_v2.jsonl`
- `data\runs\clean_v3_new_scenes\datasets\semantic_candidate_db_v0.jsonl`
- `data\runs\clean_v3_new_scenes\datasets\cowgirl_candidate_db_v0.jsonl`
- `data\runs\clean_v3_new_scenes\reports\new_scene_delta_report.md`
- `data\runs\clean_v3_new_scenes\audits\semantic_review_new_scenes_020\`

## Review Before Merge

The generated candidate DBs are audit/candidate inventories only. They are not manual training labels and must not be merged into `manual_labels.yaml`.

Open the generated review UI first:

```text
data\runs\clean_v3_new_scenes\audits\semantic_review_new_scenes_020\review_ui_static\index.html
```

For VaM scene-by-scene review, open:

```text
data\runs\clean_v3_new_scenes\audits\semantic_review_new_scenes_020\vam_review_package\vam_review_index.html
```

Only after manual review should useful new candidates be considered for promotion into the main dataset.

## Safety Rules

- The import scans only the supplied `--raw-dir`.
- Existing `clean_v3` artifacts are reference inputs only and are not overwritten.
- No ML training is run.
- No final generated Timeline animations are created.
- Person/root/world tracks remain excluded from generation-target semantics.

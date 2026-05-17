# Handmade Reference Animations

The handmade animation ZIP is a calibration set, not wild-scene ground truth.

Expected local input:

```powershell
<path-to-reference-animations.zip>
```

Repo-local convention if copied for local work:

```powershell
references\external\animations.zip
```

`references/external/*.zip` is ignored by git. Do not commit the ZIP or extracted generated artifacts.

The importer reads handmade Timeline JSON animations and paired JPG previews. Filename-derived labels are allowed only inside this handmade reference set because the files were intentionally named as labeled examples. Those labels must not be transferred directly to wild windows.

The reference pipeline writes:

- `handmade_animation_manifest.jsonl`
- `handmade_sample_index.jsonl`
- `handmade_features.jsonl`
- `handmade_reference_signatures.json`
- `wild_reference_matches.jsonl`

Cowgirl references should generally show hip/body controller motion and should not be head-only or root-only. BJ/head references are useful guards against rhythmic head-dominant false positives. Doggy references are expected to overlap with hip/thigh motion and remain a context/pose ambiguity to resolve later.

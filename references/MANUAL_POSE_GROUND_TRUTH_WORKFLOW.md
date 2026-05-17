# Manual Pose Ground Truth Workflow

This workflow creates real VaM pose examples for ontology and retargeting calibration.

1. Open VaM.
2. Load or create a correct pose manually.
3. Add `Custom/Scripts/VAMTimelineAI/SkeletonPoseCaptureTool.cs` to a scene atom or session/plugin context.
4. Select the Rider/Female atom.
5. Select the Partner/Male atom.
6. Enter `pose_family` and `pose_subtype`, for example `cowgirl` and `cowgirl_lean_back_supported`.
7. Enter motion intent or notes, such as `pose hold`, `oval grinding`, `pelvis vertical bounce`, or `head bob`.
8. Enable the skeleton overlay and labels.
9. Visually confirm that the overlay matches the real controllers.
10. Click `Capture Pose Snapshot`.
11. Send or import the resulting JSON for analysis.

## Importing Captures

Run:

```powershell
python -m vam_timeline_ai.cli import-manual-pose-captures-v1 ^
  --input-dir "<VaM>/Saves/PluginData/VAMTimelineAI/pose_captures" ^
  --out-jsonl data\runs\clean_v3\manual_pose_captures\manual_pose_captures_v1.jsonl ^
  --report data\runs\clean_v3\reports\manual_pose_capture_import_report_v1.md
```

Then:

```powershell
python -m vam_timeline_ai.cli report-manual-pose-captures-v1 ^
  --captures data\runs\clean_v3\manual_pose_captures\manual_pose_captures_v1.jsonl ^
  --out data\runs\clean_v3\reports\manual_pose_capture_report_v1.md
```

## Ground Truth Boundary

The human-created pose and human-entered label are the meaningful source. The derived distances and hints help debug ontology geometry, but they are not final truth by themselves.

Do not train from these captures until there is a deliberate supervised dataset step. Do not write these captures into `manual_labels.yaml` automatically.

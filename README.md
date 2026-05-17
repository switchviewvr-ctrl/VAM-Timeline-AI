# VaM Timeline AI

VaM Timeline AI is a semantic motion project for generative VaM animation.

The long-term target is:

```text
text prompt -> semantic motion plan -> VaM Timeline animation
```

The current phase is not ML training, motion matching, bridge playback, or loop extraction. The current phase is semantic motion understanding: building a database that can describe how an active actor moves over time so a later generative system can plan and assemble VaM Timeline animation.

## Adding New VaM Scenes

New source scenes must be imported as a separate delta run before they are considered for the main dataset. Do not drop them directly into `clean_v3` outputs.

For a local new-scene batch:

```powershell
python -m vam_timeline_ai.cli run-new-scenes-delta-import ^
  --raw-dir "<path-to-local-vam-scenes>" ^
  --base-run data\runs\clean_v3 ^
  --out-run data\runs\<your_delta_run_name>
```

The delta run scans only that folder, rebuilds technical/semantic artifacts under `data\runs\clean_v3_new_scenes`, compares the result against `clean_v3`, and exports a new-scene review package. Review those candidates manually before promoting anything into the main dataset. Candidate DB rows remain audit/candidate data, not human training truth.

See [NEW_SCENES_DELTA_IMPORT.md](references/NEW_SCENES_DELTA_IMPORT.md).

Before sharing run artifacts, neutralize local scene identifiers:

```powershell
python -m vam_timeline_ai.cli sanitize-run-scene-identifiers ^
  --run-dir data\runs\<your_run> ^
  --alias-map-out data\runs\<your_run>\local_scene_aliases.private.json ^
  --report data\runs\<your_run>\reports\scene_identifier_sanitization.md
```

The alias map is local run data and is ignored by git. Public databases should keep only neutral aliases such as `scene_000001.json`.

## Direction

The previous mocap compiler proved useful technical pieces:

- VaM scene parsing
- Timeline v283 decoding
- native MotionAnimationMaster baking
- validation and preview tooling
- offline Segment-to-Timeline export

Those ideas remain valuable references, but the goal here is different. A generative animation system does not need only perfect loops. It needs to understand behavior: posture, rhythm, contact, attention, transitions, pauses, intensity changes, and how a movement develops over time.

## Context + Semantics Before Prompt-To-Timeline

The current accepted architecture adds a context layer before any future Prompt-to-Timeline work:

- rig anatomy maps body language to VaM controllers, with Cowgirl `pelvis_hip` using `hipControl` as the visible primary driver
- motion ontology and biomechanical gates separate clean cyclic motion from pose holds, transitions, reaching, crawling, and missing-controller artifacts
- NLP lexicons map German/English prompt phrases into semantic tokens
- component intent plans represent prompts as ordered semantic phases, not Timeline exports
- external web context is candidate research only and must be reviewed before it changes ontology

No command in this layer modifies `manual_labels.yaml`, trains ML, or generates Timeline animation. See:

- [RIG_ANATOMY_ONTOLOGY.md](references/RIG_ANATOMY_ONTOLOGY.md)
- [NLP_LEXICON_AND_EXTERNAL_DICTIONARIES.md](references/NLP_LEXICON_AND_EXTERNAL_DICTIONARIES.md)
- [COMPONENT_BASED_MOTION_ONTOLOGY.md](references/COMPONENT_BASED_MOTION_ONTOLOGY.md)
- [NLP_TO_MOTION_INTENT_PLAN.md](references/NLP_TO_MOTION_INTENT_PLAN.md)
- [CONTEXT_AND_SEMANTICS_ROADMAP_TO_PROMPT.md](references/CONTEXT_AND_SEMANTICS_ROADMAP_TO_PROMPT.md)

## Loops Are A Special Case

Loop quality is technical QA. It is not semantic truth.

A good loop can be almost static or semantically unhelpful. A poor loop can contain excellent 2, 4, or 8 second movement windows. Transition and adjustment moments are not failures; they are often exactly the kind of human motion a generative system needs to learn.

This project therefore does not optimize around loop clips. It treats loops as one possible technical property of a source, while the primary semantic unit is the movement window.

## Primary Unit: Movement Window

The main unit is no longer `clip`.

The main unit is:

```text
movement_window
```

A source animation can produce overlapping windows:

- 2 seconds, stride 1 second
- 4 seconds, stride 2 seconds
- 8 seconds, stride 4 seconds

Each window can be independently analyzed and multi-labeled. A 20 second animation might contain slow upright riding, a lean-forward hand-supported section, a pause, fast shallow movement, a circular grind, and a settle transition. All of those windows are useful.

## Atom IDs Are Not Semantic Roles

Technical atom IDs such as `man`, `Person`, `Punk`, or a character name are source identifiers. They are not semantic truth.

An atom named `man` can be the active rider in a specific scene. An atom named `Person` can be active or passive. The system must infer or manually label:

- active rider/focus actor
- partner/receiver context
- role confidence
- manual review needs

Role detection must use motion, context, relative placement, controller behavior, plugin metadata, and manual labels, not atom names alone.

## First Domain: Cowgirl / Riding

Cowgirl/Riding is the first semantic domain because the active rider role is often clearer than in more ambiguous scenes, and the behavior can be studied through:

- pelvis and hip movement
- torso posture and counter-motion
- hands, arms, and support/contact behavior
- legs, knees, feet, and stance stability
- head, gaze, and attention
- rhythm, tempo, depth, pauses, and transitions

The focus actor for this domain is the active rider/female movement, with the partner treated as context. The project must not make the receiver/male body the primary motion target for Cowgirl analysis unless manual labels explicitly say otherwise.

## Semantic Labels Are Multi-Label

A movement window may have many labels:

- `cowgirl_lean_forward`
- `cowgirl_hand_supported_on_partner_chest`
- `cowgirl_deep_slow`
- `cowgirl_irregular_human_motion`

The system must not force each window into exactly one category.

## Manual Labeling Is Mandatory

Automatic inference will be imperfect. Manual YAML overrides are part of the core architecture, not a side feature.

Overrides can correct roles and labels at scene, actor, sample, and movement-window level. See:

[manual_labels.template.yaml](data/labels/manual_labels.template.yaml)

## Future Text-To-Animation Query

A future prompt such as:

```text
slow deep cowgirl, leaning forward, hands on partner chest, looking down
```

should map to semantic query filters like:

- `cowgirl_deep_slow`
- `cowgirl_lean_forward`
- `cowgirl_hand_supported_on_partner_chest`
- high `head_down_score`
- semantic role `rider`
- 4 or 8 second movement windows

This project is preparing that semantic database foundation. It does not implement the final text parser or generator yet.

## Long-Term Stages

1. Raw scan
2. Technical extraction
3. Actor role detection
4. Movement window generation
5. Cowgirl/Riding feature analysis
6. Manual labeling
7. Cowgirl semantic motion database
8. Semantic motion retrieval
9. Text prompt to semantic motion plan
10. VaM Timeline export / bridge playback / ML later

## Machine Label Proposals / Silver Labels

Codex can generate automated label proposals from numeric movement features, pair/context features, and calibrated weak-label hints. These outputs are useful for exploration and for choosing what a human should review next, but they are not human ground truth.

The project keeps four label sources separate:

- Weak labels: `weak_v2_...` threshold hints used for triage.
- Machine proposals: rule-based proposed semantic labels with evidence and confidence.
- Silver labels: high-confidence machine proposals accepted by deterministic rules.
- Manual labels: human-edited labels only.

Machine and silver labels must not be merged into `manual_labels.yaml`. They can help build a weak-supervised proxy baseline, but that model only learns feature rules and proxies. Its metrics are not semantic accuracy. Real supervised semantic ML still requires human labels distributed across multiple scenes/samples with grouped evaluation.

Run the machine-labeling workflow on a clean local run:

```powershell
python -m vam_timeline_ai.cli run-machine-labeling-v1 ^
  --run-dir data\runs\clean_v2 ^
  --min-silver-confidence 0.75 ^
  --train-silver-baseline true
```

This writes machine outputs under `data\runs\clean_v2\labels\machine_proposals\` and a review batch under `data\runs\clean_v2\labels\batches\batch_machine_review_001\`. Generated data remains local and ignored by git.

### Silver v2 Quality Control

Raw machine proposals can be intentionally overcomplete. Pair/context rules can also multiply labels because one movement window may appear in many pair windows. This is useful for review, but it is too duplicated and imbalanced for training directly.

Silver v2 fixes that by:

- auditing raw proposal duplication and conflicts
- aggregating many raw proposals into one window-label or pair-window-label score
- capping repeated pair-context evidence
- keeping role/contact candidates high-risk
- excluding `contact_unknown` and role labels such as `rider_active` from default training targets
- building a balanced proxy dataset from aggregated silver v2 labels, not raw proposals

Run the v2 workflow:

```powershell
python -m vam_timeline_ai.cli run-machine-labeling-v2 ^
  --run-dir data\runs\clean_v2 ^
  --min-silver-score 0.78 ^
  --train-silver-baseline true ^
  --allow-numpy-fallback true
```

The optional silver baseline is only a feature sanity check. If it trains, its metrics measure rule/proxy reproducibility, not human semantic accuracy. Human labels remain required before real supervised semantic ML.

### Reality Audit Before Further ML

Before doing more ML work, run a reality audit to check whether the extracted motion, controller mappings, feature proxies, pair windows, and machine/silver hints match what is actually visible in the previews.

This audit is not manual semantic labeling for training. It writes a separate annotation file under `data\runs\clean_v2\audits\...` and must not be merged into `manual_labels.yaml`.

```powershell
python -m vam_timeline_ai.cli export-reality-audit-100 ^
  --run-dir data\runs\clean_v2 ^
  --out-dir data\runs\clean_v2\audits\reality_audit_001 ^
  --count 100
```

Open:

```text
data\runs\clean_v2\audits\reality_audit_001\previews\index.html
```

Copy:

```text
data\runs\clean_v2\audits\reality_audit_001\reality_audit_annotation.stub.yaml
```

Save the edited audit answers as:

```text
data\runs\clean_v2\audits\reality_audit_001\reality_audit_annotation.edited.yaml
```

Then summarize the audit:

```powershell
python -m vam_timeline_ai.cli summarize-reality-audit ^
  --annotations data\runs\clean_v2\audits\reality_audit_001\reality_audit_annotation.edited.yaml ^
  --audit-batch data\runs\clean_v2\audits\reality_audit_001\reality_audit_batch.jsonl ^
  --out data\runs\clean_v2\audits\reality_audit_001\reality_audit_result.md
```

Only continue toward labels or ML if the audit says data extraction, feature interpretation, and pair context are trustworthy enough.

### Human Semantic Review Findings

### Semantic Families and Candidate DBs

The review pipeline now treats semantic families separately. BJ/oral motion is a valid animation family, not bad data. Cowgirl-specific filters exclude BJ/oral candidates from Cowgirl generation-safe sets, but the global semantic candidate DB preserves them with `semantic_family: bj_oral`, `excluded_from_cowgirl: true`, and `preserve_for_future_dataset: true`.

Candidate DBs remain audit inventories. They are not manual labels and must not be merged into `manual_labels.yaml`. Core-controller gates can soft-fail when visible pose/motion evidence is strong, while hand/head-only or missing-core cases remain hard failures for Cowgirl generation safety.

See:

- [SEMANTIC_FAMILIES_AND_CANDIDATE_DBS.md](references/SEMANTIC_FAMILIES_AND_CANDIDATE_DBS.md)
- [BJ_ORAL_DOMAIN_CLASSIFIER.md](references/BJ_ORAL_DOMAIN_CLASSIFIER.md)

### From Candidate DB to Generative Motion

The final goal is not clip stitching. Candidate DB entries are source material for learning relative movement principles: local trajectory shapes, rhythm, amplitude, controller coordination, anchors, and safety constraints.

The intended architecture is:

```text
text prompt -> semantic motion plan -> generated relative motion flow -> VaM Timeline animation
```

New generation-layer prototypes make that separation explicit:

- Motion primitives summarize relative movement patterns and contain no absolute world-coordinate generation targets.
- Semantic motion plans encode requested family, subtype, trajectory, tempo, depth, and safety requirements.
- Primitive retrieval inspects learned pattern space and does not concatenate Timeline clips.
- Motion flow skeletons describe future relative controller tracks and remain `export_ready: false`.

See:

- [MOTION_PRIMITIVE_ABSTRACTION.md](references/MOTION_PRIMITIVE_ABSTRACTION.md)
- [SEMANTIC_MOTION_PLAN.md](references/SEMANTIC_MOTION_PLAN.md)
- [WHY_RETRIEVAL_IS_NOT_FINAL_GENERATION.md](references/WHY_RETRIEVAL_IS_NOT_FINAL_GENERATION.md)
- [TEXT_TO_TIMELINE_ROADMAP.md](references/TEXT_TO_TIMELINE_ROADMAP.md)

### Cowgirl Motion Flow V1

Motion Flow V0 proved that generated relative deltas can be played safely in VaM through a review-only player: the Person/root stays fixed, anchors can remain stable, and reset works. V0 was intentionally simple and looked like an isolated pelvis loop when tested from a standing baseline.

Motion Flow V1 adds a Cowgirl-specific review baseline and a body coordination profile. It generates pelvis driver motion plus damped abdomen/chest/head followers, static knee/foot anchors, and optional hand support anchors. The oval grind profile reduces lateral-only hula-hoop motion by increasing forward/back and vertical components.

This is still review-only. It is not final text-to-animation, not native Timeline export, not clip stitching, and not ML training. The review player applies generated relative deltas from the current captured VaM controller baseline.

See:

- [COWGIRL_MOTION_FLOW_V1.md](references/COWGIRL_MOTION_FLOW_V1.md)
- [COWGIRL_REVIEW_BASELINE_POSE.md](references/COWGIRL_REVIEW_BASELINE_POSE.md)
- [REVIEW_PLAYER_V1.md](references/REVIEW_PLAYER_V1.md)

### Native Timeline Export

The Generated Motion Review Player was a temporary debug tool. The main generated output path is native Timeline JSON:

```text
prompt -> semantic motion plan -> generated relative motion flow -> retarget/anchor safety -> native Timeline JSON
```

Native export v0 writes Timeline-like `SerializeVersion: 283` JSON with generated controller curves, not a custom review-player schema. It remains experimental until VaM Timeline import is manually confirmed. The exporter still rejects Person/root/world tracks, does not use source-world coordinates, and does not stitch clips.

Native export v1 bakes generated relative motion onto a generated Cowgirl/kneeling baseline before writing Timeline keyframes. This matters because Timeline plays concrete controller targets; it does not add relative deltas to a captured baseline on its own.

See:

- [NATIVE_TIMELINE_EXPORT_RESEARCH.md](references/NATIVE_TIMELINE_EXPORT_RESEARCH.md)
- [NATIVE_TIMELINE_EXPORT_FROM_GENERATED_FLOW.md](references/NATIVE_TIMELINE_EXPORT_FROM_GENERATED_FLOW.md)
- [NATIVE_TIMELINE_EXPORT_V1_BASELINE_BAKING.md](references/NATIVE_TIMELINE_EXPORT_V1_BASELINE_BAKING.md)
- [REVIEW_PLAYER_IS_DEBUG_ONLY.md](references/REVIEW_PLAYER_IS_DEBUG_ONLY.md)

The first 10-item VaM semantic review showed that the machine/silver interpretation was not reliable enough for more ML. Only `review_010` was a clear Cowgirl segment. Several examples were transition/in-between motions, one looked head/BJ-domain rather than Cowgirl, and two were whole-controller/whole-person motion instead of real body/extremity animation.

Those findings live under:

```text
data\runs\clean_v2\audits\semantic_review_010
```

They are audit findings only. They are not merged into `manual_labels.yaml` and are not training labels.

### Body Motion Quality Gate

Before trusting semantic guesses, audit whether each window contains real body-controller motion:

```powershell
python -m vam_timeline_ai.cli audit-body-motion-quality ^
  --run-dir data\runs\clean_v2 ^
  --sample-index data\runs\clean_v2\baked\motion_sample_index.jsonl ^
  --features data\runs\clean_v2\features\cowgirl_window_features_v1.jsonl ^
  --controller-map data\runs\clean_v2\semantic\controller_bodypart_map.json ^
  --out-jsonl data\runs\clean_v2\audits\body_motion_quality.jsonl ^
  --report data\runs\clean_v2\audits\body_motion_quality_report.md
```

Final generated VaM Timeline animation must never output Person/root/world transform motion. Only real bodypart controller tracks such as `hipControl`, `abdomenControl`, `chestControl`, `headControl`, hands, knees, feet, and thighs are valid animation targets.

### Manual GT Timeline V4 Accepted Baseline

`data/runs/clean_v3/generation/manual_gt_timeline_examples_v4` is the accepted review-only baseline reference for future generation guardrails. It uses real manual VaM pose captures, includes controller rotations, requires `hipControl`, uses sparse 1 FPS semantic keyframes, and applies `data/config/manual_gt_motion_amplitude_profiles_v1.yaml` for readable driver motion.

For Cowgirl/Reverse Cowgirl, semantic `pelvis_hip` maps to VaM `hipControl` as the primary visible driver. `pelvisControl` is a light follower or static support controller, not the sole primary driver. BJ/HJ keep hip/pelvis static, static anchors stay static in position and rotation, and Person/root/world tracks remain forbidden.

### Handmade Reference Animations

The handmade reference ZIP calibrates Cowgirl vs BJ/head-dominant vs Doggy vs hand/head gesture motion:

```powershell
python -m vam_timeline_ai.cli import-handmade-reference-animations ^
  --zip "<path-to-reference-animations.zip>" ^
  --out-dir data\runs\clean_v2\references\handmade_animations
```

Filename labels are allowed only for this handmade reference set. They are not used as wild-scene semantic truth.

### Safer Semantic Review V3

After body-motion quality and handmade reference matching, export a new small VaM review:

```powershell
python -m vam_timeline_ai.cli export-semantic-review-010 ^
  --run-dir data\runs\clean_v2 ^
  --out-dir data\runs\clean_v2\audits\semantic_review_010_v3 ^
  --count 10 ^
  --attempt-timeline-export true ^
  --use-body-motion-quality true ^
  --prefer-clean-body-motion true ^
  --use-handmade-reference-matches true
```

This is another semantic check, not training. Timeline exports strip unsafe Person/root/world tracks and write teleport-risk metadata. If no safe body-controller tracks remain, no fake export is created.

### Semantic Review V4: Micro-Motion Filtering

The V3 VaM review improved over V1 but still allowed too many static or micro-motion examples. V4 adds:

- per-window bodypart displacement metrics
- `static_or_micro_motion`
- `minimal_head_motion_only`
- `minimal_hand_jitter_only`
- clean Cowgirl candidate score v2
- 4s/8s preference for likely Cowgirl review examples

Score candidates:

```powershell
python -m vam_timeline_ai.cli score-cowgirl-candidates-v2 ^
  --run-dir data\runs\clean_v2 ^
  --wild-reference-matches data\runs\clean_v2\references\handmade_animations\wild_reference_matches.jsonl ^
  --body-quality data\runs\clean_v2\audits\body_motion_quality.jsonl ^
  --features data\runs\clean_v2\features\cowgirl_window_features_v1.jsonl ^
  --out-jsonl data\runs\clean_v2\audits\cowgirl_candidate_scores_v2.jsonl ^
  --report data\runs\clean_v2\audits\cowgirl_candidate_score_v2_report.md
```

Export V4:

```powershell
python -m vam_timeline_ai.cli export-semantic-review-010 ^
  --run-dir data\runs\clean_v2 ^
  --out-dir data\runs\clean_v2\audits\semantic_review_010_v4 ^
  --count 10 ^
  --attempt-timeline-export true ^
  --use-body-motion-quality true ^
  --prefer-clean-body-motion true ^
  --use-handmade-reference-matches true ^
  --prefer-longer-cowgirl-windows true ^
  --min-cowgirl-window-seconds 4 ^
  --use-cowgirl-candidate-score-v2 true
```

V4 is still a review batch, not training data.

## Commands

Print project and reference status:

```powershell
python -m vam_timeline_ai.cli info
```

Run the lightweight raw scan:

```powershell
python -m vam_timeline_ai.cli scan-raw-folder ^
  --raw-dir "<path-to-local-vam-scenes>" ^
  --out data\audits\raw_scan
```

The scan writes:

- `raw_scan_index.json`
- `raw_scan_report.md`

It does not bake motion, infer deep semantics, export Timeline clips, run VaM, build bridge playback, build motion matching, or train a model.

## Practical Data Pipeline v0/v1

Run from the project root:

```powershell
cd "<project-root>"
$env:PYTHONPATH="src"
```

1. Scan raw scenes:

```powershell
python -m vam_timeline_ai.cli scan-raw-folder ^
  --raw-dir "<path-to-local-vam-scenes>" ^
  --out data\audits\raw_scan
```

2. Build the technical motion source index:

```powershell
python -m vam_timeline_ai.cli build-motion-source-index ^
  --raw-dir "<path-to-local-vam-scenes>" ^
  --out data\semantic\motion_source_index.jsonl ^
  --report data\semantic\motion_source_index_report.md ^
  --recursive
```

3. Extract and bake technical motion samples:

```powershell
python -m vam_timeline_ai.cli extract-motion-samples ^
  --source-index data\semantic\motion_source_index.jsonl ^
  --out-dir data\baked\samples ^
  --index-out data\baked\motion_sample_index.jsonl ^
  --fps 60
```

4. Build overlapping movement windows:

```powershell
python -m vam_timeline_ai.cli build-movement-windows ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --out data\semantic\movement_windows.jsonl
```

5. Extract Cowgirl/Riding numeric features v0:

```powershell
python -m vam_timeline_ai.cli extract-cowgirl-features-v0 ^
  --windows data\semantic\movement_windows.jsonl ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --out-jsonl data\features\cowgirl_window_features_v0.jsonl ^
  --out-npz data\features\cowgirl_window_features_v0.npz ^
  --report data\features\cowgirl_feature_report_v0.md
```

6. Apply real manual labels if available:

```powershell
python -m vam_timeline_ai.cli apply-manual-labels ^
  --windows data\semantic\movement_windows.jsonl ^
  --labels data\labels\manual_labels.yaml ^
  --out data\semantic\movement_windows_labeled.jsonl ^
  --report data\semantic\manual_label_report.md
```

If `manual_labels.yaml` does not exist, the template is not used as real labels.

7. Build the ML-ready dataset:

```powershell
python -m vam_timeline_ai.cli build-ml-dataset-v0 ^
  --features data\features\cowgirl_window_features_v0.jsonl ^
  --windows data\semantic\movement_windows_labeled.jsonl ^
  --out data\ml\datasets\cowgirl_ml_dataset_v0.npz ^
  --report data\ml\reports\cowgirl_ml_dataset_v0_report.md
```

8. Run ML readiness and clustering analysis:

```powershell
python -m vam_timeline_ai.cli analyze-ml-v0 ^
  --dataset data\ml\datasets\cowgirl_ml_dataset_v0.npz ^
  --out-dir data\ml\reports
```

This is still not final generative AI. It is the first data and learning-oriented layer.

## Learning Pipeline v1

The v1 pipeline adds QA and richer Cowgirl/Riding features before any supervised ML. It keeps technical facts, weak labels, and manual labels separate.

1. Audit baked samples before trusting them:

```powershell
python -m vam_timeline_ai.cli audit-baked-samples ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --out-jsonl data\audits\baked_sample_audit.jsonl ^
  --report data\audits\baked_sample_audit_report.md
```

2. Discover controller names and conservative body-part mappings:

```powershell
python -m vam_timeline_ai.cli discover-controller-map ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --out data\semantic\controller_name_inventory.json ^
  --map-out data\semantic\controller_bodypart_map.json ^
  --report data\semantic\controller_mapping_report.md
```

3. Extract richer Cowgirl/Riding features v1:

```powershell
python -m vam_timeline_ai.cli extract-cowgirl-features-v1 ^
  --windows data\semantic\movement_windows.jsonl ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --controller-map data\semantic\controller_bodypart_map.json ^
  --out-jsonl data\features\cowgirl_window_features_v1.jsonl ^
  --out-npz data\features\cowgirl_window_features_v1.npz ^
  --report data\features\cowgirl_feature_report_v1.md
```

4. Build possible context pairs without assigning rider/receiver roles:

```powershell
python -m vam_timeline_ai.cli build-context-pair-candidates ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --out data\semantic\context_pair_candidates.jsonl ^
  --report data\semantic\context_pair_candidates_report.md
```

5. Generate weak labels for review prioritization only:

```powershell
python -m vam_timeline_ai.cli generate-weak-labels-v1 ^
  --features data\features\cowgirl_window_features_v1.jsonl ^
  --out data\semantic\weak_labels_v1.jsonl ^
  --report data\semantic\weak_labels_report_v1.md
```

Every weak label is prefixed with `weak_`. Weak labels are not manual labels and are not semantic ground truth.

6. Build a manual review queue:

```powershell
python -m vam_timeline_ai.cli build-review-queue-v1 ^
  --features data\features\cowgirl_window_features_v1.jsonl ^
  --weak-labels data\semantic\weak_labels_v1.jsonl ^
  --clusters data\ml\reports\cowgirl_cluster_assignments_v0.jsonl ^
  --windows data\semantic\movement_windows.jsonl ^
  --out data\labels\review_queue_v1.jsonl ^
  --markdown data\labels\review_queue_v1.md
```

7. Manually label selected windows in `data\labels\manual_labels.yaml`, then rebuild the labeled window file:

```powershell
python -m vam_timeline_ai.cli apply-manual-labels ^
  --windows data\semantic\movement_windows.jsonl ^
  --labels data\labels\manual_labels.yaml ^
  --out data\semantic\movement_windows_labeled.jsonl ^
  --report data\semantic\manual_label_report.md
```

8. Build the v1 ML dataset with manual and weak labels separated:

```powershell
python -m vam_timeline_ai.cli build-ml-dataset-v1 ^
  --features data\features\cowgirl_window_features_v1.jsonl ^
  --windows data\semantic\movement_windows_labeled.jsonl ^
  --weak-labels data\semantic\weak_labels_v1.jsonl ^
  --out data\ml\datasets\cowgirl_ml_dataset_v1.npz ^
  --report data\ml\reports\cowgirl_ml_dataset_v1_report.md
```

9. Run leakage-aware readiness reports:

```powershell
python -m vam_timeline_ai.cli analyze-ml-v1 ^
  --dataset data\ml\datasets\cowgirl_ml_dataset_v1.npz ^
  --out-dir data\ml\reports
```

10. Optionally cluster with richer features if `scikit-learn` is installed:

```powershell
python -m vam_timeline_ai.cli cluster-ml-v1 ^
  --dataset data\ml\datasets\cowgirl_ml_dataset_v1.npz ^
  --out-dir data\ml\reports
```

If `scikit-learn` is missing, the command writes an honest dependency/status report instead of pretending clustering ran.

## Split Safety

Movement windows overlap heavily. Random train/test splits by window are invalid because nearby windows from the same sample and scene leak motion patterns into both sides. Any future supervised ML split must group by scene, sample, or source.

## Weak Labels Versus Manual Labels

Weak labels are numeric review hints such as `weak_high_vertical_bounce` or `weak_pause_or_hold`. They help choose windows for human review and clustering inspection. They must not be mixed into manual semantic labels unless a person confirms them.

## Manual Label Workflow v2

The current priority is making human review practical before supervised ML. Run these after the v1 feature dataset exists:

1. Audit data integrity:

```powershell
python -m vam_timeline_ai.cli audit-data-integrity ^
  --source-index data\semantic\motion_source_index.jsonl ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --windows data\semantic\movement_windows.jsonl ^
  --features data\features\cowgirl_window_features_v1.jsonl ^
  --dataset data\ml\datasets\cowgirl_ml_dataset_v1.npz ^
  --out data\audits\data_integrity_report.md
```

2. Calibrate weak labels:

```powershell
python -m vam_timeline_ai.cli calibrate-weak-labels-v2 ^
  --features data\features\cowgirl_window_features_v1.jsonl ^
  --weak-labels data\semantic\weak_labels_v1.jsonl ^
  --out data\semantic\weak_labels_v2.jsonl ^
  --report data\semantic\weak_label_calibration_report_v2.md
```

3. Build aligned pair windows and pair/context features:

```powershell
python -m vam_timeline_ai.cli build-pair-windows-v1 ^
  --pair-candidates data\semantic\context_pair_candidates.jsonl ^
  --windows data\semantic\movement_windows.jsonl ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --out data\semantic\pair_windows_v1.jsonl ^
  --report data\semantic\pair_windows_report_v1.md

python -m vam_timeline_ai.cli extract-pair-features-v0 ^
  --pair-windows data\semantic\pair_windows_v1.jsonl ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --controller-map data\semantic\controller_bodypart_map.json ^
  --out-jsonl data\features\cowgirl_pair_features_v0.jsonl ^
  --out-npz data\features\cowgirl_pair_features_v0.npz ^
  --report data\features\cowgirl_pair_feature_report_v0.md
```

4. Write the manual label schema/template/guide:

```powershell
python -m vam_timeline_ai.cli write-manual-label-schema-v2 ^
  --out data\labels\manual_labels.schema_v2.yaml ^
  --template data\labels\manual_labels.template_v2.yaml ^
  --guide references\MANUAL_LABELING_GUIDE_COWGIRL_V1.md
```

5. Build a balanced review batch and render static previews:

```powershell
python -m vam_timeline_ai.cli build-review-batch-v2 ^
  --windows data\semantic\movement_windows.jsonl ^
  --features data\features\cowgirl_window_features_v1.jsonl ^
  --weak-labels data\semantic\weak_labels_v2.jsonl ^
  --pair-windows data\semantic\pair_windows_v1.jsonl ^
  --pair-features data\features\cowgirl_pair_features_v0.jsonl ^
  --clusters data\ml\reports\cowgirl_cluster_assignments_v1.jsonl ^
  --out-dir data\labels\batches\batch_001 ^
  --batch-size 120 ^
  --max-per-scene 15 ^
  --max-per-sample 3 ^
  --prefer-pair-context true

python -m vam_timeline_ai.cli render-review-previews-v1 ^
  --review-batch data\labels\batches\batch_001\review_batch.jsonl ^
  --sample-index data\baked\motion_sample_index.jsonl ^
  --controller-map data\semantic\controller_bodypart_map.json ^
  --out-dir data\labels\batches\batch_001\previews
```

6. A human edits `data\labels\batches\batch_001\manual_labels.stub.yaml` into `manual_labels.edited.yaml`. Then validate and merge:

```powershell
python -m vam_timeline_ai.cli validate-manual-labels-v2 ^
  --labels data\labels\manual_labels.yaml ^
  --schema data\labels\manual_labels.schema_v2.yaml ^
  --windows data\semantic\movement_windows.jsonl ^
  --pair-windows data\semantic\pair_windows_v1.jsonl ^
  --out data\labels\manual_label_validation_report.md

python -m vam_timeline_ai.cli merge-manual-label-batch ^
  --base data\labels\manual_labels.yaml ^
  --batch data\labels\batches\batch_001\manual_labels.edited.yaml ^
  --out data\labels\manual_labels.yaml ^
  --backup true ^
  --report data\labels\manual_label_merge_report.md
```

7. Summarize labels and plan leakage-safe splits:

```powershell
python -m vam_timeline_ai.cli summarize-manual-labels ^
  --labels data\labels\manual_labels.yaml ^
  --windows data\semantic\movement_windows.jsonl ^
  --pair-windows data\semantic\pair_windows_v1.jsonl ^
  --out data\labels\manual_label_summary.md

python -m vam_timeline_ai.cli plan-ml-splits-v1 ^
  --dataset data\ml\datasets\cowgirl_ml_dataset_v1.npz ^
  --labels data\labels\manual_labels.yaml ^
  --out data\ml\datasets\split_plan_v1.json ^
  --report data\ml\reports\split_plan_v1_report.md
```

Weak labels and filenames are review hints only. The review batch never pre-fills semantic labels from weak labels.

## Clean Identity Runs

Top-level `data\...` outputs may contain earlier experimental v0/v1 artifacts. Those files are useful for development history, but they are not the recommended source for manual labels if a strict clean run exists.

The first identity-clean run is:

```text
data\runs\clean_v2
```

Use `clean_v2` for manual labeling because its `source_id`, `sample_id`, `window_id`, `pair_id`, `pair_window_id`, feature IDs, and review IDs are deterministic and unique. The strict integrity report for the label-ready batch is:

```text
data\runs\clean_v2\audits\data_integrity_with_batch_002_report.md
```

Do not label the older `data\labels\batches\batch_001` review batch unless it is regenerated or independently validated against the clean ID set. `batch_001` was created before the identity cleanup and should be treated as experimental/stale.

To start a clean run:

```powershell
python -m vam_timeline_ai.cli prepare-clean-run ^
  --data-root data ^
  --run-name clean_v2 ^
  --backup-existing true ^
  --out-manifest data\runs\clean_v2\run_manifest.json ^
  --report data\runs\clean_v2\prepare_clean_run_report.md
```

After rebuilding the pipeline under `data\runs\clean_v2`, run strict integrity before making a review batch:

```powershell
python -m vam_timeline_ai.cli audit-data-integrity ^
  --source-index data\runs\clean_v2\semantic\motion_source_index.jsonl ^
  --sample-index data\runs\clean_v2\baked\motion_sample_index.jsonl ^
  --windows data\runs\clean_v2\semantic\movement_windows.jsonl ^
  --features data\runs\clean_v2\features\cowgirl_window_features_v1.jsonl ^
  --dataset data\runs\clean_v2\ml\datasets\cowgirl_ml_dataset_v1.npz ^
  --pair-windows data\runs\clean_v2\semantic\pair_windows_v1.jsonl ^
  --pair-features data\runs\clean_v2\features\cowgirl_pair_features_v0.jsonl ^
  --out data\runs\clean_v2\audits\data_integrity_report.md ^
  --strict true
```

Use `find-latest-review-batch` to discover the current human review entry point. Batch numbers are local workflow artifacts, not final truth:

```powershell
python -m vam_timeline_ai.cli find-latest-review-batch ^
  --run-dir data\runs\clean_v2 ^
  --out data\runs\clean_v2\labels\latest_review_batch_report.md
```

Then write the exact human next-step file:

```powershell
python -m vam_timeline_ai.cli write-labeling-next-step ^
  --run-dir data\runs\clean_v2 ^
  --out data\runs\clean_v2\labels\human_labeling_next_step.md
```

See [DATA_IDENTITY_AND_CLEAN_RUNS.md](references/DATA_IDENTITY_AND_CLEAN_RUNS.md) and [LATEST_REVIEW_BATCH_WORKFLOW.md](references/LATEST_REVIEW_BATCH_WORKFLOW.md) for the ID rules and batch discovery workflow.

## Manual Label Ingestion

After a human edits the latest review-batch stub, save it as `manual_labels.edited.yaml` inside that batch folder. Then let the orchestration command discover and ingest the latest edited batch safely:

```powershell
python -m vam_timeline_ai.cli ingest-latest-edited-batch ^
  --run-dir data\runs\clean_v2 ^
  --schema data\labels\manual_labels.schema_v2.yaml ^
  --stop-if-missing true
```

If no edited labels exist, ingestion stops safely and writes `human_labeling_next_step.md`. Empty stubs are ignored. Weak labels are rejected as manual labels. Unknown IDs are treated as stale/non-clean-run references.

Then rebuild the labeled windows and v2 dataset:

```powershell
python -m vam_timeline_ai.cli apply-manual-labels ^
  --windows data\runs\clean_v2\semantic\movement_windows.jsonl ^
  --labels data\runs\clean_v2\labels\manual_labels.yaml ^
  --out data\runs\clean_v2\semantic\movement_windows_labeled.jsonl ^
  --report data\runs\clean_v2\semantic\manual_label_report.md

python -m vam_timeline_ai.cli build-ml-dataset-v2 ^
  --features data\runs\clean_v2\features\cowgirl_window_features_v1.jsonl ^
  --windows data\runs\clean_v2\semantic\movement_windows_labeled.jsonl ^
  --weak-labels data\runs\clean_v2\semantic\weak_labels_v2.jsonl ^
  --manual-labels data\runs\clean_v2\labels\manual_labels.yaml ^
  --out data\runs\clean_v2\ml\datasets\cowgirl_ml_dataset_v2.npz ^
  --report data\runs\clean_v2\ml\reports\cowgirl_ml_dataset_v2_report.md
```

`cowgirl_ml_dataset_v2.npz` separates manual positive, manual negative, manual uncertain, and weak labels. Uncertain labels are not positives. Weak labels are not training targets.

Before any supervised baseline:

```powershell
python -m vam_timeline_ai.cli plan-ml-splits-v1 ^
  --dataset data\runs\clean_v2\ml\datasets\cowgirl_ml_dataset_v2.npz ^
  --labels data\runs\clean_v2\labels\manual_labels.yaml ^
  --out data\runs\clean_v2\ml\datasets\split_plan_v1.json ^
  --report data\runs\clean_v2\ml\reports\split_plan_v1_report.md

python -m vam_timeline_ai.cli analyze-supervised-readiness ^
  --dataset data\runs\clean_v2\ml\datasets\cowgirl_ml_dataset_v2.npz ^
  --labels data\runs\clean_v2\labels\manual_labels.yaml ^
  --split-plan data\runs\clean_v2\ml\datasets\split_plan_v1.json ^
  --out data\runs\clean_v2\ml\reports\supervised_readiness_report.md
```

Supervised ML starts only if real manual labels are sufficient across grouped scenes/samples with negative/control examples. See [MANUAL_LABEL_INGESTION_AND_SUPERVISED_READINESS.md](references/MANUAL_LABEL_INGESTION_AND_SUPERVISED_READINESS.md), [SUPERVISED_BASELINE_V0.md](references/SUPERVISED_BASELINE_V0.md), and [ACTIVE_LABELING_BATCHES.md](references/ACTIVE_LABELING_BATCHES.md).

## Relative Motion And Trajectory Shape

Raw Timeline coordinates are not reusable motion knowledge. They can encode source-scene placement, Person/root transforms, and world-space pose data. The relative motion layer strips Person/root/world-like tracks, keeps only allowed body controllers, and converts motion into local deltas before extracting features.

Trajectory shape analysis inspects relative pelvis/hip paths for audit signals such as oval/circular grinding, vertical bounce, forward/back rock, transition paths, and jitter/static micro-motion. These signals are used to build safer semantic review batches, not to create ground truth.

Timeline segment exports in semantic review folders are for VaM inspection only unless explicitly marked generation-safe. Current review exports preserve source controller coordinates and are marked `safe_for_generation_template: false`.

References:

- [Relative Motion Representation](references/RELATIVE_MOTION_REPRESENTATION.md)
- [Coordinate Space And Teleport Safety](references/COORDINATE_SPACE_AND_TELEPORT_SAFETY.md)
- [Why Raw Timeline Coordinates Are Not Motion](references/WHY_RAW_TIMELINE_COORDINATES_ARE_NOT_MOTION.md)
- [Trajectory Shape Analysis](references/TRAJECTORY_SHAPE_ANALYSIS.md)

## Generated Relative Motion Flow

Motion primitives are now used to synthesize relative controller curves. V0 synthesis creates a generated motion flow in `relative_body_motion` space from primitive group statistics such as trajectory shape, amplitude ranges, rhythm, and controller roles.

This is not Timeline export and not clip stitching. Generated flows keep `export_ready: false`, exclude Person/root/world tracks, and are technical intermediates for future retargeting and safety validation.

References:

- [Motion Flow Synthesis V0](references/MOTION_FLOW_SYNTHESIS_V0.md)
- [Generated Relative Motion Flow](references/GENERATED_RELATIVE_MOTION_FLOW.md)

## Pose And Partner-Relative Semantics

clean_v3 introduces a fuller Semantic Action model. Motion alone is not enough:
Cowgirl generation now tracks actor pose, motion subtype, partner relation,
contact/support targets, phase, and generation safety as separate fields before
combining them.

BJ/oral remains a valid semantic family. Cowgirl-specific filters exclude it
from Cowgirl generation-safe sets, but the global semantic candidate inventory
preserves it for future BJ/oral dataset work.

For prompts such as `slow cowgirl grinding, leaning forward, hands on partner
chest`, the plan must include rider/receiver roles, partner-pelvis-local frame,
hand targets on `partner.chest`, knees/feet anchors, and no Person/root/world
tracks.

References:

- [Pose Semantics](references/POSE_SEMANTICS.md)
- [Semantic Actions: Pose Plus Motion](references/SEMANTIC_ACTIONS_POSE_PLUS_MOTION.md)
- [Partner-Relative Interaction Semantics](references/PARTNER_RELATIVE_INTERACTION_SEMANTICS.md)
- [Contact Targets And Support Constraints](references/CONTACT_TARGETS_AND_SUPPORT_CONSTRAINTS.md)
- [Prompt To Interaction Plan](references/PROMPT_TO_INTERACTION_PLAN.md)
- [clean_v3 Semantic Rescan](references/CLEAN_V3_SEMANTIC_RESCAN.md)

## clean_v3 QA And Review Workflow

clean_v3 is stricter and interaction-aware, but its candidate DBs are still
audit inventories rather than ground truth. Human review findings are collected
in an audit-only ledger so calibration can target recurring errors such as
low-motion Cowgirl pose context, BJ/oral mistaken as Cowgirl, ambiguous contact
support, duplicate review selection, and foot-anchor weirdness.

The overnight QA workflow writes a dashboard, DB invariant report, clean_v2 to
clean_v3 drift report, larger review batch plan, prompt capability matrix, and
operator status report. It does not train ML, generate new Timeline animations,
or modify `manual_labels.yaml`.

Run:

```powershell
python -m vam_timeline_ai.cli run-clean-v3-overnight-qa ^
  --run-dir data\runs\clean_v3 ^
  --include-runs data\runs\clean_v2,data\runs\clean_v3
```

Review v16 should be checked before exporting any larger batch. Use the ledger
and invariant report to decide the next calibration target, not as final labels.

References:

- [clean_v3 Status And QA](references/CLEAN_V3_STATUS_AND_QA.md)
- [Human Review Memory](references/HUMAN_REVIEW_MEMORY.md)
- [Semantic DB Invariants](references/SEMANTIC_DB_INVARIANTS.md)
- [Review Batch Planning](references/REVIEW_BATCH_PLANNING.md)

## Local Review UI

The local Semantic Review Workbench helps validate clean_v3 review batches,
candidate DB categories, hypotheses, error trends, and evidence scores without
internet access or heavy dependencies. It is audit-only and does not modify
`manual_labels.yaml`.

Build the static fallback:

```powershell
python -m vam_timeline_ai.cli build-static-review-ui ^
  --run-dir data\runs\clean_v3 ^
  --review-dir data\runs\clean_v3\audits\semantic_review_010_v16 ^
  --out-dir data\runs\clean_v3\audits\semantic_review_010_v16\review_ui_static
```

Open `review_ui_static\index.html`, review the cards, then export answers from
the Export Answers tab. To append UI answers into the audit ledger:

```powershell
python -m vam_timeline_ai.cli ingest-review-ui-answers ^
  --answers data\runs\clean_v3\audits\semantic_review_010_v16\human_review_ui_answers.jsonl ^
  --review-dir data\runs\clean_v3\audits\semantic_review_010_v16 ^
  --out-ledger data\runs\clean_v3\audits\human_review_ledger.jsonl ^
  --report data\runs\clean_v3\audits\review_ui_answer_ingestion_report.md
```

References:

- [Local Review UI](references/LOCAL_REVIEW_UI.md)
- [Review Answer Ingestion](references/REVIEW_ANSWER_INGESTION.md)

## Cowgirl Pose Subtypes

Cowgirl pose semantics now distinguish frontal lean-forward, upright/kneeling,
squat, and frontal lean-back supported contexts. `cowgirl_lean_back_supported`
means the rider leans backward while remaining in a frontal Cowgirl relation,
with hands behind her body for support, often on the partner legs or thighs.
This is not reverse Cowgirl unless facing-away evidence is explicit.

Support/contact labels also distinguish `hands_behind_support` and
`hands_on_partner_legs_or_thighs`; these are not treated as hands-free or
unknown support when evidence is present. All such labels remain audit
calibration candidates until manually reviewed.

References:

- [Cowgirl Lean-back Supported Pose](references/COWGIRL_LEAN_BACK_SUPPORTED_POSE.md)
- [Cowgirl Support Contexts](references/COWGIRL_SUPPORT_CONTEXTS.md)
- [Front vs Reverse Cowgirl](references/FRONT_VS_REVERSE_COWGIRL.md)

## ML Baseline Is Review-Assist Only

Cowgirl ML Baseline v1 trains only on human-reviewed audit artifacts and uses
grouped scene-level splits to avoid repeated-window leakage. Weak, silver,
machine, and heuristic labels are not ground-truth targets.

The model may rank candidates for review and highlight model-vs-heuristic
disagreements. It must not write `manual_labels.yaml`, self-label the dataset,
choose generation-safe clips automatically, or generate Timeline animation.

References:

- [Cowgirl ML Baseline V1](references/COWGIRL_ML_BASELINE_V1.md)
- [ML Review Assistant Workflow](references/ML_REVIEW_ASSISTANT_WORKFLOW.md)

## Local Visual Judge

The local Visual Judge pipeline is a review-assist layer for VaM screenshots,
contact sheets, GIFs, and MP4 previews. It prefers real VaM captures over
digital-twin skeleton previews and uses local LM Studio only; no cloud APIs are
called and no models are downloaded by this project.

The preferred model for current experiments is `nsfwvision-v4-qwen3.5-9b`.
Generic `Qwen2.5-VL-7B-Instruct` is documented only as a fallback because user
testing showed unreliable VaM pose/family classification.

Visual judge output is never ground truth by itself. Human review remains the
only truth source, and the visual judge is used only to prioritize or explain
manual review.

References:

- [LM Studio VLM Judge](references/LMSTUDIO_VLM_JUDGE.md)
- [Visual Judge Result Schema](references/VISUAL_JUDGE_RESULT_SCHEMA.md)
- [Visual Judge Model Calibration](references/VISUAL_JUDGE_MODEL_CALIBRATION.md)
- [VaM Reality Capture Bridge](references/VAM_REALITY_CAPTURE_BRIDGE.md)
- [Multisignal Triage With VLM](references/MULTISIGNAL_TRIAGE_WITH_VLM.md)

## Top-Down Semantic Motion Architecture

Generation now starts from semantic meaning, not controller movement. The
intended direction is prompt -> motion intent -> pose/role/partner/contact
requirements -> motion grammar -> parameterized relative motion -> controller
curves -> Timeline export.

Bottom-up features, heuristic candidate DBs, the Cowgirl ML ranker, and VLM
judge outputs are review/calibration aids only. They do not define truth and do
not write `manual_labels.yaml`. Controller curves are the output renderer. VaM
Person/root/world transforms are forbidden generation targets; "root" in motion
notes means pelvis/hip/abdomen body controls.

References:

- [Semantik Master Sourcebook](references/SEMANTIK_MASTER_SOURCEBOOK.md)
- [Architecture Reset: Top-Down Semantics](references/ARCHITECTURE_RESET_TOP_DOWN_SEMANTICS.md)
- [Semantic Motion Ontology](references/SEMANTIC_MOTION_ONTOLOGY.md)
- [Motion Intent Translator](references/MOTION_INTENT_TRANSLATOR.md)
- [Pose-First Semantic Resolver](references/POSE_FIRST_SEMANTIC_RESOLVER.md)
- [Pose Motion Grammar](references/POSE_MOTION_GRAMMAR.md)
- [Motion Parameter Calibration](references/MOTION_PARAMETER_CALIBRATION.md)

## Semantic Stickman Previews

Semantic Stickman previews are schematic ontology sanity checks. They do not use
VaM, source-scene world coordinates, Person/root tracks, or native Timeline
export. They test whether the top-down grammar can visually express:

concept -> pose -> primary driver -> followers -> anchors/support -> motion.

The current v2 gallery is written locally to
`data/runs/clean_v3/generation/semantic_stickman_previews_v2/index.html`.
It labels actor bodyparts, partner reference points, pelvis alignment targets,
support/contact targets, and bed/floor anchors so Cowgirl/Doggy/BJ/Missionary
concepts do not look like motion floating in empty space.

The contact-aware v3 gallery is written to
`data/runs/clean_v3/generation/semantic_stickman_previews_v3/index.html`.
v3 builds motion examples around interaction constraints first, then renders the
stickman pose. It shows the contact/alignment tolerance zone, target distance,
and valid/invalid alignment status for each concept.

References:

- [Semantic Stickman Previews](references/SEMANTIC_STICKMAN_PREVIEWS.md)
- [Ontology Visual Sanity Check](references/ONTOLOGY_VISUAL_SANITY_CHECK.md)

## First Generated Motion Review

The first generated motion review layer retargets synthesized relative motion onto a synthetic neutral controller baseline. Retargeting applies generated deltas to the baseline pose, keeps foot/knee anchors stable, validates distances and jumps, and renders technical previews.

Any Timeline-style artifact from this layer is review-only. It does not use source scene world coordinates, does not include Person/root tracks, does not stitch clips, and does not claim production readiness.

The current review-flow JSON is not native VaM Timeline plugin JSON and is not importable through Timeline. To test generated motion in VaM, use the generated `GeneratedMotionReviewPlayer.cs` script with the review-player JSON, which applies relative deltas from the current controller baseline.

References:

- [Relative Flow Retargeting V0](references/RELATIVE_FLOW_RETARGETING_V0.md)
- [First Generated Motion Review](references/FIRST_GENERATED_MOTION_REVIEW.md)

## Public GitHub Safety

This repository is public and should contain code, docs, tests, schemas, templates, and lightweight folder placeholders only. Generated local data, raw VaM scenes, baked arrays, previews, model files, and human labels must stay out of Git.

Before pushing:

```powershell
python -m vam_timeline_ai.cli audit-repo-safety ^
  --project-root . ^
  --out data\runs\clean_v2\audits\repo_safety_report.md
```

See [GITHUB_REPO_DATA_SAFETY.md](references/GITHUB_REPO_DATA_SAFETY.md).

## Tests

Normal development uses pytest from the dev dependency group:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests
```

The code and tests do not require local VaM scene data for unit tests. Synthetic tests cover source records, baked sample audit, controller mapping, feature extraction v1, weak/manual label separation, review queue de-duplication, and leakage-aware dataset metadata.

## Reference Projects

Configured local references are intentionally not stored in the public README.
Use environment-specific paths for:

- a technical compiler reference checkout
- the Timeline source reference checkout
- local raw VaM scenes
- optional local bridge/runtime references

These references should be treated as read-only project setup and kept out of committed run databases.

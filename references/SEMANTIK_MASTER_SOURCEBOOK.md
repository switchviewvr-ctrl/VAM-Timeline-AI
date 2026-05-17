# Semantik Master Sourcebook

`Semantik_Master_Konsolidiert.docx` is now the canonical sourcebook for the top-down motion ontology. Keep the local sourcebook path private; the repository stores only extracted trace metadata and derived ontology files.

The extracted local trace is:

- `data/ontology/sourcebooks/semantik_master_konsolidiert_extracted_v1.txt`
- `data/ontology/sourcebooks/semantik_master_konsolidiert_manifest_v1.json`
- `data/ontology/sourcebook_trace_v2.yaml`

The derived current ontology layer is v2:

- `data/ontology/motion_families_v2.yaml`
- `data/ontology/motion_subtypes_v2.yaml`
- `data/ontology/pose_subtypes_v2.yaml`
- `data/ontology/contact_support_v2.yaml`
- `data/ontology/motion_phrases_v2.yaml`
- `data/ontology/pose_first_motion_rules_v2.yaml`
- `data/ontology/exclusion_rules_v2.yaml`

Important interpretation rule: when the sourcebook says root, root node, pelvis root, or root-driven, this project maps that to pelvis/hip/abdomen body controllers. It never maps to the VaM Person atom root, world transform, or scene transform.

The sourcebook defines meaning. Review data, ML rankers, VLM judgments, and heuristic candidate DBs only calibrate or rank examples. They are not truth and must not be written into `manual_labels.yaml`.

## What Changed In V2

V2 moves more of the sourcebook into data:

- Axis priorities for X/Y/Z and pitch/yaw/roll.
- Driver/follower/anchor hierarchy per family.
- Hand state machine: `IK_Locked_Static`, `IK_Locked_Dynamic`, `FK_Soft_Floating`.
- Lower-body anchor and leg-state concepts.
- Cowgirl variants: bounce, grind, corkscrew, sequential spine wave, lean-forward, lean-back.
- Reverse Cowgirl as explicit inverted-yaw/back-to-partner, not just lean-back.
- Doggy as Z-axis quadruped/elevated/vertical-kneeling with partner-behind context.
- BJ/oral as top-down head/chest driver with pelvis static.
- Missionary as supine reactive/counter-motion with leg hooking and impact absorption.
- Edge cases and exclusion rules are encoded as anomaly guards rather than positive truth.

## Safe Usage

Use v2 as the source for prompt-to-intent and semantic resolver work:

```powershell
python -m vam_timeline_ai.cli translate-motion-intent-v1 ^
  --prompt "cowgirl zurueckgelehnt, haende auf seinen oberschenkeln" ^
  --ontology data\ontology\motion_families_v2.yaml ^
  --phrases data\ontology\motion_phrases_v2.yaml ^
  --out data\runs\clean_v3\generation\motion_intent_plan_lean_back_example_v2.json
```

This creates a motion intent plan only. It does not generate Timeline animation.

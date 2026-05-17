# Rig Anatomy Ontology v1

This layer maps human semantic body regions to VaM controller names.

Key rule:
- `pelvis_hip` / lower-body-core semantic motion maps primarily to `hipControl`.
- `pelvisControl` is a secondary/follower/static controller for Cowgirl and Reverse Cowgirl, not the sole primary driver.
- `Person`, `root`, world, and scene transforms are forbidden as generation targets.

The files are:
- `data/ontology/rig_anatomy_v1.yaml`
- `data/ontology/rig_anatomy_roles_by_family_v1.yaml`

This ontology is used for analysis, feature grouping, future NLP resolution, and export guardrails. It does not create labels and does not generate Timeline clips.

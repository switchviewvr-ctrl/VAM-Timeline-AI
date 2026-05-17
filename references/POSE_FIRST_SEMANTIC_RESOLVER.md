# Pose-First Semantic Resolver

The pose-first resolver combines pose family/subtype, partner relation, primary motion center, target/contact region, motion shape, and exclusion rules.

It resolves old bottom-up candidates into ontology-aligned records with conflict flags and explanations. This helps answer "why Cowgirl" or "why not Cowgirl".

It treats existing heuristic, ML, and VLM fields as evidence, not truth.

## Manual GT V4 Expectations

`manual_gt_timeline_examples_v4` is the accepted controller-role reference for generation-facing resolver expectations.

- Cowgirl/Reverse Cowgirl clean motion still resolves semantically as `pelvis_hip`, but VaM export should target `hipControl` as the primary visible driver.
- `pelvisControl` alone is not sufficient evidence for a generation-ready Cowgirl driver mapping.
- BJ/oral requires head/chest driver evidence and should keep hip/pelvis static in generation plans.
- Handjob requires hand driver evidence and should keep hip/pelvis static in generation plans.
- Doggy receiver-response should not become Cowgirl bounce just because hip/pelvis responds.
- Static anchors in accepted manual GT examples must remain static in both position and rotation.

The resolver should keep using semantic motion centers, but downstream generation must apply the accepted v4 VaM controller mapping.

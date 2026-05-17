# Semantic Motion Ontology

The ontology defines motion meaning before controller curves are generated. Families such as Cowgirl, Doggy, BJ/oral, Missionary, hand interaction, and receiver response are described through roles, pose requirements, partner relation, primary motion centers, followers, anchors, contact/support, and exclusions.

Current canonical source: `Semantik_Master_Konsolidiert.docx`, registered in [Semantik Master Sourcebook](SEMANTIK_MASTER_SOURCEBOOK.md). The derived current data layer is `data/ontology/*_v2.yaml`. The earlier v1 files remain useful implementation history, but v2 is the sourcebook-aligned layer.

The key design rule is pose plus driver plus relation. A Cowgirl pose without pelvis/hip motion is a pose context, not clean Cowgirl motion. Kneeling alone is not Doggy. Lean-back Cowgirl is still frontal Cowgirl unless back-to-partner evidence exists.

Ontology data lives in `data/ontology`.

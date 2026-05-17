# Component-Based Motion Ontology

The component ontology represents prompt meaning as composable parts:
- base state: family, pose subtype, actor/partner roles, required anchors
- action constraints: hold, support, touch, stroke
- motion profile: slow, fast, teasing, impact
- pose modifiers: lean forward, lean back, upright
- sequence phases: prompt-driven ordered phases with transitions

This lets prompts such as:

`10 seconds fast cowgirl lean forward into 10 seconds slow teasing cowgirl upright`

become a structured intent plan with two phases. The plan is not a Timeline export. It is a semantic input object for later review and generation work.

File:
- `data/ontology/component_ontology_v1.yaml`

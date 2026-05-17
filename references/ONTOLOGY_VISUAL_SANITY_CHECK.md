# Ontology Visual Sanity Check

The sourcebook ontology defines motion meaning before controller curves. The stickman preview layer tests whether the system can express that meaning visually without VaM, source scenes, world coordinates, or Timeline export.

This is useful before generation because it exposes semantic misunderstandings early:

- wrong primary driver
- missing anchors
- confusing pose context with clean motion
- treating lean-back Cowgirl as reverse Cowgirl
- treating hand/head motion as Cowgirl
- treating BJ/oral as pelvis-driven motion

The previews are intentionally simple. If a concept looks wrong in stickman form, the ontology or translator should be corrected before attempting any VaM Timeline generation.

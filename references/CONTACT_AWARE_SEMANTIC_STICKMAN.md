# Contact-Aware Semantic Stickman

Semantic Stickman v3 treats partner-relative interaction targets as constraints, not decoration.

Pose shape alone is not enough. A Cowgirl-like stickman is not semantically valid unless the rider pelvis is close to the partner pelvis target and the support/contact relation is plausible. The same rule applies to Doggy, BJ/Oral, and Missionary: the target relation must be physically visible in the schematic preview.

## v3 Rules

- Cowgirl and Reverse Cowgirl require rider pelvis alignment to `partner_pelvis_target`.
- Lean-back Cowgirl remains front Cowgirl unless explicit back-to-partner evidence exists.
- Doggy requires front support plus a visible partner-behind pelvis reference.
- BJ/Oral requires head/chest motion toward the partner pelvis target while actor pelvis stays a static base.
- Missionary requires a supine receiver and a close partner-above/body relation.
- Floating or far-offset previews are invalid semantic previews.

The renderer shows a shaded contact zone, target-distance overlay, support/contact lines, and validation metadata for each concept.

This is still an ontology sanity check only. It is not VaM Timeline generation and does not use Person/root/world transforms.

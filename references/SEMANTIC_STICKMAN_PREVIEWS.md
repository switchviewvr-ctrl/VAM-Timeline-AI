# Semantic Stickman Previews

Semantic Stickman previews are schematic visual sanity checks for the top-down motion ontology.

They render:

`semantic concept -> stickman pose -> driver/follower/anchor motion preview`

They are not VaM production controller targets, not native Timeline exports, and not final animation generation.

## What To Check

- Cowgirl should visibly use pelvis/hip as the driver, with feet/knees/hands as anchors/supports and chest/head as delayed followers.
- BJ/oral should visibly use head/chest as the top-down driver while the pelvis remains static.
- Doggy should visibly use all-fours or equivalent support distribution with Z-axis pelvis motion.
- Missionary should be supine with chest/head low or grounded and pelvis reactive/counter-moving.
- Lean-back Cowgirl should remain front Cowgirl unless back-to-partner/facing-away evidence exists.

## Outputs

The v2 gallery is written to:

`data/runs/clean_v3/generation/semantic_stickman_previews_v2/index.html`

The contact-aware v3 gallery is written to:

`data/runs/clean_v3/generation/semantic_stickman_previews_v3/index.html`

v2 adds explicit semantic context that v1 intentionally kept minimal:

- bodypart labels for pelvis, chest, head, hands, knees, feet, and thighs.
- partner reference labels for partner pelvis, chest, head, thighs, and legs.
- interaction/alignment target markers.
- rider pelvis to partner pelvis alignment vector for Cowgirl/Reverse Cowgirl.
- support/contact lines for hands, floor/bed anchors, partner chest, and partner legs/thighs.
- bed/floor plane and unsupported/floating warnings.

v3 adds interaction constraints:

- shaded partner-pelvis contact/alignment zone.
- target distance measurement.
- valid/invalid alignment overlay.
- failed constraint reporting in metadata and gallery.

Each concept has:

- `preview.gif`
- `contact_sheet.png`
- `metadata.json`
- per-frame PNGs

Human review decides whether the schematic meaning is correct.

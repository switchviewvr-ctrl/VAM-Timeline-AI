# Semantic DB Invariants

The global semantic candidate DB and Cowgirl DB have strict invariants.

Core rules:

- `semantic_family` must be present.
- `unknown` records cannot be generation-safe.
- BJ/oral records must be preserved as BJ/oral candidates, not discarded.
- Cowgirl generation-safe records must be Cowgirl, not BJ/oral, receiver
  response, standing gestures, low-motion holds, or intro/setup phases.
- Contact claims such as hands-on-partner-chest require contact evidence.
- Anchor or pose unsafe records must not be generation-safe.

Run:

`python -m vam_timeline_ai.cli validate-semantic-dbs ...`


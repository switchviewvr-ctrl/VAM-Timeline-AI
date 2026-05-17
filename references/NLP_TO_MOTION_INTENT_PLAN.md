# NLP To Motion Intent Plan

The v1 NLP resolver is intentionally conservative.

It performs:
- phrase matching against the active lexicon
- ordered style and pose modifier extraction
- target/action extraction, for example hands holding partner shoulders
- phase construction for `then`, `dann`, and `into`
- unresolved requirement reporting

It does not:
- train ML
- auto-label
- modify `manual_labels.yaml`
- export Timeline clips
- move VaM controllers

CLI examples:

```powershell
python -m vam_timeline_ai.cli resolve-nlp-tokens-v1 --prompt "Die Frau hält sich an den Schultern des Mannes fest und reitet erst langsam, dann schnell" --lexicon data\ontology\nlp_lexicon_v1.yaml --component-ontology data\ontology\component_ontology_v1.yaml --out data\runs\clean_v3_new_scenes\nlp\token_resolution_example_v1.json
```

```powershell
python -m vam_timeline_ai.cli build-motion-intent-from-prompt-v1 --prompt "10 seconds fast cowgirl lean forward into 10 seconds slow teasing cowgirl upright" --lexicon data\ontology\nlp_lexicon_v1.yaml --component-ontology data\ontology\component_ontology_v1.yaml --out data\runs\clean_v3_new_scenes\nlp\motion_intent_example_v1.json
```

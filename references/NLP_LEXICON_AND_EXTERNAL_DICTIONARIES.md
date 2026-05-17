# NLP Lexicon And External Dictionaries

The NLP lexicon translates prompt words into ontology tokens:
- anatomy terms, for example `hip`, `hüfte`, `head`, `hand`
- actions, for example `hold`, `festhalten`, `stützen`, `stroke`
- pose modifiers, for example `lean forward`, `zurückgelehnt`, `upright`
- families, for example `cowgirl`, `bj`, `handjob`, `doggy`
- style modifiers, for example `slow`, `langsam`, `fast`, `schnell`, `teasing`

Manual lexicon entries are active. External web-derived terms are always inactive candidates until human review accepts them.

Files:
- `data/ontology/nlp_lexicon_manual_v1.yaml`
- `data/ontology/lexicon_sources_v1.yaml`
- generated merged output: `data/ontology/nlp_lexicon_v1.yaml`

Safety:
- VLM and web text are never truth.
- No manual labels are modified.
- No Timeline output is produced by lexicon commands.

# clean_v3 Semantic Rescan

clean_v3 keeps clean_v2 technical artifacts where schema-compatible and rebuilds
semantic artifacts around the fuller action model:

- pose semantics
- partner-relative interaction features
- contact/support semantics
- Semantic Actions
- multi-family Semantic Candidate DB
- Cowgirl DB v5
- motion primitives v1 with pose/contact requirements

BJ/oral remains a valid semantic family. It is excluded from Cowgirl generation
sets only when Cowgirl-specific filtering is being built.

No ML training, no manual label mutation, no source-world generation targets.

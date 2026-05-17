# Context And Semantics Roadmap To Prompt

This roadmap establishes the missing layer between semantic analysis and future Prompt-to-Timeline generation.

Order of responsibility:
1. Rig anatomy defines what body words mean in VaM controller terms.
2. Motion ontology defines family meaning and driver/follower/anchor roles.
3. NLP lexicon maps user language to semantic tokens.
4. Component ontology builds ordered motion-intent plans.
5. Motion cycle and biomechanical gates keep clean motion separate from transitions, holds, crawling, reaching, and missing-controller artifacts.
6. Only after these layers are trusted should Prompt-to-Timeline be attempted.

Accepted guardrails:
- Cowgirl visible driver is `hipControl`.
- `pelvisControl` is never the sole Cowgirl driver.
- feet/hands/knees remain anchors unless semantics explicitly allow motion.
- rotations are required for VaM Timeline export.
- sparse keyframes are the future export default.
- manual human review remains final truth.

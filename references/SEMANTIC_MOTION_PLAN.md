# Semantic Motion Plan

A semantic motion plan is the internal representation between a prompt and future generated motion.

The intended flow is:

```text
Text prompt -> semantic motion plan -> generated relative motion flow -> VaM Timeline animation
```

The plan contains phases such as intro, clean motion, transition, hold, and outro. Each phase has a primitive query with family, subtype, trajectory shape, tempo, intensity, depth, amplitude, duration range, and safety requirements.

The current `draft-motion-plan-v0` command is rule-based and only produces this internal JSON plan. It is not final text-to-animation.

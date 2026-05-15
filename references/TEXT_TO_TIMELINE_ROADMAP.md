# Text To Timeline Roadmap

The project target is generative motion understanding:

```text
Text prompt
-> semantic motion plan
-> generated relative motion flow
-> VaM Timeline animation
```

Current stage:

- Build reliable semantic candidate DBs.
- Extract relative motion primitives.
- Group primitive families.
- Draft internal semantic plans.
- Retrieve primitives for inspection.
- Create motion-flow skeletons.

Future missing pieces:

- Relative curve synthesis from primitive parameters.
- Pose/current-scene retargeting.
- Anchor/contact constraint solving.
- Controller validity validation after synthesis.
- Timeline export safety validation.

The final system must never blindly copy raw Timeline coordinates, Person/root transforms, source-scene poses, or stitch old clips as the final architecture.

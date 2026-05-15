# Body Motion Quality And Root Filtering

The first VaM semantic review showed that some high-confidence machine labels were not real body/extremity motion. They were whole-controller or whole-person/root-like transforms.

Final generated VaM Timeline animation must not animate:

- Person atom transform
- atom/root/world transform
- scene-level placement
- whole-person controller motion

Allowed final animation targets are real bodypart controllers such as:

- `hipControl`
- `abdomenControl`
- `chestControl`
- `headControl`
- `lHandControl`
- `rHandControl`
- `lKneeControl`
- `rKneeControl`
- `lFootControl`
- `rFootControl`

`audit-body-motion-quality` classifies windows as:

- `good_body_motion`
- `partial_body_motion`
- `root_only_motion`
- `controller_only_whole_person_motion`
- `static_or_empty`
- `unknown`

Root/controller-only windows are useful as failure examples, but they must be suppressed as clean Cowgirl positives. Transition and adjustment windows are also valuable audit material, but should not be treated as clean repetitive riding motion.

# Visual Judge Result Schema

The visual judge schema is evidence-first. It asks for visible pose, torso lean,
facing/orientation, partner visibility, motion visibility, dominant motion, and
contact/support before any family suggestion.

Critical rule:

If the visual input is a single frame and no partner is visible and no motion is
visible, the normalized result must not claim Cowgirl, Doggy, BJ/oral, or any
other interaction family. It can describe pose context, but
`suggested_family` is normalized to `unknown` with confidence at most `0.35`.

This prevents single-image hallucinations from becoming review or training
truth.

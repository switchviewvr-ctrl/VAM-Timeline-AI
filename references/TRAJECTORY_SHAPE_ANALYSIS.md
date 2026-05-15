# Trajectory Shape Analysis

Pelvis and hip trajectory shape is a useful audit signal for Cowgirl and grinding motion.

The trajectory-shape layer uses relative pelvis/hip motion and computes:

- path length
- start/end displacement
- closed-loop ratio
- ellipse and oval scores
- circularity
- linearity
- jitter/static score
- dominant motion plane
- cycle/repetition estimate
- grind, bounce, forward/back rock, and transition scores

Interpretation is still an audit hypothesis:

- Grinding often has an oval, elliptical, circular, or repeated curved local path.
- Vertical bounce is more one-axis vertical repetition.
- Forward/back rock is more linear or arced forward/back repetition.
- Transition motion tends to have high start-to-end displacement and weak repetition.
- Jitter/static paths have tiny area or displacement.

These features improve review selection, but they are not human ground truth and must not be merged into manual labels.


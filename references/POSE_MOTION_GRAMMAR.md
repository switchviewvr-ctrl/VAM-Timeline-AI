# Pose Motion Grammar

Generation uses motion grammar:

- primary driver: the body region that carries semantic motion
- followers: body regions that lag/dampen/follow
- anchors/support: controllers that stabilize contact
- coordinate frame: partner-local or body-relative motion space
- exclusions: motions that must not be confused with the target family

For Cowgirl, pelvis/hip/thighs are primary. Abdomen/chest/head follow. Feet/knees/hands are anchors/supports. Person/root/world transforms are forbidden animation targets.

# Motion Intent Translator

The motion intent translator turns a prompt into a top-down `MotionIntentPlan`.

Example:

`cowgirl zurückgelehnt, hände auf seinen oberschenkeln`

maps to Cowgirl with `cowgirl_lean_back_supported`, front-facing context, backward torso lean, and hand support on partner legs/thighs. It must not become reverse Cowgirl unless the prompt explicitly says reverse or back-to-partner.

The translator is not a generator. It produces the semantic plan that future generation can consume.

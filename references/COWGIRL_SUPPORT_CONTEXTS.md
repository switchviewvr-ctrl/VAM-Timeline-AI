# Cowgirl Support Contexts

Cowgirl support/contact is part of the semantic action, separate from motion.

Supported contexts now include:

- `hands_on_partner_chest`
- `hands_on_partner_hips`
- `hands_on_partner_legs`
- `hands_on_partner_thighs`
- `hands_on_partner_legs_or_thighs`
- `hands_behind_support`
- `hands_behind_on_floor_or_bed`
- `ambiguous_behind_support`
- `hands_free`

Behind-body support must not be collapsed into hands-free. Partner leg/thigh support requires partner lower-body target evidence; otherwise the classifier should emit possible or ambiguous support rather than overclaiming contact.

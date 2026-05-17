# Interaction Alignment Constraints

Interaction families are defined by body pose plus target alignment.

## Cowgirl

- Actor anchor: rider `pelvis`
- Partner target: `partner_pelvis_target`
- Requirement: rider pelvis stays inside the contact/alignment tolerance zone.
- Support: knees, feet, or hands must visibly anchor the pose.
- Invalid: pelvis floating far above or away from the partner target.

## Reverse Cowgirl

- Same pelvis alignment as Cowgirl.
- Additional orientation requirement: `back_to_partner` or clear facing-away evidence.
- Leaning backward alone does not imply Reverse Cowgirl.

## Doggy

- Actor anchor: receiver/front actor pelvis.
- Partner target: partner-behind pelvis reference.
- Requirement: front support plus close behind relation.
- Invalid: kneeling or bent stickman without partner-behind context.

## BJ/Oral

- Actor anchor: head/chest path.
- Partner target: `partner_pelvis_target`.
- Requirement: head/chest path points to the target while actor pelvis remains static/base.
- Invalid: pelvis-driven riding motion.

## Missionary

- Actor anchor: receiver pelvis in supine pose.
- Partner target: partner-above/front pelvis reference.
- Requirement: close supine body relation with chest/head low.

All visual checks remain review-assist only. Human review decides whether the semantic concept is correct.

# Cowgirl ML Baseline v1

Cowgirl ML Baseline v1 is a small supervised review-assist ranker.

It is not a generative model, not a text-to-animation system, and not a
source of automatic truth labels. Targets are derived only from human review
artifacts such as the human review ledger and review UI answers.

## Targets

- `cowgirl_candidate`
- `cowgirl_clean_motion`
- `cowgirl_generation_safe_candidate`, only when enough human labels exist

Unknown labels are excluded per target. Weak, silver, machine, and heuristic
labels are never used as ground-truth targets.

## Split Rule

Splits are grouped by scene by default. Random window splits are invalid
because adjacent windows from the same source leak motion and pose context.

## Intended Use

- Review ranking: allowed
- Candidate DB scoring: experimental review assist only
- Automatic labeling: not allowed
- Generation-safe selection without human review: not allowed
- Timeline generation: not part of this model

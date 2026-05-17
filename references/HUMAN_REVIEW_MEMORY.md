# Human Review Memory

The human review ledger collects structured findings from semantic review
folders, human notes, and answer sheets.

It is a calibration memory, not a training label store. Records preserve:

- system semantic/pose/motion/contact guesses
- human corrections when known
- audit-only error tags
- notes and verdicts

The ledger must not be merged into `manual_labels.yaml`. Its purpose is to
explain why calibration rules changed and to avoid forgetting repeated review
failures such as BJ/oral-as-Cowgirl, low-motion pose context, and contact target
overconfidence.


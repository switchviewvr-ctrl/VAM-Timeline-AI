# Multisignal Triage With VLM

Signal hierarchy:

1. Human review = truth.
2. Visual judge = review-assist signal.
3. Cowgirl ML ranker = review-assist signal.
4. Heuristic semantic pipeline = review-assist signal.

Rules:

- Human answer overrides all other signals.
- Disabled or dry-run VLM results are not used as family labels.
- If `evidence_sufficient_for_family=false`, VLM family guesses are ignored.
- Heuristic + ML + trusted VLM agreement can become `spot_check`.
- VLM disagreement with heuristic/ML becomes `must_review`.
- Contact/support disagreement becomes `must_review`.

The triage output is a review-priority queue, not a label file.

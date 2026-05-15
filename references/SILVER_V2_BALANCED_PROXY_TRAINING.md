# Silver v2 Balanced Proxy Training

Silver v2 labels are generated from aggregated machine-label scores. They are not human ground truth.

Silver v2 differs from silver v1:

- built from aggregated score rows, not raw proposals
- much less duplicated
- separates window labels from pair-window labels
- stores final score and evidence summary
- excludes `contact_unknown` from positive training labels
- marks role labels such as `rider_active` and `partner_context_static` as high-risk

Dataset v4 stores:

- manual positive/negative/uncertain labels, if any
- weak labels
- silver v2 window labels
- silver v2 pair labels
- silver scores
- default trainable silver label mask
- excluded labels and reasons

The balanced silver baseline v1 uses grouped scene/sample/source splits and balanced positive/negative sampling. It never uses random overlapping-window splits.

If scikit-learn is unavailable, the command can use a small NumPy fallback:

- median imputation
- feature standardization
- one-vs-rest ridge-style linear classifier
- grouped train/test split
- balanced positives and negatives per label

This baseline is only a proxy sanity check. Because silver labels come from feature rules, a model can learn the rules without learning real semantics. Metrics must be read as rule/proxy reproducibility, not semantic accuracy.

Use `batch_machine_review_002` to inspect the corrected v2 suggestions. The manual stub remains empty and must be edited by a human before any real supervised semantic learning can begin.

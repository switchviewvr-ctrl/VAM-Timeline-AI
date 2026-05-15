# Weak Label Calibration v2

Weak labels are review hints. They are not semantic truth and must stay separate from manual labels.

v1 weak labels were intentionally broad, producing many labels per window. v2 calibrates the most useful proxies using feature distributions and percentile thresholds so review batches can be less noisy.

## Namespace

All calibrated labels use the `weak_v2_` prefix. The calibration step must not create labels such as `cowgirl_deep_slow`; those are manual semantic labels only.

## Calibration Inputs

- Cowgirl/Riding feature v1 JSONL
- old weak label v1 JSONL for count/co-occurrence comparison

## Outputs

- `data/semantic/weak_labels_v2.jsonl`
- `data/semantic/weak_label_calibration_report_v2.md`

The report includes old/new counts, threshold values, broad labels, co-occurrences, and suggested manual review quotas.

## Human Review

Use v2 weak labels to select diverse manual-review windows. Do not copy them into `manual_labels.yaml` unless a human confirms the behavior from previews or VaM visual review.

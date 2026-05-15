# Machine Label Proposals v1

Machine label proposals are Codex-generated suggestions from numeric features and pair/context proxies.

They are not manual labels and not human ground truth. Every proposal carries:

- `source`: `machine_rule_v1`, `machine_pair_rule_v1`, or `machine_cluster_rule_v1`
- `confidence`
- `rule_id`
- evidence feature names and values
- `is_human_ground_truth: false`

The proposal generator uses current clean-run feature distributions and percentile thresholds. Existing `weak_v2_...` labels may support confidence, but weak labels are not copied as final labels.

The rules avoid atom-name and filename truth. Technical atom IDs and filenames can appear in records for traceability, but they do not create semantic labels.

Outputs:

```powershell
data\runs\clean_v2\labels\machine_proposals\machine_label_proposals_v1.jsonl
data\runs\clean_v2\labels\machine_proposals\machine_label_proposals_v1.yaml
data\runs\clean_v2\labels\machine_proposals\machine_label_proposals_report_v1.md
```

Use the report to see proposal counts, silver-candidate counts, confidence bins, thresholds, and proxy-only warnings.

No visual inspection is performed by this command. If preview images are not explicitly inspected by a human, the system must not claim visual semantic certainty.

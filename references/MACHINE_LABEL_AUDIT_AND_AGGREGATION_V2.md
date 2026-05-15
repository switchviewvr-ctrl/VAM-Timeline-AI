# Machine Label Audit And Aggregation v2

The v1 machine-label workflow produced hundreds of thousands of raw proposals. That is expected: the proposal generator is designed to be generous so it can surface review candidates. The problem is that raw proposals are not a training dataset.

The main source of inflation is pair-context multiplication. One movement window can appear in many pair windows, so role/contact candidates such as `rider_active` and `partner_context_static` can repeat many times for the same actor window.

v2 adds two steps before silver labels:

1. `audit-machine-labels-v1`
   - counts proposals and silver records
   - reports proposal distributions per window and pair window
   - detects duplicate proposal keys
   - detects contradictory labels such as fast/slow, pause/fast, or rider/context on the same window
   - lists labels dominated by pair-window multiplication

2. `aggregate-machine-labels-v2`
   - groups raw proposals by `window_id + label`
   - groups pair proposals by `pair_window_id + label`
   - computes max/mean confidence, rule count, evidence count, and pair-context support
   - caps pair-context evidence with a log-scale contribution
   - penalizes conflicts
   - writes one canonical score row per window-label or pair-label

Aggregation makes machine labels stable enough for review and weak-supervised experiments, while keeping provenance and conflict flags visible.

Role labels remain high-risk proxies. They may be useful review hints, but they must not become semantic truth without human confirmation.

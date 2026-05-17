# Review Player Is Debug Only

`GeneratedMotionReviewPlayer.cs` was built to prove that generated relative deltas could be played safely in VaM. It remains useful for diagnosing motion quality, controller selection, reset behavior, and axis scaling.

It is not the final target.

The final target is native Timeline JSON import:

```text
generated relative motion flow -> native VaM Timeline animation JSON
```

Use the review player only when:

- debugging generated relative curves before export
- checking axis scale or follower behavior
- inspecting safety without touching Timeline import

For generated animation review, prefer native Timeline export once the exporter validates.

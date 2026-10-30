# Results

Ground truth: `data/annotations/events.csv` (local only), 31 clips, ~61 min.
27 POSITIVE events, 46 documented hard negatives, 9 IGNORE intervals.

How it was built (2026-10-30): every gesture cycle found by the pipeline was
reviewed visually on small person crops and labeled (smoking, vaping,
drinking, lollipop, phone, other, unclear). Then per-person overview sheets of
the short "smoking" clips were reviewed to add events the pipeline missed.
Long clips (`1`, `3`, `sm_1`, `2025_10_22_*`, `2025_12_08_*`) were only
reviewed around detected cycles, so they may contain unlabeled smoking:
recall is likely overestimated and FP/hour slightly pessimistic.

| run | settings | TP | FP | FN | precision | recall | F1 | FP/h |
|-----|----------|----|----|----|-----------|--------|----|------|
| try1 | near 0.6, far 1.0, threshold 0.5 | 8 | 9 | 19 | 0.47 | 0.30 | 0.36 | 9.9 |
| tuned | near 0.75, far 1.0, threshold 0.3 | 14 | 12 | 13 | 0.54 | 0.52 | 0.53 | 13.2 |

The tuned values were picked on the same clips (no held-out split yet), so
the second row is optimistic.

Main error sources seen while labeling:
- hard negatives with the same gesture: drinking from a can or bottle,
  a lollipop on a stick, touching the face, adjusting a headband;
- smokers who keep the cigarette in the mouth without a hand cycle;
- people the tracker misses or whose id switches mid-event.

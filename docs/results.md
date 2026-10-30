# Results

Data: 31 clips, ~61 min, from several fixed cameras (local only, not in the
repository). Ground truth `data/annotations/events.csv`: 27 POSITIVE events,
46 documented hard negatives, 9 IGNORE intervals.

The clips are split into **dev** (18 clips, 38 min, 15 positive events) and
**test** (13 clips, 22 min, 12 positive events). All thresholds were chosen
on dev only; test was looked at only to report the numbers below.

## Event-level metrics

Matching as in the task: GT extended by ±3 s, overlap >= 1 s, box IoU >= 0.3,
one-to-one.

| version | split | TP | FP | FN | precision | recall | F1 | FP / hour |
|---------|-------|----|----|----|-----------|--------|----|-----------|
| gestures only | dev | 8 | 6 | 7 | 0.57 | 0.53 | 0.55 | 10.2 |
| gestures + CLIP | dev | 7 | 5 | 8 | 0.58 | 0.47 | 0.52 | 8.5 |
| gestures only | **test** | 6 | 6 | 6 | 0.50 | 0.50 | 0.50 | 18.7 |
| gestures + CLIP | **test** | 6 | 4 | 6 | 0.60 | 0.50 | **0.55** | **12.5** |

CLIP threshold (`classifier.min_score: 0.1`) rule: the lowest FP/hour on dev
that loses at most one true event. On test it removed a third of the false
alarms without losing any true event.

These are small numbers: one event more or less changes F1 by ~0.04, so the
results show the direction, not a precise value.

## How the ground truth was built

Every gesture cycle found by an early version of the pipeline was reviewed
visually on small person crops and labeled (smoking, vaping, drinking,
lollipop, phone, other, unclear). Then per-person overview sheets of the
short "smoking" clips were reviewed to add events the pipeline missed.
Long clips (`1`, `3`, `sm_1`, `2025_10_22_*`, `2025_12_08_*`) were only
reviewed around detected cycles, so they may contain unlabeled smoking:
recall is likely overestimated and FP/hour slightly pessimistic.

## What did not work

- **COCO object detector** (bottle, cup, phone near the wrist): rarely finds
  small bottles on surveillance footage and "finds" phones in smoking clips.
- **Trained crop classifier** (ImageNet features + logistic regression on the
  labeled cycles): AUC 0.74 with leave-one-clip-out, but ~0.5 with
  leave-one-camera-out. It learned the camera, because most cameras show
  only smokers or only non-smokers.
- **Filtering single cycles with CLIP**: a 2-cycle event is lost when one of
  its cycles is filtered, so the filter now works on whole events.

## Main error sources

- a lollipop on a stick (CLIP sees it as a cigarette);
- smokers who keep the cigarette in the mouth without a hand cycle;
- people the tracker misses or whose id switches mid-event;
- remaining gesture look-alikes: touching the face, adjusting a headband.

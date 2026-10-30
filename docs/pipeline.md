# Pipeline

```
video ──► pose.py ──► cycles.py ──► classifier.py ──► events.py ──► events.csv + manifest.json
          people,      hand-to-mouth   CLIP: does the     event rules,      │
          tracks,      signal and      head crop look     confidence,       ▼
          keypoints    cycles          like smoking?      box at peak   evaluation.py ──► metrics
```

| step | module | what it does |
|------|--------|--------------|
| 1 | `pose.py` | Samples the video at `sample_fps`, runs YOLO pose + ByteTrack, keeps nose, shoulders and wrists per track. Results are cached in `outputs/poses/`. |
| 2 | `cycles.py` | For every track: distance from the nose to the closer wrist, in shoulder widths. |
| 3 | `cycles.py` | Finds "away -> at the mouth for 0.5-5 s -> away" with two thresholds (hysteresis). |
| 3b | `classifier.py` | Cuts a full-resolution crop around the head at three moments of the pause and asks CLIP (zero-shot, no training) how much it looks like "a person smoking / vaping" versus "drinking, eating, talking on the phone, touching the face". |
| 4 | `events.py` | Merges cycles of one track with gaps <= 15 s, keeps groups with 2 cycles in 20 s, scores them. An event is dropped if even its most smoking-like cycle scores below `classifier.min_score` (filtering single cycles instead broke too many real events, since one missed cycle is enough to lose a 2-cycle event). |
| 5 | `evaluation.py` | Event matching and metrics exactly as in the task. |

All settings live in `configs/default.yaml`. An optional region-of-interest mask
can be put in `configs/roi/<camera_id>.png` (white = analyse, black = ignore).
`tools/render_overlay.py` and the web app (`app.py`) draw tracks, cycles and events
on a clip for review (`overlay.py`).

## Why zero-shot CLIP and not a trained classifier

A small classifier (ImageNet features + logistic regression) trained on the
89 labeled cycles looked fine with leave-one-clip-out validation (AUC 0.74),
but dropped to chance level (AUC ~0.5) when a whole camera was held out:
almost every camera in the data shows only smokers or only non-smokers, so
the model learned the scene, not the object. CLIP is not trained on our data
and cannot learn the camera. On the same cycles it scores drinking, phone and
face touching far below smoking.

## Known limitations (next steps)

- A lollipop on a stick looks like a cigarette to CLIP and is not filtered.
- "1 cycle + direct cue" (cigarette, vape, smoke) is not implemented yet.
- Smokers who keep the cigarette in the mouth make no hand cycle and are missed.
- A tracker id switch splits one person's event into two.
- The COCO object filter (`objects.py`, `objects.enabled`) is off: on the
  available footage it rarely finds bottles in drinking clips and often
  "finds" phones and bottles in smoking clips.

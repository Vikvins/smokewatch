# Pipeline

```
video ──► pose.py ──► cycles.py ──► objects.py ──► events.py ──► events.csv + manifest.json
          people,      hand-to-mouth   drop cycles     event rules,      │
          tracks,      signal and      with a bottle,  confidence,       ▼
          keypoints    cycles          cup or phone    box at peak   evaluation.py ──► metrics.json
```

| step | module | what it does |
|------|--------|--------------|
| 1 | `pose.py` | Samples the video at `sample_fps`, runs YOLO pose + ByteTrack, keeps nose, shoulders and wrists per track. Results are cached in `outputs/poses/`. |
| 2 | `cycles.py` | For every track: distance from the nose to the closer wrist, in shoulder widths. |
| 3 | `cycles.py` | Finds "away -> at the mouth for 0.5-5 s -> away" with two thresholds (hysteresis). |
| 3b | `objects.py` | Runs a standard COCO detector (bottle, cup, wine glass, cell phone) on a full-resolution crop around each person's head, so small objects stay visible. A cycle with such an object next to the wrist is a hard negative (drinking, phone) and is dropped. Detections are cached in `outputs/hand_objects/`. |
| 4 | `events.py` | Merges cycles of one track with gaps <= 15 s, keeps groups with 2 cycles in 20 s, scores them. |
| 5 | `evaluation.py` | Event matching and metrics exactly as in the task. |

All settings live in `configs/default.yaml`. An optional region-of-interest mask
can be put in `configs/roi/<camera_id>.png` (white = analyse, black = ignore).

## Known limitations (next steps)

- "1 cycle + direct cue" (cigarette, vape, smoke) is not implemented yet.
- A tracker id switch splits one person's event into two.
- Thresholds are first guesses; tune them on the `dev` split only.
- The COCO object filter (`objects.enabled`) is off by default: on the
  available footage it rarely finds bottles in drinking clips and often
  "finds" phones and bottles in smoking clips (run try3, 2026-10-30).

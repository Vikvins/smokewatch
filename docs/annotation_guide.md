# Annotation guide

Ground truth is two CSV files in `data/annotations/` (git-ignored, like all data).
The format is defined in `src/smokewatch/annotations.py`.

## Workflow

1. `python tools/make_clip_index.py` builds `clips.csv` from `data/raw/`.
   Fill in `camera_id` by hand (clips showing the same scene share one id) and
   set `split` to `dev` or `test`. Split by camera, never inside one camera.
2. Watch a clip in any player that shows time with tenths of a second
   (IINA or mpv on macOS). Write one row per event into `events.csv`.
3. For each POSITIVE event, get the person box at the peak:
   `python tools/draw_box.py data/raw/smoking/1.mp4 15.0`
   and paste the printed `peak_sec,x1,y1,x2,y2` into the row.
4. `python tools/check_annotations.py` until it reports 0 problems.

## Labels

| label | when | box |
|-------|------|-----|
| `POSITIVE` | 2 full cycles within 20 s, or 1 full cycle plus a direct cue | required |
| `NEGATIVE` | a hard negative worth documenting: drinking, phone at face, eating, touching face, adjusting a mask, pen at lips, holding a cigarette without bringing it to the mouth | optional |
| `IGNORE` | cannot decide: person too small, occluded, leaves the frame mid-cycle | optional |

Everything not covered by a row counts as negative video.

## Rules for POSITIVE events

- A **cycle**: hand moves to the mouth, stays ~0.5-5 s, moves away.
- `start_sec`: the hand starts moving to the mouth in the first cycle.
- `end_sec`: the hand has moved away after the last cycle.
- Cycles of the same person with gaps of 15 s or less belong to **one** event.
- Different people are **never** merged: two smokers at once are two rows.
- `peak_sec`: the clearest moment with the hand at the mouth.
- Box: the whole visible person at `peak_sec`, in original pixels.
- `cycles`: number of full cycles you saw; `cue`: `none`, `cigarette`, `vape` or `smoke`.

## Example

```csv
clip_id,event_id,label,start_sec,end_sec,peak_sec,x1,y1,x2,y2,cycles,cue,negative_type,comment
sm_2,e01,POSITIVE,3.4,21.0,5.1,812,240,1010,760,3,cigarette,,
drinking_1,e01,NEGATIVE,10.0,14.5,,,,,,0,none,drinking,bottle
```

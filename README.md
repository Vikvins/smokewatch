# SmokeWatch

[![CI](https://github.com/Vikvins/smokewatch/actions/workflows/ci.yml/badge.svg)](https://github.com/Vikvins/smokewatch/actions/workflows/ci.yml)

Event-level detection of smoking-like behavior in video from stationary surveillance cameras.

The pipeline tracks people, estimates their pose and recognizes the hand-to-mouth gesture
cycle (hand moves to the mouth, pauses for 0.5-5 s, moves away). Cycles are combined into
events with start/end time, confidence and a person bounding box. A zero-shot CLIP check of
the head crop drops events that look like drinking, eating or a phone call. Events are
evaluated with event-level precision, recall, F1 and false positives per hour.

```
video -> YOLO11 pose + ByteTrack -> hand-to-mouth cycles -> event rules -> CLIP check -> events.csv
```

Everything runs locally on a laptop (Apple M4, `mps`): no video leaves the machine.

## Results

Held-out test split (13 clips, 22 min; thresholds chosen on the dev split only):

| version | precision | recall | F1 | false positives / hour |
|---------|-----------|--------|----|------------------------|
| gestures only | 0.50 | 0.50 | 0.50 | 18.7 |
| gestures + CLIP check | 0.60 | 0.50 | 0.55 | 12.5 |

Details, how the ground truth was built and what did not work: [docs/results.md](docs/results.md).

Based on a case by Videofor at the Sirius University hackathon (October 2026).
This repository is an independent re-implementation.

## Data

Videos are confidential and are **not** part of this repository.
Put them in `data/raw/` locally; the whole `data/` folder is git-ignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
nbstripout --install
```

## Web app

```bash
python -m streamlit run app.py
```

Pick a clip from `clips.csv` or upload a video, press **Analyze** and get a player with
tracked people, hand-to-mouth gestures and detected events drawn on the video, a table of
events (with a jump-to-event selector) and a CSV download. It runs on this machine only:
uploads go to `data/uploads/`, results to `outputs/`.

## Command line

```bash
python tools/run_pipeline.py --run baseline          # all clips in clips.csv
python tools/run_pipeline.py --run baseline --clip sm_2
python tools/evaluate.py --run baseline --split test
python tools/render_overlay.py --run baseline --clip sm_2   # video with boxes and events
```

How it works: [docs/pipeline.md](docs/pipeline.md).

## Annotation

See [docs/annotation_guide.md](docs/annotation_guide.md).

## Project layout

```
app.py        Streamlit web app
configs/      YAML configs (thresholds, model names, event rules)
data/         raw videos and annotations (local only, git-ignored)
docs/         annotation guide and other notes
notebooks/    exploration; outputs are stripped on commit
outputs/      predictions, reports, debug videos (git-ignored)
src/smokewatch/  the Python package
tests/        unit tests
tools/        command-line helpers (clip index, box drawing, annotation checks)
```

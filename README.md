# SmokeWatch

Event-level detection of smoking-like behavior in video from stationary surveillance cameras.

The pipeline tracks people, estimates their pose and recognizes the hand-to-mouth gesture
cycle (hand moves to the mouth, pauses for 0.5-5 s, moves away). Cycles are combined into
events with start/end time, confidence and a person bounding box, and evaluated with
event-level precision, recall, F1 and false positives per hour.

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

## Project layout

```
configs/      YAML configs (thresholds, model names, event rules)
data/         raw videos and annotations (local only, git-ignored)
notebooks/    exploration; outputs are stripped on commit
outputs/      predictions, reports, debug videos (git-ignored)
src/smokewatch/  the Python package
tests/        unit tests
```

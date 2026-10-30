"""Ground-truth annotation format: loading and validation.

Annotations live in two CSV files under data/annotations/ (git-ignored):

* clips.csv  - one row per video file (see CLIP_COLUMNS).
* events.csv - one row per labeled interval (see EVENT_COLUMNS).

All times are in seconds from the start of the clip, never frame indices,
because frame rates differ between cameras.
"""

from enum import Enum
from pathlib import Path

import pandas as pd


class Label(str, Enum):
    POSITIVE = "POSITIVE"  # smoking-like event
    NEGATIVE = "NEGATIVE"  # documented hard negative (drinking, phone, ...)
    IGNORE = "IGNORE"  # ambiguous; excluded from TP/FP/FN


CLIP_COLUMNS = [
    "clip_id",  # unique, filename-safe id derived from the file name
    "path",  # path relative to data/raw/
    "source",  # top-level folder in data/raw/ (e.g. smoking, hard_negatives)
    "camera_id",  # filled in by hand; clips from one camera share it
    "split",  # dev or test; split by camera to avoid leakage
    "fps",
    "duration_sec",
    "width",
    "height",
]

EVENT_COLUMNS = [
    "clip_id",
    "event_id",  # unique within a clip, e.g. e01
    "label",  # one of Label
    "start_sec",  # hand starts moving to the mouth (first cycle)
    "end_sec",  # hand has moved away (last cycle)
    "peak_sec",  # clearest moment with the hand at the mouth
    "x1",  # person box at peak_sec, in original pixels
    "y1",
    "x2",
    "y2",
    "cycles",  # number of full hand-to-mouth cycles seen
    "cue",  # none, cigarette, vape or smoke
    "negative_type",  # for NEGATIVE: drinking, phone, eating, face_touch, ...
    "comment",
]

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]
CUES = {"none", "cigarette", "vape", "smoke"}


def load_clips(path: Path) -> pd.DataFrame:
    """Read clips.csv and check that all expected columns are present."""
    clips = pd.read_csv(path, dtype={"clip_id": str, "camera_id": str})
    _require_columns(clips, CLIP_COLUMNS, path)
    return clips


def load_events(path: Path) -> pd.DataFrame:
    """Read events.csv and check that all expected columns are present."""
    events = pd.read_csv(path, dtype={"clip_id": str, "event_id": str})
    _require_columns(events, EVENT_COLUMNS, path)
    return events


def validate_events(events: pd.DataFrame, clips: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems; an empty list means valid."""
    problems = []
    clip_by_id = clips.set_index("clip_id")

    duplicated = events.duplicated(subset=["clip_id", "event_id"], keep=False)
    for _, row in events[duplicated].iterrows():
        problems.append(f"{_where(row)}: duplicated event_id")

    for _, row in events.iterrows():
        where = _where(row)

        if row["clip_id"] not in clip_by_id.index:
            problems.append(f"{where}: unknown clip_id")
            continue
        clip = clip_by_id.loc[row["clip_id"]]

        if row["label"] not in {label.value for label in Label}:
            problems.append(f"{where}: unknown label {row['label']!r}")
            continue

        if not 0 <= row["start_sec"] < row["end_sec"] <= clip["duration_sec"] + 0.5:
            problems.append(f"{where}: need 0 <= start_sec < end_sec <= clip duration")

        if row["label"] == Label.POSITIVE.value:
            problems.extend(_check_positive(row, clip, where))

    return problems


def _check_positive(row: pd.Series, clip: pd.Series, where: str) -> list[str]:
    problems = []

    if pd.isna(row["peak_sec"]) or not row["start_sec"] <= row["peak_sec"] <= row["end_sec"]:
        problems.append(f"{where}: peak_sec must lie inside [start_sec, end_sec]")

    if row[BOX_COLUMNS].isna().any():
        problems.append(f"{where}: POSITIVE event needs a person box")
    else:
        x1, y1, x2, y2 = row[BOX_COLUMNS]
        if not (0 <= x1 < x2 <= clip["width"] and 0 <= y1 < y2 <= clip["height"]):
            problems.append(f"{where}: box is empty or outside the frame")

    if row["cue"] not in CUES:
        problems.append(f"{where}: cue must be one of {sorted(CUES)}")

    if row["cycles"] < 2 and row["cue"] == "none":
        problems.append(f"{where}: POSITIVE needs 2 cycles or 1 cycle plus a direct cue")

    return problems


def _require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def _where(row: pd.Series) -> str:
    return f"{row['clip_id']}/{row['event_id']}"

"""Step 4: turn gesture cycles into smoking-like events.

Rules from the task:
* cycles of the same person with gaps <= merge_gap_sec form one event;
* an event needs min_cycles full cycles within window_sec;
* different people are never merged.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smokewatch.cycles import Cycle

EVENT_LABEL = "smoking_like"
OUTPUT_COLUMNS = [
    "camera_id",
    "clip_id",
    "event_id",
    "start_sec",
    "end_sec",
    "confidence",
    "label",
    "person_track_id",
    "peak_sec",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass
class Event:
    track_id: int
    cycles: list[Cycle]
    confidence: float

    @property
    def start_sec(self) -> float:
        return self.cycles[0].start_sec

    @property
    def end_sec(self) -> float:
        return self.cycles[-1].end_sec

    @property
    def peak_sec(self) -> float:
        """Peak of the cycle where the hand came closest to the mouth."""
        return min(self.cycles, key=lambda cycle: cycle.min_distance).peak_sec


def cycle_score(cycle: Cycle, near: float) -> float:
    """How convincing one cycle is, from 0 to 1: the closer to the mouth, the higher."""
    return float(np.clip(1 - cycle.min_distance / near, 0.0, 1.0))


def group_cycles(cycles: list[Cycle], merge_gap_sec: float) -> list[list[Cycle]]:
    """Split one person's cycles into groups separated by long gaps."""
    groups = []
    for cycle in sorted(cycles, key=lambda c: c.start_sec):
        if groups and cycle.start_sec - groups[-1][-1].end_sec <= merge_gap_sec:
            groups[-1].append(cycle)
        else:
            groups.append([cycle])
    return groups


def has_enough_cycles(group: list[Cycle], min_cycles: int, window_sec: float) -> bool:
    """True if some `min_cycles` consecutive cycles fit into `window_sec`."""
    for first, last in zip(group, group[min_cycles - 1 :], strict=False):
        if last.end_sec - first.start_sec <= window_sec:
            return True
    return False


def build_events(cycles: list[Cycle], config: dict, near: float) -> list[Event]:
    """Apply the event rules to all cycles of a clip."""
    events = []
    track_ids = sorted({cycle.track_id for cycle in cycles})
    for track_id in track_ids:
        track_cycles = [cycle for cycle in cycles if cycle.track_id == track_id]
        for group in group_cycles(track_cycles, config["merge_gap_sec"]):
            if not has_enough_cycles(group, config["min_cycles"], config["window_sec"]):
                continue
            # Noisy-OR: each extra convincing cycle makes the event more certain.
            miss = np.prod([1 - cycle_score(cycle, near) for cycle in group])
            confidence = float(1 - miss)
            if confidence >= config["threshold"]:
                events.append(Event(track_id, group, confidence))
    return sorted(events, key=lambda event: event.start_sec)


def box_at(poses: pd.DataFrame, track_id: int, time_sec: float) -> list[float]:
    """Person box of `track_id` at the frame closest to `time_sec`."""
    track = poses[poses["track_id"] == track_id]
    nearest = track.iloc[(track["time_sec"] - time_sec).abs().argmin()]
    return [round(float(nearest[c]), 1) for c in ("x1", "y1", "x2", "y2")]


def events_to_rows(
    events: list[Event], poses: pd.DataFrame, camera_id: str, clip_id: str
) -> list[dict]:
    """Format events as rows of the output CSV."""
    rows = []
    for number, event in enumerate(events, start=1):
        x1, y1, x2, y2 = box_at(poses, event.track_id, event.peak_sec)
        rows.append(
            {
                "camera_id": camera_id,
                "clip_id": clip_id,
                "event_id": f"{clip_id}_e{number:02d}",
                "start_sec": round(event.start_sec, 2),
                "end_sec": round(event.end_sec, 2),
                "confidence": round(event.confidence, 3),
                "label": EVENT_LABEL,
                "person_track_id": event.track_id,
                "peak_sec": round(event.peak_sec, 2),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return rows

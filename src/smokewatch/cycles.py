"""Steps 2 and 3: the hand-to-mouth signal and gesture cycles.

For every track we compute how far the closer wrist is from the nose,
measured in shoulder widths so the value does not depend on how far the
person is from the camera. A cycle is the pattern
"away -> at the mouth for 0.5-5 s -> away".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cycle:
    track_id: int
    start_sec: float  # last moment the hand was away before reaching the mouth
    end_sec: float  # first moment the hand was away again
    pause_sec: float  # time spent at the mouth
    peak_sec: float  # middle of the pause
    min_distance: float  # closest the hand got, in shoulder widths


def keypoint_distance(poses: pd.DataFrame, a: str, b: str) -> pd.Series:
    return np.hypot(poses[f"{a}_x"] - poses[f"{b}_x"], poses[f"{a}_y"] - poses[f"{b}_y"])


def body_scale(poses: pd.DataFrame, min_keypoint_conf: float) -> pd.Series:
    """Shoulder width in pixels, NaN where a shoulder is not visible.

    Seen from the side, shoulders overlap, so a fraction of the box height
    is used as a lower bound.
    """
    shoulder_width = keypoint_distance(poses, "left_shoulder", "right_shoulder")
    scale = np.maximum(shoulder_width, 0.15 * (poses["y2"] - poses["y1"]))
    shoulders_visible = (poses["left_shoulder_c"] >= min_keypoint_conf) & (
        poses["right_shoulder_c"] >= min_keypoint_conf
    )
    return scale.where(shoulders_visible)


def hand_to_mouth_distance(poses: pd.DataFrame, min_keypoint_conf: float) -> pd.Series:
    """Distance from the nose to the closer reliable wrist, in shoulder widths.

    NaN where the nose, both shoulders or both wrists are not visible.
    """

    def visible(name: str) -> pd.Series:
        return poses[f"{name}_c"] >= min_keypoint_conf

    scale = body_scale(poses, min_keypoint_conf)
    wrist_distances = [
        keypoint_distance(poses, f"{side}_wrist", "nose").where(visible(f"{side}_wrist"))
        for side in ("left", "right")
    ]
    closest = pd.concat(wrist_distances, axis=1).min(axis=1)
    return (closest / scale).where(visible("nose"))


def find_cycles(
    times: np.ndarray,
    distances: np.ndarray,
    track_id: int,
    near: float,
    far: float,
    min_pause_sec: float,
    max_pause_sec: float,
) -> list[Cycle]:
    """Find away -> mouth -> away cycles in one track's distance signal.

    Two thresholds (near < far) give hysteresis: the hand must come closer
    than `near` to count as "at the mouth" and move farther than `far` to
    count as "away", so noise around a single threshold creates no cycles.
    NaN samples are skipped.
    """
    cycles = []
    last_away_time = None  # most recent time the hand was clearly away
    mouth_start = None  # first time at the mouth in the current approach
    min_distance = np.inf

    for time, value in zip(times, distances, strict=True):
        if np.isnan(value):
            continue

        if mouth_start is None:
            if value > far:
                last_away_time = time
            elif value < near and last_away_time is not None:
                mouth_start = time
                min_distance = value
            continue

        if value > far:
            pause = time - mouth_start
            if min_pause_sec <= pause <= max_pause_sec:
                cycles.append(
                    Cycle(
                        track_id=track_id,
                        start_sec=last_away_time,
                        end_sec=time,
                        pause_sec=pause,
                        peak_sec=mouth_start + pause / 2,
                        min_distance=min_distance,
                    )
                )
            last_away_time = time
            mouth_start = None
        else:
            min_distance = min(min_distance, value)

    return cycles


def detect_cycles(poses: pd.DataFrame, signal_config: dict, cycle_config: dict) -> list[Cycle]:
    """Run cycle detection for every track in a clip."""
    if poses.empty:
        return []

    poses = poses.assign(distance=hand_to_mouth_distance(poses, signal_config["min_keypoint_conf"]))
    cycles = []
    for track_id, track in poses.groupby("track_id"):
        track = track.sort_values("time_sec")
        distance = _bridge_short_gaps(track, signal_config["max_gap_sec"])
        cycles += find_cycles(
            track["time_sec"].to_numpy(),
            distance.to_numpy(),
            track_id=int(track_id),
            near=signal_config["near_mouth"],
            far=signal_config["far_from_mouth"],
            min_pause_sec=cycle_config["min_pause_sec"],
            max_pause_sec=cycle_config["max_pause_sec"],
        )
    return cycles


def _bridge_short_gaps(track: pd.DataFrame, max_gap_sec: float) -> pd.Series:
    """Fill NaN runs shorter than `max_gap_sec` with the last known value."""
    distance = track["distance"]
    known_time = track["time_sec"].where(distance.notna()).ffill()
    filled = distance.ffill()
    too_long = (track["time_sec"] - known_time) > max_gap_sec
    return filled.mask(too_long)

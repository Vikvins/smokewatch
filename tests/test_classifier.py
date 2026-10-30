from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from smokewatch.classifier import cycle_crops, frame_times, score_cycles
from smokewatch.cycles import Cycle
from smokewatch.pipeline import process_clip

CONFIG = yaml.safe_load((Path(__file__).parents[1] / "configs/default.yaml").read_text())
CYCLE = Cycle(track_id=1, start_sec=0.5, end_sec=2.5, pause_sec=1.0, peak_sec=1.5, min_distance=0.2)


def make_poses(times, wrist_y) -> pd.DataFrame:
    """One person, nose at (550, 250), shoulders 100 px apart, left wrist at (550, wrist_y)."""
    return pd.DataFrame(
        {
            "time_sec": times,
            "track_id": 1,
            "x1": 400,
            "y1": 100,
            "x2": 700,
            "y2": 600,
            "nose_x": 550,
            "nose_y": 250,
            "nose_c": 0.9,
            "left_shoulder_x": 500,
            "left_shoulder_y": 350,
            "left_shoulder_c": 0.9,
            "right_shoulder_x": 600,
            "right_shoulder_y": 350,
            "right_shoulder_c": 0.9,
            "left_wrist_x": 550,
            "left_wrist_y": wrist_y,
            "left_wrist_c": 0.9,
            "right_wrist_x": 0,
            "right_wrist_y": 0,
            "right_wrist_c": 0.0,
        }
    )


class FakeClassifier:
    def __init__(self, value: float):
        self.value = value
        self.calls = []

    def score(self, crops):
        self.calls.append(crops)
        return self.value


def test_frames_are_taken_inside_the_pause():
    assert frame_times(CYCLE) == [1.25, 1.5, 1.75]


def test_cycle_crops_cut_the_head_from_the_video(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 700))
    for _ in range(30):
        writer.write(np.full((700, 800, 3), 128, dtype=np.uint8))
    writer.release()

    poses = make_poses(np.arange(30) / 10, 450)
    crops = cycle_crops(video, CYCLE, poses, crop_size=3.0)
    assert len(crops) == 3
    assert crops[0].shape == (300, 300, 3)  # 3 shoulder widths around the nose


def test_cycle_without_readable_frames_gets_nan(tmp_path):
    poses = make_poses([0.0], 450)
    [score] = score_cycles(FakeClassifier(0.9), tmp_path / "missing.mp4", poses, 3.0, [CYCLE])
    assert np.isnan(score)


def two_cycle_poses() -> pd.DataFrame:
    # away 2 s, at the mouth 1 s, away 2 s, at the mouth 1 s, away 2 s
    pattern = [(2, 450), (1, 270), (2, 450), (1, 270), (2, 450)]
    wrist_y = np.concatenate([np.full(seconds * 10, y) for seconds, y in pattern])
    return make_poses(np.arange(len(wrist_y)) / 10, wrist_y)


@pytest.mark.parametrize(
    "scores, expected_events",
    [([0.9, 0.9], 1), ([0.01, 0.01], 0), ([0.01, 0.9], 1), ([np.nan, np.nan], 1)],
)
def test_event_is_dropped_when_no_cycle_looks_like_smoking(scores, expected_events):
    clip = pd.Series({"camera_id": "cam", "clip_id": "c1"})
    rows, table = process_clip(clip, two_cycle_poses(), None, CONFIG, lambda cycles: scores)
    assert len(table) == 2
    assert len(rows) == expected_events
    np.testing.assert_array_equal(table["smoking_score"], scores)


def test_clip_without_cycles_gives_empty_tables_with_columns():
    clip = pd.Series({"camera_id": "cam", "clip_id": "c1"})
    rows, table = process_clip(clip, make_poses([0.0, 0.1], 450), None, CONFIG)
    assert rows == []
    assert {"track_id", "start_sec", "end_sec", "smoking_score"} <= set(table.columns)

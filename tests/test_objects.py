import pandas as pd

from smokewatch.cycles import Cycle
from smokewatch.objects import OBJECT_COLUMNS, head_crop, object_in_hand

CONFIG = {"max_time_gap_sec": 0.1, "max_wrist_distance": 1.0, "min_fraction": 0.3}
CYCLE = Cycle(track_id=1, start_sec=0.0, end_sec=1.0, pause_sec=0.6, peak_sec=0.5, min_distance=0.2)


def make_poses() -> pd.DataFrame:
    """One person, shoulders 100 px apart, left wrist at (500, 300)."""
    rows = []
    for i in range(11):
        rows.append(
            {
                "time_sec": i / 10,
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
                "left_wrist_x": 500,
                "left_wrist_y": 300,
                "left_wrist_c": 0.9,
                "right_wrist_x": 650,
                "right_wrist_y": 550,
                "right_wrist_c": 0.9,
            }
        )
    return pd.DataFrame(rows)


def make_objects(centre_x: float, times: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time_sec": t,
                "track_id": 1,
                "name": "bottle",
                "conf": 0.8,
                "x1": centre_x - 10,
                "y1": 290,
                "x2": centre_x + 10,
                "y2": 310,
            }
            for t in times
        ],
        columns=OBJECT_COLUMNS,
    )


def held(objects: pd.DataFrame):
    return object_in_hand(CYCLE, make_poses(), objects, CONFIG, min_keypoint_conf=0.4)


def test_bottle_at_wrist_is_found():
    assert held(make_objects(510, [0.0, 0.2, 0.4, 0.6])) == "bottle"


def test_bottle_far_from_wrist_is_ignored():
    assert held(make_objects(900, [0.0, 0.2, 0.4, 0.6])) is None


def test_brief_detection_is_not_enough():
    assert held(make_objects(510, [0.4])) is None


def test_no_objects():
    assert held(make_objects(510, [])) is None


def test_other_persons_object_is_ignored():
    objects = make_objects(510, [0.0, 0.2, 0.4, 0.6]).assign(track_id=2)
    assert held(objects) is None


def test_head_crop_is_centred_on_nose_and_clipped_to_frame():
    pose = make_poses().iloc[0]
    assert head_crop(pose, (1080, 1920), crop_size=3.0) == (400, 100, 700, 400)
    assert head_crop(pose, (300, 1920), crop_size=3.0) == (400, 100, 700, 300)

"""Hard-negative filter: objects in the hand during a gesture cycle.

Drinking or talking on the phone produce the same hand-to-mouth gesture as
smoking. A standard COCO detector already knows bottles, cups and phones, so
a cycle where such an object stays next to the wrist is dropped.

On surveillance footage a bottle is only 15-20 px wide and disappears when
the whole frame is resized for the detector. So the detector looks at a
crop around each person's head at full resolution instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from smokewatch.cycles import Cycle, body_scale
from smokewatch.pose import iter_sampled_frames

OBJECT_COLUMNS = ["time_sec", "track_id", "name", "conf", "x1", "y1", "x2", "y2"]


def head_crop(pose: pd.Series, frame_shape: tuple, crop_size: float) -> tuple[int, int, int, int]:
    """Square around the head, `crop_size` shoulder widths wide, inside the frame.

    Wrists at the mouth fall inside it, together with whatever they hold.
    """
    box_height = pose["y2"] - pose["y1"]
    scale = max(
        np.hypot(
            pose["left_shoulder_x"] - pose["right_shoulder_x"],
            pose["left_shoulder_y"] - pose["right_shoulder_y"],
        ),
        0.15 * box_height,
    )
    if pose["nose_c"] >= 0.3:
        centre_x, centre_y = pose["nose_x"], pose["nose_y"]
    else:  # head not visible: use the top part of the person box
        centre_x, centre_y = (pose["x1"] + pose["x2"]) / 2, pose["y1"] + 0.15 * box_height

    half = crop_size * scale / 2
    height, width = frame_shape[:2]
    x1, y1 = max(0, int(centre_x - half)), max(0, int(centre_y - half))
    x2, y2 = min(width, int(centre_x + half)), min(height, int(centre_y + half))
    return x1, y1, x2, y2


def extract_objects(
    video_path: Path, config: dict, poses: pd.DataFrame, roi_mask=None
) -> pd.DataFrame:
    """Detect the configured COCO classes near each tracked person's head.

    Returns one row per detection, in full-frame pixel coordinates.
    """
    from ultralytics import YOLO

    model = YOLO(config["model"])
    class_ids = [i for i, name in model.names.items() if name in config["classes"]]
    rows = []
    for time_sec, frame in iter_sampled_frames(video_path, config["sample_fps"], roi_mask):
        people = poses[(poses["time_sec"] - time_sec).abs() <= config["max_time_gap_sec"]]
        people = people.drop_duplicates("track_id")
        crops = [
            head_crop(person, frame.shape, config["crop_size"]) for _, person in people.iterrows()
        ]
        crops = [
            (track_id, c)
            for track_id, c in zip(people["track_id"], crops, strict=True)
            if c[2] - c[0] >= 8 and c[3] - c[1] >= 8
        ]
        if not crops:
            continue

        results = model.predict(
            [frame[y1:y2, x1:x2] for _, (x1, y1, x2, y2) in crops],
            classes=class_ids,
            conf=config["min_conf"],
            imgsz=config["imgsz"],
            device=config["device"],
            verbose=False,
        )
        for (track_id, (crop_x, crop_y, _, _)), result in zip(crops, results, strict=True):
            boxes = result.boxes
            for class_id, conf, (x1, y1, x2, y2) in zip(
                boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            ):
                rows.append(
                    {
                        "time_sec": round(time_sec, 3),
                        "track_id": int(track_id),
                        "name": model.names[class_id],
                        "conf": conf,
                        "x1": x1 + crop_x,
                        "y1": y1 + crop_y,
                        "x2": x2 + crop_x,
                        "y2": y2 + crop_y,
                    }
                )
    return pd.DataFrame(rows, columns=OBJECT_COLUMNS)


def object_in_hand(
    cycle: Cycle, poses: pd.DataFrame, objects: pd.DataFrame, config: dict, min_keypoint_conf: float
) -> str | None:
    """Name of the object held near a wrist during `cycle`, or None.

    For every pose sample of the cycle we look for a detection within
    max_time_gap_sec whose centre is closer than max_wrist_distance (in
    shoulder widths) to a visible wrist. If that happens in at least
    min_fraction of the samples, the object is considered held.
    """
    track = poses[
        (poses["track_id"] == cycle.track_id)
        & poses["time_sec"].between(cycle.start_sec, cycle.end_sec)
    ]
    objects = objects[objects["track_id"] == cycle.track_id]
    if track.empty or objects.empty:
        return None

    scale = body_scale(track, min_keypoint_conf).to_numpy()
    centres_x = ((objects["x1"] + objects["x2"]) / 2).to_numpy()
    centres_y = ((objects["y1"] + objects["y2"]) / 2).to_numpy()
    times = objects["time_sec"].to_numpy()

    hits = []
    for (_, pose), pose_scale in zip(track.iterrows(), scale, strict=True):
        nearby = np.abs(times - pose["time_sec"]) <= config["max_time_gap_sec"]
        held = None
        for side in ("left", "right"):
            if np.isnan(pose_scale) or pose[f"{side}_wrist_c"] < min_keypoint_conf:
                continue
            distance = np.hypot(
                centres_x - pose[f"{side}_wrist_x"], centres_y - pose[f"{side}_wrist_y"]
            )
            close = nearby & (distance / pose_scale <= config["max_wrist_distance"])
            if close.any():
                held = objects["name"].to_numpy()[close][0]
        hits.append(held)

    found = [name for name in hits if name is not None]
    if len(found) >= config["min_fraction"] * len(hits):
        return max(set(found), key=found.count)  # the most frequent object
    return None

"""Hard-negative filter: objects in the hand during a gesture cycle.

Drinking or talking on the phone produce the same hand-to-mouth gesture as
smoking. A standard COCO detector already knows bottles, cups and phones, so
a cycle where such an object stays next to the wrist is dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from smokewatch.cycles import Cycle, body_scale
from smokewatch.pose import iter_sampled_frames

OBJECT_COLUMNS = ["time_sec", "name", "conf", "x1", "y1", "x2", "y2"]


def extract_objects(video_path: Path, config: dict, roi_mask=None) -> pd.DataFrame:
    """Detect the configured COCO classes; return one row per detection."""
    from ultralytics import YOLO

    model = YOLO(config["model"])
    class_ids = [i for i, name in model.names.items() if name in config["classes"]]
    rows = []
    for time_sec, frame in iter_sampled_frames(video_path, config["sample_fps"], roi_mask):
        result = model.predict(
            frame,
            classes=class_ids,
            conf=config["min_conf"],
            device=config["device"],
            verbose=False,
        )[0]
        boxes = result.boxes
        for class_id, conf, box in zip(
            boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
        ):
            rows.append(
                {"time_sec": round(time_sec, 3), "name": model.names[class_id], "conf": conf}
                | dict(zip(["x1", "y1", "x2", "y2"], box, strict=True))
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

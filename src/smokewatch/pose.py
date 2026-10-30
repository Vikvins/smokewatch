"""Step 1 of the pipeline: people, track ids and pose keypoints.

A YOLO pose model finds people and their 17 COCO keypoints; ByteTrack links
detections over time into tracks. Only the keypoints the later steps need
are kept.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# COCO keypoint indices used downstream.
KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
}
PERSON_CLASS = 0
POSE_COLUMNS = ["time_sec", "track_id", "conf", "x1", "y1", "x2", "y2"] + [
    f"{name}_{field}" for name in KEYPOINTS for field in ("x", "y", "c")
]


def iter_sampled_frames(video_path: Path, sample_fps: float, roi_mask: np.ndarray | None = None):
    """Yield (time_sec, frame) at roughly `sample_fps` frames per second.

    Times come from the video timestamps, so clips with different frame rates
    give comparable results. Pixels outside `roi_mask` (if given) are blacked out.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    next_time = 0.0
    while capture.grab():
        time_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if time_sec + 1e-6 < next_time:
            continue
        ok, frame = capture.retrieve()
        if not ok:
            break
        if roi_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        yield time_sec, frame
        next_time = time_sec + 1 / sample_fps

    capture.release()


def extract_poses(
    video_path: Path, config: dict, roi_mask: np.ndarray | None = None
) -> pd.DataFrame:
    """Run pose estimation and tracking; return one row per (time, person)."""
    from ultralytics import YOLO  # imported here so the rest of the package works without it

    model = YOLO(config["model"])
    rows = []
    for time_sec, frame in iter_sampled_frames(video_path, config["sample_fps"], roi_mask):
        result = model.track(
            frame,
            persist=True,
            tracker=config["tracker"],
            conf=config["min_person_conf"],
            classes=[PERSON_CLASS],
            device=config["device"],
            verbose=False,
        )[0]
        rows.extend(_result_to_rows(result, time_sec))

    return pd.DataFrame(rows, columns=POSE_COLUMNS)


def _result_to_rows(result, time_sec: float) -> list[dict]:
    boxes = result.boxes
    if boxes.id is None or result.keypoints is None:
        return []  # no confirmed tracks in this frame

    track_ids = boxes.id.int().tolist()
    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist()
    keypoints = result.keypoints.data.tolist()  # (people, 17, [x, y, conf])

    rows = []
    for track_id, box, conf, person_keypoints in zip(
        track_ids, xyxy, confs, keypoints, strict=True
    ):
        row = {"time_sec": round(time_sec, 3), "track_id": track_id, "conf": conf}
        row.update(dict(zip(["x1", "y1", "x2", "y2"], box, strict=True)))
        for name, index in KEYPOINTS.items():
            x, y, c = person_keypoints[index]
            row.update({f"{name}_x": x, f"{name}_y": y, f"{name}_c": c})
        rows.append(row)
    return rows

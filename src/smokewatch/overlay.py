"""Draw tracks, gesture cycles and detected events on top of a video.

Colours: grey box = tracked person, yellow = hand at the mouth (inside a
cycle), red = person inside a detected event, green bar = ground truth.

The result is written as H.264 so it plays in a browser (the web app);
imageio-ffmpeg ships its own ffmpeg binary, nothing else is needed.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# BGR colours
GREY, YELLOW, RED = (160, 160, 160), (0, 220, 255), (0, 0, 255)
GREEN, BLACK = (0, 200, 0), (0, 0, 0)


def active(table: pd.DataFrame, time_sec: float) -> pd.DataFrame:
    """Rows of `table` whose [start_sec, end_sec] contains `time_sec`."""
    return table[(table["start_sec"] <= time_sec) & (time_sec <= table["end_sec"])]


def put_label(frame: np.ndarray, text: str, x: int, y: int, colour) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def draw_frame(
    frame: np.ndarray,
    time_sec: float,
    people: pd.DataFrame,
    cycles: pd.DataFrame,
    events: pd.DataFrame,
    gt: pd.DataFrame,
) -> None:
    """Draw everything that is happening at `time_sec` onto `frame` in place."""
    height, width = frame.shape[:2]
    at_mouth = set(active(cycles, time_sec)["track_id"])
    current_events = active(events, time_sec)
    in_event = set(current_events["person_track_id"])

    for _, person in people.iterrows():
        track_id = int(person["track_id"])
        colour = RED if track_id in in_event else YELLOW if track_id in at_mouth else GREY
        x1, y1, x2, y2 = (int(person[k]) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if colour == RED else 1)
        put_label(frame, f"id {track_id}", x1, max(15, y1 - 6), colour)

    for _, event in current_events.iterrows():
        put_label(frame, f"SMOKING-LIKE  conf {event['confidence']:.2f}", 10, 30, RED)
    if len(active(gt, time_sec)):
        cv2.rectangle(frame, (0, height - 12), (width, height), GREEN, -1)
    put_label(frame, f"{time_sec:6.1f} s", width - 110, 30, GREY)


def render_overlay(
    video_path: Path,
    poses: pd.DataFrame,
    cycles: pd.DataFrame,
    events: pd.DataFrame,
    out_path: Path,
    gt: pd.DataFrame | None = None,
    max_width: int | None = None,
    progress=None,
) -> Path:
    """Write a copy of the video with the overlay; return `out_path`.

    `max_width` downsizes large videos (faster, smaller file). `progress`,
    if given, is called with a number from 0 to 1.
    """
    import imageio_ffmpeg

    if gt is None:
        gt = pd.DataFrame(columns=["start_sec", "end_sec"])
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / width) if max_width else 1.0
    # H.264 needs even frame sizes
    size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out_path), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1
    )
    writer.send(None)  # start the ffmpeg process

    pose_times = np.sort(poses["time_sec"].unique())
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        time_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # poses are sampled at a lower rate: show the latest sample
        position = np.searchsorted(pose_times, time_sec, side="right") - 1
        people = poses[poses["time_sec"] == pose_times[position]] if position >= 0 else poses[:0]
        draw_frame(frame, time_sec, people, cycles, events, gt)

        frame = cv2.resize(frame, size) if size != (width, height) else frame
        writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        index += 1
        if progress is not None and index % 50 == 0:
            progress(min(1.0, index / total))

    capture.release()
    writer.close()
    return out_path

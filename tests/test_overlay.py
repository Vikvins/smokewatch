import cv2
import numpy as np
import pandas as pd

from smokewatch.overlay import active, render_overlay


def write_video(path, frames=20, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(frames):
        writer.write(np.full((size[1], size[0], 3), 128, dtype=np.uint8))
    writer.release()


def test_active_keeps_rows_around_the_time():
    table = pd.DataFrame({"start_sec": [0.0, 5.0], "end_sec": [2.0, 9.0]})
    assert len(active(table, 1.0)) == 1
    assert len(active(table, 3.0)) == 0


def test_overlay_marks_the_person_in_an_event(tmp_path):
    video = tmp_path / "in.mp4"
    write_video(video)
    poses = pd.DataFrame(
        {"time_sec": [0.0, 1.0], "track_id": 7, "x1": 50, "y1": 50, "x2": 150, "y2": 200}
    )
    cycles = pd.DataFrame(columns=["track_id", "start_sec", "end_sec"])
    events = pd.DataFrame(
        {"person_track_id": [7], "start_sec": [1.0], "end_sec": [1.9], "confidence": [0.8]}
    )

    out = render_overlay(video, poses, cycles, events, tmp_path / "out.mp4", max_width=161)

    capture = cv2.VideoCapture(str(out))
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 160  # even size for H.264
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == 20

    def red_pixels(frame):
        blue, green, red = (frame[..., i].astype(int) for i in range(3))
        return int(((red > 200) & (green < 60) & (blue < 60)).sum())

    assert red_pixels(frames[2]) == 0  # before the event
    assert red_pixels(frames[15]) > 50  # red box during the event

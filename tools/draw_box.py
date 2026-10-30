"""Show a video frame and draw the person box for events.csv.

Usage:
    python tools/draw_box.py data/raw/smoking/1.mp4 12.5

Keys in the window:
    a / d   step back / forward by 0.2 s (Shift: 1 s)
    space   draw a box with the mouse, then press Enter to confirm
    q       quit

The chosen peak_sec and box (in original pixels) are printed as a CSV
fragment ready to paste into events.csv.
"""

import argparse
from pathlib import Path

import cv2

MAX_DISPLAY_WIDTH = 1280
STEPS = {ord("a"): -0.2, ord("d"): 0.2, ord("A"): -1.0, ord("D"): 1.0}
WINDOW = "draw_box"


def read_frame_at(capture: cv2.VideoCapture, sec: float):
    """Seek to a timestamp and return (actual_sec, frame)."""
    capture.set(cv2.CAP_PROP_POS_MSEC, max(sec, 0.0) * 1000)
    ok, frame = capture.read()
    if not ok:
        raise SystemExit(f"cannot read a frame at {sec:.2f} s")
    return capture.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("sec", type=float, help="approximate peak time in seconds")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    sec = args.sec
    while True:
        actual_sec, frame = read_frame_at(capture, sec)
        scale = min(1.0, MAX_DISPLAY_WIDTH / frame.shape[1])
        view = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.putText(
            view, f"{actual_sec:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
        )
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key in STEPS:
            sec = actual_sec + STEPS[key]
        elif key == ord(" "):
            x, y, w, h = cv2.selectROI(WINDOW, view, showCrosshair=False)
            if w > 0 and h > 0:
                box = [round(v / scale) for v in (x, y, x + w, y + h)]
                print("peak_sec,x1,y1,x2,y2")
                print(f"{actual_sec:.2f},{box[0]},{box[1]},{box[2]},{box[3]}")
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

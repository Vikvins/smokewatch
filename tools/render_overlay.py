"""Draw tracks, gesture cycles and detected events on top of a clip.

Usage:
    python tools/render_overlay.py --run clip --clip sm_2

Writes outputs/overlays/<run>/<clip_id>.mp4. The video stays on this machine
like everything else under outputs/.

Colours: grey box = tracked person, yellow = hand at the mouth (inside a
cycle), red = person inside a detected event, green bar = ground truth.
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from smokewatch.annotations import load_clips

# BGR colours
GREY, YELLOW, RED = (160, 160, 160), (0, 220, 255), (0, 0, 255)
GREEN, BLACK = (0, 200, 0), (0, 0, 0)


def active(table: pd.DataFrame, time_sec: float) -> pd.DataFrame:
    return table[(table["start_sec"] <= time_sec) & (time_sec <= table["end_sec"])]


def label(frame, text: str, x: int, y: int, colour) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--clips", type=Path, default=Path("data/annotations/clips.csv"))
    parser.add_argument("--gt", type=Path, default=Path("data/annotations/events.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    clip = load_clips(args.clips).set_index("clip_id").loc[args.clip]
    run_dir = args.outputs / "runs" / args.run
    poses = pd.read_csv(args.outputs / "poses" / f"{args.clip}.csv")
    cycles = pd.read_csv(run_dir / "cycles.csv", dtype={"clip_id": str})
    events = pd.read_csv(run_dir / "events.csv", dtype={"clip_id": str})
    cycles = cycles[cycles["clip_id"] == args.clip]
    events = events[events["clip_id"] == args.clip]
    gt = pd.DataFrame(columns=["start_sec", "end_sec", "label"])
    if args.gt.exists():
        gt = pd.read_csv(args.gt, dtype={"clip_id": str})
        gt = gt[(gt["clip_id"] == args.clip) & (gt["label"] == "POSITIVE")]

    capture = cv2.VideoCapture(str(args.raw_dir / clip["path"]))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = args.outputs / "overlays" / args.run / f"{args.clip}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    pose_times = poses["time_sec"].drop_duplicates().sort_values().to_numpy()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        time_sec = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # poses are sampled at a lower rate: show the latest sample
        earlier = pose_times[pose_times <= time_sec]
        people = poses[poses["time_sec"] == earlier[-1]] if len(earlier) else poses.iloc[:0]
        at_mouth = set(active(cycles, time_sec)["track_id"])
        in_event = set(active(events, time_sec)["person_track_id"])
        for _, person in people.iterrows():
            track_id = int(person["track_id"])
            colour = RED if track_id in in_event else YELLOW if track_id in at_mouth else GREY
            x1, y1, x2, y2 = (int(person[k]) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if colour == RED else 1)
            label(frame, f"id {track_id}", x1, max(15, y1 - 6), colour)

        for _, event in active(events, time_sec).iterrows():
            label(frame, f"SMOKING-LIKE  conf {event['confidence']:.2f}", 10, 30, RED)
        if len(active(gt, time_sec)):
            cv2.rectangle(frame, (0, height - 12), (width, height), GREEN, -1)
        label(frame, f"{time_sec:6.1f} s", width - 110, 30, GREY)
        writer.write(frame)

    capture.release()
    writer.release()
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()

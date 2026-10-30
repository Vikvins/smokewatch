"""Draw tracks, gesture cycles and detected events on top of a clip.

Usage:
    python tools/render_overlay.py --run final --clip sm_2

Writes outputs/overlays/<run>/<clip_id>.mp4. The video stays on this machine
like everything else under outputs/. Colours are explained in
src/smokewatch/overlay.py.
"""

import argparse
from pathlib import Path

import pandas as pd

from smokewatch.annotations import load_clips
from smokewatch.overlay import render_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--clips", type=Path, default=Path("data/annotations/clips.csv"))
    parser.add_argument("--gt", type=Path, default=Path("data/annotations/events.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--max-width", type=int, help="downsize wider videos to this width")
    args = parser.parse_args()

    clip = load_clips(args.clips).set_index("clip_id").loc[args.clip]
    run_dir = args.outputs / "runs" / args.run
    poses = pd.read_csv(args.outputs / "poses" / f"{args.clip}.csv")
    cycles = pd.read_csv(run_dir / "cycles.csv", dtype={"clip_id": str})
    events = pd.read_csv(run_dir / "events.csv", dtype={"clip_id": str})
    gt = None
    if args.gt.exists():
        gt = pd.read_csv(args.gt, dtype={"clip_id": str})
        gt = gt[(gt["clip_id"] == args.clip) & (gt["label"] == "POSITIVE")]

    out_path = render_overlay(
        args.raw_dir / clip["path"],
        poses,
        cycles[cycles["clip_id"] == args.clip],
        events[events["clip_id"] == args.clip],
        args.outputs / "overlays" / args.run / f"{args.clip}.mp4",
        gt=gt,
        max_width=args.max_width,
    )
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()

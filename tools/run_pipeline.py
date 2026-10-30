"""Run SmokeWatch on the clips listed in clips.csv.

Usage:
    python tools/run_pipeline.py --run baseline
    python tools/run_pipeline.py --run baseline --clip sm_2

Writes outputs/runs/<run>/events.csv, cycles.csv and manifest.json.
Poses are cached in outputs/poses/, so re-running with new thresholds is fast.
"""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from smokewatch.annotations import load_clips
from smokewatch.events import OUTPUT_COLUMNS
from smokewatch.pipeline import (
    cycles_to_frame,
    load_or_extract_poses,
    load_roi_mask,
    process_clip,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="name of the output folder")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--clips", type=Path, default=Path("data/annotations/clips.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--roi-dir", type=Path, default=Path("configs/roi"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--clip", help="process only this clip_id")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    clips = load_clips(args.clips)
    if args.clip:
        clips = clips[clips["clip_id"] == args.clip]
        if clips.empty:
            raise SystemExit(f"unknown clip_id {args.clip!r}")

    all_rows, all_cycles = [], []
    for _, clip in tqdm(list(clips.iterrows()), desc="clips"):
        poses = load_or_extract_poses(
            args.raw_dir / clip["path"],
            args.outputs / "poses" / f"{clip['clip_id']}.csv",
            config["pose"],
            load_roi_mask(args.roi_dir, clip["camera_id"]),
        )
        rows, cycles = process_clip(clip, poses, config)
        all_rows += rows
        all_cycles.append(cycles_to_frame(clip["clip_id"], cycles))

    run_dir = args.outputs / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS).to_csv(run_dir / "events.csv", index=False)
    pd.concat(all_cycles).to_csv(run_dir / "cycles.csv", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip_ids": clips["clip_id"].tolist(),
        "threshold": config["events"]["threshold"],
        "merge_rules": config["events"],
        "config": config,
        "model_versions": {
            "pose_model": config["pose"]["model"],
            "ultralytics": version("ultralytics"),
            "smokewatch": version("smokewatch"),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(all_rows)} events from {len(clips)} clips -> {run_dir}")


if __name__ == "__main__":
    main()

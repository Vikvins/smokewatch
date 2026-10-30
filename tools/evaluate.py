"""Score a pipeline run against the ground truth.

Usage:
    python tools/evaluate.py --run baseline
    python tools/evaluate.py --run baseline --split test

Prints the metrics and saves them to outputs/runs/<run>/metrics.json.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from smokewatch.annotations import load_clips, load_events
from smokewatch.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--clips", type=Path, default=Path("data/annotations/clips.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/annotations/events.csv"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--split", help="evaluate only clips of this split (dev or test)")
    args = parser.parse_args()

    run_dir = args.outputs / "runs" / args.run
    config = yaml.safe_load(args.config.read_text())
    clips = load_clips(args.clips)
    predictions = pd.read_csv(run_dir / "events.csv", dtype={"clip_id": str, "camera_id": str})
    ground_truth = load_events(args.events)

    # Only clips processed in this run take part.
    manifest = json.loads((run_dir / "manifest.json").read_text())
    clips = clips[clips["clip_id"].isin(manifest["clip_ids"])]
    if args.split:
        clips = clips[clips["split"] == args.split]

    metrics = evaluate(predictions, ground_truth, clips, config["evaluation"])
    metrics["clips"] = len(clips)
    for name, value in metrics.items():
        print(f"{name:>20}: {value}")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

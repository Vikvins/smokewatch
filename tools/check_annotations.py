"""Validate data/annotations/events.csv against clips.csv.

Usage:
    python tools/check_annotations.py
"""

import argparse
from pathlib import Path

from smokewatch.annotations import load_clips, load_events, validate_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, default=Path("data/annotations/clips.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/annotations/events.csv"))
    args = parser.parse_args()

    clips = load_clips(args.clips)
    events = load_events(args.events)
    problems = validate_events(events, clips)

    for problem in problems:
        print(problem)
    counts = events["label"].value_counts().to_dict()
    print(f"{len(events)} events {counts}, {len(problems)} problems")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()

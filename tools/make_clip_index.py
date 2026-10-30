"""Scan data/raw/ and write data/annotations/clips.csv.

Re-running is safe: hand-filled columns (camera_id, split) are kept for
clips that are already in the index, and new videos are appended.

Usage:
    python tools/make_clip_index.py
"""

import argparse
import re
from pathlib import Path

import cv2
import pandas as pd

from smokewatch.annotations import CLIP_COLUMNS

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".wmv", ".mov", ".avi"}
MANUAL_COLUMNS = ["camera_id", "split"]


def make_clip_id(path: Path) -> str:
    """Turn 'smoking_Jar (1).mp4' into 'smoking_jar_1'."""
    clip_id = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    if not clip_id:
        raise ValueError(f"cannot build a clip_id from {path.name!r}; use a Latin file name")
    return clip_id


def probe_video(path: Path) -> dict:
    """Read fps, duration and frame size with OpenCV.

    Frame counts from container headers can be slightly off for some
    formats (wmv, mkv); that is fine for an index.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    info = {
        "fps": round(fps, 3),
        "duration_sec": round(n_frames / fps, 3) if fps > 0 else float("nan"),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return info


def build_index(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(raw_dir)
        rows.append(
            {
                "clip_id": make_clip_id(path),
                "path": relative.as_posix(),
                "source": relative.parts[0] if len(relative.parts) > 1 else "",
                "camera_id": "unknown",
                "split": "dev",
                **probe_video(path),
            }
        )
    return pd.DataFrame(rows, columns=CLIP_COLUMNS)


def keep_manual_columns(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Copy hand-filled values from the old index into the new one."""
    old = old.set_index("clip_id")[MANUAL_COLUMNS]
    new = new.set_index("clip_id")
    new.update(old)
    return new.reset_index()[CLIP_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations/clips.csv"))
    args = parser.parse_args()

    index = build_index(args.raw_dir)
    if index["clip_id"].duplicated().any():
        raise SystemExit("two files map to the same clip_id; rename one of them")

    if args.out.exists():
        index = keep_manual_columns(index, pd.read_csv(args.out, dtype=str))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out, index=False)
    print(f"{len(index)} clips, {index['duration_sec'].sum() / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()

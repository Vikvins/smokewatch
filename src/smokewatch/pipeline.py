"""Run the whole pipeline on one clip: poses -> cycles -> events."""

from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd

from smokewatch.cycles import Cycle, detect_cycles
from smokewatch.events import build_events, events_to_rows
from smokewatch.pose import extract_poses


def load_roi_mask(roi_dir: Path, camera_id: str):
    """Load configs/roi/<camera_id>.png (white = keep) if it exists."""
    path = roi_dir / f"{camera_id}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype("uint8") * 255


def load_or_extract_poses(
    video_path: Path, cache_path: Path, pose_config: dict, roi_mask=None
) -> pd.DataFrame:
    """Pose extraction is the slow part, so its result is cached per clip."""
    if cache_path.exists():
        return pd.read_csv(cache_path)
    poses = extract_poses(video_path, pose_config, roi_mask)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    poses.to_csv(cache_path, index=False)
    return poses


def process_clip(
    clip: pd.Series, poses: pd.DataFrame, config: dict
) -> tuple[list[dict], list[Cycle]]:
    """Return output rows and the raw cycles (useful for debugging)."""
    cycles = detect_cycles(poses, config["signal"], config["cycles"])
    events = build_events(cycles, config["events"], near=config["signal"]["near_mouth"])
    rows = events_to_rows(events, poses, clip["camera_id"], clip["clip_id"])
    return rows, cycles


def cycles_to_frame(clip_id: str, cycles: list[Cycle]) -> pd.DataFrame:
    return pd.DataFrame([{"clip_id": clip_id, **asdict(cycle)} for cycle in cycles])

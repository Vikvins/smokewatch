"""Run the whole pipeline on one clip: poses -> cycles -> filters -> events."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from smokewatch.cycles import Cycle, detect_cycles
from smokewatch.events import Event, build_events, events_to_rows
from smokewatch.objects import object_in_hand


def load_roi_mask(roi_dir: Path, camera_id: str):
    """Load configs/roi/<camera_id>.png (white = keep) if it exists."""
    path = roi_dir / f"{camera_id}.png"
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype("uint8") * 255


def load_or_extract(cache_path: Path, extract: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Model inference is the slow part, so its result is cached per clip."""
    if cache_path.exists():
        return pd.read_csv(cache_path)
    frame = extract()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def process_clip(
    clip: pd.Series,
    poses: pd.DataFrame,
    objects: pd.DataFrame | None,
    config: dict,
    score_cycles: Callable[[list[Cycle]], list[float]] | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Return output rows and a table of all cycles (useful for debugging).

    Cycles with a bottle, cup or phone in the hand (`held_object`) stay in
    the table but do not take part in events. An event is dropped when even
    its most smoking-like cycle scores below `classifier.min_score`; events
    without any score (no crops) are kept.
    """
    min_conf = config["signal"]["min_keypoint_conf"]
    cycles = detect_cycles(poses, config["signal"], config["cycles"])

    held = [
        object_in_hand(cycle, poses, objects, config["objects"], min_conf)
        if objects is not None
        else None
        for cycle in cycles
    ]
    kept = [cycle for cycle, name in zip(cycles, held, strict=True) if name is None]
    events = build_events(kept, config["events"], near=config["signal"]["near_mouth"])

    scores = score_cycles(cycles) if score_cycles is not None else [float("nan")] * len(cycles)
    score_of = {(c.track_id, c.peak_sec): s for c, s in zip(cycles, scores, strict=True)}

    def best_score(event: Event) -> float:
        event_scores = [score_of[(c.track_id, c.peak_sec)] for c in event.cycles]
        return max((s for s in event_scores if not np.isnan(s)), default=float("nan"))

    min_score = config["classifier"]["min_score"]
    events = [event for event in events if not best_score(event) < min_score]  # NaN passes
    rows = events_to_rows(events, poses, clip["camera_id"], clip["clip_id"])

    table = pd.DataFrame(
        [
            {
                "clip_id": clip["clip_id"],
                **asdict(cycle),
                "held_object": name,
                "smoking_score": score,
            }
            for cycle, name, score in zip(cycles, held, scores, strict=True)
        ]
    )
    return rows, table

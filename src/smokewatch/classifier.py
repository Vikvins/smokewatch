"""Hard-negative filter: does a gesture cycle look like smoking?

The hand-to-mouth gesture alone cannot tell smoking from drinking or talking
on the phone. CLIP compares a crop around the person's head with short text
descriptions ("a person smoking a cigarette", "a person drinking from a
bottle", ...) and needs no training data, so it cannot learn the camera
instead of the object. Known weak spot: a lollipop on a stick looks like a
cigarette to it.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from smokewatch.cycles import Cycle
from smokewatch.objects import head_crop

SMOKING_PROMPTS = [
    "a photo of a person smoking a cigarette",
    "a photo of a person vaping an e-cigarette",
    "a photo of a person holding a cigarette at the mouth",
]
OTHER_PROMPTS = [
    "a photo of a person drinking from a bottle",
    "a photo of a person drinking from a cup",
    "a photo of a person eating a lollipop",
    "a photo of a person eating food",
    "a photo of a person talking on a phone",
    "a photo of a person touching their face",
    "a photo of a person standing",
]
# Frames taken during the pause at the mouth, as offsets from its middle
# in units of the pause length.
FRAME_OFFSETS = (-0.25, 0.0, 0.25)


class SmokingClassifier:
    """Zero-shot CLIP score: probability that a set of crops shows smoking."""

    def __init__(self, config: dict):
        import open_clip  # imported here so the rest of the package works without it
        import torch

        self.torch = torch
        self.device = config["device"]
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            config["model"], pretrained=config["weights"]
        )
        self.model = model.to(self.device).eval()

        tokenizer = open_clip.get_tokenizer(config["model"])
        tokens = tokenizer(SMOKING_PROMPTS + OTHER_PROMPTS).to(self.device)
        with torch.no_grad():
            text = self.model.encode_text(tokens)
        self.text = text / text.norm(dim=-1, keepdim=True)

    def score(self, crops: list[np.ndarray]) -> float:
        """Average smoking probability over BGR crops (as read by OpenCV)."""
        from PIL import Image

        images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops]
        batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with self.torch.no_grad():
            features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        probs = (100 * features @ self.text.T).softmax(dim=-1)
        return probs[:, : len(SMOKING_PROMPTS)].sum(dim=-1).mean().item()


def frame_times(cycle: Cycle) -> list[float]:
    """Moments inside the pause at the mouth, where the held object is visible."""
    return [cycle.peak_sec + offset * cycle.pause_sec for offset in FRAME_OFFSETS]


def cycle_crops(
    video_path: Path, cycle: Cycle, poses: pd.DataFrame, crop_size: float
) -> list[np.ndarray]:
    """Head crops of the cycle's person at `frame_times(cycle)`."""
    track = poses[poses["track_id"] == cycle.track_id]
    capture = cv2.VideoCapture(str(video_path))
    crops = []
    for time_sec in frame_times(cycle):
        capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        pose = track.iloc[(track["time_sec"] - time_sec).abs().argmin()]
        x1, y1, x2, y2 = head_crop(pose, frame.shape, crop_size)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            crops.append(frame[y1:y2, x1:x2])
    capture.release()
    return crops


def score_cycles(
    classifier: SmokingClassifier,
    video_path: Path,
    poses: pd.DataFrame,
    crop_size: float,
    cycles: list[Cycle],
) -> list[float]:
    """Smoking score for every cycle; NaN when no crop could be read."""
    scores = []
    for cycle in cycles:
        crops = cycle_crops(video_path, cycle, poses, crop_size)
        scores.append(classifier.score(crops) if crops else float("nan"))
    return scores

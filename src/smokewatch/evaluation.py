"""Step 5: event-level evaluation, following the task's matching rules.

A prediction matches a POSITIVE ground-truth event when
* both are in the same camera and clip,
* the prediction overlaps the GT interval extended by gt_extend_sec on both
  sides by at least min_overlap_sec,
* the person boxes have IoU >= min_iou.
Matching is one-to-one, largest temporal overlap first. Unmatched predictions
that touch an IGNORE interval are not counted as false positives.
"""

import numpy as np
import pandas as pd

from smokewatch.annotations import Label

BOX = ["x1", "y1", "x2", "y2"]


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two intervals (0 if they do not meet)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def box_iou(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    inter_w = overlap(a[0], a[2], b[0], b[2])
    inter_h = overlap(a[1], a[3], b[1], b[3])
    intersection = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_clip(predictions: pd.DataFrame, positives: pd.DataFrame, config: dict) -> list[tuple]:
    """Return one-to-one (prediction index, gt index) pairs for one clip."""
    candidates = []
    for p_index, pred in predictions.iterrows():
        for g_index, gt in positives.iterrows():
            time_overlap = overlap(
                pred["start_sec"],
                pred["end_sec"],
                gt["start_sec"] - config["gt_extend_sec"],
                gt["end_sec"] + config["gt_extend_sec"],
            )
            if time_overlap < config["min_overlap_sec"]:
                continue
            if box_iou(pred[BOX].tolist(), gt[BOX].tolist()) < config["min_iou"]:
                continue
            candidates.append((time_overlap, p_index, g_index))

    matches, used_preds, used_gts = [], set(), set()
    for _, p_index, g_index in sorted(candidates, reverse=True):
        if p_index not in used_preds and g_index not in used_gts:
            matches.append((p_index, g_index))
            used_preds.add(p_index)
            used_gts.add(g_index)
    return matches


def negative_seconds(clip_duration: float, gt_clip: pd.DataFrame) -> float:
    """Clip duration minus the union of POSITIVE and IGNORE intervals."""
    covered = gt_clip[gt_clip["label"].isin([Label.POSITIVE.value, Label.IGNORE.value])]
    intervals = sorted(zip(covered["start_sec"], covered["end_sec"], strict=True))
    total, current_end = 0.0, -np.inf
    for start, end in intervals:
        start = max(start, current_end)
        if end > start:
            total += end - start
            current_end = end
    return max(0.0, clip_duration - total)


def evaluate(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame, clips: pd.DataFrame, config: dict
) -> dict:
    """Compute precision, recall, F1, FP per hour and median latency."""
    tp = fp = fn = 0
    latencies = []
    negative_sec = 0.0

    for _, clip in clips.iterrows():
        clip_id = clip["clip_id"]
        preds = predictions[
            (predictions["clip_id"] == clip_id) & (predictions["camera_id"] == clip["camera_id"])
        ]
        gt_clip = ground_truth[ground_truth["clip_id"] == clip_id]
        positives = gt_clip[gt_clip["label"] == Label.POSITIVE.value]
        ignores = gt_clip[gt_clip["label"] == Label.IGNORE.value]

        matches = match_clip(preds, positives, config)
        matched_preds = {p for p, _ in matches}
        tp += len(matches)
        fn += len(positives) - len(matches)
        latencies += [
            max(0.0, preds.loc[p, "start_sec"] - positives.loc[g, "start_sec"]) for p, g in matches
        ]

        for p_index, pred in preds.iterrows():
            if p_index in matched_preds:
                continue
            touches_ignore = any(
                overlap(pred["start_sec"], pred["end_sec"], s, e) > 0
                for s, e in zip(ignores["start_sec"], ignores["end_sec"], strict=True)
            )
            if not touches_ignore:
                fp += 1

        negative_sec += negative_seconds(clip["duration_sec"], gt_clip)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_hours = negative_sec / 3600
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_per_hour": round(fp / negative_hours, 3) if negative_hours else None,
        "negative_hours": round(negative_hours, 3),
        "median_latency_sec": round(float(np.median(latencies)), 2) if latencies else None,
    }

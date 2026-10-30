import pandas as pd

from smokewatch.evaluation import box_iou, evaluate

CONFIG = {"gt_extend_sec": 3, "min_overlap_sec": 1, "min_iou": 0.3}
CLIPS = pd.DataFrame([{"clip_id": "c1", "camera_id": "cam", "duration_sec": 3600.0}])
BOX = {"x1": 0, "y1": 0, "x2": 100, "y2": 200}


def gt(start, end, label="POSITIVE"):
    return {"clip_id": "c1", "label": label, "start_sec": start, "end_sec": end, **BOX}


def pred(start, end, **box):
    return {"clip_id": "c1", "camera_id": "cam", "start_sec": start, "end_sec": end, **BOX, **box}


def run(preds, gts):
    return evaluate(pd.DataFrame(preds), pd.DataFrame(gts), CLIPS, CONFIG)


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 50 / 150
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0


def test_match_within_extended_window():
    metrics = run([pred(22, 30)], [gt(10, 20)])  # GT extended to 7-23: 1 s overlap
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 0, 0)
    assert metrics["median_latency_sec"] == 12


def test_wrong_person_box_is_fp_and_fn():
    metrics = run([pred(10, 20, x1=500, x2=600)], [gt(10, 20)])
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 1, 1)


def test_matching_is_one_to_one():
    metrics = run([pred(10, 20), pred(12, 18)], [gt(10, 20)])
    assert (metrics["tp"], metrics["fp"]) == (1, 1)


def test_predictions_in_ignore_intervals_are_not_counted():
    metrics = run([pred(100, 110)], [gt(10, 20), gt(95, 120, label="IGNORE")])
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 0, 1)


def test_fp_per_hour_uses_negative_time_only():
    metrics = run([pred(2000, 2010)], [gt(0, 1800)])
    assert metrics["negative_hours"] == 0.5
    assert metrics["fp_per_hour"] == 2

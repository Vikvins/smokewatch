import numpy as np

from smokewatch.cycles import find_cycles

NEAR, FAR = 0.6, 1.0


def signal(segments: list[tuple[float, float]], fps: int = 10):
    """Build (times, distances) from (duration_sec, distance) segments."""
    values = np.concatenate([np.full(int(d * fps), v) for d, v in segments])
    return np.arange(len(values)) / fps, values


def cycles_in(times, distances):
    return find_cycles(times, distances, 1, NEAR, FAR, min_pause_sec=0.5, max_pause_sec=5.0)


def test_single_cycle_is_found():
    times, distances = signal([(2, 2.0), (1.5, 0.3), (2, 2.0)])
    [cycle] = cycles_in(times, distances)
    assert cycle.pause_sec == 1.5
    assert cycle.start_sec < 2.0 < cycle.peak_sec < 3.5 <= cycle.end_sec
    assert cycle.min_distance == 0.3


def test_too_short_and_too_long_pauses_are_rejected():
    assert cycles_in(*signal([(1, 2.0), (0.2, 0.3), (1, 2.0)])) == []
    assert cycles_in(*signal([(1, 2.0), (8, 0.3), (1, 2.0)])) == []


def test_values_between_thresholds_do_not_end_a_pause():
    times, distances = signal([(1, 2.0), (1, 0.3), (0.5, 0.8), (1, 0.3), (1, 2.0)])
    assert len(cycles_in(times, distances)) == 1


def test_hand_already_at_mouth_at_start_is_not_a_cycle():
    assert cycles_in(*signal([(1, 0.3), (1, 2.0)])) == []


def test_nan_samples_are_skipped():
    times, distances = signal([(1, 2.0), (1, 0.3), (1, 2.0)])
    distances[12:15] = np.nan
    assert len(cycles_in(times, distances)) == 1

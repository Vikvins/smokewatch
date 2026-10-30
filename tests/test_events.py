from smokewatch.cycles import Cycle
from smokewatch.events import build_events

CONFIG = {"min_cycles": 2, "window_sec": 20, "merge_gap_sec": 15, "threshold": 0.5}


def cycle(start: float, track_id: int = 1, min_distance: float = 0.1) -> Cycle:
    return Cycle(track_id, start, start + 3, 1.5, start + 1.5, min_distance)


def test_two_cycles_make_one_event():
    [event] = build_events([cycle(0), cycle(8)], CONFIG, near=0.6)
    assert (event.start_sec, event.end_sec) == (0, 11)
    assert event.confidence > 0.5


def test_single_cycle_is_not_an_event():
    assert build_events([cycle(0)], CONFIG, near=0.6) == []


def test_cycles_far_apart_are_not_merged():
    assert build_events([cycle(0), cycle(30)], CONFIG, near=0.6) == []


def test_close_cycles_merge_into_one_long_event():
    [event] = build_events([cycle(0), cycle(8), cycle(20), cycle(33)], CONFIG, near=0.6)
    assert len(event.cycles) == 4


def test_different_people_are_never_merged():
    cycles = [cycle(0, track_id=1), cycle(5, track_id=2), cycle(10, track_id=1)]
    events = build_events(cycles, CONFIG, near=0.6)
    assert [event.track_id for event in events] == [1]


def test_weak_cycles_stay_below_threshold():
    weak = [cycle(0, min_distance=0.55), cycle(8, min_distance=0.55)]
    assert build_events(weak, CONFIG, near=0.6) == []

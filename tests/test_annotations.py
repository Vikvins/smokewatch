import pandas as pd

from smokewatch.annotations import validate_events

CLIPS = pd.DataFrame([{"clip_id": "c1", "duration_sec": 60.0, "width": 1920, "height": 1080}])


def make_event(**overrides) -> dict:
    event = {
        "clip_id": "c1",
        "event_id": "e01",
        "label": "POSITIVE",
        "start_sec": 10.0,
        "end_sec": 25.0,
        "peak_sec": 15.0,
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 900,
        "cycles": 2,
        "cue": "none",
        "negative_type": None,
        "comment": None,
    }
    event.update(overrides)
    return event


def problems_for(*events: dict) -> list[str]:
    return validate_events(pd.DataFrame(list(events)), CLIPS)


def test_valid_positive_has_no_problems():
    assert problems_for(make_event()) == []


def test_one_cycle_needs_a_direct_cue():
    assert problems_for(make_event(cycles=1, cue="none"))
    assert problems_for(make_event(cycles=1, cue="cigarette")) == []


def test_peak_outside_interval_is_reported():
    assert problems_for(make_event(peak_sec=30.0))


def test_box_outside_frame_is_reported():
    assert problems_for(make_event(x2=2500))


def test_negative_does_not_need_a_box():
    negative = make_event(
        label="NEGATIVE",
        peak_sec=None,
        x1=None,
        y1=None,
        x2=None,
        y2=None,
        cycles=0,
        cue="none",
        negative_type="drinking",
    )
    assert problems_for(negative) == []


def test_unknown_clip_and_duplicate_ids_are_reported():
    assert problems_for(make_event(clip_id="missing"))
    assert len(problems_for(make_event(), make_event())) == 2

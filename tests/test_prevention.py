"""Unit tests for the PREVENTION annotation parser and sample builder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from driving_world.benchmark.prevention import (
    Camera,
    LaneChange,
    SampleConfig,
    build_events,
    hard_negatives,
    lane_index,
    lane_name,
    load_lane_changes,
    read_camera,
    sample_negatives,
)

FPS = 10.0


def _track(frames, xs, depth=20.0):
    """vehicle -> frame -> (x right, y down, z forward)."""
    return {frame: np.array([x, 0.0, depth]) for frame, x in zip(frames, xs)}


def _change(vehicle=0, f0=100, f1=130, f2=160, direction="right"):
    return LaneChange(
        vehicle=vehicle,
        counterpart=0,
        direction=direction,
        f0=f0,
        f1=f1,
        f2=f2,
        blinker=True,
    )


# ------------------------------------------------------------------------ parsers


def test_load_lane_changes_parses_seven_fields():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "lane_changes.txt"
        path.write_text(
            "000000 000537 000003 002031 002052 002088 000001\n"
            "-00001 002981 000004 004980 004992 005019 000001\n",
            encoding="utf-8",
        )
        changes = load_lane_changes(path)
    assert len(changes) == 2
    assert changes[0].direction == "left"
    assert changes[0].blinker is True
    assert changes[1].direction == "right"
    assert changes[1].vehicle == -1


def test_load_lane_changes_skips_wrong_width():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "lane_changes.txt"
        path.write_text("1 2 3\n1 2 3 4 5 6 7\n", encoding="utf-8")
        assert len(load_lane_changes(path)) == 1


def test_read_camera_rejects_wrong_frame_size():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "calib.dat"
        path.write_text(
            "ImageSize 1280 720\n"
            "IntrinsicParams 100.0 0.0 50.0 100.0 40.0\n"
            "DistortionCoefficients 0.0 0.0 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        try:
            read_camera(path)
        except ValueError as error:
            assert "frame size" in str(error)
        else:  # pragma: no cover
            raise AssertionError("expected a ValueError for a 1280x720 calibration")


def test_camera_matrix_uses_principal_point():
    camera = Camera(1460.99, 1467.05, 970.19, 314.89, (0.0, 0.0, 0.0, 0.0, 0.0))
    matrix = camera.matrix
    assert matrix[0, 0] == 1460.99
    assert matrix[0, 2] == 970.19
    assert matrix[1, 2] == 314.89


# ------------------------------------------------------------------- lane geometry


def test_lane_index_is_right_positive():
    assert lane_index(0.0) == 0
    assert lane_index(1.7) == 0  # still inside the ego lane
    assert lane_index(1.9) == 1
    assert lane_index(-1.9) == -1
    assert lane_index(3.6) == 1
    assert lane_index(5.4) == 2


def test_lane_name_falls_back_beyond_the_three_near_lanes():
    assert lane_name(0) == "ego"
    assert lane_name(-1) == "left"
    assert lane_name(2) == "far_right"
    assert lane_name(4) == "lane+4"


# ------------------------------------------------------------------ event building


def test_cut_in_from_left_adjacent_lane():
    # subject sits 3.5 m to the left and drifts right into the ego lane
    frames = list(range(0, 200))
    xs = np.interp(frames, [100, 160], [-3.5, 0.0])
    events = build_events("RECORD1_DRIVE1", [_change()], {0: _track(frames, xs)})
    assert len(events) == 1
    assert events[0].stratum == "cut_in"
    assert events[0].start_lane == "left"
    assert events[0].end_lane == "ego"
    assert events[0].label == 1


def test_cut_out_from_ego_lane():
    frames = list(range(0, 200))
    xs = np.interp(frames, [100, 160], [0.0, 3.5])
    events = build_events("RECORD1_DRIVE1", [_change()], {0: _track(frames, xs)})
    assert events[0].stratum == "cut_out"
    assert events[0].start_lane == "ego"


def test_short_dropout_at_crossing_is_interpolated():
    frames = list(range(0, 200))
    frame_map = _track(frames, np.interp(frames, [100, 160], [3.5, 0.0]))
    del frame_map[130]  # one missing crossing frame is bridged
    events = build_events("RECORD1_DRIVE1", [_change()], {0: frame_map})
    assert len(events) == 1


def test_wide_dropout_at_crossing_drops_the_event():
    frames = list(range(0, 200))
    frame_map = _track(frames, np.interp(frames, [100, 160], [3.5, 0.0]))
    for frame in range(120, 142):  # wider than tolerance_frames
        frame_map.pop(frame, None)
    assert build_events("RECORD1_DRIVE1", [_change()], {0: frame_map}) == []


def test_missing_f0_uses_a_fixed_lookback():
    frames = list(range(0, 200))
    # starts in the right lane, crosses at 130, no annotated f0
    frame_map = _track(frames, np.interp(frames, [90, 160], [3.5, 0.0]))
    change = _change(f0=-1)
    events = build_events("RECORD1_DRIVE1", [change], {0: frame_map})
    assert len(events) == 1
    assert events[0].start_lane == "right"
    assert events[0].end_lane == "ego"


def test_event_dropped_when_no_lane_change():
    frames = list(range(0, 200))
    xs = np.full(200, 3.0)  # parked in the left lane the whole time
    assert build_events("RECORD1_DRIVE1", [_change()], {0: _track(frames, xs)}) == []


def test_event_dropped_when_too_far():
    frames = list(range(0, 200))
    xs = np.interp(frames, [100, 160], [3.5, 0.0])
    events = build_events(
        "RECORD1_DRIVE1",
        [_change()],
        {0: _track(frames, xs, depth=150.0)},
        SampleConfig(max_depth_m=100.0),
    )
    assert events == []


def test_event_dropped_when_two_lanes_jumped():
    frames = list(range(0, 200))
    xs = np.interp(frames, [100, 160], [7.0, 0.0])  # two lanes in one go
    assert build_events("RECORD1_DRIVE1", [_change()], {0: _track(frames, xs)}) == []


def test_window_keeps_crossing_pre_frames_in():
    frames = list(range(0, 200))
    xs = np.interp(frames, [100, 160], [3.5, 0.0])
    config = SampleConfig(clip_frames=80, pre_frames=45)
    events = build_events("RECORD1_DRIVE1", [_change()], {0: _track(frames, xs)}, config)
    event = events[0]
    assert event.clip_start == 130 - 45
    assert event.clip_start + event.clip_frames > event.f2


def test_window_clamped_at_zero():
    frames = list(range(0, 200))
    xs = np.interp(frames, [20, 60], [3.5, 0.0])
    config = SampleConfig(clip_frames=80, pre_frames=45)
    events = build_events(
        "RECORD1_DRIVE1", [_change(f0=10, f1=40, f2=70)], {0: _track(frames, xs)}, config
    )
    assert events[0].clip_start == 0


# ----------------------------------------------------------------- negative sampler


def test_negatives_avoid_the_labelled_maneuver():
    frames = list(range(0, 400))
    tracks = {7: _track(frames, np.zeros(400))}
    change = _change(vehicle=7, f0=200, f1=230, f2=260)
    negatives = sample_negatives(
        "RECORD1_DRIVE1", [change], tracks, 5, seed=3, config=SampleConfig(clip_frames=80)
    )
    assert negatives
    for negative in negatives:
        assert negative.label == 0
        # the window plus its guard must clear the annotation
        assert negative.clip_start + 80 < 200 - 50


def test_negatives_are_reproducible_for_a_seed():
    frames = list(range(0, 400))
    tracks = {7: _track(frames, np.zeros(400))}
    change = _change(vehicle=7, f0=200, f1=230, f2=260)
    first = sample_negatives("RECORD1_DRIVE1", [change], tracks, 3, seed=11)
    second = sample_negatives("RECORD1_DRIVE1", [change], tracks, 3, seed=11)
    assert [e.clip_start for e in first] == [e.clip_start for e in second]


def test_hard_negatives_sit_before_the_maneuver():
    frames = list(range(0, 400))
    tracks = {7: _track(frames, np.interp(frames, [200, 260], [3.5, 0.0]))}
    event = build_events("RECORD1_DRIVE1", [_change(vehicle=7, f0=200, f1=230, f2=260)], tracks)[0]
    hards = hard_negatives("RECORD1_DRIVE1", [event], tracks)
    assert len(hards) == 1
    assert hards[0].clip_start + hards[0].clip_frames == 200 - 40
    assert hards[0].stratum == "negative_hard"


def test_no_hard_negative_when_the_track_starts_too_late():
    frames = list(range(150, 400))
    tracks = {7: _track(frames, np.interp(frames, [200, 260], [3.5, 0.0]))}
    event = build_events("RECORD1_DRIVE1", [_change(vehicle=7, f0=200, f1=230, f2=260)], tracks)[0]
    assert hard_negatives("RECORD1_DRIVE1", [event], tracks) == []



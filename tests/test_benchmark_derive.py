"""Unit tests for GT event derivation from object tracks."""

from __future__ import annotations

import numpy as np

from driving_world.benchmark.derive import (
    CYCLIST,
    PEDESTRIAN,
    VEHICLE,
    DeriveConfig,
    Track,
    derive_events,
)


def _track(track_id, type_, times, xs, ys):
    return Track(track_id, type_, np.asarray(times, dtype=float), np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))


def _lane_change_y(times, start, end, y0, y1):
    return np.interp(times, [start, end], [y0, y1])


def test_lane_change_detected():
    times = np.arange(0, 8, 0.1)
    ys = _lane_change_y(times, 2.0, 4.0, -3.5, 0.0)
    xs = 20 + 5 * times  # moving ahead, ahead of ego
    track = _track("v1", VEHICLE, times, xs, ys)
    events = derive_events([track])
    kinds = [e.kind for e in events]
    assert "vehicle_lane_change" in kinds
    event = next(e for e in events if e.kind == "vehicle_lane_change")
    assert event.direction == "right_to_left"


def test_no_lane_change_when_straight():
    times = np.arange(0, 8, 0.1)
    track = _track("v1", VEHICLE, times, 20 + 5 * times, np.full_like(times, -3.5))
    assert derive_events([track]) == []


def test_pedestrian_crossing_detected():
    times = np.arange(0, 8, 0.1)
    # pedestrian starts on the right sidewalk and crosses to the left lane
    ys = _lane_change_y(times, 2.0, 4.0, 4.0, -4.0)
    xs = np.full_like(times, 12.0)  # ahead, stationary-ish
    track = _track("p1", PEDESTRIAN, times, xs, ys)
    events = derive_events([track])
    kinds = [e.kind for e in events]
    assert "pedestrian_crossing" in kinds


def test_hard_brake_detected():
    times = np.arange(0, 8, 0.1)
    # forward distance plateaus around t=3 (speed drops to ~0)
    xs = 20 + 12 * times
    xs[times > 3.0] = xs[times <= 3.0][-1]
    track = _track("v1", VEHICLE, times, xs, np.full_like(times, -3.5))
    events = derive_events([track])
    kinds = [e.kind for e in events]
    assert "vehicle_hard_brake" in kinds


def test_cyclist_lane_change_detected():
    times = np.arange(0, 8, 0.1)
    ys = _lane_change_y(times, 2.0, 4.0, 0.0, 3.5)
    track = _track("c1", CYCLIST, times, 15 + 5 * times, ys)
    events = derive_events([track])
    assert any(e.kind == "vehicle_lane_change" for e in events)

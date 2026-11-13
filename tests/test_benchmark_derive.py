"""Unit tests for GT event derivation from world tracks."""

from __future__ import annotations

import numpy as np

from driving_world.benchmark.derive import (
    CYCLIST,
    PEDESTRIAN,
    VEHICLE,
    DeriveConfig,
    WorldTrack,
    derive_events,
)


def _world_track(track_id, type_, times, xs, ys, *, apparent=100.0):
    return WorldTrack(
        track_id, type_, np.asarray(times, dtype=float),
        np.asarray(xs, dtype=float), np.asarray(ys, dtype=float),
        np.full_like(times, apparent, dtype=float),
    )


def _interp(times, start, end, v0, v1):
    return np.interp(times, [start, end], [v0, v1])


def test_lane_change_from_world_y():
    times = np.arange(0, 8, 0.1)
    ys = _interp(times, 2.0, 4.0, -3.5, 0.0)  # y increases => right_to_left
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, ys)
    events = derive_events([world])
    kinds = [e.kind for e in events]
    assert "vehicle_lane_change" in kinds
    event = next(e for e in events if e.kind == "vehicle_lane_change")
    assert event.direction == "right_to_left"


def test_no_lane_change_when_straight():
    times = np.arange(0, 8, 0.1)
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, np.full_like(times, -3.5))
    assert derive_events([world]) == []


def test_too_small_ignored():
    times = np.arange(0, 8, 0.1)
    ys = _interp(times, 2.0, 4.0, -3.5, 0.0)
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, ys, apparent=10.0)
    assert derive_events([world]) == []


def test_slow_drift_ignored():
    times = np.arange(0, 14, 0.1)
    ys = _interp(times, 1.0, 13.0, -3.5, -0.5)  # 3m over 12s = drift
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, ys)
    assert derive_events([world]) == []


def test_single_point_track_ignored():
    times = np.array([1.0])
    world = _world_track("v1", VEHICLE, times, np.array([20.0]), np.array([0.0]))
    assert derive_events([world]) == []


def test_pedestrian_crossing_detected():
    times = np.arange(0, 8, 0.1)
    ys = _interp(times, 2.0, 4.0, 4.0, 0.0)  # intrudes ego lane (|y|<1.5)
    world = _world_track("p1", PEDESTRIAN, times, np.full_like(times, 12.0), ys)
    events = derive_events([world])
    assert any(e.kind == "pedestrian_crossing" for e in events)


def test_hard_brake_from_world_speed():
    times = np.arange(0, 8, 0.1)
    xs = 20 + 12 * times
    xs[times > 3.0] = xs[times <= 3.0][-1]  # forward distance plateaus
    world = _world_track("v1", VEHICLE, times, xs, np.full_like(times, -3.5))
    events = derive_events([world])
    assert any(e.kind == "vehicle_hard_brake" for e in events)


def test_cyclist_lane_change():
    times = np.arange(0, 8, 0.1)
    ys = _interp(times, 2.0, 4.0, 3.5, 0.0)  # y decreases => left_to_right
    world = _world_track("c1", CYCLIST, times, 15 + 5 * times, ys)
    events = derive_events([world])
    event = next(e for e in events if e.kind == "vehicle_lane_change")
    assert event.direction == "left_to_right"

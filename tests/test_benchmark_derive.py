"""Unit tests for GT event derivation from world + camera tracks."""

from __future__ import annotations

import numpy as np

from driving_world.benchmark.derive import (
    CYCLIST,
    PEDESTRIAN,
    VEHICLE,
    CameraTrack,
    DeriveConfig,
    WorldTrack,
    derive_events,
)


def _camera_track(track_id, type_, times, *, area=5000.0):
    size = np.sqrt(area)
    return CameraTrack(
        track_id, type_, np.asarray(times, dtype=float),
        np.full_like(times, 960.0, dtype=float),
        np.full_like(times, 640.0, dtype=float),
        np.full_like(times, size, dtype=float),
        np.full_like(times, size, dtype=float),
    )


def _world_track(track_id, type_, times, xs, ys):
    return WorldTrack(
        track_id, type_, np.asarray(times, dtype=float),
        np.asarray(xs, dtype=float), np.asarray(ys, dtype=float),
    )


def _interp(times, start, end, v0, v1):
    return np.interp(times, [start, end], [v0, v1])


def test_lane_change_from_world_y():
    times = np.arange(0, 8, 0.1)
    ys = _interp(times, 2.0, 4.0, -3.5, 0.0)  # y increases => right_to_left
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, ys)
    camera = _camera_track("v1", VEHICLE, times)
    events = derive_events([world], [camera])
    kinds = [e.kind for e in events]
    assert "vehicle_lane_change" in kinds
    event = next(e for e in events if e.kind == "vehicle_lane_change")
    assert event.direction == "right_to_left"


def test_no_lane_change_when_straight():
    times = np.arange(0, 8, 0.1)
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, np.full_like(times, -3.5))
    camera = _camera_track("v1", VEHICLE, times)
    assert derive_events([world], [camera]) == []


def test_not_visible_ignored():
    times = np.arange(0, 8, 0.1)
    ys = _interp(times, 2.0, 4.0, -3.5, 0.0)
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, ys)
    camera = _camera_track("v1", VEHICLE, times, area=100.0)  # too small/distant
    assert derive_events([world], [camera]) == []


def test_slow_drift_ignored():
    times = np.arange(0, 14, 0.1)
    ys = _interp(times, 1.0, 13.0, -3.5, -0.5)  # 3m over 12s = drift
    world = _world_track("v1", VEHICLE, times, 20 + 6 * times, ys)
    camera = _camera_track("v1", VEHICLE, times)
    assert derive_events([world], [camera]) == []


def test_pedestrian_crossing_detected():
    times = np.arange(0, 8, 0.1)
    # pedestrian ahead (x=12) whose lateral |y| drops below 1.5m (intrudes lane)
    ys = _interp(times, 2.0, 4.0, 4.0, 0.0)
    world = _world_track("p1", PEDESTRIAN, times, np.full_like(times, 12.0), ys)
    camera = _camera_track("p1", PEDESTRIAN, times)
    events = derive_events([world], [camera])
    assert any(e.kind == "pedestrian_crossing" for e in events)


def test_hard_brake_from_world_speed():
    times = np.arange(0, 8, 0.1)
    xs = 20 + 12 * times
    xs[times > 3.0] = xs[times <= 3.0][-1]  # forward distance plateaus
    world = _world_track("v1", VEHICLE, times, xs, np.full_like(times, -3.5))
    camera = _camera_track("v1", VEHICLE, times)
    events = derive_events([world], [camera])
    assert any(e.kind == "vehicle_hard_brake" for e in events)


def test_cyclist_lane_change():
    times = np.arange(0, 8, 0.1)
    ys = _interp(times, 2.0, 4.0, 3.5, 0.0)  # y decreases => left_to_right
    world = _world_track("c1", CYCLIST, times, 15 + 5 * times, ys)
    camera = _camera_track("c1", CYCLIST, times)
    events = derive_events([world], [camera])
    event = next(e for e in events if e.kind == "vehicle_lane_change")
    assert event.direction == "left_to_right"

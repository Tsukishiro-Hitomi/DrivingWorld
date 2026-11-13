"""Unit tests for GT event derivation from camera-label and world tracks."""

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


def _camera_track(track_id, type_, times, us, *, area=5000.0):
    vs = np.full_like(times, 500.0)
    ws = np.full_like(times, np.sqrt(area))
    hs = np.full_like(times, np.sqrt(area))
    return CameraTrack(
        track_id, type_, np.asarray(times, dtype=float),
        np.asarray(us, dtype=float), vs, ws, hs,
    )


def _world_track(track_id, type_, times, xs, ys=0.0):
    return WorldTrack(
        track_id, type_, np.asarray(times, dtype=float),
        np.asarray(xs, dtype=float), np.full_like(times, ys, dtype=float),
    )


def _lane_change_u(times, start, end, u0, u1):
    return np.interp(times, [start, end], [u0, u1])


def test_lane_change_from_2d():
    times = np.arange(0, 8, 0.1)
    us = _lane_change_u(times, 2.0, 4.0, 600.0, 300.0)  # u decreases => right_to_left
    track = _camera_track("v1", VEHICLE, times, us)
    events = derive_events([track], [])
    kinds = [e.kind for e in events]
    assert "vehicle_lane_change" in kinds
    event = next(e for e in events if e.kind == "vehicle_lane_change")
    assert event.direction == "right_to_left"


def test_no_lane_change_when_straight():
    times = np.arange(0, 8, 0.1)
    track = _camera_track("v1", VEHICLE, times, np.full_like(times, 600.0))
    assert derive_events([track], []) == []


def test_small_box_ignored():
    times = np.arange(0, 8, 0.1)
    us = _lane_change_u(times, 2.0, 4.0, 600.0, 300.0)
    track = _camera_track("v1", VEHICLE, times, us, area=100.0)  # too small/distant
    assert derive_events([track], []) == []


def test_pedestrian_crossing_from_2d():
    times = np.arange(0, 8, 0.1)
    us = _lane_change_u(times, 2.0, 4.0, 1000.0, 700.0)
    track = _camera_track("p1", PEDESTRIAN, times, us)
    events = derive_events([track], [])
    assert any(e.kind == "pedestrian_crossing" for e in events)


def test_hard_brake_from_world_speed():
    times = np.arange(0, 8, 0.1)
    xs = 20 + 12 * times
    xs[times > 3.0] = xs[times <= 3.0][-1]  # forward distance plateaus
    world = _world_track("v1", VEHICLE, times, xs)
    camera = _camera_track("v1", VEHICLE, times, np.full_like(times, 600.0))
    events = derive_events([camera], [world])
    assert any(e.kind == "vehicle_hard_brake" for e in events)


def test_cyclist_lane_change():
    times = np.arange(0, 8, 0.1)
    us = _lane_change_u(times, 2.0, 4.0, 300.0, 600.0)  # u increases => left_to_right
    track = _camera_track("c1", CYCLIST, times, us)
    events = derive_events([track], [])
    event = next(e for e in events if e.kind == "vehicle_lane_change")
    assert event.direction == "left_to_right"

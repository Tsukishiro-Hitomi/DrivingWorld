"""Unit tests for the shared 3D geometry."""

from __future__ import annotations

import numpy as np

from driving_world.geometry import Camera, interpolate_track, project_point, rotation


def test_rotation_identity():
    assert np.allclose(rotation(0.0), np.eye(3))


def test_rotation_yaw_maps_forward_to_left():
    # +x (forward) rotated by +90 deg yaw becomes +y (left).
    forward = np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotation(np.pi / 2) @ forward, [0.0, 1.0, 0.0], atol=1e-9)


def test_interpolate_track_midpoint():
    track = [
        {"t": 0, "position": [0, 0, 0], "yaw": 0},
        {"t": 2, "position": [10, 0, 0], "yaw": 0},
    ]
    position, yaw = interpolate_track(track, 1.0)
    assert np.allclose(position, [5, 0, 0])
    assert yaw == 0.0


def test_interpolate_track_holds_endpoints():
    track = [
        {"t": 1, "position": [1, 0, 0], "yaw": 0},
        {"t": 3, "position": [3, 0, 0], "yaw": 0},
    ]
    assert np.allclose(interpolate_track(track, 0.0)[0], [1, 0, 0])
    assert np.allclose(interpolate_track(track, 5.0)[0], [3, 0, 0])


def test_project_point_centers_forward_point():
    camera = Camera(width=832, height=480, fov_degrees=82)
    uv = project_point([10.0, 0.0, 1.5], [0.0, 0.0, 0.0], 0.0, camera)
    assert uv is not None
    assert np.allclose(uv, [416.0, 240.0], atol=1e-6)


def test_project_point_behind_camera_is_none():
    camera = Camera(width=832, height=480, fov_degrees=82)
    assert project_point([-1.0, 0.0, 1.5], [0.0, 0.0, 0.0], 0.0, camera) is None


def test_project_point_left_world_is_left_image():
    camera = Camera(width=832, height=480, fov_degrees=82)
    left = project_point([10.0, 2.0, 1.5], [0.0, 0.0, 0.0], 0.0, camera)
    right = project_point([10.0, -2.0, 1.5], [0.0, 0.0, 0.0], 0.0, camera)
    assert left is not None and right is not None
    assert left[0] < 416.0 < right[0]

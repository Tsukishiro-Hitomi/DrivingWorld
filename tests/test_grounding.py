"""Unit tests for the deterministic grounded object check."""

from __future__ import annotations

import numpy as np

from driving_world.judge.grounding import (
    GroundedConfig,
    Trajectory,
    evaluate,
    gt_trajectory,
    track_subject_by_color,
)

FPS = 24.0
WIDTH = 832.0


def _trajectory(times: np.ndarray, lateral_px: np.ndarray) -> Trajectory:
    points = np.column_stack(
        [WIDTH / 2.0 - np.asarray(lateral_px), np.full(len(times), 240.0)]
    )
    return Trajectory(times=np.asarray(times), points=points, fps=FPS)


def _lateral(times: np.ndarray, segments: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear lateral position: segments are (time, value) keyframes."""
    result = np.interp(
        times,
        [t for t, _ in segments],
        [v for _, v in segments],
    )
    return result


EVENT = {
    "event_id": "lane_change_test",
    "kind": "vehicle_lane_change",
    "subject_type": "vehicle",
    "time_window_seconds": [2.0, 4.0],
    "direction": "right_to_left",
    "target": "ego_lane",
}

_TIMES = np.arange(0.0, 5.0 + 1e-6, 1.0 / FPS)  # 121 samples


def test_observed():
    lateral = _lateral(_TIMES, [(0, -42), (2, -42), (4, 0), (5, 0)])
    result = evaluate(_trajectory(_TIMES, lateral), EVENT, WIDTH)
    assert result.status == "observed"
    assert result.checks["subject_visible"] is True
    assert result.checks["motion_observed"] is True
    assert result.checks["direction_correct"] is True
    assert result.checks["timing_valid"] is True
    assert result.checks["event_completed"] is True


def test_no_motion():
    lateral = _lateral(_TIMES, [(0, -42), (5, -42)])
    result = evaluate(_trajectory(_TIMES, lateral), EVENT, WIDTH)
    assert result.status == "not_observed"
    assert result.checks["subject_visible"] is True
    assert result.checks["motion_observed"] is False
    assert result.checks["direction_correct"] is None


def test_wrong_direction():
    lateral = _lateral(_TIMES, [(0, 42), (2, 42), (4, 0), (5, 0)])
    result = evaluate(_trajectory(_TIMES, lateral), EVENT, WIDTH)
    assert result.status == "partially_observed"
    assert result.checks["direction_correct"] is False
    assert result.checks["timing_valid"] is True
    assert result.checks["event_completed"] is True


def test_wrong_timing():
    lateral = _lateral(_TIMES, [(0, -42), (3.6, -42), (4.9, 0), (5, 0)])
    result = evaluate(_trajectory(_TIMES, lateral), EVENT, WIDTH)
    assert result.status == "partially_observed"
    assert result.checks["direction_correct"] is True
    assert result.checks["timing_valid"] is False
    assert result.checks["event_completed"] is True


def test_gt_trajectory_projects_lane_change():
    scene = {
        "duration": 5.0,
        "fps": 24,
        "camera": {
            "width": 832,
            "height": 480,
            "fov_degrees": 82,
            "position": [1.5, 0, 1.5],
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
        },
        "ego": [
            {"t": 0, "position": [0, 0, 0], "yaw": 0},
            {"t": 5.0, "position": [25.0, 0, 0], "yaw": 0},
        ],
        "actors": [
            {
                "id": "target_vehicle",
                "type": "vehicle",
                "size": [4.5, 1.8, 1.6],
                "track": [
                    {"t": 0, "position": [20, -3.5, 0.8], "yaw": 0},
                    {"t": 2.0, "position": [32.0, -3.5, 0.8], "yaw": 0},
                    {"t": 4.0, "position": [44.0, 0.0, 0.8], "yaw": 0},
                    {"t": 5, "position": [50, 0.0, 0.8], "yaw": 0},
                ],
            }
        ],
    }
    gt = gt_trajectory(scene, "target_vehicle")
    lateral = gt.lateral(WIDTH)
    # The vehicle starts right (negative lateral) and ends centered (lateral ~0).
    assert lateral[0] < -30
    assert abs(lateral[-1]) < 5
    # GT is fully visible in front of the camera across the clip.
    assert gt.visible().all()


def test_track_subject_by_color_recovers_centroid():
    frame = np.zeros((480, 832, 3), dtype=np.uint8)
    # A blue patch (proxy vehicle color) around (u, v) = (200, 300).
    frame[290:310, 190:210] = (65, 166, 255)
    tracked = track_subject_by_color([frame], (65, 166, 255), fps=FPS)
    assert np.allclose(tracked.points[0], [200.0, 300.0], atol=1.0)


def test_track_subject_by_color_ignores_background():
    frame = np.full((480, 832, 3), (5, 9, 15), dtype=np.uint8)
    tracked = track_subject_by_color([frame], (65, 166, 255), fps=FPS)
    assert np.isnan(tracked.points[0]).all()

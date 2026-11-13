"""Unit tests for the deterministic grounded object check."""

from __future__ import annotations

import numpy as np

from driving_world.judge.grounding import (
    GtTrack,
    Trajectory,
    evaluate,
    gt_track,
    track_subject_by_color,
)

FPS = 24.0

EVENT = {
    "event_id": "lane_change_test",
    "kind": "vehicle_lane_change",
    "subject_type": "vehicle",
    "time_window_seconds": [2.0, 4.0],
    "direction": "right_to_left",
    "target": "ego_lane",
}


def _scene(y_keyframes: list[tuple[float, float]]) -> dict:
    """A lane-change scene whose actor follows the given (time, world-y) keyframes.

    The actor and ego both drive forward, so the actor recedes slightly in depth;
    this reproduces the perspective drift that naive pixel-lateral checks confuse
    with a real lane change.
    """
    track = [{"t": t, "position": [20 + 6 * t, y, 0.8], "yaw": 0.0} for t, y in y_keyframes]
    return {
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
            {"t": 5.0, "position": [25, 0, 0], "yaw": 0},
        ],
        "actors": [
            {"id": "target_vehicle", "type": "vehicle", "size": [4.5, 1.8, 1.6], "track": track}
        ],
    }


def _perfect_tracked(gt: GtTrack) -> Trajectory:
    """A tracked trajectory that exactly matches the GT (a faithful render)."""
    return Trajectory(times=gt.times.copy(), points=gt.image.copy(), fps=FPS)


def _run(y_keyframes):
    gt = gt_track(_scene(y_keyframes), "target_vehicle")
    return evaluate(_perfect_tracked(gt), EVENT, gt)


def test_observed():
    result = _run([(0, -3.5), (2, -3.5), (4, 0.0), (5, 0.0)])
    assert result.status == "observed"
    assert result.checks == {
        "subject_visible": True,
        "motion_observed": True,
        "direction_correct": True,
        "timing_valid": True,
        "event_completed": True,
        "matches_proxy_trajectory": True,
    }


def test_no_motion():
    # Constant world-y: perspective drift must not be read as a lane change.
    result = _run([(0, -3.5), (5, -3.5)])
    assert result.status == "not_observed"
    assert result.checks["subject_visible"] is True
    assert result.checks["motion_observed"] is False


def test_wrong_direction():
    result = _run([(0, 3.5), (2, 3.5), (4, 0.0), (5, 0.0)])
    assert result.status == "partially_observed"
    assert result.checks["direction_correct"] is False
    assert result.checks["timing_valid"] is True


def test_wrong_timing():
    result = _run([(0, -3.5), (3.6, -3.5), (4.9, 0.0), (5, 0.0)])
    assert result.status == "partially_observed"
    assert result.checks["direction_correct"] is True
    assert result.checks["timing_valid"] is False


def test_gt_track_exposes_world_and_depth():
    gt = gt_track(_scene([(0, -3.5), (5, -3.5)]), "target_vehicle")
    assert gt.world.shape[1] == 3
    assert gt.visible().all()
    # The subject stays in the right lane (negative world-y) the whole clip.
    assert (gt.world[:, 1] < -3.0).all()


def test_track_subject_by_color_recovers_centroid():
    frame = np.zeros((480, 832, 3), dtype=np.uint8)
    frame[290:310, 190:210] = (65, 166, 255)
    tracked = track_subject_by_color([frame], (65, 166, 255), fps=FPS)
    assert np.allclose(tracked.points[0], [200.0, 300.0], atol=1.0)


def test_track_subject_by_color_ignores_background():
    frame = np.full((480, 832, 3), (5, 9, 15), dtype=np.uint8)
    tracked = track_subject_by_color([frame], (65, 166, 255), fps=FPS)
    assert np.isnan(tracked.points[0]).all()

"""Unit tests for the subject trackers (association/interpolation logic)."""

from __future__ import annotations

import numpy as np

from driving_world.judge.tracking import ColorTracker, DetectionTracker


class _FakeDetector:
    def __init__(self):
        pass

    def detect(self, image):
        return []


def _tracker(sample_step: int = 6, gate_px: float = 100.0) -> DetectionTracker:
    return DetectionTracker(_FakeDetector(), sample_step=sample_step, gate_px=gate_px)


def test_interpolates_between_detections():
    tracker = _tracker(sample_step=6)
    points = tracker._associate_and_interpolate(
        12, {0: (100.0, 200.0), 6: (160.0, 200.0)}
    )
    assert np.allclose(points[0], [100.0, 200.0])
    assert np.allclose(points[6], [160.0, 200.0])
    assert np.allclose(points[3], [130.0, 200.0])  # halfway
    assert np.isnan(points[7]).all()  # nothing detected after frame 6


def test_rejects_far_jump():
    tracker = _tracker(sample_step=6, gate_px=100.0)
    points = tracker._associate_and_interpolate(
        12, {0: (100.0, 100.0), 6: (500.0, 500.0)}
    )
    assert np.allclose(points[0], [100.0, 100.0])
    assert np.isnan(points[6]).all()  # jump rejected


def test_accepts_small_jump():
    tracker = _tracker(sample_step=6, gate_px=100.0)
    points = tracker._associate_and_interpolate(
        12, {0: (100.0, 100.0), 6: (150.0, 100.0)}
    )
    assert np.allclose(points[6], [150.0, 100.0])


def test_color_tracker_stores_color():
    tracker = ColorTracker((65, 166, 255))
    assert tracker.name == "color"
    assert tracker.color == (65, 166, 255)

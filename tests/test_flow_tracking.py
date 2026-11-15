"""Unit tests for box-seeded tracking and the image-space reproduction check."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from driving_world.judge.grounding import Trajectory, evaluate, grounded_check, gt_track
from driving_world.judge.tracking import FlowTracker, transform_box

FPS = 24.0

EVENT = {
    "event_id": "lane_change_test",
    "kind": "vehicle_lane_change",
    "subject_type": "vehicle",
    "time_window_seconds": [1.0, 4.0],
    "direction": "right_to_left",
    "target": "ego_lane",
}


def _scene(y_keyframes: list[tuple[float, float]]) -> dict:
    track = [{"t": t, "position": [20 + 4 * t, y, 0.8], "yaw": 0.0} for t, y in y_keyframes]
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
            {"t": 5.0, "position": [20, 0, 0], "yaw": 0},
        ],
        "actors": [
            {
                "id": "target_vehicle",
                "type": "vehicle",
                "size": [4.5, 1.8, 1.6],
                "track": track,
            }
        ],
    }


class _BoxSeededTracker:
    """Stand-in tracker: reports the GT boxes and records how it was seeded."""

    name = "fake"

    def __init__(self, gt, shift_px: float = 0.0):
        self.gt = gt
        self.shift_px = shift_px
        self.seed_box = None
        self.seed_frame = 0

    def track(self, video: Path) -> Trajectory:
        if self.seed_box is None:
            raise AssertionError("the judge must seed a box tracker before tracking")
        points = np.stack(
            (
                self.gt.boxes[:, 0] + self.gt.boxes[:, 2] / 2.0,
                self.gt.boxes[:, 1] + self.gt.boxes[:, 3] / 2.0,
            ),
            axis=1,
        )
        points[:, 0] += self.shift_px
        return Trajectory(
            times=self.gt.times.copy(),
            points=points,
            fps=FPS,
            widths=self.gt.boxes[:, 2].copy(),
        )


def test_transform_box_translates():
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    assert np.allclose(transform_box((10.0, 20.0, 40.0, 30.0), matrix), [15.0, 17.0, 40.0, 30.0])


def test_transform_box_scales_about_the_origin():
    matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert np.allclose(transform_box((0.0, 0.0, 10.0, 10.0), matrix), [0.0, 0.0, 20.0, 20.0])


def test_flow_tracker_requires_a_seed():
    try:
        FlowTracker().track(Path("unused.mp4"))
    except ValueError as error:
        assert "seed_box" in str(error)
    else:
        raise AssertionError("FlowTracker must refuse to run without a seed box")


def test_gt_track_projects_the_actor_box():
    gt = gt_track(_scene([(0, -3.5), (5, -3.5)]), "target_vehicle")
    assert gt.boxes is not None
    assert np.isfinite(gt.boxes).all()
    # The projected actor centre falls inside its projected box.
    inside_u = (gt.image[:, 0] >= gt.boxes[:, 0]) & (
        gt.image[:, 0] <= gt.boxes[:, 0] + gt.boxes[:, 2]
    )
    inside_v = (gt.image[:, 1] >= gt.boxes[:, 1]) & (
        gt.image[:, 1] <= gt.boxes[:, 1] + gt.boxes[:, 3]
    )
    assert (inside_u & inside_v).all()
    assert (gt.boxes[:, 2] > 10.0).all()


def test_gt_track_box_shrinks_with_distance():
    gt = gt_track(_scene([(0, -3.5), (5, -3.5)]), "target_vehicle")
    assert gt.boxes[-1, 2] < gt.boxes[0, 2]


def test_judge_seeds_and_reports_image_space_metrics():
    scene = _scene([(0, -3.5), (2, -3.5), (4, 0.0), (5, 0.0)])
    gt = gt_track(scene, "target_vehicle")
    tracker = _BoxSeededTracker(gt)
    result = grounded_check(Path("unused.mp4"), scene, EVENT, "target_vehicle", tracker)

    assert tracker.seed_box is not None
    assert result.status == "observed"
    assert abs(result.metrics["lateral_error_box_widths"]) < 1e-6
    assert abs(result.metrics["width_ratio"] - 1.0) < 1e-6


def test_image_space_metric_grows_with_tracking_drift():
    scene = _scene([(0, -3.5), (2, -3.5), (4, 0.0), (5, 0.0)])
    gt = gt_track(scene, "target_vehicle")
    width = float(np.nanmedian(gt.boxes[:, 2]))
    tracker = _BoxSeededTracker(gt, shift_px=0.5 * width)
    tracker.seed_box = tuple(float(v) for v in gt.boxes[0])
    result = evaluate(tracker.track(Path("unused.mp4")), EVENT, gt)
    assert abs(result.metrics["lateral_error_box_widths"] - 0.5) < 0.05


def test_evaluate_tolerates_a_render_shorter_than_the_scene():
    scene = _scene([(0, -3.5), (5, -3.5)])
    gt = gt_track(scene, "target_vehicle")
    tracker = _BoxSeededTracker(gt)
    tracker.seed_box = tuple(float(v) for v in gt.boxes[0])
    tracked = tracker.track(Path("unused.mp4"))
    shorter = Trajectory(
        times=tracked.times[:-4],
        points=tracked.points[:-4],
        fps=FPS,
        widths=tracked.widths[:-4],
    )
    result = evaluate(shorter, EVENT, gt)
    assert result.checks["subject_visible"] is True

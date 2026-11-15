"""Subject trackers that recover a per-frame trajectory from a video.

The grounded judge needs the subject's image-space position every frame. A
proxy can be tracked by its known render color; a photorealistic render needs
a detector plus inter-frame association; and when the scene already knows where
the subject should be, optical flow can follow it from a ground-truth box
without either. All expose :class:`SubjectTracker` so the judge is agnostic to
the video domain, and all return a
:class:`~driving_world.judge.grounding.Trajectory` in image pixels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .grounding import Trajectory, track_subject_from_video


class Detector(Protocol):
    """Return ``[(bbox_xyxy, score)]`` for a single image."""

    def detect(self, image: Image.Image) -> list[tuple[tuple[float, float, float, float], float]]: ...


class SubjectTracker(Protocol):
    name: str

    def track(self, video: Path) -> Trajectory: ...


def _decode_video(video: Path) -> tuple[float, list[np.ndarray]]:
    """Decode every frame plus the frame rate, or raise a clear error."""
    import imageio.v3 as iio  # lazy import keeps the module importable bare

    metadata = iio.immeta(video, plugin="FFMPEG")
    fps = float(metadata.get("fps", 0))
    frames = [np.asarray(frame) for frame in iio.imiter(video, plugin="FFMPEG")]
    if fps <= 0 or not frames:
        raise ValueError(f"cannot decode video frames and FPS from {video}")
    return fps, frames


def _center(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


class ColorTracker:
    """Track a subject by matching its render color (proxy domain)."""

    name = "color"

    def __init__(self, color: Sequence[float], tolerance: float = 60.0) -> None:
        self.color = tuple(color)
        self.tolerance = tolerance

    def track(self, video: Path) -> Trajectory:
        return track_subject_from_video(video, self.color, tolerance=self.tolerance)


class DetectionTracker:
    """Track a subject with a detector plus nearest-center association.

    Detection runs on every ``sample_step``-th frame and highest-score boxes
    are linearly interpolated to every frame. A forward pass rejects a
    detection that jumps farther than ``gate_px`` from the previously accepted
    position, which stops the tracker from latching onto another object.
    """

    name = "detection"

    def __init__(
        self,
        detector: Detector,
        *,
        sample_step: int = 6,
        gate_px: float = 100.0,
        min_score: float = 0.3,
        seed: tuple[float, float] | None = None,
    ) -> None:
        self.detector = detector
        self.sample_step = sample_step
        self.gate_px = gate_px
        self.min_score = min_score
        self.seed = seed

    def track(self, video: Path) -> Trajectory:
        fps, frames = _decode_video(video)

        seed = np.asarray(self.seed, dtype=float) if self.seed is not None else None
        centers: dict[int, tuple[float, float]] = {}
        for index in range(0, len(frames), self.sample_step):
            detections = [
                (box, score)
                for box, score in self.detector.detect(Image.fromarray(frames[index]))
                if score >= self.min_score
            ]
            if not detections:
                continue
            if seed is not None and not centers:
                box, _ = min(
                    detections, key=lambda item: np.linalg.norm(_center(item[0]) - seed)
                )
            else:
                box, _ = max(detections, key=lambda item: item[1])
            centers[index] = _center(box)

        points = self._associate_and_interpolate(len(frames), centers)
        times = np.arange(len(frames), dtype=float) / fps
        return Trajectory(times=times, points=points, fps=fps)

    def _associate_and_interpolate(
        self, frame_count: int, centers: dict[int, tuple[float, float]]
    ) -> np.ndarray:
        points = np.full((frame_count, 2), np.nan)
        accepted: list[tuple[int, np.ndarray]] = []
        previous: np.ndarray | None = None
        for index in sorted(centers):
            center = np.asarray(centers[index], dtype=float)
            if previous is not None and np.linalg.norm(center - previous) > self.gate_px:
                continue
            accepted.append((index, center))
            previous = center

        for index, center in accepted:
            points[index] = center
        for (a, ca), (b, cb) in zip(accepted, accepted[1:]):
            if b - a <= 1:
                continue
            for j in range(a + 1, b):
                alpha = (j - a) / (b - a)
                points[j] = (1.0 - alpha) * ca + alpha * cb
        return points


def _require_cv2():
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional heavy dependency
        raise ImportError(
            "FlowTracker needs OpenCV; install opencv-python to track photorealistic renders"
        ) from error
    return cv2


def _gray(cv2, frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) if array.ndim == 3 else array


def _box_center(box: np.ndarray) -> tuple[float, float]:
    return float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0)


def transform_box(box: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    """Move an ``(u, v, w, h)`` box by a ``2x3`` affine, keeping it axis-aligned."""
    x, y, w, h = (float(v) for v in box)
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)
    moved = corners @ matrix[:, :2].T + matrix[:, 2]
    low, high = moved.min(axis=0), moved.max(axis=0)
    return np.array(
        [low[0], low[1], max(1.0, high[0] - low[0]), max(1.0, high[1] - low[1])]
    )


class FlowTracker:
    """Follow a subject seeded with its ground-truth image box.

    Photorealistic renders carry no render-specific colour, and a detector is
    both slow and unreliable, but the scene already states which actor to watch
    and where its 3D box lands on screen. Seeding from that box lets plain
    optical flow carry the subject through the clip, which kept the lateral
    error at roughly a tenth of the subject's width on real driving video.

    Each frame fits a similarity transform to the forward-backward filtered LK
    inliers. The fitted scale is capped per frame, and a frame whose inliers
    collapse is left ``NaN`` rather than drifting silently off the subject.
    """

    name = "flow"

    def __init__(
        self,
        seed_box: Sequence[float] | None = None,
        *,
        seed_frame: int = 0,
        margin: float = 0.15,
        max_points: int = 150,
        min_points: int = 8,
        max_scale_step: float = 1.12,
        forward_backward_px: float = 1.0,
        max_lost_frames: int = 2,
    ) -> None:
        self.seed_box = tuple(float(v) for v in seed_box) if seed_box is not None else None
        self.seed_frame = int(seed_frame)
        self.margin = margin
        self.max_points = max_points
        self.min_points = min_points
        self.max_scale_step = max_scale_step
        self.forward_backward_px = forward_backward_px
        self.max_lost_frames = max_lost_frames

    def track(self, video: Path) -> Trajectory:
        if self.seed_box is None:
            raise ValueError("FlowTracker needs a seed_box to start from")
        cv2 = _require_cv2()
        fps, frames = _decode_video(video)
        if not 0 <= self.seed_frame < len(frames):
            raise ValueError(
                f"seed_frame {self.seed_frame} is outside the {len(frames)} decoded frames"
            )

        times = np.arange(len(frames), dtype=float) / fps
        points = np.full((len(frames), 2), np.nan)
        widths = np.full(len(frames), np.nan)
        box = np.asarray(self.seed_box, dtype=float)
        previous = _gray(cv2, frames[self.seed_frame])
        features = self._features(cv2, previous, box)
        points[self.seed_frame] = _box_center(box)
        widths[self.seed_frame] = box[2]

        lost = 0
        for index in range(self.seed_frame + 1, len(frames)):
            current = _gray(cv2, frames[index])
            moved = self._step(cv2, previous, current, box, features)
            previous = current
            if moved is None:
                lost += 1
                if lost > self.max_lost_frames:
                    break
                features = self._features(cv2, current, box)
                continue
            box = moved
            features = self._features(cv2, current, box)
            points[index] = _box_center(box)
            widths[index] = box[2]
            lost = 0
        return Trajectory(times=times, points=points, fps=fps, widths=widths)

    def _features(self, cv2, gray: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x, y, w, h = box
        mask = np.zeros_like(gray)
        x0 = int(max(0.0, x + self.margin * w))
        x1 = int(min(float(gray.shape[1]), x + w - self.margin * w))
        y0 = int(max(0.0, y + self.margin * h))
        y1 = int(min(float(gray.shape[0]), y + h - self.margin * h))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        mask[y0:y1, x0:x1] = 255
        return cv2.goodFeaturesToTrack(gray, self.max_points, 0.01, 4, mask=mask)

    def _step(self, cv2, previous, current, box, features) -> np.ndarray | None:
        if features is None or len(features) < self.min_points:
            return None
        params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        }
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, features, None, **params
        )
        backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, forward, None, **params
        )
        good = (status.ravel() == 1) & (status_back.ravel() == 1)
        good &= np.linalg.norm(backward - features, axis=2).ravel() < self.forward_backward_px
        if good.sum() < self.min_points:
            return None
        matrix, _ = cv2.estimateAffinePartial2D(
            features[good].reshape(-1, 2),
            forward[good].reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        if matrix is None:
            return None
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not math.isfinite(scale) or scale <= 0.0:
            return None
        capped = min(max(scale, 1.0 / self.max_scale_step), self.max_scale_step)
        if capped != scale:
            matrix = matrix / scale * capped
        return transform_box(box, matrix)

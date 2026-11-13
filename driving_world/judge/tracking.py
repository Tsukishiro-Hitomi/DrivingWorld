"""Subject trackers that recover a per-frame trajectory from a video.

The grounded judge needs the subject's image-space position every frame. A
proxy can be tracked by its known render color; a photorealistic render needs
a detector plus inter-frame association. Both expose :class:`SubjectTracker`
so the judge is agnostic to the video domain, and both return a
:class:`~driving_world.judge.grounding.Trajectory` in image pixels.
"""

from __future__ import annotations

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
    ) -> None:
        self.detector = detector
        self.sample_step = sample_step
        self.gate_px = gate_px
        self.min_score = min_score

    def track(self, video: Path) -> Trajectory:
        import imageio.v3 as iio  # lazy import keeps the module importable bare

        metadata = iio.immeta(video, plugin="FFMPEG")
        fps = float(metadata.get("fps", 0))
        frames = [np.asarray(frame) for frame in iio.imiter(video, plugin="FFMPEG")]
        if fps <= 0 or not frames:
            raise ValueError(f"cannot decode video frames and FPS from {video}")

        centers: dict[int, tuple[float, float]] = {}
        for index in range(0, len(frames), self.sample_step):
            detections = [
                (box, score)
                for box, score in self.detector.detect(Image.fromarray(frames[index]))
                if score >= self.min_score
            ]
            if not detections:
                continue
            (x1, y1, x2, y2), _ = max(detections, key=lambda item: item[1])
            centers[index] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

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

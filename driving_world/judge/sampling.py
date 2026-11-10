"""Deterministic frame sampling for temporal event inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
from PIL import Image


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp_seconds: float
    image: Image.Image


@dataclass(frozen=True)
class VideoSample:
    fps: float
    frame_count: int
    duration_seconds: float
    frames: list[SampledFrame]


def sample_video(
    video: Path,
    *,
    event_window_seconds: tuple[float, float],
    context_frames: int = 4,
    event_frames: int = 8,
) -> VideoSample:
    """Sample context and event-window frames, always returning sorted unique frames."""
    metadata = iio.immeta(video, plugin="FFMPEG")
    fps = float(metadata.get("fps", 0))
    frames = list(iio.imiter(video, plugin="FFMPEG"))
    if fps <= 0 or not frames:
        raise ValueError(f"cannot decode video frames and FPS from {video}")
    duration = len(frames) / fps
    start, end = event_window_seconds
    if not 0 <= start < end <= duration + 0.05:
        raise ValueError("event time window lies outside the video")

    def positions(first: int, last: int, count: int) -> set[int]:
        if count <= 1 or first == last:
            return {first}
        return {round(first + index * (last - first) / (count - 1)) for index in range(count)}

    first_event_frame = min(len(frames) - 1, max(0, round(start * fps)))
    last_event_frame = min(len(frames) - 1, max(0, round(end * fps)))
    selected = positions(0, len(frames) - 1, context_frames)
    selected |= positions(first_event_frame, last_event_frame, event_frames)
    return VideoSample(
        fps=fps,
        frame_count=len(frames),
        duration_seconds=duration,
        frames=[
            SampledFrame(index, index / fps, Image.fromarray(frames[index]))
            for index in sorted(selected)
        ],
    )

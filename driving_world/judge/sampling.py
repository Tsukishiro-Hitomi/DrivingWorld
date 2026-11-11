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


@dataclass(frozen=True)
class AlignedFrame:
    index: int
    timestamp_seconds: float
    rendered: Image.Image
    proxy: Image.Image


@dataclass(frozen=True)
class AlignedSample:
    fps: float
    frame_count: int
    duration_seconds: float
    frames: list[AlignedFrame]


def _positions(first: int, last: int, count: int) -> set[int]:
    if count <= 1 or first == last:
        return {first}
    return {round(first + index * (last - first) / (count - 1)) for index in range(count)}


def _decode(video: Path) -> tuple[float, list]:
    metadata = iio.immeta(video, plugin="FFMPEG")
    fps = float(metadata.get("fps", 0))
    frames = list(iio.imiter(video, plugin="FFMPEG"))
    if fps <= 0 or not frames:
        raise ValueError(f"cannot decode video frames and FPS from {video}")
    return fps, frames


def sample_video(
    video: Path,
    *,
    event_window_seconds: tuple[float, float],
    context_frames: int = 4,
    event_frames: int = 8,
) -> VideoSample:
    """Sample context and event-window frames, always returning sorted unique frames."""
    fps, frames = _decode(video)
    duration = len(frames) / fps
    start, end = event_window_seconds
    if not 0 <= start < end <= duration + 0.05:
        raise ValueError("event time window lies outside the video")
    first_event_frame = min(len(frames) - 1, max(0, round(start * fps)))
    last_event_frame = min(len(frames) - 1, max(0, round(end * fps)))
    selected = _positions(0, len(frames) - 1, context_frames)
    selected |= _positions(first_event_frame, last_event_frame, event_frames)
    return VideoSample(
        fps=fps,
        frame_count=len(frames),
        duration_seconds=duration,
        frames=[
            SampledFrame(index, index / fps, Image.fromarray(frames[index]))
            for index in sorted(selected)
        ],
    )


def sample_aligned(
    rendered: Path,
    proxy: Path,
    *,
    event_window_seconds: tuple[float, float],
    context_frames: int = 4,
    event_frames: int = 8,
) -> AlignedSample:
    """Sample timestamp-aligned frame pairs from the rendered and proxy videos."""
    rendered_fps, rendered_frames = _decode(rendered)
    proxy_fps, proxy_frames = _decode(proxy)
    duration = min(len(rendered_frames) / rendered_fps, len(proxy_frames) / proxy_fps)
    start, end = event_window_seconds
    if not 0 <= start < end <= duration + 0.05:
        raise ValueError("event time window lies outside the aligned videos")
    first_event_frame = min(len(rendered_frames) - 1, max(0, round(start * rendered_fps)))
    last_event_frame = min(len(rendered_frames) - 1, max(0, round(end * rendered_fps)))
    selected = _positions(0, len(rendered_frames) - 1, context_frames)
    selected |= _positions(first_event_frame, last_event_frame, event_frames)
    frames = []
    for index in sorted(selected):
        timestamp = index / rendered_fps
        proxy_index = min(len(proxy_frames) - 1, max(0, round(timestamp * proxy_fps)))
        frames.append(
            AlignedFrame(
                index,
                timestamp,
                Image.fromarray(rendered_frames[index]),
                Image.fromarray(proxy_frames[proxy_index]),
            )
        )
    return AlignedSample(
        fps=rendered_fps,
        frame_count=len(rendered_frames),
        duration_seconds=duration,
        frames=frames,
    )


def side_by_side(
    rendered: Image.Image, proxy: Image.Image, *, divider_width: int = 4
) -> Image.Image:
    """Concatenate a rendered frame and its proxy frame horizontally."""
    width = rendered.width + divider_width + proxy.width
    height = max(rendered.height, proxy.height)
    combined = Image.new("RGB", (width, height), (255, 255, 255))
    combined.paste(rendered, (0, 0))
    combined.paste(proxy, (rendered.width + divider_width, 0))
    return combined

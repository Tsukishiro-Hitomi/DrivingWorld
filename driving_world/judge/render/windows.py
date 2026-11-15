"""Timestamp-based sliding windows for per-frame render measurements."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """One non-empty scoring window and its source frame indices."""

    index: int
    start_seconds: float
    end_seconds: float
    frame_indices: tuple[int, ...]


def build_windows(
    frame_count: int,
    fps: float,
    *,
    window_seconds: float = 1.0,
    stride_seconds: float = 0.5,
) -> list[Window]:
    """Build sliding windows from zero through the final frame timestamp."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    duration = frame_count / fps
    last_timestamp = (frame_count - 1) / fps
    windows = []
    step = 0
    while step * stride_seconds <= last_timestamp + 1e-12:
        start = step * stride_seconds
        end = min(start + window_seconds, duration)
        is_last = end >= duration - 1e-12
        indices = tuple(
            index
            for index in range(frame_count)
            if index / fps >= start - 1e-12
            and (
                index / fps < end - 1e-12
                or (is_last and index / fps <= end + 1e-12)
            )
        )
        if indices:
            windows.append(Window(len(windows), start, end, indices))
        step += 1
    return windows

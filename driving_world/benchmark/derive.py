"""Derive ground-truth driving events from Waymo-style object tracks.

A track is a per-object sequence of ``(t_seconds, x_m, y_m)`` in the ego frame
(``x`` forward, ``y`` left, matching the renderer). Tracks come from a
detection+tracking source — for a real-video benchmark that source is the
Waymo ``laser_labels`` (3D boxes), so the events below are the dataset's own
ground truth, independent of any generative renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Waymo object types (``dataset_pb2.Label.Type``).
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 4


@dataclass(frozen=True)
class Track:
    track_id: str
    type: int
    times: np.ndarray  # (T,) seconds
    xs: np.ndarray  # (T,) forward meters
    ys: np.ndarray  # (T,) left meters

    @classmethod
    def from_points(cls, track_id: str, type: int, points: list[tuple[float, float, float]]):
        times, xs, ys = zip(*points)
        return cls(track_id, type, np.asarray(times), np.asarray(xs), np.asarray(ys))

    def subject_type(self) -> str:
        return {VEHICLE: "vehicle", PEDESTRIAN: "pedestrian", CYCLIST: "cyclist"}.get(
            self.type, "vehicle"
        )


@dataclass(frozen=True)
class Event:
    kind: str  # vehicle_lane_change | pedestrian_crossing | vehicle_hard_brake
    track_id: str
    subject_type: str
    start_seconds: float
    end_seconds: float
    direction: str | None  # left_to_right | right_to_left, where applicable


@dataclass(frozen=True)
class DeriveConfig:
    lane_displacement_m: float = 2.5  # lateral travel that counts as a lane change
    window_seconds: float = 4.0  # max span over which a lane change completes
    min_ahead_m: float = 2.0  # the subject must be ahead of ego during the event
    crossing_max_lateral_m: float = 1.8  # |y| below this = in the ego lane
    crossing_max_ahead_m: float = 40.0  # only count pedestrians this close ahead
    brake_speed_drop_mps: float = 3.0  # forward speed drop that counts as hard braking
    brake_window_seconds: float = 2.0
    smooth_window: int = 3  # median-filter window for lateral/velocity estimates


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(float)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.array(
        [np.median(padded[i : i + window]) for i in range(len(values))]
    )


def _windows(times: np.ndarray, window_seconds: float) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) windows no wider than ``window_seconds``."""
    windows = []
    start = 0
    for end in range(len(times)):
        while times[end] - times[start] > window_seconds:
            start += 1
        if end > start:
            windows.append((start, end))
    return windows


def lane_change_events(track: Track, config: DeriveConfig) -> list[Event]:
    if track.type not in (VEHICLE, CYCLIST) or len(track.ys) < 3:
        return []
    ys = _smooth(track.ys, config.smooth_window)
    events: list[Event] = []
    for start, end in _windows(track.times, config.window_seconds):
        ahead = float(track.xs[start:end].min()) > config.min_ahead_m
        displacement = float(ys[end] - ys[start])
        if ahead and abs(displacement) >= config.lane_displacement_m:
            direction = "right_to_left" if displacement > 0 else "left_to_right"
            events.append(
                Event(
                    kind="vehicle_lane_change",
                    track_id=track.track_id,
                    subject_type=track.subject_type(),
                    start_seconds=float(track.times[start]),
                    end_seconds=float(track.times[end]),
                    direction=direction,
                )
            )
    return _merge(events, config)


def pedestrian_crossing_events(track: Track, config: DeriveConfig) -> list[Event]:
    if track.type != PEDESTRIAN or len(track.ys) < 3:
        return []
    ys = _smooth(track.ys, config.smooth_window)
    ahead = (track.xs > 0) & (track.xs < config.crossing_max_ahead_m)
    in_lane = np.abs(ys) < config.crossing_max_lateral_m
    crossing = ahead & in_lane
    if crossing.sum() < 2:
        return []
    indices = np.nonzero(crossing)[0]
    first, last = int(indices[0]), int(indices[-1])
    entry_y = float(ys[first])
    direction = "right_to_left" if entry_y > 0 else "left_to_right"
    return [
        Event(
            kind="pedestrian_crossing",
            track_id=track.track_id,
            subject_type="pedestrian",
            start_seconds=float(track.times[first]),
            end_seconds=float(track.times[last]),
            direction=direction,
        )
    ]


def hard_brake_events(track: Track, config: DeriveConfig) -> list[Event]:
    if track.type != VEHICLE or len(track.xs) < 3:
        return []
    xs = _smooth(track.xs, config.smooth_window)
    velocity = np.gradient(xs, track.times)
    events: list[Event] = []
    for start, end in _windows(track.times, config.brake_window_seconds):
        ahead = float(track.xs[start:end].min()) > config.min_ahead_m
        speed_drop = float(velocity[start] - velocity[end])
        if ahead and speed_drop >= config.brake_speed_drop_mps:
            events.append(
                Event(
                    kind="vehicle_hard_brake",
                    track_id=track.track_id,
                    subject_type="vehicle",
                    start_seconds=float(track.times[start]),
                    end_seconds=float(track.times[end]),
                    direction=None,
                )
            )
    return _merge(events, config)


def derive_events(tracks: list[Track], config: DeriveConfig = DeriveConfig()) -> list[Event]:
    """Detect every event across all tracks of a clip."""
    events: list[Event] = []
    for track in tracks:
        events.extend(lane_change_events(track, config))
        events.extend(pedestrian_crossing_events(track, config))
        events.extend(hard_brake_events(track, config))
    return sorted(events, key=lambda e: (e.start_seconds, e.kind))


def _merge(events: list[Event], config: DeriveConfig) -> list[Event]:
    """Collapse overlapping windows of the same track+kind into one event."""
    merged: list[Event] = []
    for event in sorted(events, key=lambda e: (e.track_id, e.kind, e.start_seconds)):
        if merged and (
            merged[-1].track_id == event.track_id
            and merged[-1].kind == event.kind
            and event.start_seconds <= merged[-1].end_seconds
        ):
            previous = merged[-1]
            merged[-1] = Event(
                previous.kind,
                previous.track_id,
                previous.subject_type,
                previous.start_seconds,
                max(previous.end_seconds, event.end_seconds),
                previous.direction or event.direction,
            )
        else:
            merged.append(event)
    return merged

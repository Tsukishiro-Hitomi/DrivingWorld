"""Derive ground-truth driving events from Waymo camera-label tracks.

Events are derived in *image space* from the 2D camera boxes so that a detected
event is, by construction, visible in the front camera the VLM actually sees.
Hard braking is the one exception: it is measured from 3D forward-speed and
then required to have a visible 2D box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Waymo object types (``dataset_pb2.Label.Type``).
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 4

_SUBJECT = {VEHICLE: "vehicle", PEDESTRIAN: "pedestrian", CYCLIST: "cyclist"}


@dataclass(frozen=True)
class CameraTrack:
    """Per-frame 2D box of one object in the front camera (pixels)."""

    track_id: str
    type: int
    times: np.ndarray  # (T,) seconds
    us: np.ndarray  # (T,) box center x
    vs: np.ndarray  # (T,) box center y
    ws: np.ndarray  # (T,) box width (x extent)
    hs: np.ndarray  # (T,) box height (y extent)

    def area(self) -> np.ndarray:
        return self.ws * self.hs

    def subject_type(self) -> str:
        return _SUBJECT.get(self.type, "vehicle")


@dataclass(frozen=True)
class WorldTrack:
    """Per-frame 3D position of one object in the ego frame (x forward, y left)."""

    track_id: str
    type: int
    times: np.ndarray  # (T,) seconds
    xs: np.ndarray  # (T,) forward meters
    ys: np.ndarray  # (T,) left meters


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
    min_box_area_px: float = 800.0  # drop objects too small/distant to judge
    min_lateral_shift_px: float = 80.0  # lateral travel that counts as a maneuver
    margin_seconds: float = 0.6  # context required before/after a maneuver
    brake_speed_drop_mps: float = 3.0  # forward-speed drop that counts as braking
    brake_window_seconds: float = 2.0
    min_ahead_m: float = 2.0
    smooth_window: int = 3


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(float)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def _interior_span(u: np.ndarray, config: DeriveConfig) -> tuple[int, int] | None:
    """Return (start, end) of the maneuver if it is interior and large enough."""
    total = float(u[-1] - u[0])
    if abs(total) < config.min_lateral_shift_px:
        return None
    cumulative = np.abs(u - u[0])
    threshold = abs(total)
    start = int(np.nonzero(cumulative >= 0.1 * threshold)[0][0])
    end = int(np.nonzero(cumulative >= 0.9 * threshold)[0][0])
    if start <= 0 or end >= len(u) - 1:
        return None
    return start, end


def lateral_events(track: CameraTrack, config: DeriveConfig) -> list[Event]:
    if track.type not in (VEHICLE, PEDESTRIAN, CYCLIST) or len(track.us) < 5:
        return []
    if float(np.median(track.area())) < config.min_box_area_px:
        return []
    u = _smooth(track.us, config.smooth_window)
    span = _interior_span(u, config)
    if span is None:
        return []
    start, end = span
    dt = float(track.times[end] - track.times[start])
    margin = config.margin_seconds
    if track.times[start] < margin or track.times[-1] - track.times[end] < margin:
        return []
    direction = "right_to_left" if u[end] < u[start] else "left_to_right"
    kind = (
        "vehicle_lane_change"
        if track.type in (VEHICLE, CYCLIST)
        else "pedestrian_crossing"
    )
    return [
        Event(
            kind=kind,
            track_id=track.track_id,
            subject_type=track.subject_type(),
            start_seconds=float(track.times[start]),
            end_seconds=float(track.times[end]),
            direction=direction,
        )
    ]


def hard_brake_events(track: WorldTrack, config: DeriveConfig) -> list[Event]:
    if track.type != VEHICLE or len(track.xs) < 5:
        return []
    xs = _smooth(track.xs, config.smooth_window)
    velocity = np.gradient(xs, track.times)
    events: list[Event] = []
    for start in range(len(track.times)):
        for end in range(start + 1, len(track.times)):
            if track.times[end] - track.times[start] > config.brake_window_seconds:
                break
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
                break
    return _merge(events)


def derive_events(
    camera_tracks: list[CameraTrack],
    world_tracks: list[WorldTrack],
    config: DeriveConfig = DeriveConfig(),
) -> list[Event]:
    """Detect every visible event across a clip's camera and world tracks."""
    events: list[Event] = []
    for track in camera_tracks:
        events.extend(lateral_events(track, config))
    world_by_id = {t.track_id: t for t in world_tracks}
    for track in camera_tracks:
        world = world_by_id.get(track.track_id)
        if world is not None and world.type == VEHICLE:
            events.extend(hard_brake_events(world, config))
    return sorted(events, key=lambda e: (e.start_seconds, e.kind))


def _merge(events: list[Event]) -> list[Event]:
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

"""Derive ground-truth driving events from Waymo tracks.

Maneuvers are measured in the *ego frame* (3D ``y`` is lateral relative to the
ego, so ego turns do not look like lane changes), and each event is then gated
on the subject's 2D camera-box area so it is actually visible to a VLM.
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


@dataclass(frozen=True)
class WorldTrack:
    """Per-frame 3D position of one object in the ego frame (x forward, y left)."""

    track_id: str
    type: int
    times: np.ndarray  # (T,) seconds
    xs: np.ndarray  # (T,) forward meters
    ys: np.ndarray  # (T,) left meters

    def subject_type(self) -> str:
        return _SUBJECT.get(self.type, "vehicle")


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
    min_box_area_px: float = 800.0  # drop subjects too small/distant to judge
    lane_displacement_m: float = 2.5  # ego-frame lateral travel for a lane change
    max_maneuver_seconds: float = 5.0  # reject slow drifts longer than this
    margin_seconds: float = 0.6  # context required before/after a maneuver
    crossing_max_lateral_m: float = 1.5  # |y| below this = inside the ego lane
    crossing_max_ahead_m: float = 30.0  # pedestrians must intrude this close ahead
    min_ahead_m: float = 2.0
    brake_speed_drop_mps: float = 3.0
    brake_window_seconds: float = 2.0
    smooth_window: int = 3


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(float)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def _interior_span(u: np.ndarray, min_shift: float) -> tuple[int, int] | None:
    """Return (start, end) of the maneuver if interior and large enough."""
    total = float(u[-1] - u[0])
    if abs(total) < min_shift:
        return None
    cumulative = np.abs(u - u[0])
    start = int(np.nonzero(cumulative >= 0.1 * abs(total))[0][0])
    end = int(np.nonzero(cumulative >= 0.9 * abs(total))[0][0])
    if start <= 0 or end >= len(u) - 1:
        return None
    return start, end


def _visible(camera: CameraTrack | None, config: DeriveConfig) -> bool:
    return camera is not None and float(np.median(camera.area())) >= config.min_box_area_px


def lane_change_events(
    world: WorldTrack, camera: CameraTrack | None, config: DeriveConfig
) -> list[Event]:
    if world.type not in (VEHICLE, CYCLIST) or not _visible(camera, config):
        return []
    ys = _smooth(world.ys, config.smooth_window)
    span = _interior_span(ys, config.lane_displacement_m)
    if span is None:
        return []
    start, end = span
    duration = float(world.times[end] - world.times[start])
    margin = config.margin_seconds
    if duration > config.max_maneuver_seconds:
        return []
    if world.times[start] < margin or world.times[-1] - world.times[end] < margin:
        return []
    direction = "right_to_left" if ys[end] > ys[start] else "left_to_right"
    return [
        Event(
            kind="vehicle_lane_change",
            track_id=world.track_id,
            subject_type=world.subject_type(),
            start_seconds=float(world.times[start]),
            end_seconds=float(world.times[end]),
            direction=direction,
        )
    ]


def pedestrian_crossing_events(
    world: WorldTrack, camera: CameraTrack | None, config: DeriveConfig
) -> list[Event]:
    if world.type != PEDESTRIAN or not _visible(camera, config):
        return []
    ys = _smooth(world.ys, config.smooth_window)
    ahead = (world.xs > config.min_ahead_m) & (world.xs < config.crossing_max_ahead_m)
    in_lane = np.abs(ys) < config.crossing_max_lateral_m
    intruding = ahead & in_lane
    if intruding.sum() < 2:
        return []
    indices = np.nonzero(intruding)[0]
    first, last = int(indices[0]), int(indices[-1])
    direction = "right_to_left" if ys[first] > 0 else "left_to_right"
    return [
        Event(
            kind="pedestrian_crossing",
            track_id=world.track_id,
            subject_type="pedestrian",
            start_seconds=float(world.times[first]),
            end_seconds=float(world.times[last]),
            direction=direction,
        )
    ]


def hard_brake_events(
    world: WorldTrack, camera: CameraTrack | None, config: DeriveConfig
) -> list[Event]:
    if world.type != VEHICLE or not _visible(camera, config):
        return []
    xs = _smooth(world.xs, config.smooth_window)
    velocity = np.gradient(xs, world.times)
    for start in range(len(world.times)):
        for end in range(start + 1, len(world.times)):
            if world.times[end] - world.times[start] > config.brake_window_seconds:
                break
            ahead = float(world.xs[start:end].min()) > config.min_ahead_m
            speed_drop = float(velocity[start] - velocity[end])
            if ahead and speed_drop >= config.brake_speed_drop_mps:
                return [
                    Event(
                        kind="vehicle_hard_brake",
                        track_id=world.track_id,
                        subject_type="vehicle",
                        start_seconds=float(world.times[start]),
                        end_seconds=float(world.times[end]),
                        direction=None,
                    )
                ]
    return []


def derive_events(
    world_tracks: list[WorldTrack],
    camera_tracks: list[CameraTrack],
    config: DeriveConfig = DeriveConfig(),
) -> list[Event]:
    """Detect every visible event across a clip's world and camera tracks."""
    camera_by_id = {t.track_id: t for t in camera_tracks}
    events: list[Event] = []
    for world in world_tracks:
        camera = camera_by_id.get(world.track_id)
        events.extend(lane_change_events(world, camera, config))
        events.extend(pedestrian_crossing_events(world, camera, config))
        events.extend(hard_brake_events(world, camera, config))
    return sorted(events, key=lambda e: (e.start_seconds, e.kind))

"""Derive ground-truth driving events from Waymo 3D tracks.

Maneuvers are measured in the *ego frame* (3D ``y`` is lateral relative to the
ego, so ego turns do not look like lane changes). Visibility is estimated
analytically from the camera focal length and the object's depth, so a subject
is only labeled when it is large enough in the front camera to be judged.
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
class WorldTrack:
    """Per-frame 3D track + visibility in the ego frame and road frame.

    ``ys`` is left-lateral in the ego frame (used for "inside the ego lane"),
    while ``ys_road`` is left-lateral in a road-aligned frame fixed at clip
    start (used for lane changes, so the ego's own maneuvers do not count).
    """

    track_id: str
    type: int
    times: np.ndarray  # (T,) seconds
    xs: np.ndarray  # (T,) forward meters
    ys: np.ndarray  # (T,) left meters (ego frame)
    ys_road: np.ndarray  # (T,) road-aligned left meters
    apparent_px: np.ndarray  # (T,) estimated on-image height in pixels

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
    min_apparent_px: float = 30.0  # subject must be this large on screen
    max_lateral_ratio: float = 0.47  # tan(half horizontal FOV); out-of-view gate
    lane_displacement_m: float = 2.5  # ego-frame lateral travel for a lane change
    max_lane_displacement_m: float = 6.0  # a lane change is ~one lane, not a drift
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


def _visible(world: WorldTrack, config: DeriveConfig) -> bool:
    if float(np.median(world.apparent_px)) < config.min_apparent_px:
        return False
    # Must be in front of the ego and inside the front camera's horizontal FOV.
    forward = world.xs > 0
    ratio = np.abs(world.ys) / np.maximum(world.xs, 1.0)
    in_frame = forward & (ratio < config.max_lateral_ratio)
    return float(np.mean(in_frame)) >= 0.5


def lane_change_events(world: WorldTrack, config: DeriveConfig) -> list[Event]:
    if world.type not in (VEHICLE, CYCLIST) or len(world.ys_road) < 3 or not _visible(world, config):
        return []
    ys = _smooth(world.ys_road, config.smooth_window)
    span = _interior_span(ys, config.lane_displacement_m)
    if span is None:
        return []
    start, end = span
    span_shift = float(ys[end] - ys[start])
    duration = float(world.times[end] - world.times[start])
    margin = config.margin_seconds
    if abs(span_shift) > config.max_lane_displacement_m:
        return []
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


def pedestrian_crossing_events(world: WorldTrack, config: DeriveConfig) -> list[Event]:
    if world.type != PEDESTRIAN or len(world.ys) < 2 or not _visible(world, config):
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


def hard_brake_events(world: WorldTrack, config: DeriveConfig) -> list[Event]:
    if world.type != VEHICLE or len(world.xs) < 3 or not _visible(world, config):
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
    world_tracks: list[WorldTrack], config: DeriveConfig = DeriveConfig()
) -> list[Event]:
    """Detect every visible event across a clip's world tracks."""
    events: list[Event] = []
    for world in world_tracks:
        events.extend(lane_change_events(world, config))
        events.extend(pedestrian_crossing_events(world, config))
        events.extend(hard_brake_events(world, config))
    return sorted(events, key=lambda e: (e.start_seconds, e.kind))

"""Deterministic, grounded object check.

Unlike the VLM checker this path measures the subject's trajectory instead of
eyeballing frames. The proxy scene already knows every actor's 3D track and
the camera, so we project the ground-truth track to image space, extract a
trajectory from the video, and compare them numerically in world meters.

Perspective is handled explicitly: a subject that only recedes in depth still
appears to drift laterally on screen. We un-project the tracked image offset
against the ground-truth depth, recovering the subject's world-lateral ``y``
(``y_tr = y_gt + (u_gt - u_tr) * depth / focal``), so a real lane change is
distinguished from pure perspective drift. Coordinates follow the renderer:
world ``x`` forward, ``y`` left (positive = left), ``z`` up.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from driving_world.contracts.events import EventSpec, validate_event_spec
from driving_world.geometry import (
    Camera,
    camera_pose,
    interpolate_track,
    project_points,
    rotation,
)


@dataclass(frozen=True)
class Trajectory:
    """Per-frame 2D image positions; ``NaN`` marks an undetected frame.

    ``widths`` is the subject's apparent box width in pixels when the tracker
    knows it. It lets the judge normalise a lateral offset by the subject's own
    screen size, which is comparable across distances and scenes.
    """

    times: np.ndarray  # (T,)
    points: np.ndarray  # (T, 2) in (u, v) image pixels
    fps: float
    widths: np.ndarray | None = None  # (T,) apparent width in pixels

    def visible(self) -> np.ndarray:
        return ~np.isnan(self.points).any(axis=1)


@dataclass(frozen=True)
class GtTrack:
    """One actor's ground-truth track projected into image and camera space.

    ``boxes`` is the full projected 3D box per frame as ``(u, v, w, h)``. It is
    the reference the tracker is seeded with and the scale the tracked lateral
    offset is measured against.
    """

    times: np.ndarray  # (T,)
    image: np.ndarray  # (T, 2) in (u, v); NaN behind the near plane
    world: np.ndarray  # (T, 3) in world (x, y, z)
    depth: np.ndarray  # (T,) camera-frame depth
    focal: float
    boxes: np.ndarray | None = None  # (T, 4) as (u, v, w, h)

    def visible(self) -> np.ndarray:
        return ~np.isnan(self.image).any(axis=1)


@dataclass(frozen=True)
class GroundedConfig:
    """World-meter thresholds for the deterministic checks."""

    motion_min_meters: float = 1.0  # lateral travel that counts as a maneuver
    completion_max_meters: float = 1.0  # |final y - target| for completion
    timing_fraction: float = 0.6  # fraction of lateral travel inside the window
    visible_fraction: float = 0.5  # fraction of window frames with a detection
    match_max_meters: float = 0.5  # mean |y_tracked - y_gt| for trajectory match


@dataclass(frozen=True)
class GroundedResult:
    """Auditable outcome of the grounded object check."""

    event_id: str
    status: str  # observed | partially_observed | not_observed
    checks: dict[str, bool | None]
    metrics: dict[str, float]
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def _find_actor(scene: dict[str, Any], actor_id: str) -> dict[str, Any]:
    for actor in scene.get("actors", []):
        if actor["id"] == actor_id:
            return actor
    raise ValueError(f"actor {actor_id!r} not found in scene")


def _box_corners(position: np.ndarray, yaw: float, size: Sequence[float]) -> np.ndarray:
    """The eight corners of an actor's box in world coordinates."""
    half = np.asarray(size, dtype=float) / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    return position + (signs * half) @ rotation(yaw).T


def gt_track(scene: dict[str, Any], actor_id: str) -> GtTrack:
    """Project one actor's ground-truth 3D track to image and camera space."""
    camera = Camera.from_dict(scene["camera"])
    fps = float(scene["fps"])
    count = math.ceil(scene["duration"] * fps)
    actor = _find_actor(scene, actor_id)
    size = actor.get("size")
    ego = scene["ego"]
    start = actor.get("start", 0.0)
    end = actor.get("end", scene["duration"])
    times = np.arange(count, dtype=float) / fps
    image = np.full((count, 2), np.nan)
    world = np.full((count, 3), np.nan)
    depth = np.full((count,), np.nan)
    boxes = np.full((count, 4), np.nan) if size is not None else None
    for index, t in enumerate(times):
        if not start <= t < end:
            continue
        ego_pos, ego_yaw = interpolate_track(ego, t)
        position, yaw = interpolate_track(actor["track"], t)
        world[index] = position
        camera_pos, camera_rot = camera_pose(ego_pos, ego_yaw, camera)
        p = camera_rot.T @ (position - camera_pos)
        if p[0] < camera.near:
            continue
        depth[index] = p[0]
        image[index] = (
            camera.width / 2.0 - camera.focal * p[1] / p[0],
            camera.height / 2.0 - camera.focal * p[2] / p[0],
        )
        if boxes is not None:
            uv, visible = project_points(
                _box_corners(position, yaw, size), ego_pos, ego_yaw, camera
            )
            if visible.all():
                low = uv.min(axis=0)
                high = uv.max(axis=0)
                boxes[index] = (low[0], low[1], high[0] - low[0], high[1] - low[1])
    return GtTrack(
        times=times,
        image=image,
        world=world,
        depth=depth,
        focal=camera.focal,
        boxes=boxes,
    )


def track_subject_by_color(
    frames: Sequence[np.ndarray],
    color: Sequence[float],
    fps: float,
    *,
    tolerance: float = 60.0,
) -> Trajectory:
    """Track a subject by matching its render color and taking the centroid."""
    target = np.asarray(color, dtype=np.float32)
    points = np.full((len(frames), 2), np.nan)
    for index, frame in enumerate(frames):
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim != 3 or array.shape[2] < 3:
            continue
        distance = np.abs(array[:, :, :3] - target).max(axis=2)
        ys, xs = np.nonzero(distance < tolerance)
        if xs.size >= 4:
            points[index] = (float(xs.mean()), float(ys.mean()))
    times = np.arange(len(frames), dtype=float) / fps
    return Trajectory(times=times, points=points, fps=fps)


def track_subject_from_video(
    video: Path, color: Sequence[float], *, tolerance: float = 60.0
) -> Trajectory:
    """Color-track a subject across every frame of a video file."""
    import imageio.v3 as iio  # lazy import keeps the module usable without media libs

    metadata = iio.immeta(video, plugin="FFMPEG")
    fps = float(metadata.get("fps", 0))
    frames = [np.asarray(frame) for frame in iio.imiter(video, plugin="FFMPEG")]
    if fps <= 0 or not frames:
        raise ValueError(f"cannot decode video frames and FPS from {video}")
    return track_subject_by_color(frames, color, fps, tolerance=tolerance)


def _world_lateral(tracked: Trajectory, gt: GtTrack) -> np.ndarray:
    """Depth-corrected world-lateral ``y`` of the tracked subject (meters).

    Trimmed to the frames both sides share: a render whose length differs from
    the scene duration must not crash the judge.
    """
    count = min(len(gt.world), len(tracked.points))
    y_gt = gt.world[:count, 1]
    u_gt = gt.image[:count, 0]
    u_tr = tracked.points[:count, 0]
    y_tr = np.full_like(y_gt, np.nan)
    common = gt.visible()[:count] & tracked.visible()[:count]
    y_tr[common] = y_gt[common] + (u_gt[common] - u_tr[common]) * gt.depth[:count][common] / gt.focal
    return y_tr


def _image_space_metrics(
    tracked: Trajectory, gt: GtTrack, common: np.ndarray
) -> dict[str, float]:
    """Scale-free tracking audit: offset and size error in subject-widths.

    Both frames of reference share one camera, so ego motion cancels here; what
    is left is purely how far the tracker drifted from the ground-truth box.
    """
    if gt.boxes is None or tracked.widths is None or not common.any():
        return {}
    count = common.size
    gt_width = gt.boxes[:count, 2][common]
    tracked_width = tracked.widths[:count][common]
    finite = np.isfinite(gt_width) & np.isfinite(tracked_width) & (gt_width > 0)
    if not finite.any():
        return {}
    # Compare against the projected box centre, not the projected 3D centre: the
    # tracker reports a bounding box, and under perspective its centre differs.
    gt_box_center_u = gt.boxes[:count, 0][common] + gt_width / 2.0
    offset = np.abs(tracked.points[:count, 0][common] - gt_box_center_u)
    return {
        "lateral_error_box_widths": float((offset[finite] / gt_width[finite]).mean()),
        "width_ratio": float(np.median(tracked_width[finite] / gt_width[finite])),
    }


def evaluate(
    tracked: Trajectory,
    event: EventSpec,
    gt: GtTrack,
    *,
    config: GroundedConfig = GroundedConfig(),
) -> GroundedResult:
    """Compare a tracked trajectory against the event spec and the GT track."""
    event = validate_event_spec(event)
    start, end = event["time_window_seconds"]
    n = min(len(tracked.times), len(gt.times))
    times = tracked.times[:n]
    tracked_visible = tracked.visible()[:n]
    window_mask = (times >= start) & (times <= end)

    subject_visible = bool(
        window_mask.any() and tracked_visible[window_mask].mean() >= config.visible_fraction
    )

    y_tr = _world_lateral(tracked, gt)[:n]
    y_gt = gt.world[:n, 1]
    common = gt.visible()[:n] & tracked_visible & window_mask
    mean_gt_error_m = (
        float(np.abs(y_tr[common] - y_gt[common]).mean()) if common.sum() else float("nan")
    )

    valid = ~np.isnan(y_tr)
    valid_indices = np.nonzero(valid)[0]
    metrics: dict[str, float] = {
        "mean_gt_error_m": mean_gt_error_m,
        "subject_visible_fraction": (
            float(tracked_visible[window_mask].mean()) if window_mask.any() else 0.0
        ),
    }
    metrics.update(_image_space_metrics(tracked, gt, common))

    if not subject_visible:
        return GroundedResult(
            event_id=event["event_id"],
            status="not_observed",
            checks={
                "subject_visible": False,
                "motion_observed": None,
                "direction_correct": None,
                "timing_valid": None,
                "event_completed": None,
                "matches_proxy_trajectory": _match(mean_gt_error_m, config),
            },
            metrics=metrics,
            rationale="Subject was not detected in the event window.",
        )

    if valid_indices.size < 2:
        return GroundedResult(
            event_id=event["event_id"],
            status="not_observed",
            checks={
                "subject_visible": True,
                "motion_observed": None,
                "direction_correct": None,
                "timing_valid": None,
                "event_completed": None,
                "matches_proxy_trajectory": _match(mean_gt_error_m, config),
            },
            metrics=metrics,
            rationale="Too few detected frames to measure lateral motion.",
        )

    first, last = valid_indices[0], valid_indices[-1]
    net = float(y_tr[last] - y_tr[first])
    motion_observed = abs(net) >= config.motion_min_meters

    in_window_indices = np.nonzero(valid & window_mask)[0]
    window_net = (
        float(y_tr[in_window_indices[-1]] - y_tr[in_window_indices[0]])
        if in_window_indices.size >= 2
        else 0.0
    )

    metrics.update(
        {
            "world_lateral_start_m": float(y_tr[first]),
            "world_lateral_end_m": float(y_tr[last]),
            "net_lateral_m": net,
            "window_net_m": window_net,
            "window_net_fraction": (abs(window_net) / abs(net)) if abs(net) > 0 else 0.0,
        }
    )

    direction_correct: bool | None = None
    timing_valid: bool | None = None
    if motion_observed:
        direction = event.get("direction")
        if direction is not None:
            expected_sign = 1.0 if direction == "right_to_left" else -1.0
            direction_correct = bool(net * expected_sign > 0)
        timing_valid = bool(abs(window_net) >= config.timing_fraction * abs(net))

    target_y = 0.0  # ego_lane center for a lane change
    event_completed = bool(abs(y_tr[last] - target_y) <= config.completion_max_meters)

    matches = _match(mean_gt_error_m, config)
    checks = {
        "subject_visible": True,
        "motion_observed": motion_observed,
        "direction_correct": direction_correct,
        "timing_valid": timing_valid,
        "event_completed": event_completed,
        "matches_proxy_trajectory": matches,
    }

    if not motion_observed:
        status = "not_observed"
        rationale = "Subject is present but does not move laterally in the event window."
    elif direction_correct is True and timing_valid is True and event_completed is True:
        status = "observed"
        rationale = "Subject performs the requested maneuver with the correct direction and timing."
    else:
        status = "partially_observed"
        rationale = (
            "Subject moves, but direction/timing/completion does not match the request."
        )

    return GroundedResult(
        event_id=event["event_id"],
        status=status,
        checks=checks,
        metrics=metrics,
        rationale=rationale,
    )


def _match(mean_gt_error_m: float, config: GroundedConfig) -> bool | None:
    if math.isnan(mean_gt_error_m):
        return None
    return bool(mean_gt_error_m <= config.match_max_meters)


def _seed_tracker(tracker: Any, gt: GtTrack) -> None:
    """Point the tracker at the subject before it starts.

    A tracker that accepts a box gets the projected ground-truth box (and the
    frame it came from); a centre-only tracker gets the first visible image
    position. Without this, multi-object scenes resolve to the wrong subject.
    """
    visible = np.nonzero(gt.visible())[0]
    if not visible.size:
        return
    first = int(visible[0])
    if hasattr(tracker, "seed_box"):
        if gt.boxes is not None and np.isfinite(gt.boxes[first]).all():
            tracker.seed_box = tuple(float(v) for v in gt.boxes[first])
            if hasattr(tracker, "seed_frame"):
                tracker.seed_frame = first
    elif hasattr(tracker, "seed"):
        tracker.seed = (float(gt.image[first, 0]), float(gt.image[first, 1]))


def grounded_check(
    video: Path,
    scene: dict[str, Any],
    event: EventSpec,
    actor_id: str,
    tracker: Any,
    *,
    config: GroundedConfig = GroundedConfig(),
) -> GroundedResult:
    """Run the grounded judge end-to-end for one video against one actor.

    ``tracker`` is any object exposing ``track(video) -> Trajectory``, e.g. a
    :class:`ColorTracker` for proxies or a :class:`FlowTracker` for
    photorealistic renders.
    """
    gt = gt_track(scene, actor_id)
    _seed_tracker(tracker, gt)
    tracked = tracker.track(video)
    return evaluate(tracked, event, gt, config=config)

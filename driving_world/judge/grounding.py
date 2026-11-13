"""Deterministic, grounded object check.

Unlike the VLM checker this path measures the subject's trajectory instead of
eyeballing frames. The proxy scene already knows every actor's 3D track and
the camera, so we project the ground-truth track to image space, extract a
trajectory from the video, and compare them numerically. The checks are
interpretable and reproducible; a VLM is only needed later for the residual
"render quality" judgement.

Coordinate convention for the comparison is "left-positive lateral position"
in image pixels: ``width / 2 - u``, which has the same sign as the world ``y``
(left) axis and therefore makes ``right_to_left`` a positive displacement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from driving_world.contracts.events import EventSpec, validate_event_spec
from driving_world.geometry import Camera, interpolate_track, project_point


@dataclass(frozen=True)
class Trajectory:
    """Per-frame 2D image positions; ``NaN`` marks an undetected frame."""

    times: np.ndarray  # (T,)
    points: np.ndarray  # (T, 2) in (u, v) image pixels
    fps: float

    def visible(self) -> np.ndarray:
        return ~np.isnan(self.points).any(axis=1)

    def lateral(self, image_width: float) -> np.ndarray:
        """Left-positive lateral position in pixels (``width / 2 - u``)."""
        return image_width / 2.0 - self.points[:, 0]


@dataclass(frozen=True)
class GroundedConfig:
    """Pixel-space thresholds for the deterministic checks."""

    motion_min_pixels: float = 15.0  # lateral travel that counts as motion
    completion_max_pixels: float = 40.0  # |final lateral - target| for completion
    timing_fraction: float = 0.6  # fraction of lateral travel inside the window
    visible_fraction: float = 0.5  # fraction of window frames with a detection
    match_max_pixels: float = 40.0  # mean pixel error vs the GT trajectory


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


def gt_trajectory(scene: dict[str, Any], actor_id: str) -> Trajectory:
    """Project one actor's ground-truth 3D track to per-frame image points."""
    camera = Camera.from_dict(scene["camera"])
    fps = float(scene["fps"])
    count = math.ceil(scene["duration"] * fps)
    actor = _find_actor(scene, actor_id)
    ego = scene["ego"]
    start = actor.get("start", 0.0)
    end = actor.get("end", scene["duration"])
    times = np.arange(count, dtype=float) / fps
    points = np.full((count, 2), np.nan)
    for index, t in enumerate(times):
        if not start <= t < end:
            continue
        ego_pos, ego_yaw = interpolate_track(ego, t)
        position, _ = interpolate_track(actor["track"], t)
        uv = project_point(position, ego_pos, ego_yaw, camera)
        if uv is not None:
            points[index] = uv
    return Trajectory(times=times, points=points, fps=fps)


def track_subject_by_color(
    frames: Sequence[np.ndarray], color: Sequence[float], fps: float, *, tolerance: float = 60.0
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


def evaluate(
    tracked: Trajectory,
    event: EventSpec,
    image_width: float,
    *,
    gt: Trajectory | None = None,
    config: GroundedConfig = GroundedConfig(),
) -> GroundedResult:
    """Compare a tracked trajectory against the event spec (and optional GT)."""
    event = validate_event_spec(event)
    start, end = event["time_window_seconds"]
    times = tracked.times
    window_mask = (times >= start) & (times <= end)
    visible = tracked.visible()

    subject_visible = bool(
        window_mask.any() and visible[window_mask].mean() >= config.visible_fraction
    )

    lateral = tracked.lateral(image_width)
    net = float(lateral[-1] - lateral[0])

    metrics: dict[str, float] = {
        "lateral_start_px": float(lateral[0]) if visible[0] else float("nan"),
        "lateral_end_px": float(lateral[-1]) if visible[-1] else float("nan"),
        "net_lateral_px": net,
        "subject_visible_fraction": (
            float(visible[window_mask].mean()) if window_mask.any() else 0.0
        ),
        "mean_gt_error_px": _mean_gt_error(tracked, gt, window_mask),
    }

    if not subject_visible:
        checks = {
            "subject_visible": False,
            "motion_observed": None,
            "direction_correct": None,
            "timing_valid": None,
            "event_completed": None,
            "matches_proxy_trajectory": _trajectory_match(tracked, gt, window_mask, config),
        }
        return GroundedResult(
            event_id=event["event_id"],
            status="not_observed",
            checks=checks,
            metrics=metrics,
            rationale="Subject was not detected in the event window.",
        )

    # Lateral motion: per-frame travel and the fraction occurring in-window.
    deltas = np.diff(lateral)
    abs_deltas = np.abs(deltas)
    mid_times = (times[:-1] + times[1:]) / 2.0
    in_window = (mid_times >= start) & (mid_times <= end)
    total_travel = float(abs_deltas.sum())
    window_travel = float(abs_deltas[in_window].sum())
    motion_observed = total_travel >= config.motion_min_pixels
    metrics["lateral_travel_total_px"] = total_travel
    metrics["lateral_travel_window_px"] = window_travel
    metrics["window_travel_fraction"] = (
        window_travel / total_travel if total_travel > 0 else 0.0
    )

    direction_correct: bool | None = None
    timing_valid: bool | None = None
    if motion_observed:
        direction = event.get("direction")
        if direction is not None and abs(net) >= config.motion_min_pixels:
            expected_sign = 1.0 if direction == "right_to_left" else -1.0
            direction_correct = bool(net * expected_sign > 0)
        timing_valid = bool(
            window_travel / total_travel >= config.timing_fraction
        )

    # Completion: the subject reached the target lane (center for ego_lane).
    target_lateral = 0.0
    event_completed = bool(
        visible[-1] and abs(lateral[-1] - target_lateral) <= config.completion_max_pixels
    )

    matches = _trajectory_match(tracked, gt, window_mask, config)
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


def _trajectory_match(
    tracked: Trajectory,
    gt: Trajectory | None,
    window_mask: np.ndarray,
    config: GroundedConfig,
) -> bool | None:
    """Mean pixel error between tracked and GT points inside the window."""
    error = _mean_gt_error(tracked, gt, window_mask)
    if error is None:
        return None
    return bool(error <= config.match_max_pixels)


def _mean_gt_error(
    tracked: Trajectory, gt: Trajectory | None, window_mask: np.ndarray
) -> float | None:
    if gt is None:
        return None
    common = tracked.visible() & gt.visible() & window_mask
    if common.sum() < 2:
        return None
    error = np.linalg.norm(tracked.points[common] - gt.points[common], axis=1)
    return float(error.mean())

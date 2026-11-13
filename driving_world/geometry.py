"""Shared 3D geometry for the proxy renderer and the grounded judge.

Coordinate convention (world and ego): x forward, y left, z up; angles in
radians. The camera mount and yaw/pitch/roll are relative to ego, and the
camera optical axis is its local +x. This mirrors :mod:`driving_world.proxy`
so the renderer and the judge always agree on where an actor lands on screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def rotation(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Return the local-to-parent rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def interpolate_track(
    track: Sequence[dict[str, Any]], t: float
) -> tuple[np.ndarray, float]:
    """Interpolate position and yaw, holding endpoint poses outside the range."""
    if not track:
        raise ValueError("A track must contain at least one keyframe")
    if t <= track[0]["t"]:
        return np.array(track[0]["position"], dtype=float), track[0].get("yaw", 0.0)
    for left, right in zip(track, track[1:]):
        if t <= right["t"]:
            alpha = (t - left["t"]) / (right["t"] - left["t"])
            position = (1 - alpha) * np.array(left["position"]) + alpha * np.array(
                right["position"]
            )
            y0, y1 = left.get("yaw", 0.0), right.get("yaw", 0.0)
            delta = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
            return position, y0 + alpha * delta
    return np.array(track[-1]["position"], dtype=float), track[-1].get("yaw", 0.0)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera mounted on the ego vehicle."""

    width: int
    height: int
    fov_degrees: float
    position: tuple[float, float, float] = (0.0, 0.0, 1.5)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    near: float = 0.1

    @property
    def focal(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov_degrees) / 2.0))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Camera":
        """Build a camera from a scene's ``camera`` object."""
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fov_degrees=float(data["fov_degrees"]),
            position=tuple(data.get("position", (0.0, 0.0, 1.5))),
            yaw=float(data.get("yaw", 0.0)),
            pitch=float(data.get("pitch", 0.0)),
            roll=float(data.get("roll", 0.0)),
        )


def camera_pose(
    ego_position: Sequence[float], ego_yaw: float, camera: Camera
) -> tuple[np.ndarray, np.ndarray]:
    """Return the camera ``(position, rotation)`` in world coordinates."""
    ego_rot = rotation(ego_yaw)
    camera_pos = np.asarray(ego_position, dtype=float) + ego_rot @ np.asarray(
        camera.position, dtype=float
    )
    camera_rot = ego_rot @ rotation(camera.yaw, camera.pitch, camera.roll)
    return camera_pos, camera_rot


def project_point(
    world_point: Sequence[float],
    ego_position: Sequence[float],
    ego_yaw: float,
    camera: Camera,
) -> tuple[float, float] | None:
    """Project a world point to ``(u, v)`` image pixels, or ``None`` behind camera."""
    camera_pos, camera_rot = camera_pose(ego_position, ego_yaw, camera)
    p = camera_rot.T @ (np.asarray(world_point, dtype=float) - camera_pos)
    if p[0] < camera.near:
        return None
    return (
        camera.width / 2.0 - camera.focal * p[1] / p[0],
        camera.height / 2.0 - camera.focal * p[2] / p[0],
    )

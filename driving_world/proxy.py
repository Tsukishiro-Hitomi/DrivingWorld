#!/usr/bin/env python3
"""Render a dataset-independent 3D driving proxy from a JSON scene.

Coordinates: world and ego use x forward, y left, z up; angles are radians.
Camera mount position and yaw/pitch/roll are relative to ego. Camera optical
axis is its local +x, with +y left and +z up. Tracks are world-coordinate
keyframes, interpolated linearly with shortest-arc yaw interpolation.
"""

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
COLORS = {
    "vehicle": (65, 166, 255),
    "pedestrian": (255, 100, 100),
    "cyclist": (255, 210, 60),
}


def rotation(yaw, pitch=0.0, roll=0.0):
    """Return local-to-parent rotation Rz(yaw) Ry(pitch) Rx(roll)."""
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


def interpolate(track, t):
    """Interpolate position and yaw, holding endpoint poses outside the range."""
    if not track:
        raise ValueError("A track must contain at least one keyframe")
    if t <= track[0]["t"]:
        return np.array(track[0]["position"], dtype=float), track[0].get("yaw", 0.0)
    for left, right in zip(track, track[1:]):
        if t <= right["t"]:
            alpha = (t - left["t"]) / (right["t"] - left["t"])
            p = (1 - alpha) * np.array(left["position"]) + alpha * np.array(
                right["position"]
            )
            y0, y1 = left.get("yaw", 0.0), right.get("yaw", 0.0)
            delta = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
            return p, y0 + alpha * delta
    return np.array(track[-1]["position"], dtype=float), track[-1].get("yaw", 0.0)


def validate(scene):
    if scene["duration"] <= 0 or scene["fps"] <= 0:
        raise ValueError("duration and fps must be positive")
    camera = scene["camera"]
    if (
        camera["width"] <= 0
        or camera["height"] <= 0
        or not 0 < camera["fov_degrees"] < 180
    ):
        raise ValueError("Invalid camera dimensions or horizontal field of view")
    ids = set()
    for obj in scene.get("actors", []):
        if obj["id"] in ids or len(obj["size"]) != 3 or min(obj["size"]) <= 0:
            raise ValueError(
                "Actor IDs must be unique and sizes must be positive [length,width,height]"
            )
        ids.add(obj["id"])
    for track in [scene["ego"]] + [a["track"] for a in scene.get("actors", [])]:
        if not track or any(len(k["position"]) != 3 for k in track):
            raise ValueError("Every track needs 3D position keyframes")
        if any(a["t"] >= b["t"] for a, b in zip(track, track[1:])):
            raise ValueError("Keyframe times must be strictly increasing")


def render(scene, t):
    camera = scene["camera"]
    width, height = camera["width"], camera["height"]
    image = Image.new("RGB", (width, height), (5, 9, 15))
    draw = ImageDraw.Draw(image)
    ego_pos, ego_yaw = interpolate(scene["ego"], t)
    ego_rot = rotation(ego_yaw)
    mount = np.array(camera.get("position", [0, 0, 1.5]), dtype=float)
    camera_pos = ego_pos + ego_rot @ mount
    camera_rot = ego_rot @ rotation(
        camera.get("yaw", 0), camera.get("pitch", 0), camera.get("roll", 0)
    )
    focal = width / (2 * math.tan(math.radians(camera["fov_degrees"]) / 2))

    def line(a, b, color, line_width=1):
        # Clip line segments against the camera near plane before projection.
        a, b = camera_rot.T @ (a - camera_pos), camera_rot.T @ (b - camera_pos)
        near = 0.1
        if a[0] < near and b[0] < near:
            return
        if a[0] < near:
            a = a + (b - a) * ((near - a[0]) / (b[0] - a[0]))
        elif b[0] < near:
            b = b + (a - b) * ((near - b[0]) / (a[0] - b[0]))

        def project(p):
            return (width / 2 - focal * p[1] / p[0], height / 2 - focal * p[2] / p[0])

        draw.line((project(a), project(b)), fill=color, width=line_width)

    grid = scene.get("grid", {})
    extent, spacing = grid.get("extent", 60), grid.get("spacing", 5)
    if extent <= 0 or spacing <= 0 or extent / spacing > 500:
        raise ValueError("Grid extent and spacing must be positive and bounded")
    # World-anchored grid; motion of ego therefore produces camera motion.
    for value in np.arange(-extent, extent + spacing / 2, spacing):
        line(np.array([value, -extent, 0]), np.array([value, extent, 0]), (55, 95, 100))
        line(np.array([-extent, value, 0]), np.array([extent, value, 0]), (55, 95, 100))
    for actor in scene.get("actors", []):
        if not actor.get("start", 0) <= t < actor.get("end", scene["duration"]):
            continue
        position, yaw = interpolate(actor["track"], t)
        length, width_box, height_box = actor["size"]
        corners = [
            position
            + rotation(yaw)
            @ np.array([x * length / 2, y * width_box / 2, z * height_box / 2])
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ]
        for a, b in EDGES:
            line(
                corners[a],
                corners[b],
                COLORS.get(actor.get("type"), (185, 110, 235)),
                2,
            )
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    validate(scene)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = math.ceil(scene["duration"] * scene["fps"])
    with imageio.get_writer(
        str(args.output), fps=scene["fps"], codec="libx264", macro_block_size=1
    ) as writer:
        for index in range(count):
            writer.append_data(render(scene, index / scene["fps"]))
    print(f"Rendered {count} frames to {args.output}")


if __name__ == "__main__":
    main()

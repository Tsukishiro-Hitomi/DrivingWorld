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

from driving_world.geometry import (
    Camera,
    camera_pose,
    interpolate_track as interpolate,
    rotation,
)

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
    cam = Camera.from_dict(scene["camera"])
    width, height = cam.width, cam.height
    image = Image.new("RGB", (width, height), (5, 9, 15))
    draw = ImageDraw.Draw(image)
    ego_pos, ego_yaw = interpolate(scene["ego"], t)
    camera_pos, camera_rot = camera_pose(ego_pos, ego_yaw, cam)
    focal = cam.focal

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

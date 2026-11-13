"""TF-free Waymo tfrecord reading: front camera frames and 3D tracks."""

from __future__ import annotations

import io
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .derive import CYCLIST, PEDESTRIAN, VEHICLE, WorldTrack

CAM_FRONT = 1
DEFAULT_FPS = 10.0


def iter_tfrecord(path: Path):
    """Yield raw record payloads without importing TensorFlow."""
    with path.open("rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"truncated TFRecord header in {path}")
            length = struct.unpack("<Q", header[:8])[0]
            payload = handle.read(length)
            handle.read(4)  # masked CRC32C
            if len(payload) != length:
                raise ValueError(f"truncated TFRecord payload in {path}")
            yield payload


def read_frames(path: Path):
    """Parse every frame of a record into protobuf objects."""
    from waymo_open_dataset import dataset_pb2 as waymo_pb  # lazy: protobuf only

    frames = []
    for payload in iter_tfrecord(path):
        frame = waymo_pb.Frame()
        frame.ParseFromString(payload)
        frames.append(frame)
    return frames


def front_image(frame) -> np.ndarray | None:
    """Decode the CAM_FRONT image as RGB, or ``None`` when absent."""
    for image in frame.images:
        if int(image.name) == CAM_FRONT:
            with Image.open(io.BytesIO(image.image)) as img:
                return np.asarray(img.convert("RGB"))
    return None


def front_focal(frames) -> float:
    """Return the CAM_FRONT vertical focal length in pixels."""
    for calibration in frames[0].context.camera_calibrations:
        if int(calibration.name) == CAM_FRONT:
            # intrinsic = [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
            return float(calibration.intrinsic[1])
    raise ValueError("no CAM_FRONT calibration in the record")


def world_tracks_from_frames(frames, fps: float = DEFAULT_FPS) -> list[WorldTrack]:
    """Build per-object 3D tracks with ego/road lateral positions and visibility."""
    focal = front_focal(frames)
    poses = [
        np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4) for frame in frames
    ]
    origin = poses[0][:3, 3]
    road_left = poses[0][:3, 1]
    road_left = road_left / np.linalg.norm(road_left)

    points: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    types: dict[str, int] = {}
    for index, frame in enumerate(frames):
        pose = poses[index]
        t = index / fps
        for label in frame.laser_labels:
            type_ = int(label.type)
            if type_ not in (VEHICLE, PEDESTRIAN, CYCLIST):
                continue
            track_id = str(label.id)
            types[track_id] = type_
            depth = max(1.0, float(np.hypot(label.box.center_x, label.box.center_y)))
            apparent_px = focal * float(label.box.height) / depth
            world = pose @ np.array(
                [label.box.center_x, label.box.center_y, label.box.center_z, 1.0]
            )
            y_road = float(np.dot(world[:3] - origin, road_left))
            points[track_id].append(
                (
                    t,
                    float(label.box.center_x),
                    float(label.box.center_y),
                    y_road,
                    apparent_px,
                )
            )

    tracks = []
    for track_id, track_points in points.items():
        times, xs, ys, ys_road, apparent = zip(*track_points)
        tracks.append(
            WorldTrack(
                track_id,
                types[track_id],
                np.asarray(times, dtype=float),
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
                np.asarray(ys_road, dtype=float),
                np.asarray(apparent, dtype=float),
            )
        )
    return tracks

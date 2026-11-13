"""TF-free Waymo tfrecord reading: front camera frames, 2D and 3D tracks."""

from __future__ import annotations

import io
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .derive import CYCLIST, PEDESTRIAN, VEHICLE, CameraTrack, WorldTrack

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


def camera_tracks_from_frames(frames, fps: float = DEFAULT_FPS) -> list[CameraTrack]:
    """Build per-object 2D-box tracks from the CAM_FRONT camera labels."""
    points: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    types: dict[str, int] = {}
    for index, frame in enumerate(frames):
        t = index / fps
        for camera in frame.camera_labels:
            if int(camera.name) != CAM_FRONT:
                continue
            for label in camera.labels:
                type_ = int(label.type)
                if type_ not in (VEHICLE, PEDESTRIAN, CYCLIST):
                    continue
                track_id = str(label.id)
                types[track_id] = type_
                points[track_id].append(
                    (
                        t,
                        float(label.box.center_x),
                        float(label.box.center_y),
                        float(label.box.length),
                        float(label.box.width),
                    )
                )

    tracks = []
    for track_id, track_points in points.items():
        times, us, vs, ws, hs = zip(*track_points)
        tracks.append(
            CameraTrack(
                track_id,
                types[track_id],
                np.asarray(times, dtype=float),
                np.asarray(us, dtype=float),
                np.asarray(vs, dtype=float),
                np.asarray(ws, dtype=float),
                np.asarray(hs, dtype=float),
            )
        )
    return tracks


def world_tracks_from_frames(frames, fps: float = DEFAULT_FPS) -> list[WorldTrack]:
    """Build per-object 3D tracks from the laser labels (ego frame)."""
    points: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    types: dict[str, int] = {}
    for index, frame in enumerate(frames):
        t = index / fps
        for label in frame.laser_labels:
            type_ = int(label.type)
            if type_ not in (VEHICLE, PEDESTRIAN, CYCLIST):
                continue
            track_id = str(label.id)
            types[track_id] = type_
            points[track_id].append(
                (t, float(label.box.center_x), float(label.box.center_y))
            )

    tracks = []
    for track_id, track_points in points.items():
        times, xs, ys = zip(*track_points)
        tracks.append(
            WorldTrack(
                track_id,
                types[track_id],
                np.asarray(times, dtype=float),
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
            )
        )
    return tracks

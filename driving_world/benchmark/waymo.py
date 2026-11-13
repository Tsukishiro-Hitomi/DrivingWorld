"""TF-free Waymo tfrecord reading: front camera frames + object tracks."""

from __future__ import annotations

import io
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .derive import CYCLIST, PEDESTRIAN, VEHICLE, Track

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


def tracks_from_frames(frames, fps: float = DEFAULT_FPS) -> list[Track]:
    """Build per-object tracks from the 3D laser labels."""
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
    return [
        Track.from_points(track_id, types[track_id], track_points)
        for track_id, track_points in points.items()
    ]

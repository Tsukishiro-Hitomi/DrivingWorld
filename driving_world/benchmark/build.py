"""Build the real-video VLM event-recognition benchmark from Waymo records.

Scans Waymo ``.tfrecord`` files, derives ground-truth events from the 3D box
tracks, and exports short front-camera clips (one per event, plus negative
clips with no event) together with a ``manifest.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .derive import DeriveConfig, derive_events
from .waymo import (
    DEFAULT_FPS,
    camera_tracks_from_frames,
    front_image,
    read_frames,
    world_tracks_from_frames,
)

DEFAULT_TARGETS = {
    "vehicle_lane_change": 8,
    "pedestrian_crossing": 6,
    "vehicle_hard_brake": 6,
    "negative": 8,
}


def _clip_bounds(
    frame_count: int, event, clip_seconds: float, fps: float
) -> tuple[int, int] | None:
    midpoint = (event.start_seconds + event.end_seconds) / 2.0
    half = clip_seconds / 2.0
    start = int(max(0, midpoint - half) * fps)
    end = int(min(frame_count, (midpoint + half) * fps))
    if end - start < int(0.5 * fps):
        return None
    return start, end


def export_clip(frames: list[np.ndarray], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {output}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _front_frames(frames, start: int, end: int) -> list[np.ndarray]:
    """Decode front-camera frames in ``[start, end)``, dropping any missing ones."""
    decoded: list[np.ndarray] = []
    for index in range(start, end):
        image = front_image(frames[index])
        if image is not None:
            decoded.append(image)
    return decoded


def build(
    data_root: Path,
    out_dir: Path,
    *,
    limit: int,
    targets: dict[str, int] | None = None,
    clip_seconds: float = 6.0,
    fps: float = DEFAULT_FPS,
) -> Path:
    targets = targets or DEFAULT_TARGETS
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = defaultdict(int)
    manifest: list[dict] = []
    records = sorted(data_root.rglob("*.tfrecord"))[:limit]

    for record in records:
        if all(counts[k] >= n for k, n in targets.items() if k != "negative"):
            break
        frames = read_frames(record)
        if not frames:
            continue
        camera_tracks = camera_tracks_from_frames(frames, fps=fps)
        world_tracks = world_tracks_from_frames(frames, fps=fps)
        events = derive_events(world_tracks, camera_tracks, DeriveConfig())

        for event in events:
            kind = event.kind
            if counts[kind] >= targets.get(kind, DEFAULT_TARGETS[kind]):
                continue
            bounds = _clip_bounds(len(frames), event, clip_seconds, fps)
            if bounds is None:
                continue
            start, end = bounds
            clip_id = f"{kind}_{counts[kind]:03d}"
            export_clip(
                _front_frames(frames, start, end),
                clips_dir / f"{clip_id}.mp4",
                fps,
            )
            manifest.append(
                {
                    "clip_id": clip_id,
                    "video": f"clips/{clip_id}.mp4",
                    "record": record.name,
                    "labels": [kind],
                    "events": [
                        {
                            "kind": kind,
                            "track_id": event.track_id,
                            "subject_type": event.subject_type,
                            "clip_start_seconds": round(event.start_seconds - start / fps, 2),
                            "clip_end_seconds": round(event.end_seconds - start / fps, 2),
                            "direction": event.direction,
                        }
                    ],
                }
            )
            counts[kind] += 1

        if not events and counts["negative"] < targets["negative"]:
            start, end = 0, int(clip_seconds * fps)
            end = min(end, len(frames))
            if end - start >= int(0.5 * fps):
                clip_id = f"negative_{counts['negative']:03d}"
                export_clip(
                    _front_frames(frames, start, end),
                    clips_dir / f"{clip_id}.mp4",
                    fps,
                )
                manifest.append(
                    {
                        "clip_id": clip_id,
                        "video": f"clips/{clip_id}.mp4",
                        "record": record.name,
                        "labels": [],
                        "events": [],
                    }
                )
                counts["negative"] += 1

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in manifest) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/xs-fd-nas/dataset/waymo"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60, help="max records to scan")
    parser.add_argument("--clip-seconds", type=float, default=6.0)
    args = parser.parse_args()
    path = build(
        args.data_root, args.out_dir, limit=args.limit, clip_seconds=args.clip_seconds
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

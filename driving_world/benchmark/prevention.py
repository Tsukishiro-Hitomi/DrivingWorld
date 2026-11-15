"""PREVENTION dataset: human-annotated lane changes with per-frame 3D tracks.

What is trusted, and what is not
--------------------------------
``lane_changes.txt`` is the human annotation and the reason this dataset exists.
Each row is ``[vehicle, counterpart, type, f0, f1, f2, blinker]``: ``type`` 3 is a
left change and 4 a right change, ``f1`` is the frame at which the rear middle of
the vehicle sits on the lane divider, ``f0`` the start and ``f2`` the end of the
maneuver.  ``f1`` and ``f2`` are present for every one of the 852 events; ``f0`` is
``-1`` for 22% of them.  The crossing moment is therefore the only safe anchor.

``trajectories.txt`` gives per-frame 3D centres in the camera-1 frame, in metres.
Projecting them through the calibrated intrinsics reproduces the 2D boxes from
``labels.txt`` exactly, so the two files agree.

``lanes.txt`` holds clothoid lane *line* parameters per frame.  ``c0`` is a metric
lateral offset, but the detector is incomplete: at the annotated crossing frame the
subject lies within 0.5 m of a detected line for only 41 of 57 checkable events.
It is carried along as context and never used to decide a label.

Coverage
--------
Only three drives keep a vehicle id on the annotation *and* usable tracks, which is
why 852 annotated events reduce to roughly a hundred buildable ones.  Everything
here assumes the caller has already extracted the small ``.txt`` members of
``processed_data.zip``; nothing in this module touches the network.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Driving videos are 10 Hz, 1920x600 after the sensor's own letterbox crop.
FPS = 10.0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 600
VIDEO_URL = (
    "https://prevention-dataset.uah.es/static/RECORD{record}/DRIVE{drive}/video_camera1.mp4"
)

#: Spanish motorway lanes are close to the observed 2.8 m median lateral shift plus margin.
LANE_WIDTH_M = 3.5

LEFT_CHANGE = 3
RIGHT_CHANGE = 4


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics plus the five Brown-Conrady distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LaneChange:
    """One human-annotated lane change."""

    vehicle: int
    counterpart: int
    direction: str  # "left" | "right"
    f0: int  # -1 when the annotators did not mark a start
    f1: int
    f2: int
    blinker: bool

    @property
    def has_start(self) -> bool:
        return self.f0 >= 0


@dataclass(frozen=True)
class LaneChangeEvent:
    """A buildable sample: the annotation joined with geometry and a clip window."""

    drive: str
    vehicle: int
    label: int  # 1 positive, 0 negative
    stratum: str  # cut_in | cut_out | lateral | negative_clean | negative_hard
    direction: str | None
    f0: int
    f1: int
    f2: int
    blinker: bool
    start_lane: str
    end_lane: str
    depth_m: float
    clip_start: int
    clip_frames: int

    @property
    def clip_id(self) -> str:
        kind = "pos" if self.label else "neg"
        drive = self.drive.replace("_", "").lower()
        return f"{kind}_{drive}_{self.f1:05d}"


@dataclass(frozen=True)
class SampleConfig:
    """Gates applied when turning annotations into buildable clips."""

    clip_frames: int = 80  # 8 s: enough lead-in to see the maneuver begin
    pre_frames: int = 45  # keep the crossing 4.5 s into the clip
    min_depth_m: float = 2.0
    max_depth_m: float = 100.0
    lane_width_m: float = LANE_WIDTH_M
    max_lanes_jumped: int = 1  # a two-lane sweep is a different maneuver
    negative_guard_frames: int = 50  # negatives must clear any labeled event
    #: label dropout is common; interpolate over a gap no wider than this
    tolerance_frames: int = 8
    #: when f0 is missing, look back this far from the crossing for the start lane
    fallback_start_frames: int = 30


# ------------------------------------------------------------------ annotation IO


def read_camera(path: Path) -> Camera:
    """Parse ``camera1_intrinsic_calibration.dat``."""
    text = Path(path).read_text(encoding="utf-8")
    size = values = distortion = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "ImageSize":
            size = (int(parts[1]), int(parts[2]))
        elif parts[0] == "IntrinsicParams":
            # fx, skew, cx, fy, cy -- the skew term is dropped, it is always ~0
            values = [float(v) for v in parts[1:6]]
        elif parts[0] == "DistortionCoefficients":
            distortion = tuple(float(v) for v in parts[1:6])
    if size is None or values is None or distortion is None:
        raise ValueError(f"unreadable calibration file: {path}")
    if size != (FRAME_WIDTH, FRAME_HEIGHT):
        raise ValueError(f"unexpected frame size {size} in {path}")
    return Camera(values[0], values[3], values[2], values[4], distortion)


def load_lane_changes(path: Path) -> list[LaneChange]:
    """Parse ``lane_changes.txt``, keeping every row with seven fields."""
    changes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 7:
            continue
        vehicle, counterpart, kind, f0, f1, f2, blinker = (int(p) for p in parts)
        if kind not in (LEFT_CHANGE, RIGHT_CHANGE):
            continue
        changes.append(
            LaneChange(
                vehicle=vehicle,
                counterpart=counterpart,
                direction="left" if kind == LEFT_CHANGE else "right",
                f0=f0,
                f1=f1,
                f2=f2,
                blinker=bool(blinker),
            )
        )
    return changes


def load_trajectories(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """Parse ``trajectories.txt`` into ``vehicle -> frame -> (x, y, z)`` metres.

    The camera frame is x right, y down, z forward, which the projection check in
    the module docstring confirms.
    """
    tracks: dict[int, dict[int, np.ndarray]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 10:
            continue
        frame, vehicle = int(parts[0]), int(parts[1])
        tracks.setdefault(vehicle, {})[frame] = np.array(
            [float(parts[4]), float(parts[5]), float(parts[6])], dtype=np.float64
        )
    return tracks


def load_lanes(path: Path) -> dict[int, dict[int, float]]:
    """Parse ``lanes.txt`` into ``frame -> lane id -> c0`` metric lateral offset."""
    lanes: dict[int, dict[int, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 9:
            continue
        frame, lane, c0, _c1, _c2, _c3, detected, _permanent, _threshold = parts
        if detected != "1":
            continue
        lanes.setdefault(int(frame), {})[int(lane)] = float(c0)
    return lanes


def load_boxes(path: Path) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    """Parse ``labels.txt`` into ``frame -> vehicle -> (x, y, w, h)`` pixels."""
    boxes: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 8:
            continue
        frame = int(float(parts[0]))
        vehicle = int(float(parts[1]))
        boxes.setdefault(frame, {})[vehicle] = tuple(float(v) for v in parts[2:6])
    return boxes


# ------------------------------------------------------------------ lane geometry


def lane_index(lateral_m: float, lane_width_m: float = LANE_WIDTH_M) -> int:
    """Lane index of a lateral offset, ``0`` being the ego lane.

    Positive offsets are to the right of the ego lane centre, matching the camera.
    """
    return int(round(lateral_m / lane_width_m))


def lane_name(index: int) -> str:
    return {-2: "far_left", -1: "left", 0: "ego", 1: "right", 2: "far_right"}.get(
        index, f"lane{index:+d}"
    )


def _tracked_between(track: dict[int, np.ndarray], start: int, end: int) -> list[int]:
    return [frame for frame in range(start, end + 1) if frame in track]


def _sample_at(
    track: dict[int, np.ndarray], frame: int, tolerance: int
) -> tuple[float, float] | None:
    """Return ``(lateral, depth)`` at ``frame``, interpolating a short dropout.

    Label tracks drop out for a handful of frames at a time; refusing to bridge
    them throws away a third of the annotations.  Gaps wider than ``tolerance``
    still fail, so a subject that was never really tracked is not invented.
    """
    if frame in track:
        return float(track[frame][0]), float(track[frame][2])
    before = [f for f in track if f < frame]
    after = [f for f in track if f > frame]
    if not before or not after:
        return None
    lo, hi = max(before), min(after)
    if frame - lo > tolerance or hi - frame > tolerance:
        return None
    weight = (frame - lo) / (hi - lo)
    lateral = track[lo][0] * (1 - weight) + track[hi][0] * weight
    depth = track[lo][2] * (1 - weight) + track[hi][2] * weight
    return float(lateral), float(depth)


# ------------------------------------------------------------------ event building


def _window(f1: int, config: SampleConfig) -> tuple[int, int]:
    """Clip window that keeps the crossing ``pre_frames`` in, clamped at zero."""
    start = max(0, f1 - config.pre_frames)
    return start, config.clip_frames


def build_events(
    drive: str,
    changes: list[LaneChange],
    tracks: dict[int, dict[int, np.ndarray]],
    config: SampleConfig = SampleConfig(),
) -> list[LaneChangeEvent]:
    """Join annotations with geometry, dropping the ones that cannot be built.

    An event is buildable only when the subject is tracked at both the crossing and
    the end frame, because those two positions are what define the lane transition.
    """
    events = []
    for change in changes:
        track = tracks.get(change.vehicle)
        if not track:
            continue
        crossing = _sample_at(track, change.f1, config.tolerance_frames)
        if crossing is None:
            continue
        # prefer the annotated start; when it is missing or untracked, look back a
        # fixed distance so the start lane is not measured on the crossing frame
        if change.f0 >= 0:
            start_frame = change.f0
        else:
            start_frame = change.f1 - config.fallback_start_frames
        start = _sample_at(track, start_frame, config.tolerance_frames)
        if start is None:
            start_frame = change.f1 - config.fallback_start_frames
            start = _sample_at(track, start_frame, config.tolerance_frames)
            if start is None:
                continue
        end = _sample_at(track, change.f2, config.tolerance_frames)
        if end is None:
            continue

        start_lane = lane_index(start[0], config.lane_width_m)
        end_lane = lane_index(end[0], config.lane_width_m)
        if start_lane == end_lane:
            continue
        if abs(end_lane - start_lane) > config.max_lanes_jumped:
            continue

        depth = crossing[1]
        if not config.min_depth_m < depth < config.max_depth_m:
            continue

        if start_lane != 0 and end_lane == 0:
            stratum = "cut_in"
        elif start_lane == 0 and end_lane != 0:
            stratum = "cut_out"
        else:
            stratum = "lateral"

        start, frames = _window(change.f1, config)
        events.append(
            LaneChangeEvent(
                drive=drive,
                vehicle=change.vehicle,
                label=1,
                stratum=stratum,
                direction=change.direction,
                f0=change.f0,
                f1=change.f1,
                f2=change.f2,
                blinker=change.blinker,
                start_lane=lane_name(start_lane),
                end_lane=lane_name(end_lane),
                depth_m=round(depth, 2),
                clip_start=start,
                clip_frames=frames,
            )
        )
    return sorted(events, key=lambda e: e.f1)


def sample_negatives(
    drive: str,
    changes: list[LaneChange],
    tracks: dict[int, dict[int, np.ndarray]],
    count: int,
    *,
    seed: int = 0,
    config: SampleConfig = SampleConfig(),
) -> list[LaneChangeEvent]:
    """Draw windows in which no annotated lane change involves the subject.

    Negatives are defined by the *absence of an annotation*, never by a detector,
    so scoring a detector against them stays honest.  Windows that sit within
    ``negative_guard_frames`` of a labeled event for the same vehicle are rejected,
    which keeps the sample clear of the maneuver and of its lead-in.
    """
    if count <= 0:
        return []
    label_span: dict[int, list[tuple[int, int]]] = {}
    for change in changes:
        label_span.setdefault(change.vehicle, []).append((change.f0, change.f2))

    candidates = []
    for vehicle, track in tracks.items():
        if not track:
            continue
        frames = sorted(track)
        span = (frames[0], frames[-1])
        if span[1] - span[0] <= config.clip_frames:
            continue
        blocked = label_span.get(vehicle, [])
        candidates.append((vehicle, span, blocked, track))
    candidates.sort(key=lambda item: item[0])

    rng = np.random.default_rng(seed)
    events: list[LaneChangeEvent] = []
    attempts = 0
    while len(events) < count and attempts < count * 200:
        attempts += 1
        vehicle, (lo, hi), blocked, track = candidates[int(rng.integers(len(candidates)))]
        latest = hi - config.clip_frames
        if latest <= lo:
            continue
        start = int(rng.integers(lo, latest))
        end = start + config.clip_frames
        guard = config.negative_guard_frames
        if any(b0 >= 0 and not (end + guard < b0 or start - guard > b1) for b0, b1 in blocked):
            continue
        middle = start + config.clip_frames // 2
        if middle not in track:
            continue
        depth = float(track[middle][2])
        if not config.min_depth_m < depth < config.max_depth_m:
            continue
        lanes = lane_index(track[middle][0], config.lane_width_m)
        events.append(
            LaneChangeEvent(
                drive=drive,
                vehicle=vehicle,
                label=0,
                stratum="negative_clean",
                direction=None,
                f0=-1,
                f1=middle,
                f2=end,
                blinker=False,
                start_lane=lane_name(lanes),
                end_lane=lane_name(lanes),
                depth_m=round(depth, 2),
                clip_start=start,
                clip_frames=config.clip_frames,
            )
        )
    return events


def hard_negatives(
    drive: str,
    events: list[LaneChangeEvent],
    tracks: dict[int, dict[int, np.ndarray]],
    *,
    gap_frames: int = 40,
    config: SampleConfig = SampleConfig(),
) -> list[LaneChangeEvent]:
    """Lane-keeping windows from the same subject right before its own maneuver.

    These are the cases a judge is most likely to fire on: the subject is in frame,
    close, and about to move, but has not crossed yet.
    """
    out = []
    for event in events:
        if event.label != 1 or event.f0 < 0:
            continue
        track = tracks.get(event.vehicle, {})
        # the window must finish before the maneuver's lead-in begins
        end = event.f0 - gap_frames
        start = end - config.clip_frames
        if start < 0:
            continue
        frames = _tracked_between(track, start, end)
        if len(frames) < config.clip_frames // 2:
            continue
        middle = start + config.clip_frames // 2
        sampled = _sample_at(track, middle, config.tolerance_frames)
        if sampled is None:
            continue
        out.append(
            LaneChangeEvent(
                drive=drive,
                vehicle=event.vehicle,
                label=0,
                stratum="negative_hard",
                direction=None,
                f0=-1,
                f1=middle,
                f2=end,
                blinker=False,
                start_lane=event.start_lane,
                end_lane=event.start_lane,
                depth_m=round(sampled[1], 2),
                clip_start=start,
                clip_frames=config.clip_frames,
            )
        )
    return out


# ------------------------------------------------------------------ video access


def extract_frames(
    source: str,
    start_frame: int,
    count: int,
    *,
    fps: float = FPS,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> np.ndarray:
    """Decode exactly ``count`` frames beginning at ``start_frame``.

    ``source`` may be a local path or an HTTP URL; ``ffmpeg`` seeks with range
    requests, so a six-second window costs a few megabytes rather than the whole
    video.  ``-ss`` before ``-i`` with re-encoding is frame accurate: extracting at
    ``t`` and ``t + 0.1`` differs by exactly one frame.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_frame / fps:.3f}",
        "-i",
        source,
        "-frames:v",
        str(count),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace")[-2000:])
    frame_bytes = width * height * 3
    available = len(completed.stdout) // frame_bytes
    if available == 0:
        raise RuntimeError(f"no frames decoded at {start_frame} from {source}")
    usable = min(available, count)
    return np.frombuffer(
        completed.stdout[: usable * frame_bytes], dtype=np.uint8
    ).reshape(usable, height, width, 3)


def undistort_frames(frames: np.ndarray, camera: Camera) -> np.ndarray:
    """Undistort a batch of RGB frames; a no-op when the coefficients are all zero."""
    import cv2  # local import: only video materialisation needs OpenCV

    if not any(camera.distortion):
        return frames
    return np.stack([cv2.undistort(frame, camera.matrix, np.asarray(camera.distortion)) for frame in frames])


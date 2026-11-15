"""Video decoding and report generation for deterministic render checks."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .probes import (
    LaplacianSharpness,
    MotionCompensatedResidual,
    Probe,
    StructuralWarpAgreement,
)
from .reference import ReferenceDistribution
from .score import AXES, AxisScore, ScoreConfig, score_series

DEFAULT_PROBES = (
    LaplacianSharpness(),
    MotionCompensatedResidual(),
    StructuralWarpAgreement(),
)


@dataclass(frozen=True)
class Canonical:
    """Decode settings that calibration and scoring must agree on.

    Raw probes are sensitive to pixel scale and frame interval: a 480p H3 clip
    and a 1280p reference clip differ by orders of magnitude in Laplacian
    variance for exactly the same content. Both sides are therefore decoded to
    one frame height and one frame rate before measurement.
    """

    fps: float = 10.0
    height: int = 480


def _resize(frames: Sequence[np.ndarray], height: int) -> list[np.ndarray]:
    from PIL import Image

    resized = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        if image.height == height:
            resized.append(np.asarray(image))
            continue
        width = max(1, round(image.width * height / image.height))
        resized.append(np.asarray(image.resize((width, height), Image.BILINEAR)))
    return resized


def _resample(frames: Sequence[np.ndarray], fps: float, target_fps: float) -> list[np.ndarray]:
    """Pick nearest frames for `target_fps`; keep the source rate if it is slower."""
    if fps <= target_fps + 1e-9 or target_fps <= 0:
        return list(frames)
    duration = len(frames) / fps
    count = max(1, int(round(duration * target_fps)))
    indices = [min(len(frames) - 1, int(round(index * fps / target_fps))) for index in range(count)]
    return [frames[index] for index in dict.fromkeys(indices)]


def decode_video(
    video: Path, *, canonical: Canonical | None = None
) -> tuple[float, list[np.ndarray]]:
    """Decode a video into its FPS and RGB frames, importing imageio on demand."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("render checking requires the 'imageio' package") from exc
    metadata = iio.immeta(video, plugin="FFMPEG")
    fps = float(metadata.get("fps", 0))
    frames = [
        np.asarray(frame, dtype=np.uint8) for frame in iio.imiter(video, plugin="FFMPEG")
    ]
    if fps <= 0 or not frames:
        raise ValueError(f"cannot decode video frames and FPS from {video}")
    if canonical is None:
        return fps, frames
    if canonical.fps <= 0 or canonical.height <= 0:
        raise ValueError("canonical fps and height must be positive")
    resampled = _resample(frames, fps, canonical.fps)
    effective_fps = canonical.fps if len(resampled) != len(frames) else fps
    return effective_fps, _resize(resampled, canonical.height)


@dataclass(frozen=True)
class RenderReport:
    """Serializable result of deterministic render-quality inspection."""

    schema_version: int
    model: str
    video: str
    video_duration_seconds: float
    fps: float
    frame_count: int
    probes: tuple[str, ...]
    axes: Mapping[str, AxisScore]
    total: float
    status: str
    findings: tuple[dict, ...]
    reasons: tuple[str, ...]
    canonical: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the report and nested score records to JSON-compatible data."""
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "video": self.video,
            "video_duration_seconds": self.video_duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "probes": list(self.probes),
            "axes": {axis: score.to_dict() for axis, score in self.axes.items()},
            "total": self.total,
            "status": self.status,
            "findings": list(self.findings),
            "reasons": list(self.reasons),
            "canonical": self.canonical,
        }


class RenderChecker:
    """Decode a video, run probes, and return a calibrated render report."""

    def __init__(
        self,
        references: Mapping[str, ReferenceDistribution],
        probes: Sequence[Probe] | None = None,
        config: ScoreConfig = ScoreConfig(),
        canonical: Canonical | None = Canonical(),
    ) -> None:
        self.references = dict(references)
        self.probes = tuple(probes) if probes is not None else DEFAULT_PROBES
        self.config = config
        self.canonical = canonical

    def check(self, video: Path) -> RenderReport:
        """Decode and score one video without eagerly importing imageio."""
        fps, frames = decode_video(video, canonical=self.canonical)
        return self.check_frames(frames, fps=fps, video=str(video))

    def check_frames(
        self, frames: Sequence[np.ndarray], *, fps: float, video: str = "<frames>"
    ) -> RenderReport:
        """Score already decoded frames so callers can avoid a second decode."""
        series: dict[str, Sequence[float]] = {}
        for probe in self.probes:
            if probe.axis in series:
                raise ValueError(f"multiple probes configured for axis: {probe.axis}")
            reference = self.references.get(probe.axis)
            if reference is not None and reference.higher_is_better != probe.higher_is_better:
                raise ValueError(
                    f"reference for {probe.axis} disagrees with {probe.name} on score direction"
                )
            values = probe.measure(frames)
            if len(values) != len(frames):
                raise ValueError(f"probe {probe.name} returned the wrong number of values")
            series[probe.axis] = values
        clip = score_series(series, self.references, fps=fps, config=self.config)
        findings = self._findings(clip.axes)
        return RenderReport(
            schema_version=1,
            model="deterministic",
            video=video,
            video_duration_seconds=len(frames) / fps,
            fps=fps,
            frame_count=len(frames),
            probes=tuple(probe.name for probe in self.probes),
            axes=clip.axes,
            total=clip.total,
            status=clip.status,
            findings=findings,
            reasons=clip.reasons,
            canonical=(
                {"fps": self.canonical.fps, "height": self.canonical.height}
                if self.canonical is not None
                else None
            ),
        )

    def _findings(self, axes: Mapping[str, AxisScore]) -> tuple[dict, ...]:
        """Localize every axis that scored below the warn threshold, worst first."""
        checks_by_axis = {
            "imaging": "artifact_free",
            "temporal": "motion_smooth",
            "structure": "geometry_plausible",
        }
        findings = []
        failing = [
            score
            for score in (axes.get(axis) for axis in AXES)
            if score is not None and score.score < self.config.warn_below
        ]
        failing.sort(key=lambda value: value.score)
        for score in failing:
            axis = score.axis
            worst = min(
                score.windows,
                key=lambda value: (value.percentile, value.window.index),
            )
            indices = worst.window.frame_indices
            evidence = list(
                dict.fromkeys((indices[0], indices[len(indices) // 2], indices[-1]))
            )
            checks: dict[str, bool | None] = {
                "artifact_free": True,
                "motion_smooth": True,
                "geometry_plausible": True,
                "identity_consistent": None,
            }
            checks[checks_by_axis[axis]] = False
            reference = self.references[axis]
            reference_median = statistics.median(reference.samples)
            findings.append(
                {
                    "finding_id": f"render-{axis}-{worst.window.index:03d}",
                    "error_type": "render",
                    "severity": min(
                        1.0,
                        max(
                            0.0,
                            (self.config.fail_below - worst.percentile)
                            / self.config.fail_below,
                        ),
                    ),
                    "temporal_segment": [
                        worst.window.start_seconds,
                        worst.window.end_seconds,
                    ],
                    "evidence_frame_indices": evidence,
                    "checks": checks,
                    "rationale": (
                        f"The {axis} raw value {worst.raw:.6g} is at percentile "
                        f"{worst.percentile:.2f}, versus reference median raw "
                        f"{reference_median:.6g}."
                    ),
                }
            )
            if len(findings) >= self.config.max_findings:
                break
        return tuple(findings)


def write_report(destination: Path, report: RenderReport) -> None:
    """Write a report as JSON, refusing to replace an existing file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_references(
    videos: Sequence[Path],
    probes: Sequence[Probe] | None = None,
    canonical: Canonical | None = Canonical(),
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, ReferenceDistribution]:
    """Measure real clips to build one empirical reference per render axis.

    The first `warmup` frame of each probe is dropped: a probe that compares a
    frame against its predecessor has no predecessor for frame zero, so keeping
    those placeholders would bias the reference toward looking good.
    """
    active = tuple(probes) if probes is not None else DEFAULT_PROBES
    samples: dict[str, list[float]] = {probe.axis: [] for probe in active}
    decoded = 0
    for index, video in enumerate(videos, 1):
        _, frames = decode_video(Path(video), canonical=canonical)
        for probe in active:
            values = probe.measure(frames)
            if len(values) != len(frames):
                raise ValueError(f"probe {probe.name} returned the wrong number of values")
            warmup = int(getattr(probe, "warmup", 0))
            samples[probe.axis].extend(values[warmup:])
        decoded += 1
        if progress is not None:
            progress(index, len(videos))
    if decoded == 0:
        raise ValueError("no videos were decoded")
    return {
        probe.axis: ReferenceDistribution.from_samples(
            probe.name, probe.higher_is_better, samples[probe.axis]
        )
        for probe in active
    }


def write_references(
    directory: Path, references: Mapping[str, ReferenceDistribution]
) -> None:
    """Write one `<axis>.json` reference file per axis."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for axis, reference in references.items():
        reference.save(directory / f"{axis}.json")


def load_references(directory: Path) -> dict[str, ReferenceDistribution]:
    """Load `<axis>.json` reference files from a calibration directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"reference directory does not exist: {directory}")
    references = {}
    for path in sorted(directory.glob("*.json")):
        axis = path.stem
        if axis not in AXES:
            raise ValueError(f"unexpected reference file for unknown axis: {path}")
        references[axis] = ReferenceDistribution.load(path)
    if not references:
        raise ValueError(f"no reference files found in {directory}")
    return references

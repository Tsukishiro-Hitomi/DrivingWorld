"""Deterministic render-quality scoring for generated driving videos."""

from __future__ import annotations

from .probes import (
    DepthConsistencyProbe,
    LaplacianSharpness,
    MotionCompensatedResidual,
    MusiqProbe,
    Probe,
    ProbeUnavailable,
    RaftWarpProbe,
    StructuralWarpAgreement,
)
from .reference import ReferenceDistribution
from .render import (
    Canonical,
    RenderChecker,
    RenderReport,
    build_references,
    decode_video,
    load_references,
    write_references,
    write_report,
)
from .score import (
    AXES,
    AxisScore,
    ClipScore,
    ScoreConfig,
    WindowScore,
    score_series,
)
from .windows import Window, build_windows

__all__ = [
    "AXES",
    "AxisScore",
    "Canonical",
    "ClipScore",
    "DepthConsistencyProbe",
    "LaplacianSharpness",
    "MotionCompensatedResidual",
    "MusiqProbe",
    "Probe",
    "ProbeUnavailable",
    "RaftWarpProbe",
    "ReferenceDistribution",
    "RenderChecker",
    "RenderReport",
    "ScoreConfig",
    "StructuralWarpAgreement",
    "Window",
    "WindowScore",
    "build_references",
    "build_windows",
    "decode_video",
    "load_references",
    "score_series",
    "write_references",
    "write_report",
]

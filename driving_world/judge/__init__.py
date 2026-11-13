"""Video-only, evidence-based judges."""

from .grounding import (
    GroundedConfig,
    GroundedResult,
    GtTrack,
    Trajectory,
    evaluate,
    grounded_check,
    gt_track,
    track_subject_by_color,
    track_subject_from_video,
)
from .tracking import ColorTracker, DetectionTracker, Detector, SubjectTracker

__all__ = [
    "ObjectChecker",
    "ObjectResult",
    "write_result",
    "GroundedConfig",
    "GroundedResult",
    "GtTrack",
    "Trajectory",
    "evaluate",
    "grounded_check",
    "gt_track",
    "track_subject_by_color",
    "track_subject_from_video",
    "ColorTracker",
    "DetectionTracker",
    "Detector",
    "SubjectTracker",
    "Qwen25VLDetector",
]

# Heavy backends are loaded lazily so the deterministic core stays importable
# without imageio, torch, or transformers.
_LAZY_EXPORTS = {
    "ObjectChecker": ("object", "ObjectChecker"),
    "ObjectResult": ("object", "ObjectResult"),
    "write_result": ("object", "write_result"),
    "Qwen25VLDetector": ("detect", "Qwen25VLDetector"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[attribute])
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

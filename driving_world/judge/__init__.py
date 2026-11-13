"""Video-only, evidence-based judges."""

from .grounding import (
    GroundedConfig,
    GroundedResult,
    GtTrack,
    Trajectory,
    evaluate,
    gt_track,
    track_subject_by_color,
    track_subject_from_video,
)

__all__ = [
    "ObjectChecker",
    "ObjectResult",
    "write_result",
    "GroundedConfig",
    "GroundedResult",
    "GtTrack",
    "Trajectory",
    "evaluate",
    "gt_track",
    "track_subject_by_color",
    "track_subject_from_video",
]

# The VLM checker pulls in imageio + transformers, so it is loaded lazily to
# keep the deterministic grounded checker importable without those dependencies.
_OBJECT_EXPORTS = {
    "ObjectChecker": ("object", "ObjectChecker"),
    "ObjectResult": ("object", "ObjectResult"),
    "write_result": ("object", "write_result"),
}


def __getattr__(name: str):
    if name in _OBJECT_EXPORTS:
        module_name, attribute = _OBJECT_EXPORTS[name]
        module = __import__(f"{__name__}.{module_name}", fromlist=[attribute])
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

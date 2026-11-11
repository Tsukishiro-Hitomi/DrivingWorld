"""Contracts for the multi-check Judge: error findings and the object verdict."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from .events import CHECK_NAMES, EventSpec, _finite_number

ErrorType = Literal["object", "render", "condition"]

VerdictStatus = Literal[
    "observed",
    "partially_observed",
    "not_observed",
    "indeterminate",
]

VERDICT_STATUSES = {
    "observed",
    "partially_observed",
    "not_observed",
    "indeterminate",
}

# The object checker answers the five event checks plus an explicit
# proxy-trajectory match. Dataset annotations keep only the five event checks.
OBJECT_CHECK_NAMES = set(CHECK_NAMES) | {"matches_proxy_trajectory"}


class EventVerdict(TypedDict, total=False):
    """Evidence-backed result for one event, shared by the object checker and
    dataset annotations."""

    event_id: str
    status: VerdictStatus
    confidence: float
    observed_interval_seconds: list[float] | None
    checks: dict[str, bool | None]
    evidence_frame_indices: list[int]
    rationale: str


class Finding(TypedDict, total=False):
    """One localized finding of the three-way error taxonomy."""

    finding_id: str
    error_type: ErrorType
    severity: float
    temporal_segment: list[float]
    evidence_frame_indices: list[int]
    checks: dict[str, bool | None]
    rationale: str


class Review(TypedDict, total=False):
    """Aggregated Judge output; findings may co-occur and overlap in time."""

    schema_version: int
    model: str
    video: str
    proxy_video: str
    proxy_args: dict
    event: EventSpec
    duration_seconds: float
    findings: list[Finding]
    status: str
    raw_responses: dict[str, str]


def _validate_verdict(
    data: Any, duration_seconds: float, check_names: set[str]
) -> EventVerdict:
    """Validate a verdict payload, rejecting unsupported claims or evidence."""
    duration = _finite_number(duration_seconds, "video duration")
    if duration <= 0 or not isinstance(data, dict):
        raise ValueError("video duration must be positive and verdict must be an object")
    if not isinstance(data.get("event_id"), str) or not data["event_id"].strip():
        raise ValueError("event_id must be a non-empty string")
    if data.get("status") not in VERDICT_STATUSES:
        raise ValueError("invalid verdict status")
    confidence = _finite_number(data.get("confidence"), "confidence")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    interval = data.get("observed_interval_seconds")
    if interval is not None:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("observed_interval_seconds must be [start, end] or null")
        start = _finite_number(interval[0], "observed interval start")
        end = _finite_number(interval[1], "observed interval end")
        if not 0 <= start <= end <= duration + 0.05:
            raise ValueError("observed interval falls outside the video")
    checks = data.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(check_names):
        raise ValueError(f"checks must contain exactly: {sorted(check_names)}")
    if any(value is not None and type(value) is not bool for value in checks.values()):
        raise ValueError("every check must be true, false, or null")
    frames = data.get("evidence_frame_indices")
    if not isinstance(frames, list) or any(
        type(index) is not int or index < 0 for index in frames
    ):
        raise ValueError("evidence_frame_indices must be non-negative integer indices")
    if not isinstance(data.get("rationale"), str) or not data["rationale"].strip():
        raise ValueError("rationale must be a non-empty string")
    if data["status"] == "observed" and any(value is not True for value in checks.values()):
        raise ValueError("observed requires every check to be true")
    return data


def validate_event_verdict(data: Any, duration_seconds: float) -> EventVerdict:
    """Validate a dataset annotation (the five event checks)."""
    return _validate_verdict(data, duration_seconds, CHECK_NAMES)


def validate_object_verdict(data: Any, duration_seconds: float) -> EventVerdict:
    """Validate the object checker output (five event checks + proxy match)."""
    return _validate_verdict(data, duration_seconds, OBJECT_CHECK_NAMES)

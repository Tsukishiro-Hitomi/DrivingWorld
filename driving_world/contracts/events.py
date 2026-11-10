"""Contracts for judging whether a requested driving event is visible in a video."""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict


EventKind = Literal[
    "vehicle_lane_change",
    "pedestrian_crossing",
    "vehicle_hard_brake",
]
VerdictStatus = Literal[
    "observed",
    "partially_observed",
    "not_observed",
    "indeterminate",
]


class EventSpec(TypedDict, total=False):
    """A user-requested event, normalized before it reaches the Judge."""

    event_id: str
    kind: EventKind
    subject_type: Literal["vehicle", "pedestrian"]
    time_window_seconds: list[float]
    direction: Literal["left_to_right", "right_to_left"]
    target: str
    attributes: dict[str, Any]


class EventVerdict(TypedDict, total=False):
    """Evidence-backed result for one :class:`EventSpec`."""

    event_id: str
    status: VerdictStatus
    confidence: float
    observed_interval_seconds: list[float] | None
    checks: dict[str, bool | None]
    evidence_frame_indices: list[int]
    rationale: str


EVENT_KINDS = {
    "vehicle_lane_change": "vehicle",
    "pedestrian_crossing": "pedestrian",
    "vehicle_hard_brake": "vehicle",
}
VERDICT_STATUSES = {
    "observed",
    "partially_observed",
    "not_observed",
    "indeterminate",
}
CHECK_NAMES = {
    "subject_visible",
    "motion_observed",
    "direction_correct",
    "timing_valid",
    "event_completed",
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def validate_event_spec(data: Any) -> EventSpec:
    """Validate an event request without imposing planner-specific geometry."""
    if not isinstance(data, dict):
        raise ValueError("event spec must be an object")
    event_id, kind, subject = (
        data.get("event_id"),
        data.get("kind"),
        data.get("subject_type"),
    )
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    if kind not in EVENT_KINDS:
        raise ValueError(f"unsupported event kind: {kind!r}")
    if subject != EVENT_KINDS[kind]:
        raise ValueError(f"{kind} requires subject_type={EVENT_KINDS[kind]!r}")
    window = data.get("time_window_seconds")
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError("time_window_seconds must contain [start, end]")
    start = _finite_number(window[0], "time window start")
    end = _finite_number(window[1], "time window end")
    if not 0 <= start < end:
        raise ValueError("time window must satisfy 0 <= start < end")
    result: EventSpec = {
        "event_id": event_id,
        "kind": kind,
        "subject_type": subject,
        "time_window_seconds": [start, end],
    }
    direction = data.get("direction")
    if direction is not None:
        if direction not in {"left_to_right", "right_to_left"}:
            raise ValueError("invalid event direction")
        result["direction"] = direction
    for field in ("target",):
        if field in data:
            if not isinstance(data[field], str) or not data[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
            result[field] = data[field]
    if "attributes" in data:
        if not isinstance(data["attributes"], dict):
            raise ValueError("attributes must be an object")
        result["attributes"] = data["attributes"]
    return result


def validate_event_verdict(data: Any, duration_seconds: float) -> EventVerdict:
    """Validate a VLM response and reject unsupported claims or evidence."""
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
    if not isinstance(checks, dict) or set(checks) != CHECK_NAMES:
        raise ValueError(f"checks must contain exactly: {sorted(CHECK_NAMES)}")
    if any(value is not None and type(value) is not bool for value in checks.values()):
        raise ValueError("every check must be true, false, or null")
    frames = data.get("evidence_frame_indices")
    if not isinstance(frames, list) or any(type(index) is not int or index < 0 for index in frames):
        raise ValueError("evidence_frame_indices must be non-negative integer indices")
    if not isinstance(data.get("rationale"), str) or not data["rationale"].strip():
        raise ValueError("rationale must be a non-empty string")
    if data["status"] == "observed" and any(value is not True for value in checks.values()):
        raise ValueError("observed requires every check to be true")
    return data

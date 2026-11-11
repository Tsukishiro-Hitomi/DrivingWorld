"""Contracts for a requested driving event."""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict


EventKind = Literal[
    "vehicle_lane_change",
    "pedestrian_crossing",
    "vehicle_hard_brake",
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


EVENT_KINDS = {
    "vehicle_lane_change": "vehicle",
    "pedestrian_crossing": "pedestrian",
    "vehicle_hard_brake": "vehicle",
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

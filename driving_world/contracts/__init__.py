"""Shared, model-independent contracts for the driving-world pipeline."""

from .events import EventSpec, EventVerdict, validate_event_spec, validate_event_verdict

__all__ = [
    "EventSpec",
    "EventVerdict",
    "validate_event_spec",
    "validate_event_verdict",
]

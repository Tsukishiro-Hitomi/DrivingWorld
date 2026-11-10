"""Planner contract. The existing constrained planner will move behind this API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlannerRequest:
    user_input: str
    previous_judge_report: dict[str, Any] | None = None


def plan(request: PlannerRequest) -> dict[str, Any]:
    """Return proxy arguments; implementation is intentionally deferred to the Planner phase."""
    raise NotImplementedError("Planner is not implemented yet; use plan_scene.py during the transition")

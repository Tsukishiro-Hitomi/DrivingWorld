"""Decider contract. Policies are deferred until the Judge has been evaluated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DecisionAction = Literal["revise_h3_prompt", "regenerate_proxy", "human_review", "deliver"]


@dataclass(frozen=True)
class Decision:
    action: DecisionAction
    rationale: str


def decide(judge_report: dict) -> Decision:
    """Select a next action; implementation follows Event Judge calibration."""
    raise NotImplementedError("Decider is not implemented yet")

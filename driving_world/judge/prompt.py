"""Prompts for the event-only VLM judge."""

import json

from driving_world.contracts.events import CHECK_NAMES, EventSpec


def build_event_prompt(event: EventSpec, sample_times: list[float]) -> str:
    """Ask for observed video evidence, never a repair proposal or a next action."""
    return f"""You are a strict event judge for a driving video. Inspect only the supplied frames.
Determine whether the requested event is visibly supported. Do not infer motion between frames.
If occlusion, frame spacing, or image quality prevents a reliable conclusion, choose indeterminate.

Requested event:
{json.dumps(event, ensure_ascii=False)}

Frame timestamps in seconds: {', '.join(f'{time:.3f}' for time in sample_times)}.

Return only one JSON object with these exact keys:
- event_id: copy the requested event_id
- status: observed, partially_observed, not_observed, or indeterminate
- confidence: number from 0 to 1
- observed_interval_seconds: [start, end] or null
- checks: object with exactly {sorted(CHECK_NAMES)}; each value is true, false, or null
- evidence_frame_indices: indexes from the supplied frame order
- rationale: concise description of directly visible evidence

Use observed only when every check is true. Do not give repair instructions or select a next action."""

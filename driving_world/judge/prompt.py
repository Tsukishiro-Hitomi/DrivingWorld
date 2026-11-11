"""Prompts for the object checker."""

import json

from driving_world.contracts.events import EventSpec
from driving_world.contracts.review import OBJECT_CHECK_NAMES


def build_object_prompt(event: EventSpec, sample_times: list[float]) -> str:
    """Ask for observed evidence against the proxy, never a repair or next action."""
    return f"""You are a strict event judge for a rendered driving video.

Each supplied frame is a side-by-side pair: the LEFT half is the proxy
(ground-truth) frame and the RIGHT half is the rendered H3 frame at the same
timestamp. The proxy is the reference the rendered video must reproduce.

Determine whether the requested event is visibly present in the rendered
(RIGHT) video and matches the proxy (LEFT). Do not infer motion between
frames. If occlusion, frame spacing, or image quality prevents a reliable
conclusion, set the relevant check to null and use status "indeterminate".

Requested event:
{json.dumps(event, ensure_ascii=False)}

Frame timestamps in seconds: {', '.join(f'{time:.3f}' for time in sample_times)}.

First reason step by step about what you observe in each frame pair. Then
return exactly one JSON object (after your reasoning) with these exact keys:
- event_id: copy the requested event_id
- status: observed, partially_observed, not_observed, or indeterminate
- confidence: number from 0 to 1
- observed_interval_seconds: [start, end] or null
- checks: object with exactly {sorted(OBJECT_CHECK_NAMES)}; each value is true, false, or null
- evidence_frame_indices: indexes from the supplied frame order
- rationale: concise description of directly visible evidence

Use observed only when every check is true. Do not give repair instructions or select a next action."""

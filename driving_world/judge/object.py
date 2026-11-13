"""Object checker: verify that H3 reproduced the proxy's requested action."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from driving_world.contracts.events import EventSpec, validate_event_spec
from driving_world.contracts.review import validate_object_verdict

from .prompt import build_object_prompt
from .qwen import VisionLanguageModel
from .sampling import sample_aligned, side_by_side


@dataclass(frozen=True)
class ObjectResult:
    """Auditable result retaining sampling metadata and raw model output."""

    schema_version: int
    model: str
    video: str
    proxy_video: str
    video_duration_seconds: float
    event: EventSpec
    sampled_frame_indices: list[int]
    sampled_frame_times_seconds: list[float]
    verdict: dict | None
    status: str
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(raw: str) -> str:
    """Extract the JSON object from a fenced block or a bare braced region."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return raw[raw.index("{") : raw.rindex("}") + 1]


class ObjectChecker:
    """Verify one requested event by comparing rendered frames against the proxy."""

    def __init__(self, model: VisionLanguageModel) -> None:
        self.model = model

    def judge(self, video: Path, proxy_video: Path, event: EventSpec) -> ObjectResult:
        event = validate_event_spec(event)
        sample = sample_aligned(
            video,
            proxy_video,
            event_window_seconds=tuple(event["time_window_seconds"]),
        )
        prompt = build_object_prompt(
            event, [frame.timestamp_seconds for frame in sample.frames]
        )
        images = [
            (
                f"t={frame.timestamp_seconds:.3f}s (left: rendered, right: proxy)",
                side_by_side(frame.rendered, frame.proxy),
            )
            for frame in sample.frames
        ]
        raw = self.model.answer(prompt, images)
        try:
            payload = json.loads(_extract_json(raw))
            verdict = validate_object_verdict(payload, sample.duration_seconds)
            if verdict["event_id"] != event["event_id"]:
                raise ValueError("response event_id does not match the request")
            status = "valid"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            verdict = None
            status = f"invalid_response: {error}"
        return ObjectResult(
            schema_version=1,
            model=self.model.name,
            video=str(video),
            proxy_video=str(proxy_video),
            video_duration_seconds=sample.duration_seconds,
            event=event,
            sampled_frame_indices=[frame.index for frame in sample.frames],
            sampled_frame_times_seconds=[
                frame.timestamp_seconds for frame in sample.frames
            ],
            verdict=verdict,
            status=status,
            raw_response=raw,
        )


def write_result(destination: Path, result: ObjectResult) -> None:
    """Write an object-check report without overwriting an existing report."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

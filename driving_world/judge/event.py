"""Event-only Judge: VLM observations over a rendered driving video."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from driving_world.contracts.events import EventSpec, EventVerdict, validate_event_spec, validate_event_verdict

from .prompt import build_event_prompt
from .qwen import VisionLanguageModel
from .sampling import VideoSample, sample_video


@dataclass(frozen=True)
class JudgeResult:
    """Auditable result retaining sampling metadata and raw model output."""

    schema_version: int
    model: str
    video: str
    video_duration_seconds: float
    event: EventSpec
    sampled_frame_indices: list[int]
    sampled_frame_times_seconds: list[float]
    verdict: EventVerdict | None
    status: str
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


class EventJudge:
    """Judge one requested event from video pixels and an event specification only."""

    def __init__(self, model: VisionLanguageModel) -> None:
        self.model = model

    def judge(self, video: Path, event: EventSpec) -> JudgeResult:
        event = validate_event_spec(event)
        start, end = event["time_window_seconds"]
        sample = sample_video(video, event_window_seconds=(start, end))
        prompt = build_event_prompt(event, [frame.timestamp_seconds for frame in sample.frames])
        raw = self.model.answer(sample, prompt)
        try:
            payload = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            verdict = validate_event_verdict(payload, sample.duration_seconds)
            if verdict["event_id"] != event["event_id"]:
                raise ValueError("response event_id does not match the request")
            status = "valid"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            verdict = None
            status = f"invalid_response: {error}"
        return JudgeResult(
            schema_version=1,
            model=self.model.name,
            video=str(video),
            video_duration_seconds=sample.duration_seconds,
            event=event,
            sampled_frame_indices=[frame.index for frame in sample.frames],
            sampled_frame_times_seconds=[frame.timestamp_seconds for frame in sample.frames],
            verdict=verdict,
            status=status,
            raw_response=raw,
        )


def write_result(destination: Path, result: JudgeResult) -> None:
    """Write an event-judge report without overwriting an existing report."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

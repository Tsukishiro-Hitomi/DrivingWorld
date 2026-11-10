"""JSONL manifest format for Event Judge training and evaluation data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from driving_world.contracts.events import EventSpec, EventVerdict, validate_event_spec, validate_event_verdict


class DatasetRecord(TypedDict):
    id: str
    split: str
    video: str
    event: EventSpec
    annotation: EventVerdict


def validate_record(data: Any) -> DatasetRecord:
    """Validate one JSONL record, including its independent human annotation."""
    if not isinstance(data, dict):
        raise ValueError("dataset record must be an object")
    if not isinstance(data.get("id"), str) or not data["id"].strip():
        raise ValueError("record id must be a non-empty string")
    if data.get("split") not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if not isinstance(data.get("video"), str) or not data["video"].strip():
        raise ValueError("video must be a relative non-empty path")
    if Path(data["video"]).is_absolute() or ".." in Path(data["video"]).parts:
        raise ValueError("video must be a relative path inside the dataset")
    event = validate_event_spec(data.get("event"))
    # Dataset manifests do not store video duration. A large finite bound checks
    # annotation shape here; the evaluation runner later checks it against video.
    annotation = validate_event_verdict(data.get("annotation"), 1_000_000.0)
    if annotation["event_id"] != event["event_id"]:
        raise ValueError("annotation event_id must match event event_id")
    return {"id": data["id"], "split": data["split"], "video": data["video"], "event": event, "annotation": annotation}


def load_manifest(path: Path) -> list[DatasetRecord]:
    """Load a non-empty, duplicate-free Event Judge JSONL manifest."""
    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = validate_record(json.loads(line))
        if record["id"] in seen:
            raise ValueError(f"duplicate record id at line {line_number}: {record['id']}")
        seen.add(record["id"])
        records.append(record)
    if not records:
        raise ValueError("dataset manifest contains no records")
    return records

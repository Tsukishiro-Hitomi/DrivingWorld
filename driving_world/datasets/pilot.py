"""Build controlled, unlabeled H3 candidate sets for Event Judge calibration."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VARIANTS = ("observed", "no_motion", "wrong_direction", "wrong_timing")


@dataclass(frozen=True)
class Candidate:
    id: str
    scene_family_id: str
    split_hint: str
    event: dict[str, Any]
    proxy_relation: str
    scene: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scene_family_id": self.scene_family_id,
            "split_hint": self.split_hint,
            "event": self.event,
            "proxy_relation": self.proxy_relation,
            "annotation_status": "pending_h3_and_human_review",
            "paths": {
                "scene": f"scenes/{self.id}.json",
                "event": f"events/{self.id}.json",
                "proxy": f"proxy/{self.id}.mp4",
            },
        }


def _base_scene(context: int) -> dict[str, Any]:
    speed = 5 + context
    duration = 5.0
    return {
        "duration": duration,
        "fps": 24,
        "camera": {
            "width": 832,
            "height": 480,
            "fov_degrees": 82 + 3 * context,
            "position": [1.5, 0, 1.5],
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
        },
        "grid": {"extent": 70, "spacing": 5},
        "ego": [
            {"t": 0, "position": [0, 0, 0], "yaw": 0},
            {"t": duration, "position": [speed * duration, 0, 0], "yaw": 0},
        ],
        "actors": [],
    }


def _actor(actor_id: str, actor_type: str, size: list[float], track: list[dict]) -> dict:
    return {"id": actor_id, "type": actor_type, "size": size, "track": track}


def _lane_change(context: int, variant: str) -> Candidate:
    scene = _base_scene(context)
    distance, speed = 20 + 3 * context, 6 + context
    if variant == "wrong_direction":
        start_y, end_y, interval = 3.5, 0.0, (2.0, 4.0)
    elif variant == "wrong_timing":
        start_y, end_y, interval = -3.5, 0.0, (3.6, 4.9)
    else:
        start_y, end_y, interval = -3.5, (-3.5 if variant == "no_motion" else 0.0), (2.0, 4.0)
    start, end = interval
    scene["actors"].append(
        _actor(
            "target_vehicle",
            "vehicle",
            [4.5, 1.8, 1.6],
            [
                {"t": 0, "position": [distance, start_y, 0.8], "yaw": 0},
                {"t": start, "position": [distance + speed * start, start_y, 0.8], "yaw": 0},
                {"t": end, "position": [distance + speed * end, end_y, 0.8], "yaw": 0},
                {"t": 5, "position": [distance + speed * 5, end_y, 0.8], "yaw": 0},
            ],
        )
    )
    event = {
        "event_id": f"lane_change_c{context}_{variant}",
        "kind": "vehicle_lane_change",
        "subject_type": "vehicle",
        "time_window_seconds": [2.0, 4.0],
        "direction": "right_to_left",
        "target": "ego_lane",
    }
    return Candidate(event["event_id"], f"lane_change_context_{context}", "train" if context == 0 else "validation", event, variant, scene)


def _pedestrian_crossing(context: int, variant: str) -> Candidate:
    scene = _base_scene(context)
    if variant == "wrong_direction":
        start_y, end_y, interval = 5.0, -5.0, (2.0, 4.0)
    elif variant == "wrong_timing":
        start_y, end_y, interval = -5.0, 5.0, (3.6, 4.9)
    else:
        start_y, end_y, interval = -5.0, (-5.0 if variant == "no_motion" else 5.0), (2.0, 4.0)
    start, end = interval
    scene["actors"].append(
        _actor(
            "target_pedestrian",
            "pedestrian",
            [0.6, 0.6, 1.7],
            [
                {"t": 0, "position": [28 + 2 * context, start_y, 0.85], "yaw": 1.57},
                {"t": start, "position": [28 + 2 * context, start_y, 0.85], "yaw": 1.57},
                {"t": end, "position": [28 + 2 * context, end_y, 0.85], "yaw": 1.57},
                {"t": 5, "position": [28 + 2 * context, end_y, 0.85], "yaw": 1.57},
            ],
        )
    )
    event = {
        "event_id": f"pedestrian_crossing_c{context}_{variant}",
        "kind": "pedestrian_crossing",
        "subject_type": "pedestrian",
        "time_window_seconds": [2.0, 4.0],
        "direction": "right_to_left",
        "target": "roadway",
    }
    return Candidate(event["event_id"], f"pedestrian_crossing_context_{context}", "train" if context == 0 else "validation", event, variant, scene)


def _hard_brake(context: int, variant: str) -> Candidate:
    scene = _base_scene(context)
    x0, velocity = 18 + 2 * context, 7 + context
    if variant == "wrong_timing":
        brake_start, brake_end = 3.6, 4.9
    else:
        brake_start, brake_end = 2.0, 4.0
    final_velocity = velocity if variant in {"no_motion", "wrong_direction"} else 1.0
    x_start = x0 + velocity * brake_start
    x_end = x_start + (velocity + final_velocity) * (brake_end - brake_start) / 2
    scene["actors"].append(
        _actor(
            "target_vehicle",
            "vehicle",
            [4.5, 1.8, 1.6],
            [
                {"t": 0, "position": [x0, 0, 0.8], "yaw": 0},
                {"t": brake_start, "position": [x_start, 0, 0.8], "yaw": 0},
                {"t": brake_end, "position": [x_end, 0, 0.8], "yaw": 0},
                {"t": 5, "position": [x_end + final_velocity * (5 - brake_end), 0, 0.8], "yaw": 0},
            ],
        )
    )
    event = {
        "event_id": f"hard_brake_c{context}_{variant}",
        "kind": "vehicle_hard_brake",
        "subject_type": "vehicle",
        "time_window_seconds": [2.0, 4.0],
        "target": "lead_vehicle",
        "attributes": {"minimum_speed_drop_mps": 4.0},
    }
    return Candidate(event["event_id"], f"hard_brake_context_{context}", "train" if context == 0 else "validation", event, variant, scene)


def build_candidates(contexts: int = 2) -> list[Candidate]:
    """Return balanced controlled candidates; they are not H3 labels."""
    if contexts < 2:
        raise ValueError("at least two contexts are needed to separate train and validation")
    builders = (_lane_change, _pedestrian_crossing, _hard_brake)
    return [builder(context, variant) for context in range(contexts) for builder in builders for variant in VARIANTS]


def prepare_collection(output: Path, *, render_proxy: bool, contexts: int = 2) -> list[Candidate]:
    """Write source scenes, event specs and a candidate manifest to a new collection."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"collection must be new or empty: {output}")
    candidates = build_candidates(contexts)
    for directory in ("scenes", "events", "proxy"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        (output / "scenes" / f"{candidate.id}.json").write_text(json.dumps(candidate.scene, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "events" / f"{candidate.id}.json").write_text(json.dumps(candidate.event, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "candidates.jsonl").write_text("\n".join(json.dumps(item.to_record(), ensure_ascii=False) for item in candidates) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# Event Judge candidate collection\n\n`proxy_relation` is only a controlled design condition. Add an H3 video and human annotation before promoting a record to the benchmark manifest.\n", encoding="utf-8")
    if render_proxy:
        renderer = Path(__file__).resolve().parent.parent.parent / "proxy.py"
        for candidate in candidates:
            subprocess.run([sys.executable, str(renderer), str(output / "scenes" / f"{candidate.id}.json"), str(output / "proxy" / f"{candidate.id}.mp4")], check=True)
    return candidates

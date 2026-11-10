# Driving World

This repository develops a controllable long-tail driving-video pipeline built from a deterministic proxy renderer, H3 Ref2VA, and a three-part Agent.

```text
user input + prior Judge report → Planner → proxy arguments → proxy video → H3 video → Judge → Decider
```

- `driving_world/planner/` accepts user intent and prior feedback, then will produce proxy arguments. The current constrained implementation remains available through `plan_scene.py` during migration.
- `driving_world/judge/` evaluates whether one requested event is visibly present in an H3 video. This is the active implementation.
- `driving_world/decider/` will choose prompt revision, proxy regeneration, human review, or delivery. It is an explicit placeholder until the Judge is calibrated.
- `driving_world/contracts/` contains role-neutral JSON contracts.
- `driving_world/datasets/` validates dataset manifests.
- `datasets/event_judge/` contains videos, immutable human annotations, and per-model reports.
- `examples/` contains source scene examples; `experiments/` retains historical runs and reports.

## Event Judge

The first Judge is deliberately narrow: given only an H3 video and an `event.json`, it determines whether the requested event occurred. It does not receive the proxy video and does not decide how to repair a failure.

```json
{
  "event_id": "cut_in_01",
  "kind": "vehicle_lane_change",
  "subject_type": "vehicle",
  "time_window_seconds": [2.0, 5.0],
  "direction": "right_to_left",
  "target": "ego_lane"
}
```

Supported event kinds are `vehicle_lane_change`, `pedestrian_crossing`, and `vehicle_hard_brake`. The Judge samples global context plus denser frames within the requested time window and writes a validated report with an event status, five atomic checks, evidence frame indices, observed time interval, confidence, and raw VLM response.

```bash
/root/miniconda3/envs/h3-ref2va/bin/python judge_event.py \
  --video /path/to/generated.mp4 \
  --event /path/to/event.json \
  --output /path/to/report.json
```

`observed` requires all checks to be true. When frame evidence cannot establish an event, the valid status is `indeterminate`.

## Dataset layout

Each line in `datasets/event_judge/manifest.jsonl` pairs a generated H3 video with an event specification and a human annotation of that generated video. Proxy trajectories are candidate targets, never final labels. Store model reports separately in `datasets/event_judge/reports/<model>/`.

The first collection should include successful events, partial events, hard negatives, and indeterminate examples across different scenes and camera configurations. This makes false acceptance measurable before the Judge becomes part of the retry loop.

## Existing proxy workflow

The deterministic renderer is dataset-independent. It uses world-coordinate actor tracks and a camera mounted to the ego vehicle.

```bash
/root/miniconda3/envs/h3-ref2va/bin/python proxy.py \
  examples/example_scene.json /path/to/proxy.mp4
```

The legacy constrained natural-language planner remains available while the dedicated `planner/` package is introduced:

```bash
/root/miniconda3/envs/h3-ref2va/bin/python plan_scene.py \
  --request '右侧车辆在2秒后切入自车道' \
  --output /path/to/new-plan
```

## Verification

```bash
/root/miniconda3/envs/h3-ref2va/bin/python -m unittest discover -s tests -v
```

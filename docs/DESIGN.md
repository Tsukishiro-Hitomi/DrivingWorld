# Design

## Goal

Optimize the middle agent of the controllable long-tail driving-video pipeline.
Through **multi-segment generation**, the agent must (1) locate and repair
defective frames to raise quality, and (2) apply scene-level control such as
changing weather, action, or appearance.

The **core innovation** is combination optimization across the generated
segments for different inputs. The **trainable object** is the skill fed to the
VLM.

## Pipeline

```text
user input (+ prior review) -> Planner -> proxy args -> proxy video -> H3 video -> Judge -> Decider -> ...
```

## Agents

### Planner

- Input: user input (prompt or image) and the prior review.
- Output: proxy args, then a proxy video.
- Responsibility: guarantee proxy-video quality. If the requested event (for
  example a lane change or pedestrian intrusion) does not occur, regenerate. On
  delivery, hand the proxy args to the Judge as ground truth.

Proxy args are grouped into four kinds:

- `camera` — fov, intrinsics/extrinsics, resolution, ...
- `ego` — ego-vehicle trajectory.
- `objects` — per-object class, trajectory, and 3D box.
- `conditions` — weather, lane-line layout, road-sign text, ...

Only the first three (`camera`, `ego`, `objects`) participate in proxy
rendering; `conditions` are semantic constraints checked against the rendered
output.

### Judge

- Input: proxy args, proxy video, and the rendered (H3) video.
- Output: a structured review containing one or more error findings.

Errors are classified into three kinds. They are a preliminary direction for
now, intentionally coarse, and may **co-occur**:

- **Object Error** — H3 does not follow the proxy and omits the required action
  (lane change, pedestrian intrusion, ...).
- **Render Error** — the action appears but its appearance is implausible,
  hurting visual quality.
- **Condition Error** — weather/scene mismatch with the request, disordered lane
  lines, or garbled road-sign text.

### Decider

- Input: the Judge's review.
- Output: one bounded action — change the H3 prompt, regenerate the proxy via the
  Planner, hand off for human review, or finalize.

Combination optimization across segments is, for now, driven by the Decider's
closed loop rather than a separate module.

## Current focus

The Judge is developed first; combination optimization and the Decider loop come
later.

## Codebase status

| Component | Current (`zxr`) | Target |
| --- | --- | --- |
| Planner | `driving_world/planner/service.py` placeholder; legacy `plan_scene.py`, `intent.py`, `scene_compiler.py` | add `conditions`; self-check proxy quality; deliver proxy args as ground truth |
| Proxy args | `driving_world/proxy.py` scene JSON has camera/ego/actors | add `conditions` |
| Judge | `driving_world/judge/event.py` judges event presence only | proxy-vs-H3 comparator with Object/Render/Condition errors |
| Decider | `driving_world/decider/policy.py` placeholder | four-action loop |
| Multi-segment | `driving_world/mvp.py` segment + concat prototype | locate defects + combination optimization |
| Trainable | — | the skill fed to the VLM |

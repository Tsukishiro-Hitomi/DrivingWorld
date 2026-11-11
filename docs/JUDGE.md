# Judge — Design & Implementation Plan

## Decisions (locked)

- **Decomposed judge**: one orchestrator drives three narrow, independently
  calibrated checkers instead of a single VLM pass.
- **Migration**: the existing `EventVerdict` (`observed`/... plus five atomic
  checks) is folded into the Object checker. There is no second, parallel event
  contract.
- **Model**: start with the wired `Qwen2.5-VL-7B` baseline; benchmark larger
  VLMs only after the baseline is calibrated on a controlled set.

## Rationale (research)

- Spotlight ([arXiv:2511.18102](https://arxiv.org/abs/2511.18102)): VLMs lag
  humans ~2x at fine-grained error localization; multi-agent decomposition
  nearly doubles Qwen3-VL and matches Gemini 2.5 Pro. -> **decompose**.
- VehDyn ([arXiv:2609.33264](https://arxiv.org/abs/2609.33264)): visual-quality
  metrics correlate weakly with vehicle-dynamics fidelity. -> **separate Object
  (action fidelity) from Render (appearance)**.
- WorldJen ([arXiv:2605.03475](https://arxiv.org/abs/2605.03475)): binary VQA
  has yes-bias. -> **Likert/rubric, allow `null`, never force boolean checks**.
- Motion-o ([arXiv:2603.18856](https://export.arxiv.org/pdf/2603.18856)): 7B
  VLMs are weak at temporal grounding; explicit motion cues help. -> **aligned
  frames + timestamps**, optionally frame-difference/flow.

## Architecture

```text
judge.Judge (orchestrator)
  ├── ObjectChecker    proxy (GT) vs H3, aligned frames -> action fidelity
  ├── RenderChecker    H3 frames                       -> appearance quality
  └── ConditionChecker conditions (text) vs H3         -> scene fidelity
        └── aggregate -> Review
```

### ObjectChecker (Object Error)

- Input: event spec + aligned proxy/H3 frame pairs, dense in the event window,
  presented side-by-side.
- Checks: `subject_visible`, `motion_observed`, `direction_correct`,
  `timing_valid`, `event_completed`, `matches_proxy_trajectory`.

### RenderChecker (Render Error)

- Input: H3 frames (proxy optional as a geometry reference).
- Checks: `geometry_plausible`, `identity_consistent`, `motion_smooth`,
  `artifact_free`.
- Render errors are short and local; sample uniformly, then refine locally.

### ConditionChecker (Condition Error)

- Input: the `conditions` dict (weather, lane layout, road-sign text) + global
  H3 frames.
- Checks: `weather_match`, `lane_layout_sane`, `road_sign_legible`.

## Contracts

`driving_world/contracts/review.py` (new):

```python
ErrorType = Literal["object", "render", "condition"]

class Finding(TypedDict, total=False):
    finding_id: str
    error_type: ErrorType
    severity: float                 # 0..1
    temporal_segment: list[float]   # [start, end] seconds
    evidence_frame_indices: list[int]
    checks: dict[str, bool | None]  # null = cannot determine
    rationale: str

class Review(TypedDict, total=False):
    schema_version: int
    model: str
    video: str                 # rendered H3 video
    proxy_video: str
    proxy_args: dict
    event: EventSpec
    duration_seconds: float
    findings: list[Finding]    # may co-occur and overlap in time
    status: str                # valid | invalid_response
    raw_responses: dict[str, str]
```

`contracts/events.py` keeps `EventSpec` and `validate_event_spec`.
`EventVerdict` / `validate_event_verdict` move here as the object-finding shape.

## Inputs & sampling

| Checker | Inputs | Sampling |
| --- | --- | --- |
| Object | event spec + proxy/H3 | dense aligned pairs in event window + a few context frames |
| Render | H3 (+ proxy reference) | uniform over full video, then refine around candidates |
| Condition | conditions dict + H3 | global frames (weather/lane are scene-level) |

## Inference strategy

- CoT then JSON extraction (reason first, parse after), not direct JSON.
- Boolean checks allow `null`; add `severity` (0..1) and confidence.
- Pluggable `VisionLanguageModel` protocol (already present); add a shared
  aligned-sampling helper.

## Evaluation & calibration

- **Object**: the pilot set
  (`experiments/event_judge/pilot_001`, 24 controlled candidates over
  `observed` / `no_motion` / `wrong_direction` / `wrong_timing`) gives a labeled
  benchmark for false-accept / false-reject and per-variant accuracy.
- **Render / Condition**: build synthetic negatives (add `conditions` fields,
  generate mismatched H3, annotate by hand).
- **Metrics** (after Spotlight): finding-level IoU precision/recall, error-type
  accuracy, reason similarity (LLM score), and human inter-annotator agreement.

## Target layout

```text
driving_world/
  contracts/events.py     keep EventSpec + validate_event_spec
  contracts/review.py     NEW ErrorType / Finding / Review
  judge/qwen.py           keep VisionLanguageModel protocol + Qwen25VL
  judge/sampling.py       extend: aligned proxy/H3 sampling
  judge/prompt.py         split into per-checker prompts
  judge/object.py         from event.py (ObjectChecker)
  judge/render.py         NEW RenderChecker
  judge/condition.py      NEW ConditionChecker
  judge/judge.py          orchestrator -> Review
```

## Implementation plan

### Phase 0 — Contracts

- Add `contracts/review.py`; migrate `EventVerdict` into the object-finding
  shape; keep `validate_event_spec`.
- Update `tests/test_event_contracts.py` (or add `tests/test_review.py`).
- Acceptance: tests green; `judge_event.py` still runs as a thin wrapper over
  the Object checker.

### Phase 1 — ObjectChecker

- Add aligned proxy/H3 sampling; migrate `judge/event.py` into
  `judge/object.py` with the six checks above.
- CoT-then-JSON prompt; calibrate on the pilot set.
- Acceptance: on the pilot variants, Object checker beats the current baseline
  (which returns `not_observed` on the `observed` variant).

### Phase 2 — RenderChecker

- Add `judge/render.py` with the four appearance checks; uniform + local
  sampling.
- Acceptance: detects synthetic render errors with reasonable IoU.

### Phase 3 — ConditionChecker

- Add `judge/condition.py`; consume the `conditions` dict.
- Acceptance: detects weather/lane/road-sign mismatches.

### Phase 4 — Orchestrator

- Add `judge/judge.py`; aggregate findings into `Review`; expose the shape the
  Decider consumes.
- Acceptance: one CLI/entry point produces a validated `Review`.

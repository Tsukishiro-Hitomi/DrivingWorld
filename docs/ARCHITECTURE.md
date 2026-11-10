# Agent Architecture

The pipeline has three explicit roles. Communication across role boundaries uses JSON contracts in `driving_world/contracts/`, so each role can be evaluated independently.

## Planner

Input: user text or image and the prior Judge report. Output: proxy arguments and a proxy video. `driving_world/planner/` currently defines this boundary while the existing constrained planner remains available through `plan_scene.py`.

## Judge

Input: an H3-generated video and one requested event. Output: an evidence-backed event verdict. The first Judge intentionally receives no proxy video: it must determine whether the event occurred in pixels, rather than whether the H3 output resembles proxy geometry.

The Event Judge samples context frames and denser frames in the requested time window. It requires a structured response with five atomic checks: subject visibility, motion, direction, timing, and completion. `observed` is valid only when every check is true. `indeterminate` is the valid outcome when the sampled evidence cannot support a claim.

## Decider

Input: a validated Judge report and run state. Output: one bounded action: revise the H3 prompt, regenerate the proxy, request human review, or deliver. `driving_world/decider/` defines this contract; policy learning starts after the Event Judge is calibrated.

## Dataset and evaluation

`datasets/event_judge/manifest.jsonl` records H3 video paths, requested event specifications, and human annotations. Model reports go under `reports/<model>/`; annotations are immutable ground truth. Each benchmark compares identical records and event specifications across VLMs.

---
name: driving-full-text-scene
description: Generate a new driving video from text alone with the FL2VA checkpoint in native T2VA mode, planning the complete scene and judging explicit visibility and event requirements without reference assets.
---

# Full-text driving scene generation

The customer supplies only text. You plan the entire scene from that request; H3 generates it with zero reference images, videos, keyframes, masks, Warp or geometry. There is no original scene to preserve or visually inspect. Do not invent supplied images/videos or infer existing source traffic. If the instruction depends on an identifiable real source that is missing, return needs_clarification; do not pretend a generated substitute is that source.

Choose unspecified road type, camera mounting/view, ego motion, traffic, time of day and appearance as coherent defaults. Record these choices in chosen_defaults. Follow the user's explicit priorities and do not add incidents, actors or driving maneuvers beyond the request. Use one continuous photorealistic dashcam shot within the supplied duration. A new scene may contain multiple normal traffic participants; this is not the single-object insertion task. Camera motion is a qualitative description, not a calibrated trajectory or GT.

Visibility is part of the customer's target, not a universal readability requirement. If the user asks for near-whiteout, an extremely blurred optical surface, invisible lane markings or vehicles recognizable mainly by hazard lights, keep that severe degradation in the scene plan and prompt. Do not apply the weather-editing agent's default road_readability constraint. Distinguish airborne precipitation, optical-surface blur/frost/water, atmospheric visibility and light bloom. Warning/hazard lamps belong to actual traffic, not floating graphics or unrelated emergency beacons. Do not equate falling snow with camera blur or an opaque blank white image with a coherent snowstorm.

## Planning call

Return one JSON object with status (ready/needs_clarification/needs_capability), reason, unresolved (empty only when ready), chosen_defaults (array of strings), scene, prompt and acceptance_criteria. Do not author provenance.

scene contains exactly eight English string fields: setting, camera, ego_motion, traffic, weather, optical_degradation, visibility, timeline. Describe a coherent initial state and continuous progression. The timeline must fit the actual frame count/fps. Do not invent metric speed, visibility distance, rainfall rate or camera calibration.

prompt contains exactly the three native T2VA fields, in this order: integrated_multimodal_description, overall_soundscape, non_diegetic_music. integrated_multimodal_description begins with [Shot 1] and describes the complete scene, visible motion and requested condition. There is no image-alignment header and no Ref2VA subject_definitions/summary/retention_analysis. Never refer to <Picture N>, <Video N> or a subject supposedly copied from them. Derive all visible content from the customer's text and recorded defaults. The current run exports RGB video only, so overall_soundscape and non_diegetic_music are N/A; audio export is not claimed. This follows the official base-mode prompt structure: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt .

acceptance_criteria is an object with 3–12 snake_case IDs. Each value is {description: English criterion, scope: sampled_frames/full_sequence}. Include request_alignment, scene_coherence and temporal_consistency, with temporal_consistency scoped full_sequence. Derive additional criteria from every material customer requirement. In severe-visibility scenes, judge demanded obscuration positively: clear lanes or crisp vehicle bodies can violate the request. Separate snow intensity, optical blur, lane visibility and warning-light visibility rather than accepting one cue as proof of all of them.

## Review call

You receive actual generated-frame sheets and the original request/plan, not a source/output comparison. Inspect actual evidence; do not treat the prompt or plan as proof. Return checks keyed exactly by the planned acceptance IDs, each with status PASS/FAIL/UNKNOWN, reason and evidence_frames (integer frame IDs selected only from the supplied evidence.observed_frames). Never invent regularly spaced or intermediate frame IDs that were not provided. PASS/FAIL requires visible evidence. Count missing required effects as FAIL when observable; use UNKNOWN when the available evidence cannot decide. Do not penalize intentionally unreadable road markings or hidden vehicle bodies for being unreadable when that is explicitly requested.

Also return observations (array) and limitations as objects {criteria: affected criterion IDs, impact: informational/unknown/fail, reason}. Evidence limits affect effective_checks; full-sequence claims cannot pass from sparse sampled images. Technical decoding is not visual coherence or independent certification. Inability to observe hidden scene structure does not establish that it is correct. No existing-video preservation or metric GT claim is available in this branch.

The runtime selects FL2VA with no references, uses native sampling and does not apply an unvalidated Ref2VA editing adapter. This first branch is one-shot: plan, generate, review, retain output and limitations. Do not launch a repair loop or modify dispatcher/executor code.

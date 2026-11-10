---
name: ood-driving-prompt
description: Write MiniMax H3 driving prompts and plan targeted vehicle replacement from natural-language requests, including target selection, tracking, first-frame editing and generation conditions. Use for continuous dashcam editing with an explicit reference inventory.
---

# OOD driving prompts for H3

Use the official [base-mode guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt) for T2VA and keyframe modes, and the [Ref2VA guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt) for reference generation or video editing. The CrossView caller supplies the LoRA trigger when the baseline used it, and sends the text to the existing prompt embedder.

## Route the request

- When the customer supplies text and no source video or image, use `../../agent_loop/full_text/SKILL.md` and `python -m agent_loop.dispatcher run --agent full-text`. This branch generates a complete scene with the FL2VA checkpoint in zero-reference T2VA mode and uses the three base-mode fields, not the six Ref2VA fields. No existing-video preservation, SAM, Warp or geometry is assumed.

- For a newly added object rather than replacement of an existing vehicle, use `../../agent_loop/insert/SKILL.md` and `python -m agent_loop.insert run`. This route currently accepts video plus text only; it does not need SAM, a reference picture, geometry or a hand-built Warp. Open-vocabulary identity, event timing and motion are planned by the insertion agent. The outer planner may compose it with registered replacement/weather operations.

- For automatic selection between vehicle replacement and global weather/lighting editing, use `python -m agent_loop.dispatcher run`. Weather-specific execution and review are defined in `../../agent_loop/weather/SKILL.md`; this route does not require SAM or a first-frame image editor.
- When the caller explicitly requests prompt rewriting, return the English prompt body alone using the format below. The existing `ood_driving_experiment.py` caller requests this mode and expects six Ref2VA sections.
- When the user requests replacement of an existing vehicle, route to the agent in `../../agent_loop/replace/SKILL.md` for the full vehicle replacement workflow. Its entry point is `python -m agent_loop.replace run`. Resolve the target and required editing operations before preparing the final H3 prompt.
- When asked for a machine-readable replacement plan, use the JSON contract in `../../agent_loop/replace/SKILL.md`. A plan requires an executor to invoke the available tools. Check actual callable capabilities and current files before choosing operations.

## Choose the format from the actual inputs

- T2VA starts with `integrated_multimodal_description`. For actual I2VA, FL2VA or L2VA keyframe conditioning, put the guide's exact frame-alignment line first, followed by a blank line. Then use `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` in that order. A scene image supplied as an ordinary Ref2VA reference does not activate I2VA.
- For Ref2VA, emit these six headings at column one, each followed by a colon: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`. Define the real media labels in the first section and keep them consistent. Reference-derived content uses `<Subject N>`; cite the actual `<Picture N>` or `<Video N>` as its source. An ordinary scene image used solely for appearance belongs inside its subject definition. Introduce newly generated subjects by name in the shot. A separate aligned CrossView Warp is a control input and has no ordinary media label. Assign audio labels only to actual audio references.
- Start the Ref2VA summary with the actual bracketed task type. For video editing, follow it with “The target video is an edited version of <Video 1>.” Describe what changes and what carries through. In retention analysis, use one line per separately defined item, with the official relation markers such as `fully_preserved` or `partially_preserved`. In detailed description, write a short style opening followed by `[Shot 1]` and describe the motion in playback order.

## Write visible motion

- Keep one uninterrupted forward-facing, ego-mounted shot. Stay within the supplied duration. In-shot times describe event phases; the opening shot has no cut timestamp. Preserve reference lane geometry, perspective, daylight and vehicle identities except for the requested edit. Describe the first clear appearance, continuous motion and end state.
- Ego lane change: describe forward travel, lateral translation across the lane line and settling in the destination lane. Use the ego viewpoint as the moving subject. Preserve the lead vehicle and the gap specified in the brief. Camera yaw and focal length follow the provided camera constraints.
- Front-car cut-in: name the existing car by its initial lane, relative position and appearance. Keep the ego camera on its supplied path. Describe the target crossing the lane divider in the specified direction while moving forward, then settling ahead as one continuous vehicle.
- Wheelchair intrusion: keep the lane clear initially. Describe a single seated older person entering from the specified curb, wheels rolling on the road, continuous scale and visibility, and the ego vehicle yielding. Match the requested clearance and crossing end state.
- Keep direction relative to the forward view and use the timing in the brief. Describe effects in image space without inventing telemetry, precise distances or frame constraints. Use concrete physical cues and omit extra traffic maneuvers, inserts, graphics and camera cuts. Use `N/A` for a silent export's soundscape and when there is no non-diegetic music. Keep any requested dialogue or visible text in its original language.

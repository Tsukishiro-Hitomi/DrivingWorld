---
name: vehicle-replacement
description: Replace a selected existing vehicle in a driving reference video, starting at frame zero, with a category specified in natural language. Ground the target, track its pixels, create the replacement first frame, prepare consistent video conditions and verify the output.
---

# Vehicle replacement agent

The dispatcher accepts a source video and user request. Its default path obtains candidate vehicle boxes from SAM3.1 detection on the first frame. Select the existing instance using that frame and the detected candidate IDs. The image-guided branch then tracks it through the source video. Scene geometry and pre-existing masks are optional legacy inputs. Keep the original movement unless a separate motion edit is requested. Return an unresolved target when selection is ambiguous.

Use one entry point: `python -m agent_loop.replace run --source VIDEO --instruction REQUEST --out OUTPUT [--replacement-image PHOTO]`. Select the generation branch from actual replacement-image presence; a source-video frame is not a replacement photo. Keep the original movement, scene perspective and unrelated content consistent.

With no replacement photo, the default is `text-only`: send the original video and replacement text directly to native Ref2VA, with zero image references, zero aligned Warp inputs and no first-frame anchor. Skip tracking propagation, first-frame image editing and appearance-guide construction. Infer the requested vehicle category and explicit attributes from the request. Generate a plausible `replacement_description` in English for missing appearance details, preserving the requested category. No vehicle class, paint color, lane position or target appearance is preset by the executor. Inspect frame-zero replacement and continuity in the result.

The routing agent authors `target_description`, `replacement_category`, `replacement_description`, `first_frame_edit_prompt` and `motion_description`. Ground the original target with its exact candidate ID and first-frame box, rather than inferring it from a previous pilot. Describe replacement appearance separately from the original target's appearance: its original paint color does not constrain the new vehicle unless requested. Do not add unrequested actions such as emergency flashing lights or sirens. Preserve actual input weather, lighting and camera viewpoint, including changes made by an earlier agent. The executor fills the six-section H3 structure from these fields; it never substitutes a fixed truck description or assumes daylight.

An outer planner may pass `task_parameters` with `target_description`, `target_selector` and `replacement_category`. The selector is a natural-language visual cue, not a pre-grounded object ID or an enum limited to one scene. Resolve it against the actual child input. Copy the supplied category exactly and generate appearance details consistent with that category and task instruction. Report ambiguous selection instead of silently changing the target or category. Open-vocabulary prompt input provides no guarantee of visual success; inspect the generated output. The automatic text-only path remains vehicle replacement, not arbitrary object insertion.

With `--replacement-image PHOTO`, select `appearance-guided`. Supply the scene frame and replacement photo to Qwen with explicit roles. Use the photo's body, color, cab/cargo layout and distinguishing details in the edit prompt and as the actual H3 appearance reference. An unreadable photo is an error; do not silently fall back to text. The image-guided executor follows these stages.

1. Ground the target in the actual first frame and bind its source object ID.
2. Reuse a matching SAM mask sequence or segment and track the selected instance. Verify source hash, frame count and target identity before reuse.
3. Read `image_edit_request.json`, including its ordered `input_images`. Send both the scene edit target and the replacement photo to the host image editor. The Python executor returns `awaiting_image_edit` at this boundary; it does not call the host image tool itself. Resume the same entry point with the same `--replacement-image` and `--edited-first-frame EDITED_SCENE`. The latter is the edited full scene, not the raw customer photo. Inspect the new object and surrounding scene before video generation.
4. Segment the replacement in the edited first frame. Expand the editable region to include both silhouettes and affected shadows. Prepare a video condition with the old vehicle removed and the replacement following the input target's image trajectory. An image-plane appearance warp is a visual guide with limited viewpoint coverage.
5. Bind the edited first frame to the H3 first-frame anchor. Feed the modified video conditions and appearance reference with a prompt describing the existing replacement appearance and motion. Use the official Ref2VA six-section format in the final H3 prompt.
6. Verify full decoding, frame-zero replacement, target localization, continuity and background preservation. Include the replacement photo in sampled-frame review and require appearance matching. Record measurements separately from visual judgments and disclose artifacts observed in the output.

Resume existing jobs with their recorded branch and unchanged input hashes. `--conditioning` remains a compatibility override for prior experiments. Preserve existing outputs instead of switching their branch in place.

Current pilot constraints are a fully visible target with a successfully tracked mask sequence, frame-zero replacement, a single shot and modest viewpoint changes. The video adapter currently accepts 124 frames at 24 FPS and 832 by 480 pixels. Image-plane warping preserves a 2D trajectory and provides no metric size or 3D reconstruction claim. Occlusion-heavy inputs require an appropriate depth-aware editing route. Stop on missing or invalid tracking and input artifacts. Bound video-generation retries to a diagnosed correction.

## Implementation

`dispatch.py` handles planning, branch selection, resume and visual review. `executor.py` handles video inputs, tracking, conditions, H3 generation and verification. `runtime.py` contains the JSON/hash helpers and SAM adapter. The shared H3 engine remains `crossview_v2v.py` in the project root, along with its installed model runtime and weights.

## Replacement plan contract

Use this contract when the caller requests a JSON plan. Fill values from the request and verified input metadata. Use `null` for unresolved paths, target IDs or timing. Record category-level visual defaults when chosen. Set `status` to `needs_target_selection` when the target is ambiguous, `needs_implementation` when required stages lack callable implementations, or `ready` only when the plan is executable with the available capabilities.

```json
{
  "task_type": "vehicle_replacement",
  "status": "needs_implementation",
  "source_video": null,
  "target": {
    "description": "front-left white car",
    "object_id": null,
    "bbox_xyxy": null,
    "mask_sequence": null
  },
  "replacement": {
    "category": "truck",
    "appearance_source": "text_generated",
    "requested_attributes": {},
    "chosen_attributes": {},
    "reference_image": null
  },
  "start_frame": 0,
  "preserve": ["target_trajectory", "ego_camera_motion", "unrelated_scene_content"],
  "operations": [
    "resolve_target",
    "generate_from_source_video_and_text",
    "validate_output"
  ],
  "missing_capabilities": [],
  "h3_prompt": null
}
```

The example shows a text-only request. For user-supplied replacement imagery, set `appearance_source` to `user_reference` and use operations `resolve_target`, `segment_and_track`, `edit_first_frame`, `prepare_video_edit_conditions`, `generate_with_edited_first_frame_anchor`, `validate_output`. Populate `missing_capabilities` after checking the runtime. Produce `h3_prompt` when the reference inventory and input bindings have been resolved.

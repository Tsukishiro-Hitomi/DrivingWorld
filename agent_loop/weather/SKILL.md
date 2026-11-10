---
name: driving-weather-editing
description: Plan, execute and review coherent driving-video weather and lighting edits, including scene atmosphere, coupled lighting and RGB camera optical contamination, with evidence-driven plan and prompt repair.
---

# Driving weather and lighting editing

Use `python -m agent_loop.weather run --source VIDEO --instruction REQUEST --out OUTPUT`. The global entry `python -m agent_loop.dispatcher run` selects this agent before object detection. `--profile` optionally pins a preset; `--plan-only` prepares a plan without generation. The current adapter accepts 124 frames at 24 FPS, 832x480. Reject unsupported shape/timing instead of silently retiming the input.

## Resolve the complete target environment

Weather editing changes a coupled environment, not just falling particles. Separate these concepts:

- `time_of_day` describes day, night or twilight. It does not describe cloud cover. Retaining daytime never means retaining a sunny sky, direct sunlight or hard sun shadows.
- `sky_condition` describes the target sky or its atmospheric obscuration.
- `illumination` describes the target light distribution and scattering.
- `visibility` describes atmospheric or glare-related visibility.
- `road_surface` and `snow_accumulation` describe target surface state separately from precipitation.

Inspect the actual source-frame sheet. Record `source_environment` with exactly `time_of_day` (`day`, `night`, `twilight`, `unknown`) and `road_surface` (`dry`, `wet`, `snow_covered`, `unknown`). Default to the observed source day/night unless the profile or customer changes it, then resolve that policy to a concrete target value. New target plans must not contain `preserve`, `unknown` or unresolved policies in `edits`. When an essential source attribute cannot be observed and the customer did not specify its target, report it in `unresolved`.

Use the following environment couplings as the defaults for the six supported severe-condition profiles. These are editing presets, not universal claims about every real weather event. A request outside the representable combinations must be marked unresolved rather than silently weakened.

| Profile | Visible effects | sky_condition | illumination | visibility | Target surface / day-night policy |
| --- | --- | --- | --- | --- | --- |
| heavy_rain | Dense rainfall, rain streaks, tire spray and reflections | storm_overcast | storm_diffuse | poor | wet; resolve source day/night unless explicitly changed |
| dense_fog | Depth-dependent fog obscuration and contrast loss | fog_obscured | fog_scattered | poor | resolve source surface and day/night |
| heavy_snow | Dense moving flakes, diffuse illumination | snow_overcast | snow_diffuse | reduced | resolve source surface and day/night; accumulation separately requested |
| strong_backlight | Strong frontal sunlight, glare and shadowed object surfaces | bright_backlit | strong_frontal_sunlight | glare_limited | resolve source surface; day |
| rainy_night | Rain, dark ambient light and existing-light reflections | night_overcast | wet_night_lighting | poor | wet; night |
| sandstorm | Windblown fine dust, warm muted light and distance-dependent obscuration | dust_obscured | dust_filtered | poor | resolve source surface and day/night |

A daytime heavy-rain target therefore means `time_of_day: day`, `sky_condition: storm_overcast`, `illumination: storm_diffuse`, `visibility: poor`, `road_surface: wet`. Source blue sky, direct sunshine and sharp sun shadows are editable appearance, not preservation constraints. At night, diffuse/scattered illumination means dark ambient light plus scattering from existing lights, not artificial conversion to daylight. Preserve road topology and object/camera movement while letting weather change colors, apparent brightness, shadows, reflections and distant visibility.

## Planner contract

Return `profile`, `source_environment`, `temporal_mode`, `transition_start_s`, `transition_end_s`, `edits`, `scene_description`, `weather_description`, `preservation_description`, `sensor_effects`, `execution_emphasis`, and `unresolved`. The descriptions are English explanations; `unresolved` is empty only when executable. The caller assigns `prompt_version: 5` after validating the result. `profiles.planning_profiles()` provides the machine-readable policies. Historical v1/v2/v3 plans remain immutable for resume; new requests use schema v5. Artifact names such as video v3 are experiment versions, separate from this schema number.

`edits` contains exactly these six fields:

- `time_of_day`: `day`, `night` or `twilight`.
- `road_surface`: `dry`, `wet` or `snow_covered`.
- `snow_accumulation`: `none`, `light` or `heavy`; new accumulation only belongs to heavy_snow.
- `sky_condition`, `illumination`, `visibility`: the selected profile's coupled values above.

Heavy rain/rainy night require wet roads. Strong frontal sunlight requires daytime; rainy_night requires nighttime. Falling snow does not automatically require thick ground cover; choose accumulation from the request. Distinguish existing ground cover from new accumulation: fog, backlight or dust may retain a source snow-covered road without requesting fresh accumulation. Do not automatically add lightning, large flying debris, wipers, new light poles, extra traffic or changed driving behavior. Represent unsupported additional requests in `unresolved`; a combined vehicle-and-weather request belongs to the global composition route.

Default `temporal_mode` to `constant_from_frame_zero` with null transition times. Use `gradual` only when requested, with a finite interval inside the actual source duration. Avoid invented telemetry, exact fog distances or rain rates.

## Camera observation and lens contamination

A weather edit affects both the world and the RGB camera observation. Separate distant atmospheric rain/fog/dust from foreground optical-surface effects. This route models the front RGB camera and its protective glass; it does not claim LiDAR/radar corruption, calibrated sensor noise or measured optical accuracy. Condensation or deposits are requested synthetic scenarios, not an assertion that every instance of fog/snow causes the same lens state.

| Profile | surface_condition | optical_effect | temporal_behavior |
| --- | --- | --- | --- |
| heavy_rain | water_droplets_and_film | local_refraction | coalesce_and_slide |
| dense_fog | condensation_haze | soft_veiling | slow_condensation |
| heavy_snow | wet_flakes_and_slush | localized_soft_occlusion | adhere_and_melt |
| strong_backlight | clean | flare_bloom | light_dependent |
| rainy_night | water_droplets_and_film | refractive_highlight_bloom | coalesce_and_slide |
| sandstorm | dust_deposit | dust_smearing | gradual_deposition |

`sensor_effects` contains exactly `sensor: rgb_camera`, `optical_surface: protective_glass`, the three profile-specific fields above, `coverage` (`light`, `moderate`, `heavy`), `strength` (`mild`, `moderate`, `strong`), and `road_readability: retained`. Default coverage and strength to moderate. In schema v5 heavy_rain/rainy_night also require `water_activity` (`high`, `very_high`), default high; this controls ongoing replenishment and runoff independently of coverage. Backlighting affects optical response and exposure/glare; it does not require invented dirt. The compiler renders only the selected camera effect, without copying irrelevant weather presets into H3.

For rain, show irregular translucent droplets and an uneven film in the extreme foreground, with localized refraction, soft edges and visible distortion of the moving scene behind them. Attached droplets remain camera-attached while their surfaces evolve. In severe rain, repeated fresh impacts, rapid merging, shedding and runoff must visibly renew the water pattern; a few slow persistent drops are insufficient evidence of high water load. Airborne rain streaks and tire spray move in the scene. Lens drops alone cannot establish outdoor rainfall intensity. A camera-surface effect should not track a particular car or road location. Keep the central road and nearby traffic recognizable through clear gaps. Use local optical degradation rather than a uniformly opaque screen, decorative circles or fixed graphic overlays. These cues follow the distinction between atmospheric rain and adherent refracting droplets in [Hao et al., ICCVW 2019](https://openaccess.thecvf.com/content_ICCVW_2019/html/PBDL/Hao_Learning_From_Synthetic_Photorealistic_Raindrop_for_Single_Image_Raindrop_Removal_ICCVW_2019_paper.html); the current executor uses H3 generation, not that paper's physical renderer.

`execution_emphasis` contains exactly `weather`, `sensor`, and `preservation`, each `standard` or `strong`. Initialize all three to standard. These and the camera coverage/strength are execution controls that a diagnosed repair may adjust; they do not change the customer's requested weather, day/night or timeline.

## Severe-rain intensity is independent of camera contamination

For schema v5 heavy_rain and rainy_night, require `rain_controls` with exactly `airborne_density: torrential`, `rain_structure` (`dense_streaks`, `wind_driven_sheets`), `surface_impact` (`strong`, `very_strong`), and `vehicle_spray` (`strong`, `very_strong`). Default to wind_driven_sheets and strong impacts/spray. Target distant visibility is poor. Do not include these controls for non-rain profiles. These remain semantic generation controls, not measured mm/hour.

Show sustained dense fast airborne precipitation in the space between camera and vehicles, overlapping wind-slanted streaks/sheets, continuous road impacts or substantial moving tire spray, and a depth-dependent moving rain veil. Keep nearby geometry intermittently discernible rather than preserving a uniformly crisp distant scene. Compile the original reference description from observed source_environment fields, preventing free-text target weather from being mislabeled as source observations.

Add `precipitation_intensity` and `surface_water_dynamics` to acceptance. Return `rain_evidence` with exactly `dense_airborne_rain`, `impact_or_spray`, `depth_rain_veil`, each containing status and reason. PASS requires clearly visible evidence for all three; FAIL means clear observable regions lack the cue; UNKNOWN means actual occlusion prevents judgment. The program derives the aggregate intensity verdict and prevents weather_match from passing on wet asphalt, overcast skies or lens drops alone. Inspect higher-resolution generated-frame views as well as source/output and camera-detail sheets. `surface_water_dynamics` separately assesses repeated new glass impacts, coalescence, runoff and renewal. Never infer rain rate from adhered drop size alone.

`--feedback-from PREVIOUS_RUN` verifies the earlier video/input hashes, audits it against the explicitly stronger target without altering its original rendering metadata, and gives the actual output and new failure reasons to the planner. The revised plan records the feedback-review hash. Keep historical automatic scores as historical evidence, not as accepted severe-rain performance after user rejection.

## Compile and execute

Separate original observations from target instructions: source appearance belongs in the `<Video 1>` reference definition, not in the target shot. Compile target sky, light, visibility, surface and camera optical state from the validated fields. Free-text planner explanations cannot override these fields or reintroduce source sunny lighting. Only serialize attributes relevant to the selected target. A non-snow target does not receive a literal `snow_accumulation: none` sentence.

Supply the complete original video and the six-section H3 Ref2VA prompt. There are zero replacement images, no original-image first-frame anchor and no aligned RGB Warp. SAM, masks, depth, scene NPZ and a first-frame image editor are unnecessary for this route. Keep real dynamic frames; never broadcast one image over time. Follow the official [Ref2VA guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt); bind `<Video 1>` to the actual source, and use `N/A` audio sections for silent export.

## Review the whole environment

Check decoding, dimensions, frame count, FPS and bound input/output hashes. Compare source/output at matched frame indices. Review weather match, onset, layout, vehicle motion, camera motion, event timing, temporal consistency, unintended edits, camera-effect presence, camera-effect temporal coherence and retained readability, using `PASS`, `FAIL` or `UNKNOWN` with reasons.

`weather_match` must assess precipitation/atmosphere, sky, illumination, visibility and surface together. A wet road alone is insufficient for heavy_rain. A daytime storm retaining clear sunny lighting is a sky/light failure. Rainy night retaining daylight is a failure. Dense fog/dust hiding distant vehicles is not by itself object deletion; mark unobservable motion unknown. Weather changes to brightness and color are expected, so do not score structural retention with raw RGB differences. Sparse-frame review does not certify exact 3D or all-frame alignment.

For schema v4 add exactly three camera checks: `sensor_effect_presence`, `sensor_temporal_coherence`, `sensor_readability`. Inspect the full-frame sheet plus matching fixed-coordinate source/output detail crops. Rainy roads alone cannot pass the camera-effect criterion. Require the planned foreground contamination or glare response, inspect local image distortion and temporal persistence, and check that useful road/traffic information remains. Avoid claiming all-frame correctness from samples; mark unobservable behavior UNKNOWN. Requested camera occlusion is not itself an unwanted new scene object.

## Autonomous plan and prompt repair

After a real FAIL, the repair agent sees the actual source/output full-frame and camera-detail sheets, the attempted plan and all criterion reasons. Return `retryable`, `reason`, positive English `correction`, `diagnosis` (criterion plus observation), and `plan_patch`. Diagnose only criteria that actually failed. Do not paste failure reports or negative lists into the target prompt.

Allowed patch paths are `sensor_effects.coverage`, `sensor_effects.strength`, and `execution_emphasis.weather`, `.sensor`, `.preservation`. Schema v5 rain also permits `sensor_effects.water_activity`, `rain_controls.rain_structure`, `.surface_impact`, `.vehicle_spray`. Airborne density remains torrential; do not weaken the requested rain to pass. A rain-intensity failure must change world-rain execution cues or weather emphasis, not merely add foreground drops. Use the declared enums. Increase visible foreground coverage/strength or sensor emphasis when the requested optical effect is absent; reduce excessive coverage/strength when readability fails; strengthen weather or preservation emphasis for corresponding scene failures. Preserve sensor type, optical surface, weather-coupled effect kind, source, task/profile, day/night, target environment, camera trajectory, timeline, seed, sampler and weights. Never disable the requested corruption or weaken the task to obtain PASS.

Validate the patched full plan before generation. Store each attempt's full `plan.json`, compiled prompt/config, output, review and repair record with input/output hashes; keep the original request and root plan immutable. The next attempt must use the revised plan, not merely save it beside an unchanged renderer call. At most one video retry is allowed. If all results remain FAIL or UNKNOWN, expose that state without manual retouching or fabricated success. Schema-format repairs are separately bounded to one correction before an invalid plan/patch fails execution.

## Reusable repair lessons

The following lessons come from the retained rain experiment: `outputs/weather_heavy_rain_20261107/attempt_00` and `attempt_01` both failed weather matching; `outputs/weather_heavy_rain_v2_20261107/attempt_00` passed the earlier eight sampled checks; those checks did not independently establish torrential rain intensity. These observations support the combined repair; individual causal contributions were not isolated.

1. **Keep irrelevant weather concepts out of the target prompt.** The failed rain prompt contained `New snow accumulation: none`; repeating prohibitions about snow in the automatic retry did not solve the failure. Prefer positive target cues and omit inapplicable fields. Keep raw planner output for provenance without forwarding every field to H3.
2. **Source observations are not target appearance constraints.** The failed target shot mixed blue-sky source observations with rain instructions. Move the source inventory into the reference definition and explicitly describe the desired replacement sky.
3. **Protect structure separately from weather appearance.** Free-form preservation text can accidentally retain source sunshine or repeat unrelated weather. Compile a fixed structure/motion contract and independently compile the requested atmosphere, lighting and surface.
4. **Treat weather/light interactions as a family-level rule.** For each profile resolve the full target state using the table, rather than fixing one rain prompt with an isolated overcast sentence. Review these same coupled attributes.
5. **Distinguish prompt correction from planner/compiler defects.** A missing effect can justify a specific prompt correction. Repeated category contamination or contradictory state fields require repairing the planner/skill/compiler and running regression checks; stronger repetitions of the same negative text are not evidence of improvement.

Camera-effect evidence: `outputs/weather_heavy_rain_v3_20261107/attempt_00` received eleven automatic sampled PASS results and visibly added adhered droplets/film. The user subsequently rejected its rain intensity. The original weather_match rationale cited overcast sky, wet road and reduced visibility without establishing dense airborne rain, so that PASS is insufficient acceptance evidence. The retained camera-effect result does not establish a torrential downpour. This real run did not trigger generation repair. The automatic plan-patch path was exercised separately with a simulated visual failure, verifying that the next attempt consumed a changed full plan and compiled prompt; do not report that test as empirical video self-repair convergence.

Retry generation at most once after an observed FAIL and a justified correction. Keep source, profile, target day/night, seed, steps, weights and timeline fixed; change only validated execution controls and the diagnosed prompt constraint. UNKNOWN alone does not justify retry. Prefer positive descriptions of the desired correction; do not paste raw failure reports or lists of unwanted weather into H3. Preserve both attempts and their hashes. Record successful fixes as generalizable skill rules with evidence and uncertainty, then test applicable profiles and historical resume before treating them as reusable. This skill records the current verified lessons; runtime self-modification of skill files is not implemented.

Independent acceptance lesson: the stricter feedback audit marked v3 airborne rain, impacts/spray and rain veil FAIL. The resulting `outputs/weather_torrential_rain_v4_20261107/attempt_00` received automatic PASS again, but assistant frame inspection still did not establish sufficiently dense airborne precipitation. Its acceptance is NOT_ESTABLISHED and the video is an unaccepted candidate. Stronger rubric wording and extra semantic fields alone do not establish a reliable verifier or successful rain-intensity control. Preserve model scores separately from independent/user acceptance; do not promote this case as successful autonomous convergence.

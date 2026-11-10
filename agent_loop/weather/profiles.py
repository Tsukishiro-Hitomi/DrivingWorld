"""Semantic weather presets; these are prompt constraints, not physical simulator controls."""

import math

PROFILES = {
    "heavy_rain": {
        "label": "Heavy Rain",
        "effects": "Heavy dense rainfall across the scene, rain streaks, tire spray, wet asphalt and plausible reflections. Reduce distant visibility while retaining nearby scene structure.",
        "road_surface": "wet", "time_of_day": "preserve", "snow_accumulation": "none",
    },
    "dense_fog": {
        "label": "Dense Fog",
        "effects": "Dense atmospheric fog with distance-dependent loss of contrast. Distant objects become obscured gradually with depth; nearby road and vehicles remain coherent. Do not depict a flat white screen or add rain.",
        "road_surface": "preserve", "time_of_day": "preserve", "snow_accumulation": "none",
    },
    "heavy_snow": {
        "label": "Heavy Snow",
        "effects": "Dense sustained snowfall with flakes at varied apparent depths and continuous downward drifting motion. Diffuse overcast illumination and reduced distant visibility. Snowfall does not by itself imply thick ground accumulation.",
        "road_surface": "preserve", "time_of_day": "preserve", "snow_accumulation": "none",
    },
    "strong_backlight": {
        "label": "Strong Backlight",
        "effects": "Strong sunlight from ahead of the camera, plausible glare and restrained lens flare, bright sky and shadowed rear-facing surfaces. Preserve the camera orientation and visible scene layout. Avoid whiteout, flashing exposure or a graphic sun overlay.",
        "road_surface": "preserve", "time_of_day": "day", "snow_accumulation": "none",
    },
    "rainy_night": {
        "label": "Rainy Night",
        "effects": "Nighttime driving in heavy rain, dark ambient illumination, visible rainfall and wet asphalt reflecting plausible vehicle lights and existing scene light sources. Preserve vehicle identities and positions; do not invent streetlight poles or extra traffic.",
        "road_surface": "wet", "time_of_day": "night", "snow_accumulation": "none",
    },
    "sandstorm": {
        "label": "Sandstorm",
        "effects": "Dense windblown airborne sand and dust with warm muted color, persistent moving dust and distance-dependent obscuration. Keep nearby scene geometry coherent. No rain, snow or newly flying large debris.",
        "road_surface": "preserve", "time_of_day": "preserve", "snow_accumulation": "none",
    },
}
PRESERVE = ["scene_layout", "vehicle_identity_and_motion", "camera_motion", "event_timing"]
CHECKS = ["weather_match", "onset_correct", "scene_layout", "vehicle_motion",
          "camera_motion", "event_timing", "temporal_consistency", "no_unrequested_edits"]

# Separate target environment from the legacy day/night preservation policy.
# Keep PROFILES and the v1/v2 compiler unchanged so existing outputs can resume.
ENVIRONMENTS = {
    "heavy_rain": dict(sky_condition="storm_overcast", illumination="storm_diffuse", visibility="reduced"),
    "dense_fog": dict(sky_condition="fog_obscured", illumination="fog_scattered", visibility="poor"),
    "heavy_snow": dict(sky_condition="snow_overcast", illumination="snow_diffuse", visibility="reduced"),
    "strong_backlight": dict(sky_condition="bright_backlit", illumination="strong_frontal_sunlight", visibility="glare_limited"),
    "rainy_night": dict(sky_condition="night_overcast", illumination="wet_night_lighting", visibility="poor"),
    "sandstorm": dict(sky_condition="dust_obscured", illumination="dust_filtered", visibility="poor"),
}


def planning_profiles(version=4):
    """Expose resolved-target rules to Qwen rather than legacy preserve fields."""
    from .sensor import sensor_defaults

    result = {
        key: dict(label=preset["label"], environment=ENVIRONMENTS[key],
                  sensor_effects=sensor_defaults(key, version),
                  time_of_day_policy="day" if key == "strong_backlight" else
                  ("night" if key == "rainy_night" else "resolve_from_source_unless_requested"),
                  road_surface_policy="wet" if key in {"heavy_rain", "rainy_night"} else
                  "resolve_from_source_unless_requested")
        for key, preset in PROFILES.items()
    }
    if version >= 5:
        from .rain import RAIN_PROFILES, defaults
        for key in RAIN_PROFILES:
            result[key]["environment"] = dict(result[key]["environment"], visibility="poor")
            result[key]["rain_controls"] = defaults()
    return result


def validate_environment(plan):
    source = plan.get("source_environment", {})
    if (set(source) != {"time_of_day", "road_surface"}
            or source["time_of_day"] not in {"day", "night", "twilight", "unknown"}
            or source["road_surface"] not in {"dry", "wet", "snow_covered", "unknown"}):
        raise ValueError("Source environment must identify observed day/night and road surface")
    edits = plan["edits"]
    keys = {"time_of_day", "road_surface", "snow_accumulation", *ENVIRONMENTS[plan["profile"]]}
    if set(edits) != keys:
        raise ValueError("Target environment must explicitly include sky, illumination and visibility")
    if (edits["time_of_day"] not in {"day", "night", "twilight"}
            or edits["road_surface"] not in {"dry", "wet", "snow_covered"}
            or edits["snow_accumulation"] not in {"none", "light", "heavy"}):
        raise ValueError("Target day/night and surface must be resolved; preserve is not a target state")
    expected_environment = dict(ENVIRONMENTS[plan["profile"]])
    if plan.get("prompt_version", 1) >= 5 and plan["profile"] in {"heavy_rain", "rainy_night"}:
        expected_environment["visibility"] = "poor"
    for key, expected in expected_environment.items():
        if edits[key] != expected:
            raise ValueError(f"{plan['profile']} requires {key}={expected}")
    if plan["profile"] in {"heavy_rain", "rainy_night"} and edits["road_surface"] != "wet":
        raise ValueError("Heavy rain and rainy night require a wet road")
    if plan["profile"] == "strong_backlight" and edits["time_of_day"] != "day":
        raise ValueError("Strong frontal sunlight requires daytime")


def validate_plan(plan, duration, requested_profile=None):
    if type(plan.get("prompt_version", 1)) is not int or plan.get("prompt_version", 1) not in (1, 2, 3, 4, 5):
        raise ValueError("Unsupported weather prompt version")
    profile = plan.get("profile")
    if profile not in PROFILES or (requested_profile and profile != requested_profile):
        raise ValueError("Planner selected an invalid or conflicting weather profile")
    if plan.get("unresolved") != []:
        raise ValueError(f"Unresolved weather request: {plan.get('unresolved')}")
    for key in ("scene_description", "weather_description", "preservation_description"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"Missing weather planner field: {key}")
    edits = plan.get("edits", {})
    options = {"road_surface": {"preserve", "wet", "snow_covered"},
               "time_of_day": {"preserve", "day", "night"},
               "snow_accumulation": {"none", "light", "heavy"}}
    if plan.get("prompt_version", 1) >= 3:
        validate_environment(plan)
    elif set(edits) != set(options) or any(edits[k] not in values for k, values in options.items()):
        raise ValueError("Invalid weather edit fields")
    if profile != "heavy_snow":
        if edits["snow_accumulation"] != "none":
            raise ValueError("Snow accumulation requires the heavy_snow profile")
        if edits["road_surface"] == "snow_covered" and not (
                plan.get("prompt_version", 1) >= 3
                and plan["source_environment"]["road_surface"] == "snow_covered"):
            raise ValueError("A non-snow profile may only retain existing ground cover")
    if profile == "rainy_night" and edits["time_of_day"] != "night":
        raise ValueError("Rainy night must use nighttime lighting")
    mode = plan.get("temporal_mode")
    start, end = plan.get("transition_start_s"), plan.get("transition_end_s")
    if mode == "constant_from_frame_zero":
        if start is not None or end is not None:
            raise ValueError("Constant weather must not contain a transition interval")
    elif mode == "gradual":
        if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                or not 0 <= start < end <= duration):
            raise ValueError("Weather transition must lie inside the source timeline")
    else:
        raise ValueError("Unsupported weather temporal mode")
    if plan.get("prompt_version", 1) >= 4:
        from .sensor import validate_sensor

        validate_sensor(plan)
    if plan.get("prompt_version", 1) >= 5:
        from .rain import validate
        validate(plan)
    return plan


def h3_prompt(plan, correction=""):
    if plan.get("prompt_version", 1) >= 4:
        from .sensor import sensor_prompt

        emphasis = plan["execution_emphasis"]
        additions = [sensor_prompt(plan)]
        if plan.get("prompt_version", 1) >= 5 and plan["profile"] in {"heavy_rain", "rainy_night"}:
            from .rain import prompt as rain_prompt
            additions.insert(0, rain_prompt(plan))
        if emphasis["weather"] == "strong":
            additions.append("Make the requested atmospheric weather, sky and road-surface changes unmistakable throughout the shot.")
        if emphasis["sensor"] == "strong":
            additions.append("The camera optical degradation is a prominent, clearly visible component from the first frame, with localized foreground distortion distinct from the distant weather.")
        if emphasis["preservation"] == "strong":
            additions.append("Strictly follow the reference's visible road geometry, vehicle identities, background progression and event ordering through the clear optical regions.")
        additions.append(correction)
        if plan.get("prompt_version", 1) >= 5:
            from copy import deepcopy
            plan = deepcopy(plan)
            observed = plan["source_environment"]
            plan["scene_description"] = (f"RGB driving reference with observed {observed['time_of_day']} "
                                         f"and {observed['road_surface']} road surface; use the actual video for scene geometry and traffic motion.")
        return environment_prompt(plan, " ".join(additions))
    if plan.get("prompt_version", 1) == 3:
        return environment_prompt(plan, correction)
    preset = PROFILES[plan["profile"]]
    modern = plan.get("prompt_version", 1) == 2
    observations = f" Original source observations: {plan['scene_description']}" if modern else ""
    shot_scene = "Retain the source road and traffic composition." if modern else plan["scene_description"]
    effects = preset["effects"]
    if modern and plan["profile"] == "heavy_rain":
        effects += " A dark overcast storm sky replaces the source sky while the horizon and tree silhouettes remain in place."
    if modern and plan["profile"] == "sandstorm":
        effects = effects.replace("No rain, snow or newly flying large debris.",
                                  "The moving dust consists of fine airborne particles.")
    if plan["temporal_mode"] == "constant_from_frame_zero":
        timing = "The requested environment is already fully present in the very first frame at 0.00 seconds and persists throughout the complete clip."
    else:
        timing = (f"Retain the initial source environment until {plan['transition_start_s']:.2f} seconds, "
                  f"then transition smoothly to the requested environment by {plan['transition_end_s']:.2f} seconds and retain it thereafter.")
    edits = plan["edits"]
    preservation = ("Retain road topology, lane markings, vehicle identities and movements, "
                    "camera trajectory and event timing." if modern else plan["preservation_description"])
    accumulation = (f"New snow accumulation: {edits['snow_accumulation']}. "
                    if not modern or plan["profile"] == "heavy_snow" else "")
    return (
        "crossview.\nsubject_definitions:\n"
        f"<Video 1> is the original driving video, providing the scene layout, traffic and camera movement.{observations}\n\n"
        "summary:\n[video editing] The target video is an edited version of <Video 1>. "
        f"Change its environment to {preset['label'].lower()} while retaining its scene and motion.\n\n"
        "retention_analysis:\n<Video 1>: partially_preserved - change weather and requested lighting, "
        "retain road layout, vehicle identities, motion, event ordering and original timing.\n\n"
        "detailed_description:\nPhotorealistic continuous forward-facing driving footage.\n"
        f"[Shot 1] {shot_scene} {timing} {effects} "
        f"{plan['weather_description']} Road surface: {edits['road_surface']}. "
        f"Time of day: {edits['time_of_day']}. {accumulation}"
        f"{preservation} Preserve the source camera trajectory, object movements, "
        "event order, duration and frame correspondence. Objects obscured by atmosphere still follow "
        "their original motion. Keep weather, lighting and reflections temporally coherent. "
        "Do not freeze motion, shift the video timeline, add traffic, change vehicle categories, "
        "or add unrequested maneuvers, lightning, debris, wipers, cuts, zooms, text or logos. "
        f"{correction}\n\noverall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    )


def environment_prompt(plan, correction=""):
    """Compile the validated target state; free-text preservation cannot override it."""
    edits = plan["edits"]
    daytime = edits["time_of_day"] != "night"
    sky = {
        "storm_overcast": "Dense storm clouds replace the original sky; the horizon and tree silhouettes remain fixed.",
        "fog_obscured": "Dense fog obscures the sky and distant scene with a continuous depth-dependent veil.",
        "snow_overcast": "An overcast sky and diffuse atmospheric illumination accompany the falling flakes.",
        "bright_backlit": "Bright sunlight comes from ahead of the camera, with plausible glare and shadowed object surfaces.",
        "night_overcast": "The environment is nighttime with a dark clouded sky and low ambient brightness.",
        "dust_obscured": "Windblown airborne dust obscures the sky and distant scene with a warm muted veil.",
    }[edits["sky_condition"]]
    illumination = {
        "storm_diffuse": "Diffuse low-contrast storm daylight replaces direct sunlight and sharp sun shadows." if daytime else
                         "Dark storm ambient light is scattered by rainfall, with reflections from existing light sources.",
        "fog_scattered": "Light is diffusely scattered by the fog, softening contrast and direct shadows." if daytime else
                         "Existing light sources scatter through the fog against dark ambient illumination.",
        "snow_diffuse": "Soft diffuse overcast daylight lights the falling flakes." if daytime else
                        "Existing lights illuminate falling flakes against dark nighttime ambient light.",
        "strong_frontal_sunlight": "Strong frontal sunlight creates backlit surfaces, glare and pronounced brightness contrast.",
        "wet_night_lighting": "Dark ambient illumination and existing vehicle or scene lights produce wet-road reflections.",
        "dust_filtered": "Warm dust-filtered light replaces clear direct sunshine." if daytime else
                         "Dark ambient light and existing lights are scattered through the moving dust.",
    }[edits["illumination"]]
    effects = PROFILES[plan["profile"]]["effects"]
    # Retain category-specific positive cues; omit negative lists of other weather.
    effects = effects.replace("Do not depict a flat white screen or add rain.", "")
    effects = effects.replace("Snowfall does not by itself imply thick ground accumulation.", "")
    effects = effects.replace("No rain, snow or newly flying large debris.", "")
    time = {"day": "The target remains daytime.", "night": "The target is nighttime.",
            "twilight": "The target has twilight ambient brightness."}[edits["time_of_day"]]
    surface = {"dry": "Retain the observed dry road material and markings.",
               "wet": "The asphalt is wet, with plausible reflections and tire spray.",
               "snow_covered": "Retain the observed existing ground cover and visible road markings." if plan["profile"] != "heavy_snow" else
                               "Ground accumulation covers the road according to the requested amount."}[edits["road_surface"]]
    visibility = {"reduced": "Distant visibility and contrast are reduced while nearby structures remain coherent.",
                  "poor": "Distant visibility is poor; atmospheric obscuration increases with depth.",
                  "glare_limited": "Bright glare limits visibility near the light source while scene structure remains coherent."}[edits["visibility"]]
    accumulation = ""
    if plan["profile"] == "heavy_snow":
        accumulation = {"none": "Falling flakes coexist with the original exposed ground surfaces.",
                        "light": "A light consistent layer accumulates on exposed ground and vegetation.",
                        "heavy": "A thick consistent layer accumulates on exposed ground and vegetation."}[edits["snow_accumulation"]]
    if plan["temporal_mode"] == "constant_from_frame_zero":
        timing = "The full target environment is present from the first frame at 0.00 seconds through the end."
    else:
        timing = (f"Keep the source environment until {plan['transition_start_s']:.2f} seconds, then transition "
                  f"smoothly to the target by {plan['transition_end_s']:.2f} seconds and retain it thereafter.")
    return (
        "crossview.\nsubject_definitions:\n"
        f"<Video 1> is the original driving reference. Original source observations: {plan['scene_description']}\n\n"
        "summary:\n[video editing] The target video is an edited version of <Video 1>. "
        f"Transform its environment to {PROFILES[plan['profile']]['label'].lower()}.\n\n"
        "retention_analysis:\n<Video 1>: partially_preserved - retain road topology, vehicle identities, "
        "object movements, camera trajectory and event timing; replace weather-dependent sky, "
        "illumination, shadows, contrast, visibility and requested surface appearance.\n\n"
        "detailed_description:\nPhotorealistic continuous forward-facing driving footage.\n"
        f"[Shot 1] Retain the source road and traffic composition. {timing} {effects} {time} "
        f"{sky} {illumination} {surface} {visibility} {accumulation} "
        "Keep the same road layout, vehicle identities and movements, camera trajectory, duration and "
        "frame correspondence. Atmospheric occlusion can hide distant objects while they continue "
        "their original motion. Maintain coherent lighting, reflections and atmospheric movement "
        f"throughout one uninterrupted shot. {correction}\n\n"
        "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    )

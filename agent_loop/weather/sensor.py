"""Camera-space weather effects and their diagnostic views; no other sensor modalities."""

from copy import deepcopy

SENSOR_PRESETS = {
    "heavy_rain": dict(surface_condition="water_droplets_and_film", optical_effect="local_refraction",
                       temporal_behavior="coalesce_and_slide"),
    "dense_fog": dict(surface_condition="condensation_haze", optical_effect="soft_veiling",
                      temporal_behavior="slow_condensation"),
    "heavy_snow": dict(surface_condition="wet_flakes_and_slush", optical_effect="localized_soft_occlusion",
                       temporal_behavior="adhere_and_melt"),
    "strong_backlight": dict(surface_condition="clean", optical_effect="flare_bloom",
                            temporal_behavior="light_dependent"),
    "rainy_night": dict(surface_condition="water_droplets_and_film", optical_effect="refractive_highlight_bloom",
                        temporal_behavior="coalesce_and_slide"),
    "sandstorm": dict(surface_condition="dust_deposit", optical_effect="dust_smearing",
                      temporal_behavior="gradual_deposition"),
}
SENSOR_CHECKS = ["sensor_effect_presence", "sensor_temporal_coherence", "sensor_readability"]


def sensor_defaults(profile, version=4):
    result = dict(sensor="rgb_camera", optical_surface="protective_glass", **SENSOR_PRESETS[profile],
                coverage="moderate", strength="moderate", road_readability="retained")
    if version >= 5 and profile in {"heavy_rain", "rainy_night"}:
        result["water_activity"] = "high"
    return result


def validate_sensor(plan):
    actual = plan.get("sensor_effects", {})
    expected = sensor_defaults(plan["profile"], plan.get("prompt_version", 1))
    if set(actual) != set(expected):
        raise ValueError("Sensor plan must contain the complete camera optical-state contract")
    for key, value in expected.items():
        if key not in {"coverage", "strength", "water_activity"} and actual[key] != value:
            raise ValueError(f"{plan['profile']} requires sensor {key}={value}")
    if actual["coverage"] not in {"light", "moderate", "heavy"}:
        raise ValueError("Invalid sensor coverage")
    if actual["strength"] not in {"mild", "moderate", "strong"}:
        raise ValueError("Invalid sensor optical strength")
    if "water_activity" in expected and actual["water_activity"] not in {"high", "very_high"}:
        raise ValueError("Severe rain camera water activity must be high or very_high")
    emphasis = plan.get("execution_emphasis", {})
    if (set(emphasis) != {"weather", "sensor", "preservation"}
            or any(value not in {"standard", "strong"} for value in emphasis.values())):
        raise ValueError("Execution emphasis must explicitly cover weather, sensor and preservation")


def sensor_prompt(plan):
    state = plan["sensor_effects"]
    timing = ("These camera effects are already visible in the first frame at 0.00 seconds and persist throughout the clip."
              if plan["temporal_mode"] == "constant_from_frame_zero" else
              "These camera effects develop coherently during the requested weather transition and persist afterward.")
    surface = {
        "water_droplets_and_film": "Irregular translucent water droplets adhere to the camera's protective glass in the extreme foreground, with soft meniscus edges, localized background refraction and a thin uneven water film.",
        "condensation_haze": "A thin uneven condensation layer on the protective glass creates localized soft veiling and loss of microcontrast, distinct from distant atmospheric fog.",
        "wet_flakes_and_slush": "A few wet flakes and melting slush patches adhere to the protective glass in the extreme foreground, creating localized out-of-focus soft occlusion.",
        "clean": "Strong frontal light produces visible camera-optical glare, bloom and plausible flare around the bright source.",
        "dust_deposit": "Uneven fine dust deposits and soft smears adhere to the protective glass in the extreme foreground, locally veiling and softening the scene.",
    }[state["surface_condition"]]
    motion = {
        "coalesce_and_slide": "Droplets persist in camera-image coordinates as the road moves behind them, then slowly merge and slide downward under gravity; local refraction changes with their shape.",
        "slow_condensation": "The camera-attached condensation changes slowly and coherently while background objects continue their original motion.",
        "adhere_and_melt": "Adhered patches persist in camera coordinates, slowly soften and melt as the background moves behind them.",
        "light_dependent": "Glare and bloom vary coherently with the existing bright source and camera motion.",
        "gradual_deposition": "Camera-attached smears persist and slowly accumulate while airborne dust moves separately in the scene.",
    }[state["temporal_behavior"]]
    if state.get("water_activity"):
        motion = ("Fresh impacts continuously replenish the camera glass. Droplets rapidly merge, "
                  "shed and reform into running rivulets and an uneven flowing film. Local refraction "
                  "changes continuously as water moves in camera-image coordinates, independently of the road.")
        if state["water_activity"] == "very_high":
            motion += " Frequent forceful impacts and brisk runoff repeatedly renew the foreground water pattern."
    coverage = {"light": "Sparse small affected patches occupy a limited part of the image.",
                "moderate": "Several clearly visible affected patches are distributed across the foreground, including peripheral and lower-frame regions.",
                "heavy": "Numerous clearly visible foreground patches affect a substantial part of the image, leaving useful clear gaps."}[state["coverage"]]
    strength = {"mild": "Optical degradation is subtle but discernible in the affected patches.",
                "moderate": "Optical degradation is plainly visible, with localized soft distortion and contrast loss.",
                "strong": "Optical degradation is pronounced, with unmistakable local distortion, blurred edges and contrast loss."}[state["strength"]]
    effect = ""
    if state["optical_effect"] == "refractive_highlight_bloom":
        effect = "Existing lights refract and bloom through the adhered droplets against dark nighttime illumination."
    return (f"The RGB camera is affected by the weather as well as the outdoor scene. {timing} {surface} {coverage} "
            f"{strength} {motion} {effect} The central road, lane layout and nearby traffic remain "
            "recognizable through clear gaps. Treat these as optical foreground effects, distinct "
            "from atmospheric rain streaks, road spray or new physical scene objects.")


def apply_plan_patch(plan, patch, duration):
    """Allow agent changes to execution controls, never to the customer's task or timeline."""
    from .profiles import validate_plan

    allowed = {"sensor_effects": {"coverage", "strength"},
               "execution_emphasis": {"weather", "sensor", "preservation"}}
    if plan.get("prompt_version", 1) >= 5 and plan["profile"] in {"heavy_rain", "rainy_night"}:
        allowed["sensor_effects"].add("water_activity")
        allowed["rain_controls"] = {"rain_structure", "surface_impact", "vehicle_spray"}
    if not isinstance(patch, dict) or set(patch) - set(allowed):
        raise ValueError("Repair attempted to change an immutable plan field")
    result = deepcopy(plan)
    for section, values in patch.items():
        if not isinstance(values, dict) or set(values) - allowed[section]:
            raise ValueError("Repair attempted to change an immutable plan control")
        result[section].update(values)
    validate_plan(result, duration, plan["profile"])
    return result


def review_checks(plan):
    from .profiles import CHECKS

    extra = ["precipitation_intensity", "surface_water_dynamics"] if (
        plan.get("prompt_version", 1) >= 5 and plan["profile"] in {"heavy_rain", "rainy_night"}) else []
    return CHECKS + (SENSOR_CHECKS if plan.get("prompt_version", 1) >= 4 else []) + extra


def sensor_sheet(path, source, output, fps=24):
    """Matching RGB crops in fixed image coordinates, with no object tracking or masks."""
    import numpy as np
    from PIL import Image, ImageDraw

    _, height, width, _ = source.shape
    indices = np.linspace(0, len(source) - 1, 4).round().astype(int)
    regions = [
        ("LEFT-LOW", (0, 2 * height // 3, width // 3, height)),
        ("CENTER", (width // 3, height // 3, 2 * width // 3, 2 * height // 3)),
        ("RIGHT-LOW", (2 * width // 3, 2 * height // 3, width, height)),
    ]
    sheet = Image.new("RGB", (256 * len(indices), 180 * 2 * len(regions)), "#141923")
    draw = ImageDraw.Draw(sheet)
    for region, (label, (x0, y0, x1, y1)) in enumerate(regions):
        for row, frames in enumerate((source, output)):
            for col, i in enumerate(indices):
                crop = Image.fromarray(frames[i, y0:y1, x0:x1])
                crop.thumbnail((256, 152))
                x, y = col * 256, (region * 2 + row) * 180
                sheet.paste(crop, (x, y + 28))
                draw.text((x + 3, y + 7), f"{'SRC' if row == 0 else 'OUT'} {label} {i/fps:.2f}s", fill="white")
    sheet.save(path, quality=94)

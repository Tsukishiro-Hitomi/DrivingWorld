"""Observable heavy-rain intensity controls, independent of wet roads and lens drops."""

from copy import deepcopy

RAIN_PROFILES = {"heavy_rain", "rainy_night"}
RAIN_EVIDENCE = {"dense_airborne_rain", "impact_or_spray", "depth_rain_veil"}


def enabled(plan):
    return plan.get("prompt_version", 1) >= 5 and plan["profile"] in RAIN_PROFILES


def defaults():
    return dict(airborne_density="torrential", rain_structure="wind_driven_sheets",
                surface_impact="strong", vehicle_spray="strong")


def validate(plan):
    if not enabled(plan):
        if "rain_controls" in plan:
            raise ValueError("Rain controls are inapplicable to this profile/schema")
        return
    actual = plan.get("rain_controls", {})
    if set(actual) != set(defaults()) or actual.get("airborne_density") != "torrential":
        raise ValueError("Severe rain needs explicit torrential airborne density and complete rain controls")
    if (actual["rain_structure"] not in {"dense_streaks", "wind_driven_sheets"}
            or actual["surface_impact"] not in {"strong", "very_strong"}
            or actual["vehicle_spray"] not in {"strong", "very_strong"}):
        raise ValueError("Invalid rain execution controls")


def prompt(plan):
    control = plan["rain_controls"]
    structure = {"dense_streaks": "Dense fast rain streaks visibly fill the space between the camera and traffic.",
                 "wind_driven_sheets": "Dense wind-slanted rain streaks overlap into fast-moving sheets across near and middle distances."}[control["rain_structure"]]
    impact = {"strong": "Raindrops continuously strike the asphalt, producing plainly visible splashes and a low splash haze.",
              "very_strong": "Vigorous continuous impacts produce dense splashes and an actively churning splash layer over the asphalt."}[control["surface_impact"]]
    spray = {"strong": "Moving tires throw substantial water spray into the air behind the vehicles.",
             "very_strong": "Moving tires drive large continuous plumes of water spray behind the vehicles."}[control["vehicle_spray"]]
    return (f"A sustained torrential downpour is the dominant scene event. {structure} {impact} {spray} "
            "A moving veil of dense rainfall strongly obscures distant traffic and vegetation. "
            "Clearly show the high volume of rain throughout the 3D scene, including regions viewed "
            "through gaps between camera droplets. Nearby road geometry remains discernible intermittently.")


def audit_target(old_plan):
    """A stricter acceptance target, explicitly separate from the original rendering inputs."""
    from .sensor import sensor_defaults

    result = deepcopy(old_plan)
    result["prompt_version"] = 5
    result["edits"]["visibility"] = "poor"
    result["rain_controls"] = defaults()
    result["sensor_effects"] = sensor_defaults(result["profile"], version=5)
    return result


def normalize_review(result):
    evidence = result.get("rain_evidence", {})
    if set(evidence) != RAIN_EVIDENCE:
        raise ValueError("Rain review must provide separate airborne, impact/spray and depth evidence")
    for item in evidence.values():
        if (not isinstance(item, dict) or item.get("status") not in {"PASS", "FAIL", "UNKNOWN"}
                or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
            raise ValueError("Invalid observable rain evidence")
    statuses = [item["status"] for item in evidence.values()]
    verdict = "FAIL" if "FAIL" in statuses else ("UNKNOWN" if "UNKNOWN" in statuses else "PASS")
    result["checks"]["precipitation_intensity"] = {
        "status": verdict,
        "reason": "Required independent rain evidence: " + "; ".join(f"{key}: {item['reason']}" for key, item in evidence.items()),
    }
    if verdict != "PASS" and result["checks"]["weather_match"]["status"] == "PASS":
        result["checks"]["weather_match"] = dict(status=verdict, reason="Severe rain is not established by the independent intensity evidence.")
    return result


def intensity_sheet(path, frames, fps=24):
    import numpy as np
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (1280, 2 * 400), "#141923")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(np.linspace(0, len(frames)-1, 4).round().astype(int)):
        picture = Image.fromarray(frames[index])
        picture.thumbnail((640, 370))
        x, y = (position % 2) * 640, (position // 2) * 400
        sheet.paste(picture, (x, y + 28))
        draw.text((x+4, y+7), f"OUTPUT RAIN EVIDENCE {index} / {index/fps:.2f}s", fill="white")
    sheet.save(path, quality=95)

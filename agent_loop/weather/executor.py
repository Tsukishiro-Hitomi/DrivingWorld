"""Weather video generation and evidence-bound output verification."""

import argparse
from pathlib import Path
import subprocess
import sys

from ..qwen import run_worker
from ..runtime import ROOT, dump, load, sha
from ..video import comparison, read_frames, sample_sheet
from .profiles import CHECKS, PROFILES, h3_prompt
from .sensor import review_checks, sensor_sheet


def generate(out, attempt, correction="", plan=None):
    request = load(out / "request.json")
    plan_path = out / "plan.json"
    if plan is None:
        plan = load(plan_path)
    else:
        plan_path = attempt / "plan.json"
        if plan_path.exists() and load(plan_path) != plan:
            raise ValueError("Existing attempt plan has changed")
        dump(plan_path, plan)
    if sha(Path(request["source"])) != request["source_sha256"]:
        raise ValueError("Weather source video has changed")
    config = {
        "agent": "weather", "source": request["source"], "source_sha256": request["source_sha256"],
        "weather_plan": plan, "plan_sha256": sha(plan_path), "prompt": h3_prompt(plan, correction),
        "correction": correction, "frames": request["frames"], "fps": request["fps"],
        "width": request["width"], "height": request["height"],
        "seed": request["seed"], "steps": request["steps"],
        "reference_video_count": 1, "reference_image_count": 0,
        "aligned_warp": None, "first_frame_anchor": None,
        "video_shift": 12, "audio_shift": 3, "crossview_strength": .8,
        "acceleration": "Larry v4 step600 EMA",
    }
    if (attempt / "config.json").exists() and load(attempt / "config.json") != config:
        raise ValueError("Existing weather attempt inputs have changed")
    dump(attempt / "config.json", config)
    with (attempt / "h3.log").open("a") as log:
        subprocess.run([sys.executable, "-m", "agent_loop.weather.executor", "infer", "--out", str(attempt)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def verify(attempt):
    config = load(attempt / "config.json")
    if sha(Path(config["source"])) != config["source_sha256"]:
        raise ValueError("Weather source video changed before verification")
    generation = load(attempt / "generation.json")
    if (generation["generated_sha256"] != sha(attempt / "generated.mp4")
            or any(generation.get(k) != v for k, v in config.items())):
        raise ValueError("Weather generation metadata differs from the bound inputs")
    source = read_frames(config["source"])
    generated = read_frames(attempt / "generated.mp4")
    comparison(attempt / "comparison.mp4", source, generated, PROFILES[config["weather_plan"]["profile"]]["label"])
    sample_sheet(attempt / "review_frames.jpg", source, generated)
    sensor_enabled = config["weather_plan"].get("prompt_version", 1) >= 4
    if sensor_enabled:
        sensor_sheet(attempt / "review_sensor_frames.jpg", source, generated)
    from .rain import enabled as rain_enabled, intensity_sheet
    if rain_enabled(config["weather_plan"]):
        intensity_sheet(attempt / "review_rain_frames.jpg", generated)
    dump(attempt / "technical_checks.json", {
        "frames": len(generated), "fps": config["fps"], "width": config["width"], "height": config["height"],
        "duration_s": len(generated) / config["fps"], "generated_sha256": sha(attempt / "generated.mp4"),
        "source_sha256": sha(Path(config["source"])),
        "reference_image_count": 0, "aligned_warp": None, "first_frame_anchor": None,
        "note": "Media integrity checks do not certify scene, 3D or all-frame temporal preservation.",
    })
    if not (attempt / "visual_review.json").exists():
        run_worker("agent_loop.weather", "review", attempt, "review.log")
    review = load(attempt / "visual_review.json")
    if (review["generated_sha256"] != sha(attempt / "generated.mp4")
            or review["sheet_sha256"] != sha(attempt / "review_frames.jpg")
            or review["config_sha256"] != sha(attempt / "config.json")):
        raise ValueError("Weather review belongs to changed output or inputs")
    if sensor_enabled and review.get("sensor_sheet_sha256") != sha(attempt / "review_sensor_frames.jpg"):
        raise ValueError("Sensor review belongs to a changed detail sheet")
    if rain_enabled(config["weather_plan"]) and review.get("rain_sheet_sha256") != sha(attempt / "review_rain_frames.jpg"):
        raise ValueError("Rain review belongs to a changed intensity sheet")
    statuses = [review["checks"][key]["status"] for key in review_checks(config["weather_plan"])]
    if any(status not in {"PASS", "FAIL", "UNKNOWN"} for status in statuses):
        raise ValueError("Invalid cached weather review status")
    return "FAIL" if "FAIL" in statuses else ("UNKNOWN" if "UNKNOWN" in statuses else "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from ..ref2va import infer

    infer(args.out.resolve())


if __name__ == "__main__":
    main()

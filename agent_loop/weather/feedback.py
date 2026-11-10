"""Re-evaluate an earlier output under an explicit, stricter acceptance target."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from ..qwen import run_worker
from ..runtime import ROOT, dump, load, sha
from ..video import read_frames
from .rain import RAIN_PROFILES, audit_target, intensity_sheet
from .profiles import validate_plan


def feedback_binding(path, request):
    baseline = path.resolve()
    config_path = baseline / "attempt_00/config.json"
    if (baseline / "state.json").exists():
        selected = load(baseline / "state.json").get("selected_attempt", 0)
        config_path = baseline / f"attempt_{selected:02d}/config.json"
    config = load(config_path)
    generation = load(config_path.parent / "generation.json")
    if (config["source_sha256"] != request["source_sha256"]
            or config["weather_plan"]["profile"] not in RAIN_PROFILES
            or (request["profile"] and config["weather_plan"]["profile"] != request["profile"])):
        raise ValueError("Feedback must belong to the same source and rain profile")
    video = config_path.parent / "generated.mp4"
    if sha(video) != generation["generated_sha256"]:
        raise ValueError("Feedback video differs from its generation metadata")
    return dict(run=str(baseline), attempt=str(config_path.parent), generated_sha256=sha(video),
                config_sha256=sha(config_path), generation_sha256=sha(config_path.parent / "generation.json"))


def audit(out):
    request = load(out / "request.json")
    binding = request["feedback"]
    baseline = Path(binding["attempt"])
    for key, name in [("generated_sha256", "generated.mp4"), ("config_sha256", "config.json"),
                      ("generation_sha256", "generation.json")]:
        if binding[key] != sha(baseline / name):
            raise ValueError("Feedback inputs changed before re-evaluation")
    config = load(baseline / "config.json")
    target = audit_target(config["weather_plan"])
    validate_plan(target, config["frames"]/config["fps"], config["weather_plan"]["profile"])
    dump(out / "feedback_target.json", target)
    with TemporaryDirectory(prefix="rain-intensity-audit-", dir=ROOT / "logs") as directory:
        stage = Path(directory)
        for name in ("config.json", "generation.json", "review_frames.jpg", "review_sensor_frames.jpg"):
            shutil.copyfile(baseline / name, stage / name)
        (stage / "generated.mp4").symlink_to(baseline / "generated.mp4")
        dump(stage / "review_target.json", target)
        intensity_sheet(stage / "review_rain_frames.jpg", read_frames(baseline / "generated.mp4"))
        try:
            run_worker("agent_loop.weather", "review", stage, str(out / "feedback_review.log"))
        finally:
            for response in stage.glob("review_response_*.json"):
                shutil.copyfile(response, out / ("feedback_" + response.name))
        result = load(stage / "visual_review.json")
        result.update(parent_run=binding["run"], previous_plan=config["weather_plan"],
                      acceptance_target=target, note="A stricter acceptance target; original rendering metadata and review are unchanged.")
        dump(out / "feedback_review.json", result)
        shutil.copyfile(stage / "review_rain_frames.jpg", out / "feedback_rain_frames.jpg")

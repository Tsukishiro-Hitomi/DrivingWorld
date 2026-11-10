"""Plan, execute and review a bounded weather-editing run."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil

from ..qwen import qwen_json, run_worker
from ..runtime import dump, load, sha
from ..video import read_frames, sample_sheet
from .profiles import CHECKS, PROFILES, planning_profiles, validate_plan
from .sensor import apply_plan_patch, review_checks

SKILL = Path(__file__).with_name("SKILL.md")


def initialize(args, out):
    if args.max_retries not in (0, 1):
        raise ValueError("Weather retries must be bounded to zero or one")
    if getattr(args, "replacement_image", None) or getattr(args, "edited_first_frame", None):
        raise ValueError("Weather v1 accepts source video and text only; no image editor input")
    request = {
        "agent": "weather", "instruction": args.instruction,
        "source": str(args.source.resolve()), "source_sha256": sha(args.source),
        "profile": getattr(args, "profile", None), "max_retries": args.max_retries,
        "frames": 124, "fps": 24, "width": 832, "height": 480,
        "seed": 20261028, "steps": 8,
    }
    saved = load(out / "request.json") if (out / "request.json").exists() else None
    if saved is None or "plan_schema_version" in saved:
        request["plan_schema_version"] = saved["plan_schema_version"] if saved else 5
    if getattr(args, "feedback_from", None):
        from .feedback import feedback_binding
        request["feedback"] = feedback_binding(args.feedback_from, request)
    if (out / "request.json").exists():
        if request != load(out / "request.json"):
            raise ValueError("Resume weather inputs differ from the saved request")
        return
    frames = read_frames(args.source)
    sample_sheet(out / "source_frames.jpg", frames)
    dump(out / "request.json", request)


def plan(out):
    request = load(out / "request.json")
    version = request.get("plan_schema_version", 3)
    presets = planning_profiles(version)
    if request["profile"]:
        presets = {request["profile"]: presets[request["profile"]]}
    instruction = (
        SKILL.read_text() + "\nReturn one JSON plan only. The supplied sheet shows source frames with "
        "timestamps. Do not invent objects or motion. Select one of the listed presets. "
        "Return unresolved as an array, empty if executable. Include all required fields; "
        "Describe target weather and preserved content positively; do not repeat unrelated weather categories in the prose. "
        "default transition_start_s and transition_end_s to null for constant_from_frame_zero. "
        "Return source_environment with observed time_of_day and road_surface. Resolve target "
        "time_of_day and road_surface to concrete values, not preserve. The target edits must "
        "also contain sky_condition, illumination, visibility and snow_accumulation. "
        "Follow the selected profile's coupled sky/light/visibility values. "
        "The descriptions explain the plan; validated structured target fields drive H3. "
        "For schema version 4 include sensor_effects (the selected profile's complete camera state) "
        "and execution_emphasis with weather, sensor, preservation all initially standard. "
        "The request plan_schema_version selects the contract; absent means legacy version 3. "
        "For schema 5 rain profiles include rain_controls, sensor_effects.water_activity and poor "
        "target visibility. If feedback_review identifies inadequate rain intensity or runoff, "
        "use the evidence to revise execution controls and emphasis, not merely rename the task. "
        "Profile policies: " + str(presets)
    )
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": [
            {"type": "image", "image": str(out / "source_frames.jpg")},
            {"type": "text", "text": str(request)},
        ]},
    ]
    if (out / "feedback_review.json").exists():
        messages[1]["content"] += [
            {"type": "image", "image": str(out / "feedback_rain_frames.jpg")},
            {"type": "text", "text": "The following is the earlier generated output and its stricter re-evaluation, not the original scene. Repair the plan based on its actual failures: " + json.dumps(load(out / "feedback_review.json"))},
        ]
    for index in range(2):
        result = qwen_json(messages, out, "plan" if index == 0 else "plan_schema_fix", max_tokens=2200)
        result["prompt_version"] = version
        try:
            validate_plan(result, request["frames"] / request["fps"], request["profile"])
            break
        except ValueError as error:
            dump(out / f"plan_validation_{index}.json", {"error": str(error)})
            if index:
                raise
            messages += [{"role": "assistant", "content": json.dumps(result)},
                         {"role": "user", "content": f"The plan violates its contract: {error}. Return a complete corrected plan for the unchanged request."}]
    result.update(planner="Qwen/Qwen3.8-27B", skill_sha256=sha(SKILL),
                  request_sha256=sha(out / "request.json"))
    if (out / "feedback_review.json").exists():
        result["feedback_review_sha256"] = sha(out / "feedback_review.json")
    dump(out / "plan.json", result)


def review(attempt):
    config = load(attempt / "config.json")
    target = load(attempt / "review_target.json") if (attempt / "review_target.json").exists() else config["weather_plan"]
    criteria = review_checks(target)
    instruction = (
        "Review the labelled source/output full-frame samples at matching frame indices. "
        "This is global driving-video weather/lighting editing. Judge the visible evidence. "
        "Weather can change colors, brightness, reflections and distant visibility. Atmospheric "
        "occlusion is not by itself object removal. Mark a criterion UNKNOWN when it cannot "
        "be observed. weather_match must assess target sky, illumination, visibility and road "
        "surface as well as precipitation; wet asphalt alone is not sufficient for heavy rain. "
        "Preserve day/night does not mean preserve source sunshine. Sparse samples cannot certify exact camera poses, metric trajectories "
        "or all-frame temporal consistency. Do not replace uncertainty with a success. "
        "Return JSON with checks (an object containing exactly the named criteria, each an object "
        "with status PASS/FAIL/UNKNOWN and reason), observations (array), limitations (array). "
        "For sensor_effect_presence, require visible camera-glass droplets, film, haze, smears "
        "or the profile's optical glare response; wet roads and airborne rain alone do not qualify. "
        "For sensor_temporal_coherence inspect image-plane persistence, local refraction and slow "
        "surface motion separately from moving road/traffic. sensor_readability requires nearby "
        "road/traffic to remain recognizable; use UNKNOWN if evidence is insufficient. "
        "Criteria: " + ", ".join(criteria) + ".\n" + SKILL.read_text()
        + "\nFor this call you are the reviewer. Return the checks/observations/limitations JSON, not an editing plan."
    )
    from .rain import enabled as rain_enabled, normalize_review
    if rain_enabled(target):
        instruction += (
            "\nStrict torrential-rain acceptance: overcast skies, wet roads, lens droplets and color "
            "grading do not establish high rainfall intensity. Inspect clear scene regions for "
            "dense airborne streaks/sheets, active road impacts or substantial moving tire spray, "
            "and a depth-dependent moving rain veil. Return rain_evidence with exactly "
            "dense_airborne_rain, impact_or_spray, depth_rain_veil, each {status, reason}. "
            "PASS only when that cue is clearly visible; FAIL when observable regions lack it; "
            "UNKNOWN when occlusion prevents observation. precipitation_intensity is derived "
            "from these independent criteria. For surface_water_dynamics require repeated "
            "fresh glass impacts, merging, runoff and water-pattern renewal, not just a few "
            "persistent slow droplets. Do not assume the output satisfies its prompt."
        )
    content = [{"type": "image", "image": str(attempt / "review_frames.jpg")}]
    if target.get("prompt_version", 1) >= 4:
        content.append({"type": "image", "image": str(attempt / "review_sensor_frames.jpg")})
    if rain_enabled(target):
        content.append({"type": "image", "image": str(attempt / "review_rain_frames.jpg")})
    content.append({"type": "text", "text": str(target)})
    result = qwen_json([
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ], attempt, "review", max_tokens=3400 if rain_enabled(target) else 2400)
    checks = result.get("checks", {})
    if set(checks) != set(criteria):
        raise ValueError("Weather review is missing required criteria")
    for item in checks.values():
        if (not isinstance(item, dict) or item.get("status") not in {"PASS", "FAIL", "UNKNOWN"}
                or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
            raise ValueError("Weather reviewer returned an invalid criterion")
    if rain_enabled(target):
        normalize_review(result)
    result.update(reviewer="Qwen/Qwen3.8-27B, qualitative sampled-frame review",
                  generated_sha256=sha(attempt / "generated.mp4"),
                  sheet_sha256=sha(attempt / "review_frames.jpg"),
                  config_sha256=sha(attempt / "config.json"))
    if target.get("prompt_version", 1) >= 4:
        result["sensor_sheet_sha256"] = sha(attempt / "review_sensor_frames.jpg")
    if rain_enabled(target):
        result["rain_sheet_sha256"] = sha(attempt / "review_rain_frames.jpg")
    if (attempt / "review_target.json").exists():
        result["acceptance_target_sha256"] = sha(attempt / "review_target.json")
    dump(attempt / "visual_review.json", result)


def refine(attempt):
    config = load(attempt / "config.json")
    current = config["weather_plan"]
    evaluated = load(attempt / "visual_review.json")
    failed = {key for key, item in evaluated["checks"].items() if item["status"] == "FAIL"}
    if not failed:
        raise ValueError("A visual repair requires an observed FAIL criterion")
    structured = current.get("prompt_version", 1) >= 4
    instruction = (
        "Diagnose the observed FAIL criteria and propose one targeted correction for the unchanged request. "
        "Return JSON with retryable (boolean), correction (positive English target instruction), reason (string). "
        "Keep source, profile, target day/night, sky/light contract, timeline, seed, steps and weights fixed. "
        "UNKNOWN is not a diagnosed failure. Prefer describing wanted effects positively.\n" + SKILL.read_text()
    )
    if structured:
        instruction += (
            "\nAlso return diagnosis (array of objects with criterion and observation) and plan_patch (object). "
            "Every diagnosis criterion must be one of these actual failed checks: " + str(sorted(failed)) + ". "
            "Allowed patch fields ONLY: sensor_effects.coverage (light/moderate/heavy), "
            "sensor_effects.strength (mild/moderate/strong), and execution_emphasis.weather, "
            "execution_emphasis.sensor, execution_emphasis.preservation (standard/strong). "
            "Use nested objects for the patch. Strengthen inadequate camera effects; reduce excessive "
            "coverage when readability fails. A scene-weather failure can strengthen weather emphasis. "
            "Use a nonempty valid patch and a specific correction for a justified retry; never disable "
            "the requested sensor effect to pass a check."
        )
        if current.get("prompt_version", 1) >= 5 and current["profile"] in {"heavy_rain", "rainy_night"}:
            instruction += (
                " Additional allowed patch fields: sensor_effects.water_activity (high/very_high), "
                "rain_controls.rain_structure (dense_streaks/wind_driven_sheets), "
                "rain_controls.surface_impact and rain_controls.vehicle_spray (strong/very_strong). "
                "Dense airborne rainfall cannot be repaired solely by adding lens drops."
            )
    instruction += "\nFor this call return the repair JSON only, not a new task plan or review."
    content = [{"type": "image", "image": str(attempt / "review_frames.jpg")}]
    if structured:
        content.append({"type": "image", "image": str(attempt / "review_sensor_frames.jpg")})
    content.append({"type": "text", "text": json.dumps({"plan": current, "review": evaluated})})
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": content}]
    for index in range(2):
        result = qwen_json(messages, attempt, "refine" if index == 0 else "repair_schema_fix", max_tokens=1400)
        try:
            if type(result.get("retryable")) is not bool or not isinstance(result.get("reason"), str):
                raise ValueError("Invalid weather retry decision")
            if result["retryable"]:
                if not isinstance(result.get("correction"), str) or not result["correction"].strip():
                    raise ValueError("Retry requires a concrete correction")
                if structured:
                    diagnosis = result.get("diagnosis")
                    if (not isinstance(diagnosis, list) or not diagnosis
                            or any(not isinstance(item, dict) or item.get("criterion") not in failed
                                   or not isinstance(item.get("observation"), str) or not item["observation"].strip()
                                   for item in diagnosis)):
                        raise ValueError("Repair diagnosis must cite observed FAIL evidence")
                    patch = result.get("plan_patch")
                    proposed = apply_plan_patch(current, patch, config["frames"] / config["fps"])
                    if proposed == current:
                        raise ValueError("A structured retry must change an allowed plan control")
            break
        except ValueError as error:
            if index:
                raise
            messages += [{"role": "assistant", "content": json.dumps(result)},
                         {"role": "user", "content": f"Repair contract violation: {error}. Correct the repair JSON for the unchanged request."}]
    if structured:
        result.update(base_plan_sha256=sha(attempt / "plan.json"),
                      review_sha256=sha(attempt / "visual_review.json"),
                      config_sha256=sha(attempt / "config.json"),
                      generated_sha256=sha(attempt / "generated.mp4"))
    dump(attempt / "repair.json", result)


def run(args):
    from .executor import generate, verify

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".executor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(args, out)
        if "feedback" in load(out / "request.json") and not (out / "feedback_review.json").exists():
            from .feedback import audit
            dump(out / "state.json", {"stage": "reviewing_feedback", "agent": "weather"})
            audit(out)
        if not (out / "plan.json").exists():
            dump(out / "state.json", {"stage": "planning", "agent": "weather"})
            run_worker("agent_loop.weather", "plan", out, "planner.log")
        request, planned = load(out / "request.json"), load(out / "plan.json")
        validate_plan(planned, request["frames"] / request["fps"], request["profile"])
        if planned["request_sha256"] != sha(out / "request.json"):
            raise ValueError("Weather plan belongs to a different request")
        if args.plan_only:
            dump(out / "state.json", {"stage": "planned", "agent": "weather"})
            return
        correction = ""
        for index in range(args.max_retries + 1):
            attempt = out / f"attempt_{index:02d}"
            attempt.mkdir(exist_ok=True)
            dump(out / "state.json", {"stage": "generating", "agent": "weather", "attempt": index})
            if planned.get("prompt_version", 1) >= 4:
                generate(out, attempt, correction, plan=planned)
            else:
                generate(out, attempt, correction)
            dump(out / "state.json", {"stage": "verifying", "agent": "weather", "attempt": index})
            verdict = verify(attempt)
            if verdict != "FAIL" or index == args.max_retries:
                break
            if not (attempt / "repair.json").exists():
                run_worker("agent_loop.weather", "refine", attempt, "repair.log")
            repair = load(attempt / "repair.json")
            if not repair["retryable"]:
                break
            if planned.get("prompt_version", 1) >= 4:
                for key, file in [("base_plan_sha256", "plan.json"), ("review_sha256", "visual_review.json"),
                                  ("config_sha256", "config.json"), ("generated_sha256", "generated.mp4")]:
                    if repair.get(key) != sha(attempt / file):
                        raise ValueError("Repair belongs to a changed plan or output")
                planned = apply_plan_patch(planned, repair["plan_patch"], request["frames"] / request["fps"])
                planned["repair_history"] = planned.get("repair_history", []) + [{
                    "from_attempt": index, "repair_sha256": sha(attempt / "repair.json"),
                    "reason": repair["reason"], "diagnosis": repair["diagnosis"],
                }]
            correction = repair["correction"]
        for name in ("generated.mp4", "comparison.mp4"):
            shutil.copyfile(attempt / name, out / name)
        if (out / "acceptance_feedback.json").exists():
            acceptance = load(out / "acceptance_feedback.json")
            if acceptance.get("status") == "FAIL" and acceptance.get("generated_sha256") == sha(out / "generated.mp4"):
                verdict = "FAIL"
        dump(out / "state.json", {
            "stage": {"PASS": "complete", "FAIL": "completed_with_visual_issues", "UNKNOWN": "needs_review"}[verdict],
            "agent": "weather", "profile": planned["profile"], "selected_attempt": index,
            "technical_checks_passed": True, "sampled_visual_verdict": verdict,
            "video": str(out / "generated.mp4"), "generated_sha256": sha(out / "generated.mp4"),
        })
        print(f"Weather run finished: {planned['profile']}, sampled review {verdict}", flush=True)


def add_run_arguments(parser, source_required=True):
    parser.add_argument("--source", type=Path, required=source_required)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--max-retries", type=int, choices=(0, 1), default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--feedback-from", type=Path, help="Re-evaluate an earlier rain run before revising the plan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_run_arguments(sub.add_parser("run"))
    for name in ("plan", "review", "refine"):
        sub.add_parser(name).add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args)
        else:
            {"plan": plan, "review": review, "refine": refine}[args.command](args.out.resolve())
    except BlockingIOError:
        raise
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        dump(args.out / "state.json", {"stage": "failed", "agent": "weather", "error": str(error)})
        raise


if __name__ == "__main__":
    main()

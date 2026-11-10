#!/usr/bin/env python3
"""Plan a driving scene with Qwen and render a validated geometric proxy."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .intent import compare, extract
from .scene_compiler import compile_plan, inspect_scene
from .config import VLM_WEIGHTS

MODEL = str(VLM_WEIGHTS)
SYSTEM = """You are a driving scene planner. Output ONLY one JSON object.
Schema: {"duration": number, "ego_speed": number, "assumptions": [string], "events": [
  {"actor_id": string, "action": "cut_in" or "brake" or "cross", "start": number,
   "end": number, "distance": number, "speed": number, "direction": string, "final_speed": number}
]}. Use ONLY supported actions. At most one action per actor. duration is 1-20 seconds,
ego_speed is 0-30 meters/second. Event start and end satisfy 0 <= start < end <= duration.
distance is 3-80 meters in front of ego at event start. Vehicle speed is 0-35 m/s.
For cut_in direction is right_to_left or left_to_right; speed is vehicle forward m/s.
For brake speed is initial forward m/s and final_speed is 0..speed. If explicitly requested
as emergency braking, (speed-final_speed)/(end-start) must be at least 4 m/s².
For cross direction is right_to_left or left_to_right. Cross travels 10 meters laterally;
allow at least 4 seconds. Do not include speed for cross, it is computed from distance and time.
Only include relevant fields. Use unique actor IDs. Note unspecified
details as assumptions. Do not claim the scene has been rendered or validated.
Example for 'a right vehicle cuts in': {"duration":6,"ego_speed":5,"assumptions":[],"events":[
{"actor_id":"car","action":"cut_in","start":2,"end":5,"distance":15,"speed":5,"direction":"right_to_left"}]}"""


def load_model():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    return processor, model


def ask(processor, model, request):
    import torch

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": request},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1400, do_sample=False)
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]


def parse(raw):
    plan = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    if not isinstance(plan.get("assumptions"), list) or not all(
        isinstance(x, str) for x in plan["assumptions"]
    ):
        raise ValueError("assumptions must be a list of strings")
    return plan


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def semantic_warnings(request, plan):
    """Flag a few explicit Chinese directions; this is not a general language verifier."""
    warnings = []
    for event in plan["events"]:
        action = event["action"]
        direction = event.get("direction")
        if action == "cut_in":
            if (
                "右侧" in request or "右边" in request
            ) and direction == "left_to_right":
                warnings.append(
                    "Possible mismatch: user specified right-side vehicle but plan starts on left"
                )
            if (
                "左侧" in request or "左边" in request
            ) and direction == "right_to_left":
                warnings.append(
                    "Possible mismatch: user specified left-side vehicle but plan starts on right"
                )
        if action == "cross":
            if (
                "从右向左" in request or "由右向左" in request
            ) and direction == "left_to_right":
                warnings.append(
                    "Possible mismatch: requested right-to-left crossing but plan reversed it"
                )
            if (
                "从左向右" in request or "由左向右" in request
            ) and direction == "right_to_left":
                warnings.append(
                    "Possible mismatch: requested left-to-right crossing but plan reversed it"
                )
        if action == "brake" and "急刹" in request:
            warnings.append(
                "Urgent braking uses a project default threshold of 4 m/s²; review suitability"
            )
    warnings.append(
        "Natural-language intent completeness is not automatically verified"
    )
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request", required=True, help="Natural-language scene description"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New or empty directory"
    )
    parser.add_argument(
        "--plan", type=Path, help="Optional existing plan JSON; skips model inference"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "request.txt").write_text(args.request, encoding="utf-8")
    intent = extract(args.request)
    write_json(output / "intent.json", intent)
    if not intent["events"]:
        raise ValueError(
            "No supported explicit action recognized; inspect intent.json and clarify the request"
        )
    processor = model = None
    feedback = ""
    for attempt in range(3):
        if args.plan:
            raw = args.plan.read_text(encoding="utf-8")
        else:
            if model is None:
                processor, model = load_model()
            raw = ask(
                processor,
                model,
                args.request
                + "\nRequired explicit intent facts: "
                + json.dumps(intent["events"], ensure_ascii=False)
                + feedback,
            )
        (output / f"model_response_{attempt:02d}.txt").write_text(raw, encoding="utf-8")
        try:
            plan = parse(raw)
            scene, events = compile_plan(plan)
            intent_errors = compare(intent, plan)
            if intent_errors:
                raise ValueError("; ".join(intent_errors))
            break
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            write_json(output / f"failure_{attempt:02d}.json", {"error": str(exc)})
            feedback = (
                f"\nYour previous output was:\n{raw}\nValidation error: {exc}. "
                "Correct the invalid values while preserving the original user's action and direction. "
                "If an event ends after duration, increase duration or change the event times. "
                "Return the entire corrected JSON only."
            )
            if args.plan or attempt == 2:
                raise ValueError(f"Planning failed: {exc}; see {output}") from exc
    report = inspect_scene(scene, events)
    report["assumptions"] = plan["assumptions"]
    report["semantic_warnings"] = semantic_warnings(args.request, plan)
    report["intent_verification"] = {
        "status": "matched_extracted_facts",
        "coverage": intent["coverage"],
        "unverified": intent["unverified"],
    }
    write_json(output / "plan.json", plan)
    write_json(output / "scene.json", scene)
    write_json(output / "events.json", events)
    write_json(output / "validation.json", report)
    if not all(check["trajectory_goal_met"] for check in report["checks"]):
        raise ValueError("Trajectory goal check failed; inspect validation.json")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "proxy.py"),
            str(output / "scene.json"),
            str(output / "proxy.mp4"),
        ],
        check=True,
    )
    print(
        f"Scene rendered to {output}; geometry warnings: {len(report['warnings'])}; "
        f"semantic warnings: {len(report['semantic_warnings'])}. Review validation.json."
    )


if __name__ == "__main__":
    main()

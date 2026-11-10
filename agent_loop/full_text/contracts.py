"""Reference-free scene plans and native T2VA prompt contracts."""

import json
import re

from ..runtime import load, sha

PROMPT_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
SCENE_FIELDS = ("setting", "camera", "ego_motion", "traffic", "weather", "optical_degradation", "visibility", "timeline")
CORE_CHECKS = {"request_alignment", "scene_coherence", "temporal_consistency"}


def evaluate_review(result, plan, evidence):
    """Quarantine invalid model reviews without changing their recorded judgments."""
    from ..acceptance import assess

    full = [key for key, goal in plan["acceptance_criteria"].items() if goal["scope"] == "full_sequence"]
    try:
        assessed = assess(result, plan["acceptance_criteria"], observed_frames=evidence["observed_frames"],
                          frame_count=evidence["frame_count"], full_sequence_ids=full)
    except ValueError as error:
        return dict(review_valid=False, raw_result=result, review_error=str(error),
                    sampled_verdict="UNKNOWN", acceptance_verdict="UNKNOWN")
    return dict(assessed, review_valid=True)


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing " + name)


def validate(plan):
    if plan.get("status") not in {"ready", "needs_clarification", "needs_capability"}:
        raise ValueError("Invalid full-text readiness")
    text(plan.get("reason"), "reason")
    unresolved = plan.get("unresolved")
    if not isinstance(unresolved, list) or any(not isinstance(v, str) or not v.strip() for v in unresolved):
        raise ValueError("Invalid unresolved list")
    if (plan["status"] == "ready") != (unresolved == []):
        raise ValueError("Readiness conflicts with unresolved requirements")
    if plan["status"] != "ready":
        return
    scene, prompt, goals = plan.get("scene"), plan.get("prompt"), plan.get("acceptance_criteria")
    if not isinstance(scene, dict) or set(scene) != set(SCENE_FIELDS):
        raise ValueError("Full-text scene fields do not match the scene contract")
    for key in SCENE_FIELDS:
        text(scene[key], key)
    if not isinstance(prompt, dict) or set(prompt) != set(PROMPT_FIELDS):
        raise ValueError("T2VA uses exactly three prompt fields")
    for key in PROMPT_FIELDS:
        text(prompt[key], key)
    combined = compile_prompt(plan)
    if "[Shot 1]" not in prompt["integrated_multimodal_description"]:
        raise ValueError("T2VA description must identify the first continuous shot")
    if re.search(r"<(?:Picture|Video|Subject)\s*\d+>", combined, flags=re.I):
        raise ValueError("Full-text cannot reference nonexistent input assets")
    if not isinstance(plan.get("chosen_defaults"), list) or any(not isinstance(v, str) for v in plan["chosen_defaults"]):
        raise ValueError("Record full-text defaults separately from user requirements")
    if not isinstance(goals, dict) or not CORE_CHECKS <= set(goals) or not 3 <= len(goals) <= 12:
        raise ValueError("Missing full-text acceptance goals")
    for key, value in goals.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", key) or not isinstance(value, dict) or set(value) != {"description", "scope"}:
            raise ValueError("Invalid full-text acceptance goal")
        text(value["description"], "acceptance description")
        if value["scope"] not in {"sampled_frames", "full_sequence"}:
            raise ValueError("Invalid acceptance evidence scope")
    if goals["temporal_consistency"]["scope"] != "full_sequence":
        raise ValueError("Full-sequence temporal consistency must keep its evidence scope")


def compile_prompt(plan):
    return "\n\n".join(key + ":\n" + plan["prompt"][key] for key in PROMPT_FIELDS)


def verify_origin(out, plan):
    origin = plan["provenance"]
    name = origin["raw_response_file"]
    if not re.fullmatch(r"full_text_plan_response_[0-9]+\.json", name):
        raise ValueError("Invalid full-text planner response path")
    for key, filename in (("request_sha256", "request.json"), ("planner_input_sha256", "planner_input.json"),
                          ("raw_response_sha256", name)):
        if origin[key] != sha(out / filename):
            raise ValueError("Full-text planner input/response changed")
    raw = load(out / name)["text"]
    parsed = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    if parsed != {k: v for k, v in plan.items() if k != "provenance"}:
        raise ValueError("Full-text plan was modified after the agent response")

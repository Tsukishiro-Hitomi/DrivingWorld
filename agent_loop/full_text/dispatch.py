"""Plan and run a reference-free full-text driving scene."""

import argparse
import fcntl
import json
from pathlib import Path

from ..qwen import qwen_json, run_worker
from ..runtime import dump, load, sha
from .contracts import evaluate_review, validate, verify_origin

SKILL = Path(__file__).with_name("SKILL.md")


def initialize(args, out):
    if any(getattr(args, key, None) for key in ("source", "replacement_image", "edited_first_frame", "profile")):
        raise ValueError("full-text accepts text only; no source video, images or weather-profile override")
    request = dict(agent="full-text", instruction=args.instruction, source=None, source_sha256=None,
        frames=124, fps=24, width=832, height=480, seed=20261028, steps=50, output_audio=False,
        reference_image_count=0, reference_video_count=0, keyframe_count=0, max_retries=0)
    if getattr(args, "task_parameters", None) is not None:
        request["task_parameters"] = args.task_parameters
    if (out / "request.json").exists() and load(out / "request.json") != request:
        raise ValueError("Full-text resume inputs changed")
    dump(out / "request.json", request)


def plan(out):
    if (out / "planner_input.json").exists():
        raise ValueError("A full-text planning attempt is already recorded")
    messages = [{"role": "system", "content": SKILL.read_text() +
                 "\nThis is a planning call. Return your own full JSON scene plan and native T2VA prompt. "
                 "There are no images or videos. The user instruction is passed verbatim; "
                 "task_parameters, if supplied, are additional constraints from the outer agent."},
                {"role": "user", "content": json.dumps(load(out / "request.json"), ensure_ascii=False)}]
    dump(out / "planner_input.json", messages)
    result = qwen_json(messages, out, "full_text_plan", max_tokens=3000)
    if "provenance" in result:
        raise ValueError("The planning agent must not author provenance")
    try:
        validate(result)
    except ValueError:
        dump(out / "rejected_plan.json", result)
        raise
    raw = sorted(out.glob("full_text_plan_response_*.json"))[-1]
    result["provenance"] = dict(planner="Qwen/Qwen3.8-27B", authorship="model_generated",
        raw_response_file=raw.name, raw_response_sha256=sha(raw), planner_input_sha256=sha(out / "planner_input.json"),
        request_sha256=sha(out / "request.json"), skill_sha256=sha(SKILL))
    dump(out / "plan.json", result)


def review(out):
    planned, evidence = load(out / "plan.json"), load(out / "review_manifest.json")
    messages = [{"role": "system", "content": SKILL.read_text() +
        "\nThis is a review call, not planning. Return only checks, observations and structured limitations. "
        "The images are actual generated frames, with no supplied source video. Judge visibility against "
        "the customer's requested severity, not a generic requirement to keep lanes or traffic bodies readable. "
        "evidence_frames must contain integer IDs from evidence.observed_frames only. Never invent "
        "intermediate frame IDs or assume unshown frames were inspected."},
        {"role": "user", "content": [
            {"type": "image", "image": str(out / "review_frames.jpg")},
            {"type": "image", "image": str(out / "review_sequence.jpg")},
            {"type": "text", "text": json.dumps(dict(request=load(out / "request.json"), plan=planned,
                                                       evidence=evidence), ensure_ascii=False)},
        ]}]
    dump(out / "review_input.json", messages)
    result = qwen_json(messages, out, "full_text_review", max_tokens=3000)
    record_review(out, result)


def record_review(out, result):
    planned, evidence = load(out / "plan.json"), load(out / "review_manifest.json")
    result = evaluate_review(result, planned, evidence)
    result.update(generated_sha256=sha(out / "generated.mp4"), manifest_sha256=sha(out / "review_manifest.json"),
                  plan_sha256=sha(out / "plan.json"), review_type="Qwen sampled-frame review, not independent acceptance")
    dump(out / "visual_review.json", result)


def reassess(out):
    """Validate the recorded response only; do not call the model or sample video."""
    responses = sorted(out.glob("full_text_review_response_*.json"))
    if not responses:
        raise ValueError("No recorded full-text review response")
    raw = load(responses[-1])["text"]
    result = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    record_review(out, result)


def run(args):
    from .executor import generate
    from .verifier import verify

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".full_text.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(args, out)
        if not (out / "plan.json").exists():
            if (out / "planner_input.json").exists():
                raise ValueError("A previous full-text plan failed; preserve it in this run")
            dump(out / "state.json", dict(stage="planning", agent="full-text", generation_started=False))
            run_worker("agent_loop.full_text", "plan", out, "planner.log")
        planned = load(out / "plan.json")
        validate(planned)
        verify_origin(out, planned)
        if planned["status"] != "ready" or getattr(args, "plan_only", False):
            dump(out / "state.json", dict(stage="planned" if planned["status"] == "ready" else planned["status"],
                agent="full-text", generation_started=False, unresolved=planned["unresolved"]))
            return
        dump(out / "state.json", dict(stage="generating", agent="full-text", feedback_loop=False, max_retries=0))
        generate(out)
        dump(out / "state.json", dict(stage="verifying", agent="full-text", feedback_loop=False))
        result = verify(out)
        verdict = result["acceptance_verdict"]
        dump(out / "state.json", dict(stage={"PASS": "complete", "FAIL": "completed_with_visual_issues", "UNKNOWN": "needs_review"}[verdict],
            agent="full-text", technical_checks_passed=True, sampled_visual_verdict=result["sampled_verdict"],
            acceptance_verdict=verdict, independent_acceptance="NOT_ESTABLISHED", feedback_loop=False, max_retries=0,
            review_valid=result["review_valid"], review_error=result.get("review_error"),
            generated_sha256=sha(out / "generated.mp4"), video=str(out / "generated.mp4")))
        print(json.dumps(load(out / "state.json"), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "review", "reassess"):
        sub.add_parser(name).add_argument("--out", type=Path, required=True)
    command = sub.add_parser("run")
    command.add_argument("--instruction", required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args) if args.command == "run" else {"plan": plan, "review": review, "reassess": reassess}[args.command](args.out.resolve())
    except BlockingIOError:
        raise
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        dump(args.out / "state.json", dict(stage="failed", agent="full-text", error=str(error)))
        raise


if __name__ == "__main__":
    main()

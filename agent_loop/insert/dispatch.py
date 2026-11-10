"""Plan, generate and assess one video-plus-text object insertion."""

import argparse
import fcntl
import json
from pathlib import Path

from ..qwen import qwen_json, run_worker
from ..runtime import dump, load, sha
from ..video import read_frames, sample_sheet
from .contracts import CHECKS, validate, verify_origin

SKILL = Path(__file__).with_name("SKILL.md")


def initialize(args, out):
    if getattr(args, "replacement_image", None) or getattr(args, "edited_first_frame", None):
        raise ValueError("Insertion v1 accepts the original video and text only")
    limit = getattr(args, "max_repairs", 0)
    if type(limit) is not int or not 0 <= limit <= 3:
        raise ValueError("Insertion max_repairs must be between zero and three")
    request = dict(agent="insert", instruction=args.instruction, source=str(args.source.resolve()),
                   source_sha256=sha(args.source), frames=124, fps=24, width=832, height=480,
                   seed=20261028, steps=8, feedback_loop=limit > 0, max_retries=0)
    if limit:
        request["max_repairs"] = limit
    if getattr(args, "task_parameters", None) is not None:
        request["task_parameters"] = args.task_parameters
    if (out / "request.json").exists():
        if load(out / "request.json") != request:
            raise ValueError("Insertion resume inputs changed")
        return
    frames = read_frames(args.source)
    sample_sheet(out / "source_frames.jpg", frames)
    dump(out / "request.json", request)


def plan(out):
    if (out / "planner_input.json").exists() or list(out.glob("insert_plan_response_*.json")):
        raise ValueError("An insertion planning attempt is already recorded; no automatic replanning")
    request = load(out / "request.json")
    messages = [{"role": "system", "content": SKILL.read_text() +
                 "\nThis is a planning call. Return your own JSON plan for the supplied request and actual scene. "
                 "The host has not chosen the appearance, entry point, trajectory or timing. "
                 "task_parameters, when present, are constraints from the outer planning agent; preserve their meaning."},
                {"role": "user", "content": [
                    {"type": "image", "image": str(out / "source_frames.jpg")},
                    {"type": "text", "text": json.dumps(request, ensure_ascii=False)},
                ]}]
    dump(out / "planner_input.json", messages)
    result = qwen_json(messages, out, "insert_plan", max_tokens=3000)
    try:
        if "provenance" in result:
            raise ValueError("The agent must not author provenance metadata")
        validate(result, request)
    except ValueError:
        dump(out / "rejected_plan.json", result)
        raise
    raw = sorted(out.glob("insert_plan_response_*.json"))[-1]
    result["provenance"] = dict(planner="Qwen/Qwen3.8-27B", authorship="model_generated",
        request_sha256=sha(out / "request.json"), planner_input_sha256=sha(out / "planner_input.json"),
        source_sheet_sha256=sha(out / "source_frames.jpg"), raw_response_file=raw.name,
        raw_response_sha256=sha(raw), skill_sha256=sha(SKILL))
    dump(out / "plan.json", result)


def review(out):
    from .verifier import assess_insertion

    planned, evidence = load(out / "plan.json"), load(out / "review_manifest.json")
    content = [{"type": "image", "image": str(out / filename)}
               for filename in ("review_frames.jpg", "review_event_frames.jpg")]
    content.append({"type": "text", "text": json.dumps({"request": load(out / "request.json"),
        "plan": planned, "evidence": evidence, "acceptance_criteria": planned["acceptance_criteria"]}, ensure_ascii=False)})
    messages = [{"role": "system", "content": SKILL.read_text() +
        "\nThis is a review call, NOT planning. Return checks, observations and structured limitations only. "
        "The first sheet contains source/output full-frame pairs. The second shows actual output event frames, "
        "with frame IDs and seconds. Inspect these literally. Verify the requested mechanics (rolling versus "
        "sliding, articulation versus a rigid sprite, wheeled motion versus floating) rather than just category. "
        "Supplied frame IDs are the only observed visual evidence. Do not claim all-frame temporal verification."},
        {"role": "user", "content": content}]
    dump(out / "review_input.json", messages)
    result = qwen_json(messages, out, "insert_review", max_tokens=3200)
    assessed = assess_insertion(result, evidence)
    assessed.update(generated_sha256=sha(out / "generated.mp4"), plan_sha256=sha(out / "plan.json"),
                    manifest_sha256=sha(out / "review_manifest.json"),
                    review_type="Qwen qualitative sampled-frame review, with limitation and coverage gates")
    dump(out / "visual_review.json", assessed)


def run(args):
    from .executor import generate
    from .verifier import verify

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".insertion.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize(args, out)
        if not (out / "plan.json").exists():
            if (out / "planner_input.json").exists():
                raise ValueError("Previous insertion planning failed; automatic replanning is disabled")
            dump(out / "state.json", dict(stage="planning", agent="insert", generation_started=False))
            run_worker("agent_loop.insert", "plan", out, "planner.log")
        planned = load(out / "plan.json")
        validate(planned, load(out / "request.json"))
        verify_origin(out, planned)
        if planned["status"] != "ready" or getattr(args, "plan_only", False):
            dump(out / "state.json", dict(stage="planned" if planned["status"] == "ready" else planned["status"],
                                          agent="insert", generation_started=False, unresolved=planned["unresolved"]))
            return
        if getattr(args, "max_repairs", 0):
            from .loop import run as repair_loop

            repair_loop(out, max_repairs=args.max_repairs)
            return
        dump(out / "state.json", dict(stage="generating", agent="insert", feedback_loop=False, max_retries=0))
        generate(out)
        dump(out / "state.json", dict(stage="verifying", agent="insert", feedback_loop=False))
        assessed = verify(out)
        verdict = assessed["acceptance_verdict"]
        dump(out / "state.json", dict(stage={"PASS": "complete", "FAIL": "completed_with_visual_issues",
             "UNKNOWN": "needs_review"}[verdict], agent="insert", technical_checks_passed=True,
             sampled_visual_verdict=assessed["sampled_verdict"], acceptance_verdict=verdict,
             independent_acceptance="NOT_ESTABLISHED", feedback_loop=False, max_retries=0,
             generated_sha256=sha(out / "generated.mp4"), video=str(out / "generated.mp4")))
        print(json.dumps(load(out / "state.json"), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "review", "repair-plan", "compare"):
        sub.add_parser(name).add_argument("--out", type=Path, required=True)
    command = sub.add_parser("run")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--instruction", required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--plan-only", action="store_true")
    command.add_argument("--max-repairs", type=int, choices=range(4), default=3)
    repair_command = sub.add_parser("repair")
    repair_command.add_argument("--from-run", type=Path, required=True)
    repair_command.add_argument("--out", type=Path, required=True)
    repair_command.add_argument("--max-repairs", type=int, choices=range(4), default=3)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args)
        elif args.command == "repair":
            from .loop import run as repair_loop

            repair_loop(args.out, args.max_repairs, baseline=args.from_run)
        elif args.command in {"repair-plan", "compare"}:
            from .loop import repair, compare

            {"repair-plan": repair, "compare": compare}[args.command](args.out.resolve())
        else:
            {"plan": plan, "review": review}[args.command](args.out.resolve())
    except BlockingIOError:
        raise
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        previous = load(args.out / "state.json") if (args.out / "state.json").exists() else {}
        previous.update(stage="failed", agent="insert", error=str(error))
        dump(args.out / "state.json", previous)
        raise


if __name__ == "__main__":
    main()

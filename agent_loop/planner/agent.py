"""Planning-only default entry, plus explicit one-shot execution and joint review."""

import argparse
import fcntl
import json
from pathlib import Path

from ..qwen import qwen_json, run_worker
from ..runtime import dump, load, sha
from ..video import read_frames, sample_sheet
from .contracts import validate, verify_origin
from .registry import capabilities

SKILL = Path(__file__).with_name("SKILL.md")


def build(out):
    if (out / "planner_input.json").exists() or list(out.glob("outer_plan_response_*.json")):
        raise ValueError("A planning attempt is already recorded; preserve it and use a new run for another agent plan")
    request, registry = load(out / "request.json"), load(out / "capabilities.json")
    content = ([{"type": "image", "image": str(out / "source_frames.jpg")}] if request.get("source_video") else [])
    if request.get("replacement_image"):
        content.append({"type": "image", "image": request["replacement_image"]})
    content.append({"type": "text", "text": json.dumps({"request": request, "capabilities": registry}, ensure_ascii=False)})
    messages = [{"role": "system", "content": SKILL.read_text() + "\nReturn your own task decomposition and dependencies as one JSON object. The host has not selected a task order."},
                {"role": "user", "content": content}]
    dump(out / "planner_input.json", messages)
    result = qwen_json(messages, out, "outer_plan", max_tokens=2600)
    try:
        if "provenance" in result:
            raise ValueError("Planner must not author its own provenance metadata")
        validation = validate(result, registry, request)
    except ValueError as error:
        dump(out / "rejected_plan.json", result)
        dump(out / "state.json", {"stage": "invalid_plan", "error": str(error)})
        raise
    raw = sorted(out.glob("outer_plan_response_*.json"))[-1]
    result["provenance"] = {
        "planner": "Qwen/Qwen3.8-27B", "authorship": "model_generated",
        "raw_response_file": raw.name, "raw_response_sha256": sha(raw), "planner_input_sha256": sha(out / "planner_input.json"),
        "source_sheet_sha256": sha(out / "source_frames.jpg") if request.get("source_video") else None,
        "request_sha256": sha(out / "request.json"), "capabilities_sha256": sha(out / "capabilities.json"),
        "skill_sha256": sha(SKILL),
    }
    dump(out / "plan.json", result)
    dump(out / "validation.json", validation)


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".planner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        photo = getattr(args, "replacement_image", None)
        source = getattr(args, "source", None)
        request = {"instruction": args.instruction, "source_video": str(source.resolve()) if source else None,
                   "source_sha256": sha(source) if source else None,
                   "profile_hint": getattr(args, "profile", None),
                   "replacement_image": str(photo.resolve()) if photo else None,
                   "replacement_image_sha256": sha(photo) if photo else None,
                   "execution_policy": {"feedback_loop": False, "max_retries": 0}}
        registry = capabilities()
        for name, data in (("request.json", request), ("capabilities.json", registry)):
            if (out / name).exists() and load(out / name) != data:
                raise ValueError(f"Planner inputs changed: {name}; preserve this run and use a new output directory")
            dump(out / name, data)
        if not (out / "plan.json").exists():
            if (out / "planner_input.json").exists():
                raise ValueError("A previous planning attempt failed; automatic replanning is disabled")
            if source:
                sample_sheet(out / "source_frames.jpg", read_frames(source))
            dump(out / "state.json", {"stage": "planning", "generation_started": False})
            run_worker("agent_loop.planner", "build", out, "planner.log")
        plan = load(out / "plan.json")
        validation = validate(plan, registry, request)
        verify_origin(out, plan)
        dump(out / "state.json", {"stage": "planned" if plan["status"] == "ready" else plan["status"],
                                   "generation_started": False, "plan_sha256": sha(out / "plan.json"),
                                   "execution_order": validation["execution_order"],
                                   "capability_violations": validation["capability_violations"]})
        print(json.dumps(load(out / "state.json"), ensure_ascii=False), flush=True)


def review(out):
    from ..acceptance import assess

    plan = load(out / "plan.json")
    manifest = load(out / "joint_review/manifest.json")
    criteria = {item["id"]: item["description"] for item in plan["final_acceptance"]}
    content = [{"type": "image", "image": item["sheet"]} for item in manifest["references"]]
    content.append({"type": "text", "text": json.dumps({"request": load(out / "request.json"),
                     "plan": plan, "references": manifest, "criteria": criteria}, ensure_ascii=False)})
    messages = [{"role": "system", "content": (
        "Review the actual original/intermediate/final video frame sheets against every final goal. "
        "When no original source exists, evaluate generated content against the text and scene goals, "
        "not against invented original-video preservation. Severe requested obscuration is an intended feature. "
        "A later stage can undo an earlier edit. Distinguish allowed changes and optical/atmospheric "
        "occlusion from unintended changes. Treat the plan as desired behavior, never proof of success. "
        "Return JSON with checks keyed by the exact supplied criterion IDs, each {status: PASS/FAIL/UNKNOWN, reason}, "
        "observations (array), limitations (array of {criteria: affected criterion IDs, impact: "
        "informational/unknown/fail, reason: string}). Missing evidence for a required goal maps to unknown; "
        "an observed violation maps to fail. These limitations affect acceptance, not just notes. "
        "Sampled frames cannot certify all-frame timing or metric 3D. "
        "Use UNKNOWN for unobservable evidence. Do not infer severe rain from wet asphalt or lens droplets alone. "
        "This stage only reviews and must not replan or generate." )},
                {"role": "user", "content": content}]
    result = qwen_json(messages, out, "joint_review", max_tokens=2400)
    result = assess(result, criteria)
    result.update(manifest_sha256=sha(out / "joint_review/manifest.json"), plan_sha256=sha(out / "plan.json"),
                  review_type="Qwen qualitative sampled-frame assessment; not independent certification")
    dump(out / "joint_review.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planned = sub.add_parser("plan")
    planned.add_argument("--source", type=Path)
    planned.add_argument("--instruction", required=True)
    planned.add_argument("--out", type=Path, required=True)
    planned.add_argument("--replacement-image", type=Path)
    for action in ("build", "execute", "review"):
        sub.add_parser(action).add_argument("--out", type=Path, required=True)
    recovery = sub.add_parser("reassess")
    recovery.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            prepare(args)
        elif args.command in {"execute", "reassess"}:
            from .executor import execute
            execute(args.out.resolve(), assessment_only=args.command == "reassess")
        else:
            {"build": build, "review": review}[args.command](args.out.resolve())
    except BlockingIOError:
        raise
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "state.json"
        state = load(path) if path.exists() else {}
        state.update(error=str(error))
        if state.get("stage") not in {"invalid_plan", "needs_capability", "needs_clarification"}:
            state["stage"] = "failed"
        dump(path, state)
        raise


if __name__ == "__main__":
    main()

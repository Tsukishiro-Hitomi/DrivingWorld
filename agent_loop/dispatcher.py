"""Route a driving edit before invoking any task-specific detector or renderer."""

import argparse
from argparse import Namespace
import fcntl
import json
from pathlib import Path

from .qwen import qwen_json, run_worker
from .runtime import dump, load, sha
from .weather.profiles import PROFILES


def classify(out):
    request = load(out / "dispatch_request.json")
    result = qwen_json([
        {"role": "system", "content": (
            "Route the meaning of the user's driving video edit. Return JSON with agent "
            "(full-text, weather, replace, insert, composite or unsupported), profile (one supported weather preset "
            "or null), and reason (string). If no source video is supplied, choose full-text for creating "
            "an entire new scene from text, including any requested weather or event. It uses zero references. "
            "Return profile null for full-text; its complete environment is planned from the original instruction. "
            "Never treat missing input as an existing editable video. If a source video exists, route "
            "editing operations as follows. weather edits global environment or illumination: "
            "heavy rain, dense fog, heavy snow, frontal strong backlighting, rainy night or "
            "sandstorm. replace changes the identity/category/appearance of an existing vehicle. "
            "Weather requests include camera-lens/protective-glass contamination or optical effects "
            "coupled to these environments: droplets, water film, condensation, adhered slush, "
            "dust smears, glare and bloom. Route such camera-weather requests to weather, not vehicle replacement. "
            "This route edits RGB camera video; LiDAR/radar sensor simulation is unsupported. "
            "insert adds one new object that did not exist in the source, including unusual objects, "
            "animals, strollers or rolling rocks; it does not replace an existing vehicle. "
            "Requests combining two or more edit kinds use composite; never discard an operation. "
            "Camera/motion changes of existing objects and unsupported weather variants use unsupported. "
            "Select by meaning, not by an isolated weather word in a vehicle description. "
            "Available weather profiles: " + ", ".join(PROFILES))},
        {"role": "user", "content": json.dumps({"instruction": request["instruction"],
                                                  "has_source_video": bool(request.get("source")),
                                                  "requested_profile": request.get("profile")}, ensure_ascii=False)},
    ], out, "dispatch", max_tokens=500)
    validate_route(result, request.get("profile"), has_source=bool(request.get("source")))
    result.update(request_sha256=sha(out / "dispatch_request.json"), planner="Qwen/Qwen3.8-27B")
    dump(out / "dispatch.json", result)


def validate_route(route, requested_profile=None, has_source=True):
    if route.get("agent") not in {"full-text", "weather", "replace", "insert", "composite", "unsupported"}:
        raise ValueError("Unknown dispatcher agent")
    if not isinstance(route.get("reason"), str) or not route["reason"].strip():
        raise ValueError("Dispatcher must explain its route")
    if route["agent"] == "full-text" and has_source:
        raise ValueError("full-text cannot silently discard a supplied source video")
    if not has_source and route["agent"] in {"weather", "replace", "insert"}:
        raise ValueError("Editing agents require an actual source video")
    if route["agent"] == "weather":
        if route.get("profile") not in PROFILES:
            raise ValueError("Unsupported dispatcher weather profile")
        if requested_profile and route["profile"] != requested_profile:
            raise ValueError("Dispatcher profile conflicts with --profile")
    elif route["agent"] == "composite":
        if requested_profile and route.get("profile") not in {None, requested_profile}:
            raise ValueError("Composite route conflicts with the requested weather profile")
    elif requested_profile:
        raise ValueError("--profile is only valid for a weather request")


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".dispatcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = getattr(args, "source", None)
        request = {"instruction": args.instruction, "source": str(source.resolve()) if source else None,
                   "source_sha256": sha(source) if source else None, "agent": args.agent, "profile": args.profile}
        if getattr(args, "feedback_from", None):
            request["feedback_from"] = str(args.feedback_from.resolve())
        path = out / "dispatch_request.json"
        if path.exists() and load(path) != request:
            raise ValueError("Dispatcher resume inputs have changed")
        dump(path, request)
        if not (out / "dispatch.json").exists():
            if args.agent == "auto":
                run_worker("agent_loop.dispatcher", "classify", out, "dispatch.log")
            else:
                if args.agent == "replace" and args.profile:
                    raise ValueError("--profile is only valid for weather")
                dump(out / "dispatch.json", {
                    "agent": args.agent, "profile": args.profile,
                    "reason": "Explicit agent selection", "request_sha256": sha(path),
                })
        route = load(out / "dispatch.json")
        if route["request_sha256"] != sha(path):
            raise ValueError("Dispatcher plan belongs to different inputs")
        if args.agent == "auto":
            validate_route(route, args.profile, has_source=bool(source))
        if route["agent"] in {"composite", "planner"}:
            from .planner.agent import prepare
            prepare(args)
            return
        if route["agent"] == "unsupported":
            dump(out / "state.json", {"stage": "unsupported",
                                       "reason": route["reason"]})
            raise ValueError(f"No executable single-agent route: {route['reason']}")
        values = vars(args).copy()
        values.update(out=out, profile=route.get("profile"))
        if route["agent"] == "full-text":
            from .full_text.dispatch import run as full_text

            full_text(Namespace(**values))
        elif not source:
            raise ValueError("Editing agents require --source; use full-text to create a new scene")
        elif route["agent"] == "weather":
            from .weather.dispatch import run as weather

            weather(Namespace(**values))
        elif route["agent"] == "replace":
            if args.plan_only:
                raise ValueError("--plan-only currently supports weather; replacement was not executed")
            from .replace.dispatch import run as replace

            values.update(conditioning="auto", geometry=None, tracking_cache=None)
            replace(Namespace(**values))
        elif route["agent"] == "insert":
            from .insert.dispatch import run as insert

            insert(Namespace(**values))
        else:
            raise ValueError("Agent is outside the fixed executor allowlist")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("classify").add_argument("--out", type=Path, required=True)
    run_parser = sub.add_parser("run")
    from .weather.dispatch import add_run_arguments

    add_run_arguments(run_parser, source_required=False)
    run_parser.add_argument("--agent", choices=("auto", "full-text", "replace", "insert", "weather", "planner"), default="auto")
    run_parser.add_argument("--replacement-image", type=Path)
    run_parser.add_argument("--edited-first-frame", type=Path)
    run_parser.add_argument("--max-repairs", type=int, choices=range(4), default=3,
                            help="Insertion-only local repair limit; outer task graph is not replanned")
    args = parser.parse_args()
    try:
        classify(args.out.resolve()) if args.command == "classify" else run(args)
    except BlockingIOError:
        raise
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        existing = load(args.out / "state.json") if (args.out / "state.json").exists() else {}
        if existing.get("stage") not in {"needs_composition", "unsupported"}:
            dump(args.out / "state.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()

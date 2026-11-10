"""Dispatch a natural-language driving edit to a resumable task executor."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from .runtime import ROOT, dump, load, sha

AGENT_SKILL = Path(__file__).with_name("SKILL.md")
from ..qwen import QWEN_PYTHON, qwen_json


def route(out):
    request = load(out / "request.json")
    candidates = load(out / "candidates.json")
    instruction = (
        "Route the user's driving edit and visually ground the selected vehicle in the supplied frame. "
        "Available execution agent: vehicle_replacement. Other task kinds are object_insertion, "
        "object_motion_editing, ego_motion_editing, unsupported. Choose the task from the meaning "
        "of the request. Use the candidate vehicle boxes and IDs to identify the intended object. "
        "Return JSON with task_type, target_id (exact candidate ID or null if ambiguous), "
        "target_description (English), replacement_category, replacement_description (English), "
        "appearance_source (text_generated or user_reference), "
        "start_frame (integer), first_frame_edit_prompt (English), motion_description (English), reason. "
        "The edit prompt must identify the exact target by position, current appearance and pixel bbox, "
        "describe the replacement category, and preserve surrounding scene and other vehicles. "
        "The motion description preserves the selected input vehicle's existing movement. "
        "Infer frame zero when the user requests replacement from the first frame. "
        "replacement_description describes the replacement body, color and distinguishing visual features, "
        "without scene-edit commands or invented reference images. Honor the requested vehicle category "
        "and all explicit attributes. For unspecified attributes choose plausible category-level defaults; "
        "do not substitute another category or add unrequested emergency actions, sirens or flashing lights. "
        "Preserve the actual source weather and illumination, including any prior edits. Do not demand a user reference. "
        "When task_parameters are supplied by the outer planning agent, use them as the task constraints: "
        "copy replacement_category exactly, and visually ground the target_description and target_selector "
        "against actual candidate boxes. target_selector is a natural-language selection cue, not a fixed object ID. "
        "Return null target_id when visual selection is ambiguous."
    )
    has_reference = bool(request.get("replacement_image"))
    appearance_source = "user_reference" if has_reference else "text_generated"
    instruction += (
        f" The actual appearance_source is {appearance_source}. "
        "Image 1 is the source frame used to select the existing target."
    )
    content = [{"type": "image", "image": str(out / "source_first_frame.png")}]
    if has_reference:
        instruction += (
            " Image 2 is the customer-supplied replacement vehicle appearance. "
            "Use its body type, color and distinguishing details in first_frame_edit_prompt. "
            "Ask the image editor to match Image 2 while retaining Image 1's camera perspective. "
            "Infer the replacement category from the customer image when needed."
        )
        content.append({"type": "image", "image": str(out / "replacement_reference.png")})
    payload = json.dumps({"request": request["instruction"], "candidates": candidates,
                          "task_parameters": request.get("task_parameters"),
                          "appearance_source": appearance_source}, ensure_ascii=False)
    content.append({"type": "text", "text": payload})
    messages = [
        {"role": "system", "content": instruction + "\n\n" + AGENT_SKILL.read_text()},
        {"role": "user", "content": content},
    ]
    answer = qwen_json(messages, out, "route")
    if answer.get("task_type") != "vehicle_replacement":
        dump(out / "route.json", answer)
        raise ValueError(f"Selected agent is not implemented: {answer.get('task_type')}")
    valid_ids = {item["object_id"] for item in candidates}
    if answer.get("target_id") not in valid_ids:
        raise ValueError("Target could not be uniquely grounded in candidate vehicles")
    if answer.get("start_frame") != 0:
        raise ValueError("This executor currently supports frame-zero replacement")
    answer["appearance_source"] = appearance_source
    for key in ("target_description", "first_frame_edit_prompt", "motion_description",
                "replacement_category", "replacement_description"):
        if not isinstance(answer.get(key), str) or not answer[key].strip():
            raise ValueError(f"Missing planner field: {key}")
    constraints = request.get("task_parameters")
    if constraints and answer["replacement_category"] != constraints["replacement_category"]:
        raise ValueError("Replacement agent changed the outer planner's requested category")
    answer["agent"] = "vehicle_replacement"
    answer["planner"] = "Qwen/Qwen3.8-27B"
    answer["skill_sha256"] = sha(AGENT_SKILL)
    answer["request_sha256"] = sha(out / "request.json")
    answer["candidates_sha256"] = sha(out / "candidates.json")
    dump(out / "route.json", answer)


def review(out):
    request = load(out / "request.json")
    route = load(out / "route.json")
    instruction = (
        "Review the actual generated video samples against the source. The first image is the "
        "generated first frame. The second is a labelled sheet of corresponding source/output crops. "
        f"The selected vehicle in the source crops should be replaced with a {route['replacement_category']} "
        "from frame zero and retain coherent appearance throughout. Inspect the images literally. "
        "Return JSON with boolean replacement_visible_in_first_frame, replacement_category_correct, "
        "target_identity_consistent, wrong_vehicle_replaced, plus observations (array of strings), "
        "visible_artifacts (array of strings), and limitations (array of strings). "
        "Use false for a success field when the evidence is unclear. Do not infer exact 3D dimensions, "
        "metric trajectory, physical safety or unobserved frames from these images."
    )
    content = [
        {"type": "image", "image": str(out / "generated_first_frame.png")},
        {"type": "image", "image": str(out / "review_frames.jpg")},
    ]
    if request.get("replacement_image"):
        instruction += (
            " Image 3 is the customer-supplied vehicle reference. Also return a boolean "
            "reference_appearance_matched. Compare visible body type, color, cargo/cab layout and "
            "distinctive features, allowing the viewpoint to differ."
        )
        content.append({"type": "image", "image": str(out / "replacement_reference.png")})
    content.append({"type": "text", "text": json.dumps({
        "request": request["instruction"], "target_description": route["target_description"],
        "replacement_description": route.get("replacement_description"),
        "motion_description": route["motion_description"],
    }, ensure_ascii=False)})
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]
    result = qwen_json(messages, out, "review", max_tokens=1000)
    for key in ("replacement_visible_in_first_frame", "replacement_category_correct",
                "target_identity_consistent", "wrong_vehicle_replaced"):
        if type(result.get(key)) is not bool:
            raise ValueError(f"Invalid reviewer field: {key}")
    if request.get("replacement_image") and type(result.get("reference_appearance_matched")) is not bool:
        raise ValueError("Invalid reviewer field: reference_appearance_matched")
    result["reviewer"] = "Qwen/Qwen3.8-27B, qualitative sampled-frame review"
    dump(out / "visual_review.json", result)


def resolve_conditioning(args):
    requested = getattr(args, "conditioning", "auto")
    reference = getattr(args, "replacement_image", None)
    edited = getattr(args, "edited_first_frame", None)
    if reference is not None:
        if requested == "text-only":
            raise ValueError("--replacement-image conflicts with --conditioning text-only")
        return "appearance-guided"
    if requested != "auto":
        if requested == "text-only" and edited is not None:
            raise ValueError("Text-only mode accepts no edited first-frame image")
        return requested
    saved = args.out / "request.json"
    if saved.exists():
        previous = load(saved)
        if previous.get("replacement_image"):
            raise ValueError("Resume requires the original --replacement-image")
        return previous.get("conditioning", "appearance-guided")
    return "appearance-guided" if edited is not None else "text-only"


def run(args):
    from .executor import initialize, prepare, generate, generate_text_only, verify

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".executor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.conditioning = resolve_conditioning(args)
        initialize(args, out)
        if not (out / "route.json").exists():
            print("Routing request with local Qwen3.8-27B", flush=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["HF_ENABLE_PARALLEL_LOADING"] = "true"
            env["HF_PARALLEL_LOADING_WORKERS"] = "6"
            with (out / "planner.log").open("w") as log:
                subprocess.run(
                    [QWEN_PYTHON, "-m", "agent_loop.replace", "route", "--out", str(out)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                )
        dispatch = load(out / "route.json")
        if dispatch["agent"] != "vehicle_replacement":
            raise ValueError("Unknown execution agent")
        for key, filename in (("request_sha256", "request.json"), ("candidates_sha256", "candidates.json")):
            if key in dispatch and dispatch[key] != sha(out / filename):
                raise ValueError(f"Replacement route belongs to different inputs: {filename}")
        constraints = getattr(args, "task_parameters", None)
        if constraints and dispatch.get("replacement_category") != constraints["replacement_category"]:
            raise ValueError("Replacement route conflicts with the outer planner's category")
        if args.conditioning == "text-only":
            if args.edited_first_frame is not None:
                raise ValueError("Text-only mode accepts no edited first-frame image")
            generate_text_only(out)
            verify(out)
            return
        prepare(out)
        if args.edited_first_frame is None and not (out / "edited_first_frame.png").exists():
            dump(out / "state.json", {
                "stage": "awaiting_image_edit", "agent": dispatch["agent"],
                "request": str(out / "image_edit_request.json"),
            })
            print(f"Image editing request ready: {out / 'image_edit_request.json'}", flush=True)
            return
        generate(out, args.edited_first_frame)
        verify(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("route", "review"):
        plan = sub.add_parser(name)
        plan.add_argument("--out", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--instruction", required=True)
    run_parser.add_argument("--source", type=Path, required=True)
    run_parser.add_argument("--geometry", type=Path,
                            help="Optional legacy scene geometry for projecting candidate boxes")
    run_parser.add_argument("--tracking-cache", type=Path)
    run_parser.add_argument("--edited-first-frame", type=Path)
    run_parser.add_argument("--replacement-image", type=Path,
                            help="Optional customer vehicle appearance photo")
    run_parser.add_argument("--conditioning", choices=("auto", "appearance-guided", "text-only"),
                            default="auto", help="Default selects image guidance when a replacement image is supplied")
    run_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "route":
        route(args.out.resolve())
    elif args.command == "review":
        review(args.out.resolve())
    else:
        try:
            run(args)
        except BlockingIOError:
            raise
        except Exception as error:
            dump(args.out.resolve() / "state.json", {
                "stage": "failed", "error": str(error), "error_type": type(error).__name__,
            })
            raise


if __name__ == "__main__":
    main()

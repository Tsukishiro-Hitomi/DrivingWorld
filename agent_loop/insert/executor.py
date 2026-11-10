"""Generic source-video/text H3 formatting and one-shot inference."""

import argparse
from pathlib import Path
import subprocess
import sys

from ..runtime import ROOT, dump, load, sha
from .contracts import validate, verify_origin


def h3_prompt(plan, fps):
    obj, event, place = plan["object"], plan["event"], plan["placement"]
    start, end = event["start_frame"], event["end_frame"]
    opening = ("The added object is already present in frame zero. " if start == 0 else
               f"The added object is absent before frame {start} ({start / fps:.3f} seconds). ")
    return (
        "crossview.\nsubject_definitions:\n"
        "<Video 1> is the original driving video, containing the original road, traffic and camera motion.\n"
        f"The newly generated object is one {obj['description']}. Its appearance is specified by text, "
        "not copied from any existing source object or a picture reference.\n\n"
        "summary:\n[video editing] The target video is an edited version of <Video 1>. "
        f"Add exactly one {obj['description']} to the scene, without replacing existing objects.\n\n"
        "retention_analysis:\n<Video 1>: partially_preserved - retain the original scene except for "
        "the new object and its necessary local shadows and occlusions.\n"
        + "\n".join(plan["preserve"]) + "\n\n"
        "detailed_description:\nPhotorealistic uninterrupted driving footage with the source viewpoint, "
        "weather and illumination.\n[Shot 1] " + opening +
        f"Appearance: {obj['appearance']} Placement: {place['description']} "
        f"Contact: {place['contact_description']} Scale: {place['scale_description']} "
        f"Appearance mode: {event['appearance_mode']}. Entry/reveal: {event['entry_region']} "
        f"From frame {start} ({start / fps:.3f} seconds) to frame {end} ({end / fps:.3f} seconds), "
        f"path: {event['path_description']} Motion mechanics: {event['motion_description']} "
        f"End state: {event['end_state']} "
        + (f"Execution emphasis: {plan['execution_directive']} " if plan.get("execution_directive") else "") +
        "Keep one consistent identity, coherent scene contact, plausible perspective and occlusion. "
        "Preserve the source ego camera and the original vehicles' motion, existing weather and lighting. "
        "No cuts, camera zooms, duplicate added objects, text overlays or logos.\n\n"
        "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    )


def generate(out):
    request, planned = load(out / "request.json"), load(out / "plan.json")
    validate(planned, request)
    verify_origin(out, planned)
    if planned["status"] != "ready":
        raise ValueError("Insertion plan is not ready")
    if sha(Path(request["source"])) != request["source_sha256"]:
        raise ValueError("Insertion source video changed")
    config = {key: request[key] for key in ("source", "source_sha256", "frames", "fps", "width", "height", "seed", "steps")}
    config.update(agent="insert", plan_sha256=sha(out / "plan.json"), prompt=h3_prompt(planned, request["fps"]),
                  reference_video_count=1, reference_image_count=0, aligned_warp=None, first_frame_anchor=None,
                  video_shift=12, audio_shift=3, crossview_strength=.8, acceleration="Larry v4 step600 EMA")
    if (out / "config.json").exists() and load(out / "config.json") != config:
        raise ValueError("Insertion generation inputs changed")
    dump(out / "config.json", config)
    marker = out / "generation_attempt.json"
    if marker.exists() and not (out / "generated.mp4").exists():
        raise ValueError("Previous generation is incomplete or failed; automatic retry is disabled")
    if not marker.exists():
        dump(marker, dict(config_sha256=sha(out / "config.json"), max_retries=0))
    if load(marker)["config_sha256"] != sha(out / "config.json"):
        raise ValueError("Generation attempt belongs to changed inputs")
    with (out / "h3.log").open("a") as log:
        subprocess.run([sys.executable, "-m", "agent_loop.insert.executor", "infer", "--out", str(out)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from ..ref2va import infer

    infer(args.out.resolve())


if __name__ == "__main__":
    main()

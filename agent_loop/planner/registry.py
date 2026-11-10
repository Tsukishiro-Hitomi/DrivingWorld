"""Live backend capabilities; declarations do not prescribe a task order."""

from argparse import Namespace
import json

from ..runtime import ROOT, sha
from ..weather.profiles import PROFILES


def capabilities():
    from ..full_text.executor import checkpoint_ready

    def fingerprint(*files):
        return {file: sha(ROOT / file) for file in files}

    return {
        "framework_sha256": fingerprint("agent_loop/planner/registry.py", "agent_loop/planner/contracts.py", "agent_loop/planner/executor.py", "agent_loop/acceptance.py"),
        "input_contract": {"frames": 124, "fps": 24, "width": 832, "height": 480},
        "agents": {
            "full-text": {
                "description": "Generate an entire new driving scene from text alone with native FL2VA/T2VA and zero reference media.",
                "required_parameters": ["scene_description"], "source_policy": "none",
                "available": checkpoint_ready(),
                "limitations": [
                    "No original source video is required or invented. input_video must be null and depends_on empty.",
                    "The scene agent fills missing scene defaults and compiles native three-field T2VA text.",
                    "An explicit request for severe blur or unreadable lanes is retained; no generic road-readability constraint applies.",
                    "One-shot RGB video generation; no metric GT, camera calibration or independent certification.",
                ],
                "backend_sha256": fingerprint("agent_loop/full_text/dispatch.py", "agent_loop/full_text/executor.py",
                    "agent_loop/full_text/contracts.py", "agent_loop/full_text/verifier.py", "agent_loop/full_text/SKILL.md"),
            },
            "insert": {
                "description": "Add one newly generated object of an open-vocabulary category to the actual input video, using video plus text only.",
                "required_parameters": ["object_description", "event_description"],
                "limitations": [
                    "The insert child can run up to three local plan/text repair rounds, with paired best selection. Outer task graph/code are not rewritten.",
                    "One object and one continuous shot per task; no reference images, image editor, Warp or metric geometry inputs.",
                    "The child agent chooses appearance defaults, placement, first visible frame, path and motion from the actual scene and request. Do not invent pixel tracks in the outer planner.",
                    "New object insertion must retain existing objects; changing an existing vehicle belongs to replace.",
                    "Rolling/articulated motion and temporal continuity are requested through text, not guaranteed controls; missing evidence remains UNKNOWN.",
                    "The reviewer maps limitations to affected checks. All-frame continuity cannot pass from sparse sampled images.",
                ],
                "backend_sha256": fingerprint("agent_loop/insert/dispatch.py", "agent_loop/insert/executor.py",
                    "agent_loop/insert/contracts.py", "agent_loop/insert/verifier.py", "agent_loop/insert/loop.py", "agent_loop/insert/SKILL.md",
                    "agent_loop/acceptance.py", "agent_loop/ref2va.py"),
            },
            "replace": {
                "description": "Replace a visually selected existing vehicle, starting at frame zero, using a text-only appearance request.",
                "required_parameters": ["target_description", "target_selector", "replacement_category"],
                "target_selection": "Natural-language description and selection cue; child agent grounds one detected source vehicle in the actual first frame.",
                "category_input": "Open-vocabulary vehicle category; child agent generates appearance details from the request, with no fixed vehicle-class allowlist.",
                "replacement_image_supported": False,
                "limitations": [
                    "Open-vocabulary prompt support is not a guarantee of category fidelity, target grounding or temporal consistency; these require actual output review.",
                    "The target must be a uniquely identifiable existing vehicle visible in the source first frame; start-frame zero only.",
                    "The image-guided route needs a host image-editor handoff and is not an unattended executor here.",
                    "Target grounding is performed by the child agent against its actual input frames, not invented by the outer planner.",
                ],
                "backend_sha256": fingerprint("agent_loop/replace/executor.py", "agent_loop/replace/dispatch.py", "agent_loop/replace/SKILL.md", "agent_loop/ref2va.py"),
            },
            "weather": {
                "description": "Edit global atmosphere, lighting and RGB-camera optical response in the input video.",
                "required_parameters": ["profile"], "profiles": list(PROFILES),
                "limitations": [
                    "Only qualitative RGB video editing; no LiDAR/radar simulation or metric GT certification.",
                    "Automatic visual reviews can falsely accept severity; report them as sampled judgments, not proven acceptance.",
                ],
                "backend_sha256": fingerprint("agent_loop/weather/profiles.py", "agent_loop/weather/dispatch.py", "agent_loop/weather/sensor.py", "agent_loop/weather/rain.py", "agent_loop/ref2va.py"),
            },
        },
        "execution_policy": {"feedback_loop": False, "max_retries": 0, "max_tasks": 4,
                             "child_repair_limits": {"insert": 3, "replace": 0, "weather": 0, "full-text": 0}},
        "artifact_contract": {"input": "source video for editing; no video for full-text creation", "output": "generated_video"},
    }


def invoke(task, source, out):
    """Use a fixed adapter allowlist; never execute a model-authored shell command."""
    retained = "\n".join(task["preserve"])
    checks = "\n".join(task["acceptance_criteria"])
    instruction = (task["instruction"] + "\nTask parameters:\n" + json.dumps(task["parameters"])
                   + "\nRequired retention:\n" + retained + "\nAcceptance goals:\n" + checks)
    if task["agent"] == "full-text":
        from ..full_text.dispatch import run

        run(Namespace(source=source, instruction=instruction, out=out, plan_only=False,
                      task_parameters=task["parameters"], replacement_image=None, edited_first_frame=None, profile=None))
    elif task["agent"] == "replace":
        from ..replace.dispatch import run

        run(Namespace(source=source, instruction=instruction, out=out, conditioning="text-only",
                      geometry=None, tracking_cache=None, replacement_image=None, edited_first_frame=None,
                      task_parameters=task["parameters"]))
    elif task["agent"] == "weather":
        from ..weather.dispatch import run

        run(Namespace(source=source, instruction=instruction, out=out,
                      profile=task["parameters"]["profile"], max_retries=0, plan_only=False,
                      feedback_from=None, replacement_image=None, edited_first_frame=None))
    elif task["agent"] == "insert":
        from ..insert.dispatch import run

        run(Namespace(source=source, instruction=instruction, out=out, plan_only=False,
                      task_parameters=task["parameters"], replacement_image=None, edited_first_frame=None, max_repairs=3))
    else:
        raise ValueError("Unregistered execution agent")

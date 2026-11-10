"""Execute validated model dependencies once, without feedback or regeneration loops."""

import fcntl
from pathlib import Path
import shutil

from ..qwen import run_worker
from ..runtime import dump, load, sha
from ..video import read_frames, sample_sheet
from .contracts import validate, verify_origin
from .registry import capabilities, invoke


def run_task(task, source, out):
    invoke(task, source, out)
    video = out / "generated.mp4"
    if not video.is_file():
        raise ValueError(f"Agent {task['agent']} did not produce its registered video artifact")
    child_request = load(out / "request.json")
    if child_request["source_sha256"] != (sha(source) if source else None):
        raise ValueError("Child agent consumed a different input video")
    state = load(out / "state.json")
    verdict = state.get("acceptance_verdict", state.get("sampled_visual_verdict"))
    if verdict is None:
        verdict = "PASS" if state.get("visual_check_passed") is True else "FAIL"
    return {"path": str(video.resolve()), "sha256": sha(video), "local_verdict": verdict,
            "sampled_visual_verdict": state.get("sampled_visual_verdict")}


def reassess_task(task, source, out):
    """Reassess an already generated insertion output; never call an edit executor."""
    if task["agent"] != "insert":
        raise ValueError("Recorded assessment recovery currently supports insert only")
    from ..insert.contracts import verify_origin as verify_child_origin
    from ..insert.verifier import assess_recorded, verify

    if load(out / "request.json")["source_sha256"] != sha(source):
        raise ValueError("Recorded insertion used a different source")
    verify_child_origin(out, load(out / "plan.json"))
    prior_state = load(out / "state.json")
    if not (out / "assessment_previous_state.json").exists():
        dump(out / "assessment_previous_state.json", prior_state)
    assess_recorded(out)
    result = verify(out)
    verdict = result["acceptance_verdict"]
    dump(out / "state.json", dict(stage={"PASS": "complete", "FAIL": "completed_with_visual_issues",
        "UNKNOWN": "needs_review"}[verdict], agent="insert", technical_checks_passed=True,
        sampled_visual_verdict=result["sampled_verdict"], acceptance_verdict=verdict,
        assessment_recovered=True, generation_retried=False, feedback_loop=False, max_retries=0,
        independent_acceptance="NOT_ESTABLISHED", generated_sha256=sha(out / "generated.mp4"),
        video=str(out / "generated.mp4")))
    return dict(path=str((out / "generated.mp4").resolve()), sha256=sha(out / "generated.mp4"),
                local_verdict=verdict, sampled_visual_verdict=result["sampled_verdict"])


def joint_review(out, request, plan, journal):
    directory = out / "joint_review"
    directory.mkdir(exist_ok=True)
    references = ([{"role": "original_source", "path": request["source_video"], "sha256": request["source_sha256"]}]
                  if request.get("source_video") else [])
    for task_id in journal["execution_order"]:
        artifact = journal["tasks"][task_id]["artifact"]
        references.append(dict(role=task_id + (" (final output)" if task_id == plan["final_task"] else " (intermediate)"), **artifact))
    for index, item in enumerate(references):
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("Joint-review input changed")
        picture = directory / f"{index:02d}.jpg"
        sample_sheet(picture, read_frames(item["path"]))
        item["sheet"] = str(picture)
        item["sheet_sha256"] = sha(picture)
    dump(directory / "manifest.json", {"references": references})
    run_worker("agent_loop.planner", "review", out, "joint_review.log")
    result = load(out / "joint_review.json")
    if (result["manifest_sha256"] != sha(directory / "manifest.json")
            or result["plan_sha256"] != sha(out / "plan.json")):
        raise ValueError("Joint review is bound to different artifacts or plan")
    return result


def execute(out, runner=None, reviewer=None, assessment_only=False):
    runner = runner or (reassess_task if assessment_only else run_task)
    reviewer = reviewer or joint_review
    with (out / ".execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request, plan = load(out / "request.json"), load(out / "plan.json")
        registry = load(out / "capabilities.json")
        if not assessment_only and registry != capabilities():
            raise ValueError("Backend capabilities changed; generate a new agent plan in a new run")
        validated = validate(plan, registry, request)
        if plan["status"] != "ready":
            raise ValueError("The planning agent has not declared an executable plan")
        verify_origin(out, plan)
        if request.get("source_video") and sha(Path(request["source_video"])) != request["source_sha256"]:
            raise ValueError("Original source video changed")
        journal_path = out / "execution.json"
        journal = dict(plan_sha256=sha(out / "plan.json"), request_sha256=sha(out / "request.json"),
                       execution_order=validated["execution_order"], tasks={}, feedback_loop=False, max_retries=0)
        if journal_path.exists():
            saved = load(journal_path)
            if any(saved[key] != journal[key] for key in ["plan_sha256", "request_sha256", "execution_order"]):
                raise ValueError("Cannot resume execution with a changed plan or request")
            journal = saved
        by_id = {task["id"]: task for task in plan["tasks"]}
        for task_id in journal["execution_order"]:
            task = by_id[task_id]
            prior = journal["tasks"].get(task_id)
            if prior:
                if prior["stage"] != "done":
                    if not assessment_only:
                        raise ValueError("A previous attempt is unfinished or failed; automatic retry is disabled")
                else:
                    if sha(Path(prior["artifact"]["path"])) != prior["artifact"]["sha256"]:
                        raise ValueError("Completed task artifact changed")
                    continue
            source = (Path(request["source_video"]) if request.get("source_video") else None) if not task["depends_on"] else Path(journal["tasks"][task["depends_on"][0]]["artifact"]["path"])
            stage_out = out / "tasks" / task_id
            stage_out.mkdir(parents=True, exist_ok=True)
            if assessment_only:
                if not prior or not (stage_out / "generated.mp4").exists():
                    raise ValueError("Assessment-only mode cannot start or regenerate a missing task")
                journal.setdefault("assessment_history", {}).setdefault(task_id, prior)
            journal["tasks"][task_id] = dict(stage="running", input_video=str(source) if source else None,
                                           input_sha256=sha(source) if source else None)
            dump(journal_path, journal)
            dump(out / "state.json", dict(stage="executing", task=task_id, feedback_loop=False))
            try:
                artifact = runner(task, source, stage_out)
                if sha(Path(artifact["path"])) != artifact["sha256"] or artifact["local_verdict"] not in {"PASS", "FAIL", "UNKNOWN"}:
                    raise ValueError("Agent returned an invalid artifact/result")
                journal["tasks"][task_id].update(stage="done", artifact=artifact)
            except Exception as error:
                journal["tasks"][task_id].update(stage="failed", error=str(error))
                dump(journal_path, journal)
                raise
            dump(journal_path, journal)
        final = journal["tasks"][plan["final_task"]]["artifact"]
        shutil.copyfile(final["path"], out / "generated.mp4")
        verdicts = [item["artifact"]["local_verdict"] for item in journal["tasks"].values()]
        reviewed = reviewer(out, request, plan, journal)
        if set(reviewed.get("checks", {})) != {item["id"] for item in plan["final_acceptance"]}:
            raise ValueError("Joint review does not cover the planned acceptance goals")
        if any(item.get("status") not in {"PASS", "FAIL", "UNKNOWN"} for item in reviewed["checks"].values()):
            raise ValueError("Invalid joint-review result")
        effective = reviewed.get("effective_checks", reviewed["checks"])
        if set(effective) != set(reviewed["checks"]):
            raise ValueError("Effective joint review has different acceptance goals")
        verdicts += [item["status"] for item in effective.values()]
        verdict = "FAIL" if "FAIL" in verdicts else ("UNKNOWN" if "UNKNOWN" in verdicts else "PASS")
        dump(out / "state.json", dict(stage="completed_with_sampled_pass" if verdict == "PASS" else "completed_with_issues",
             sampled_verdict=verdict, acceptance_verdict=verdict, independent_acceptance="NOT_ESTABLISHED", feedback_loop=False,
             max_retries=0, generated_sha256=sha(out / "generated.mp4"), video=str(out / "generated.mp4")))

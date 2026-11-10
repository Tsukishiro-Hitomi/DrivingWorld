"""Validate agent-authored data dependencies without selecting an edit strategy."""

import re
import json


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {label}")


def strings(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing {label}")
    for item in value:
        nonempty(item, label)


def validate(plan, registry, request):
    required = {"status", "reason", "unresolved", "tasks", "final_task", "final_acceptance"}
    if not required <= set(plan):
        raise ValueError("Planner output is missing required contract fields")
    if set(plan) - required - {"provenance"}:
        raise ValueError("Planner supplied fields outside the task-plan contract")
    if plan["status"] not in {"ready", "needs_clarification", "needs_capability"}:
        raise ValueError("Invalid planner readiness status")
    nonempty(plan["reason"], "planning reason")
    if not isinstance(plan["unresolved"], list):
        raise ValueError("unresolved must be a list")
    if (plan["status"] == "ready") != (plan["unresolved"] == []):
        raise ValueError("Readiness conflicts with unresolved requirements")
    for item in plan["unresolved"]:
        nonempty(item, "unresolved requirement")
    tasks = plan["tasks"]
    if not isinstance(tasks, list) or len(tasks) > registry["execution_policy"]["max_tasks"]:
        raise ValueError("Invalid task count")
    if not tasks:
        if plan["status"] == "ready" or plan["final_task"] is not None:
            raise ValueError("A ready plan needs an executable final task")
        return {"execution_order": [], "capability_violations": []}
    by_id = {}
    violations = []
    for task in tasks:
        expected = {"id", "agent", "depends_on", "input_video", "instruction", "parameters", "preserve", "acceptance_criteria", "reason"}
        if not isinstance(task, dict) or set(task) != expected:
            raise ValueError("Task fields must match the execution contract exactly")
        task_id = task["id"]
        if not isinstance(task_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", task_id) or task_id in by_id:
            raise ValueError("Invalid or duplicate task ID")
        by_id[task_id] = task
        if task["agent"] not in registry["agents"]:
            raise ValueError("Unknown task agent")
        for key in ("instruction", "reason"):
            nonempty(task[key], key)
        strings(task["preserve"], "task preservation goals")
        strings(task["acceptance_criteria"], "task acceptance goals")
        parameters = task["parameters"]
        cap = registry["agents"][task["agent"]]
        if not isinstance(parameters, dict) or set(parameters) != set(cap["required_parameters"]):
            raise ValueError("Task parameters do not match the registered interface")
        for key in cap["required_parameters"]:
            nonempty(parameters[key], key)
        if task["agent"] == "full-text":
            if not cap["available"]:
                violations.append(f"{task_id}: FL2VA deployment is not ready")
            if request.get("replacement_image"):
                violations.append(f"{task_id}: full-text accepts no supplied image")
        elif task["agent"] == "replace":
            if request.get("replacement_image"):
                violations.append(f"{task_id}: unattended image-guided replacement is unavailable")
        elif task["agent"] == "insert":
            if request.get("replacement_image"):
                violations.append(f"{task_id}: insertion accepts video and text only")
        elif task["agent"] == "weather":
            if parameters["profile"] not in cap["profiles"]:
                violations.append(f"{task_id}: unavailable weather profile: {parameters['profile']}")
            elif request.get("profile_hint") and parameters["profile"] != request["profile_hint"]:
                violations.append(f"{task_id}: profile conflicts with the user's explicit profile hint")
    final = plan["final_task"]
    if final not in by_id:
        raise ValueError("Unknown final task")
    for task in tasks:
        dependencies = task["depends_on"]
        if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
            raise ValueError("depends_on must contain task IDs")
        reference = task["input_video"]
        if task["agent"] == "full-text":
            if reference is not None or dependencies:
                raise ValueError("full-text creates from text: input_video must be null with no video dependency")
            if request.get("source_video"):
                raise ValueError("full-text cannot discard a supplied source video")
            expected_dependencies = []
        elif reference == "request.source_video":
            if "source_video" in request and request["source_video"] is None:
                raise ValueError("No source video exists; a full-text task must create one first")
            expected_dependencies = []
        elif isinstance(reference, str) and reference.endswith(".generated_video"):
            parent = reference[:-len(".generated_video")]
            if parent not in by_id or parent == task["id"]:
                raise ValueError("Invalid video artifact dependency")
            expected_dependencies = [parent]
        else:
            raise ValueError("Video inputs must reference a registered artifact, never an invented path")
        if dependencies != expected_dependencies:
            raise ValueError("Declared dependencies conflict with video data flow")
    order, visiting, visited = [], set(), set()
    def visit(task_id):
        if task_id in visiting:
            raise ValueError("Task dependency cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for parent in by_id[task_id]["depends_on"]:
            visit(parent)
        visiting.remove(task_id)
        visited.add(task_id)
        order.append(task_id)
    visit(final)
    if len(visited) != len(tasks):
        raise ValueError("Some edits do not contribute to the final video's dependency chain")
    goals = plan["final_acceptance"]
    if not isinstance(goals, list) or not goals:
        raise ValueError("Final acceptance goals are required")
    goal_ids = set()
    for goal in goals:
        if not isinstance(goal, dict) or set(goal) != {"id", "description"}:
            raise ValueError("Invalid final acceptance goal")
        nonempty(goal["id"], "acceptance ID")
        nonempty(goal["description"], "acceptance description")
        if goal["id"] in goal_ids:
            raise ValueError("Duplicate acceptance goal")
        goal_ids.add(goal["id"])
    if plan["status"] == "ready" and violations:
        raise ValueError("Plan falsely claims readiness: " + "; ".join(violations))
    return {"execution_order": order, "capability_violations": violations}


def verify_origin(out, plan):
    from ..runtime import load, sha

    provenance = plan["provenance"]
    name = provenance["raw_response_file"]
    if not re.fullmatch(r"outer_plan_response_[0-9]+\.json", name):
        raise ValueError("Invalid model-response provenance path")
    for key, path in [("request_sha256", out / "request.json"),
                      ("capabilities_sha256", out / "capabilities.json"),
                      ("raw_response_sha256", out / name),
                      ("planner_input_sha256", out / "planner_input.json"),
                      ]:
        if provenance[key] != sha(path):
            raise ValueError("Plan provenance differs from recorded agent inputs or response")
    sheet = out / "source_frames.jpg"
    if provenance.get("source_sheet_sha256") is None:
        if sheet.exists():
            raise ValueError("Unexpected source image in a text-only planning run")
    elif provenance["source_sheet_sha256"] != sha(sheet):
        raise ValueError("Planner source sheet changed")
    raw = load(out / name)["text"]
    parsed = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    if parsed != {key: value for key, value in plan.items() if key != "provenance"}:
        raise ValueError("Task plan was modified after the agent response")

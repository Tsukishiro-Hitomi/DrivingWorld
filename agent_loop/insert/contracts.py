"""Structural insertion contracts; the agent owns appearance, timing and placement."""

import json
import re
from copy import deepcopy

from ..runtime import load, sha

CHECKS = ("object_identity", "object_count", "event_timing", "path_and_direction",
          "motion_mechanics", "ground_contact_and_occlusion", "appearance_stability",
          "scene_preservation", "temporal_continuity")


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing " + label)


def texts(value, label, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError("Invalid " + label)
    for entry in value:
        text(entry, label)


def validate(plan, request):
    if plan.get("schema_version") != 1 or plan.get("status") not in {"ready", "needs_clarification", "needs_capability"}:
        raise ValueError("Invalid insertion plan status/schema")
    text(plan.get("reason"), "reason")
    texts(plan.get("unresolved"), "unresolved", allow_empty=True)
    if (plan["status"] == "ready") != (plan["unresolved"] == []):
        raise ValueError("Readiness conflicts with unresolved requirements")
    if plan["status"] != "ready":
        return
    obj, event, placement = plan.get("object"), plan.get("event"), plan.get("placement")
    if not isinstance(obj, dict) or type(obj.get("count")) is not int or obj["count"] != 1:
        raise ValueError("This insertion executor accepts exactly one new object")
    for key in ("description", "appearance"):
        text(obj.get(key), "object." + key)
    texts(obj.get("chosen_defaults"), "chosen_defaults", allow_empty=True)
    text(plan.get("source_observations"), "source_observations")
    if not isinstance(event, dict) or event.get("appearance_mode") not in {"present_from_start", "enter_frame", "uncover"}:
        raise ValueError("Invalid appearance mode")
    start, end = event.get("start_frame"), event.get("end_frame")
    if type(start) is not int or type(end) is not int or not 0 <= start < end < request["frames"]:
        raise ValueError("Insertion event frames must lie inside the actual clip")
    if event["appearance_mode"] == "present_from_start" and start != 0:
        raise ValueError("present_from_start must start at frame zero")
    for key in ("entry_region", "path_description", "motion_description", "end_state"):
        text(event.get(key), "event." + key)
    if not isinstance(placement, dict):
        raise ValueError("Missing placement")
    for key in ("description", "contact_description", "scale_description"):
        text(placement.get(key), "placement." + key)
    texts(plan.get("preserve"), "preserve")
    goals = plan.get("acceptance_criteria")
    if not isinstance(goals, dict) or set(goals) != set(CHECKS):
        raise ValueError("Insertion plan must define every acceptance criterion")
    for value in goals.values():
        text(value, "acceptance description")


def verify_origin(out, plan):
    provenance = plan["provenance"]
    name = provenance["raw_response_file"]
    kind = provenance.get("response_kind", "plan")
    if kind not in {"plan", "repair"} or not re.fullmatch(r"insert_" + kind + r"_response_[0-9]+\.json", name):
        raise ValueError("Invalid insertion model-response filename")
    files = {"request_sha256": "request.json", "planner_input_sha256": "repair_input.json" if kind == "repair" else "planner_input.json",
             "source_sheet_sha256": "source_frames.jpg", "raw_response_sha256": name}
    if kind == "repair":
        files.update(loop_context_sha256="loop_context.json", parent_plan_sha256="parent_plan.json", base_plan_sha256="base_plan.json")
    for key, filename in files.items():
        if provenance[key] != sha(out / filename):
            raise ValueError("Insertion planner input/response changed")
    raw = load(out / name)["text"]
    parsed = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    if kind == "repair":
        parsed = apply_repair(load(out / "parent_plan.json"), load(out / "base_plan.json"), parsed, load(out / "request.json"))
    if parsed != {k: v for k, v in plan.items() if k != "provenance"}:
        raise ValueError("Insertion plan was modified after the agent response")


EDITABLE = (("object", "appearance"), ("placement", "description"), ("placement", "contact_description"),
            ("placement", "scale_description"), ("event", "entry_region"), ("event", "path_description"),
            ("event", "motion_description"), ("event", "end_state"))


def validate_repair(base, revised, request):
    validate(revised, request)
    if revised["status"] != "ready" or "provenance" in revised:
        raise ValueError("A repair must be an executable agent-authored plan")
    text(revised.get("execution_directive"), "execution_directive")
    def frozen(plan):
        result = deepcopy(plan)
        for key in ("provenance", "execution_directive", "reason"):
            result.pop(key, None)
        for section, key in EDITABLE:
            result[section][key] = "<agent-editable description>"
        return result
    if frozen(base) != frozen(revised):
        raise ValueError("Repair changed a frozen requirement, timing, identity or acceptance goal")


def apply_repair(parent, base, proposal, request):
    patch = proposal.get("plan_patch")
    if not isinstance(patch, dict):
        raise ValueError("Repair plan_patch must be an object")
    allowed = {}
    for section, key in EDITABLE:
        allowed.setdefault(section, set()).add(key)
    revised = deepcopy(parent)
    revised.pop("provenance", None)
    for section, changes in patch.items():
        if section not in allowed or not isinstance(changes, dict) or set(changes) - allowed[section]:
            raise ValueError("Repair patch contains an unregistered field")
        for key, value in changes.items():
            text(value, section + "." + key)
            revised[section][key] = value
    text(proposal.get("execution_directive"), "execution_directive")
    text(proposal.get("reason"), "repair reason")
    revised.update(execution_directive=proposal["execution_directive"], reason=proposal["reason"])
    validate_repair(base, revised, request)
    return revised

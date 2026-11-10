"""Agent-authored repairs and paired best selection, capped at three video repairs."""

from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil

from ..qwen import qwen_json, run_worker
from ..runtime import dump, load, sha
from .contracts import CHECKS, apply_repair, verify_origin

SKILL = Path(__file__).with_name("SKILL.md")
COPY_FILES = ("request.json", "plan.json", "planner_input.json", "source_frames.jpg", "repair_input.json",
    "parent_plan.json", "base_plan.json", "loop_context.json",
    "config.json", "generation.json", "generation_attempt.json", "generated.mp4", "comparison.mp4",
    "visual_review.json", "review_manifest.json", "review_input.json", "review_frames.jpg", "review_event_frames.jpg")


def only_evidence_limit(review):
    checks = review["effective_checks"]
    return (checks["temporal_continuity"]["status"] == "UNKNOWN"
            and all(v["status"] == "PASS" for k, v in checks.items() if k != "temporal_continuity"))


def selection(best, candidate, judgment):
    if judgment.get("decision") not in {"candidate", "best", "inconclusive"}:
        raise ValueError("Invalid paired-comparison decision")
    for key in ("improved_criteria", "regressions"):
        if not isinstance(judgment.get(key), list) or any(v not in CHECKS for v in judgment[key]):
            raise ValueError("Invalid paired-comparison criterion IDs")
    if not isinstance(judgment.get("reason"), str) or not judgment["reason"].strip():
        raise ValueError("Paired comparison needs visible reasons")
    rank = {"FAIL": 0, "UNKNOWN": 1, "PASS": 2}
    regressions = [k for k in CHECKS if rank[candidate["effective_checks"][k]["status"]]
                   < rank[best["effective_checks"][k]["status"]]]
    promoted = (judgment["decision"] == "candidate" and bool(judgment["improved_criteria"])
                and not judgment["regressions"] and not regressions)
    if best.get("generated_sha256") and best.get("generated_sha256") == candidate.get("generated_sha256"):
        promoted = False
    return dict(promoted=promoted, formal_regressions=regressions, paired_judgment=judgment,
                interpretation="Qualitative paired preference, not independent acceptance")


def _context(out):
    context = load(out / "loop_context.json")
    for filename, digest in context["bindings"].items():
        if sha(Path(filename)) != digest:
            raise ValueError("Repair/comparison context changed: " + filename)
    return context


def make_context(root, target, base, best, history):
    files = []
    for directory in {base, best}:
        files += [directory / f for f in ("plan.json", "request.json", "config.json", "generated.mp4",
                                           "visual_review.json", "review_frames.jpg", "review_event_frames.jpg")]
    maximum = load(root / "loop_request.json")["max_repairs"]
    index = int(target.name[-2:])
    dump(target / "loop_context.json", dict(root=str(root), baseline=str(base), best=str(best), history=history,
        loop_policy=dict(max_repairs=maximum, current_repair=index, remaining_after_this=maximum - index,
                         generation_retries_within_attempt=0),
        bindings={str(p): sha(p) for p in files}))


def repair(out):
    if (out / "repair_input.json").exists():
        raise ValueError("This repair proposal is already recorded; no hidden proposal retry")
    context = _context(out)
    base, best = Path(context["baseline"]), Path(context["best"])
    request, original = load(out / "request.json"), load(base / "plan.json")
    observed = load(best / "visual_review.json")
    content = [{"type": "image", "image": str(best / name)}
               for name in ("source_frames.jpg", "review_frames.jpg", "review_event_frames.jpg")]
    content.append({"type": "text", "text": json.dumps(dict(request=request, baseline_plan=original,
        best_plan=load(best / "plan.json"), best_prompt=load(best / "config.json")["prompt"],
        best_review=observed, previous_rounds=context["history"], loop_policy=context["loop_policy"]), ensure_ascii=False)})
    messages = [{"role": "system", "content": SKILL.read_text() +
        "\nThis is a REPAIR call. Diagnose actual failures from the best output and earlier rounds. "
        "loop_policy is the current repair authorization; the historical request may record a previous one-shot policy. "
        "Return {retryable: boolean, reason: string, diagnosis: [{criterion, observation, action}], "
        "plan_patch: object, execution_directive: string}. "
        "When retryable, return ONLY the changed fields in nested plan_patch, plus execution_directive "
        "as a concrete positive English instruction that changes how H3 realizes the unchanged goal. "
        "Only object.appearance, placement.description/contact_description/scale_description, "
        "event.entry_region/path_description/motion_description/end_state, reason and execution_directive may change. "
        "Do not include reason or execution_directive inside plan_patch. "
        "All other fields, including identity, count, chosen defaults, first-visible frame, event end frame, "
        "appearance mode, preservation and acceptance criteria, are frozen by the host and cannot be patched. "
        "Do not redefine early appearance as success, replace rolling with sliding, or claim missing tools. "
        "Do not copy provenance. The unchanged original source video will be edited again, not the previous generated video. "
        "Stop with retryable false when only unavailable evidence remains, required controls are unavailable, "
        "or another text correction is not justified. Do not invent new scene-level requirements."},
        {"role": "user", "content": content}]
    dump(out / "repair_input.json", messages)
    result = qwen_json(messages, out, "insert_repair", max_tokens=2400)
    if type(result.get("retryable")) is not bool or not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ValueError("Invalid repair decision")
    if not isinstance(result.get("diagnosis"), list):
        raise ValueError("Repair diagnosis must be an array")
    actionable = {k for k, v in observed["effective_checks"].items() if v["status"] != "PASS"}
    for item in result["diagnosis"]:
        if (not isinstance(item, dict) or item.get("criterion") not in actionable
                or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("observation", "action"))):
            raise ValueError("Repair diagnosis must cite actual nonpassing criteria")
    if result["retryable"]:
        if not result["diagnosis"]:
            raise ValueError("A video repair needs an evidence-based diagnosis")
        revised = apply_repair(load(best / "plan.json"), original, result, request)
        shutil.copyfile(best / "plan.json", out / "parent_plan.json")
        shutil.copyfile(base / "plan.json", out / "base_plan.json")
        raw = sorted(out.glob("insert_repair_response_*.json"))[-1]
        revised = deepcopy(revised)
        revised["provenance"] = dict(response_kind="repair", planner="Qwen/Qwen3.8-27B", authorship="model_generated",
            raw_response_file=raw.name, raw_response_sha256=sha(raw), planner_input_sha256=sha(out / "repair_input.json"),
            request_sha256=sha(out / "request.json"), source_sheet_sha256=sha(out / "source_frames.jpg"),
            skill_sha256=sha(SKILL), loop_context_sha256=sha(out / "loop_context.json"),
            parent_plan_sha256=sha(out / "parent_plan.json"), base_plan_sha256=sha(out / "base_plan.json"))
        dump(out / "plan.json", revised)
    dump(out / "repair.json", result)


def compare(out):
    if (out / "comparison_input.json").exists():
        raise ValueError("A paired comparison is already recorded; no hidden comparison retry")
    context = _context(out)
    base, best = Path(context["baseline"]), Path(context["best"])
    content = [{"type": "image", "image": str(best / "source_frames.jpg")}]
    for directory in (best, out):
        content += [{"type": "image", "image": str(directory / name)}
                    for name in ("review_frames.jpg", "review_event_frames.jpg")]
    content.append({"type": "text", "text": json.dumps(dict(request=load(base / "request.json"),
        baseline_plan=load(base / "plan.json"), best_review=load(best / "visual_review.json"),
        candidate_review=load(out / "visual_review.json")), ensure_ascii=False)})
    messages = [{"role": "system", "content": (
        "Compare actual insertion outputs under the SAME original task, event timeline and acceptance criteria. "
        "Image 1 is the source. Images 2-3 are incumbent best full-frame/event sheets; images 4-5 are candidate sheets. "
        "Use visible evidence and stable original requirements, not a count of PASS labels or round number. "
        "Judge timing, identity, requested mechanics, preservation and artifacts. Partial improvement can be real "
        "even if a criterion still fails; explain observable differences. Do not prefer a new prompt on wording alone. "
        "Return JSON {decision: candidate/best/inconclusive, improved_criteria: array of criterion IDs, "
        "regressions: array of criterion IDs, reason: string}. Criterion IDs: " + ", ".join(CHECKS) +
        ". No visible improvement or insufficient evidence means best or inconclusive. "
        "Do not certify all-frame continuity or metric geometry from sample sheets.")}, {"role": "user", "content": content}]
    dump(out / "comparison_input.json", messages)
    result = qwen_json(messages, out, "insert_compare", max_tokens=1600)
    selected = selection(load(best / "visual_review.json"), load(out / "visual_review.json"), result)
    selected.update(best_generated_sha256=sha(best / "generated.mp4"), candidate_generated_sha256=sha(out / "generated.mp4"),
                    context_sha256=sha(out / "loop_context.json"))
    dump(out / "paired_comparison.json", selected)


def _publish(root, directory, index, journal):
    config, result = load(directory / "config.json"), load(directory / "visual_review.json")
    for source, target in (("generated.mp4", "generated.mp4"), ("comparison.mp4", "comparison.mp4"),
                           ("plan.json", "best_plan.json"), ("config.json", "best_config.json"),
                           ("visual_review.json", "visual_review.json")):
        shutil.copyfile(directory / source, root / target)
    (root / "prompt.txt").write_text(config["prompt"] + "\n")
    selected = dict(attempt=index, path=str(directory), generated_sha256=sha(root / "generated.mp4"),
                    prompt_sha256=sha(root / "prompt.txt"), acceptance_verdict=result["acceptance_verdict"])
    dump(root / "best.json", selected)
    updates = load(root / "prompt_history.json") if (root / "prompt_history.json").exists() else []
    if not updates or updates[-1]["attempt"] != index:
        updates.append(dict(selected, promotion="baseline" if index == 0 else "agent paired improvement",
                            comparison=str(directory / "paired_comparison.json") if index else None))
        dump(root / "prompt_history.json", updates)
    journal["best_attempt"] = index
    journal["best_generated_sha256"] = selected["generated_sha256"]
    journal["best_prompt_sha256"] = selected["prompt_sha256"]
    print("BEST_UPDATED", json.dumps(selected, ensure_ascii=False), flush=True)


def _finish(root, journal, reason):
    best = root / f"attempt_{journal['best_attempt']:02d}"
    result = load(best / "visual_review.json")
    journal.update(stage="finished", stop_reason=reason)
    dump(root / "loop.json", journal)
    verdict = result["acceptance_verdict"]
    dump(root / "state.json", dict(stage={"PASS": "complete", "FAIL": "completed_with_visual_issues", "UNKNOWN": "needs_review"}[verdict],
        agent="insert", feedback_loop=True, max_repairs=journal["max_repairs"], repairs_attempted=len(journal["rounds"]),
        best_attempt=journal["best_attempt"], stop_reason=reason, sampled_visual_verdict=result["sampled_verdict"],
        new_generations=sum(bool(v.get("generated")) for v in journal["rounds"].values()) + (not journal["baseline_imported"]),
        acceptance_verdict=verdict, independent_acceptance="NOT_ESTABLISHED", technical_checks_passed=True,
        generated_sha256=sha(root / "generated.mp4"), video=str(root / "generated.mp4"),
        prompt=str(root / "prompt.txt"), prompt_sha256=sha(root / "prompt.txt")))


def run(root, max_repairs=3, baseline=None):
    from .executor import generate
    from .verifier import verify

    if type(max_repairs) is not int or not 0 <= max_repairs <= 3:
        raise ValueError("Insertion repair limit must be between zero and three")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".repair.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        origin = baseline.resolve() if baseline else root
        if baseline and origin == root:
            raise ValueError("Use a new loop output directory; do not overwrite the baseline run")
        if baseline and (origin / "best.json").exists():
            selected = load(origin / "best.json")
            if sha(origin / "generated.mp4") != selected["generated_sha256"]:
                raise ValueError("Imported loop's published video changed")
            origin = Path(selected["path"])
        setup = dict(max_repairs=max_repairs, baseline=str(origin), imported=baseline is not None,
                     initial_request_sha256=sha(origin / "request.json"), initial_plan_sha256=sha(origin / "plan.json"))
        setup_file = root / "loop_request.json"
        if setup_file.exists() and load(setup_file) != setup:
            raise ValueError("Loop inputs changed; preserve this run and use a new directory")
        dump(setup_file, setup)
        first = root / "attempt_00"
        if not first.exists():
            first.mkdir()
            for file in origin.iterdir():
                if file.is_file() and (file.name in COPY_FILES or file.name.startswith(("insert_plan_response_", "insert_repair_response_"))):
                    shutil.copyfile(file, first / file.name)
        verify_origin(first, load(first / "plan.json"))
        journal_file = root / "loop.json"
        journal = load(journal_file) if journal_file.exists() else dict(stage="running", max_repairs=max_repairs,
            baseline_imported=baseline is not None, rounds={}, best_attempt=0)
        if journal["max_repairs"] != max_repairs:
            raise ValueError("Cannot change the repair budget on resume")
        if journal["stage"] == "finished":
            if sha(root / "generated.mp4") != journal["best_generated_sha256"] or sha(root / "prompt.txt") != journal["best_prompt_sha256"]:
                raise ValueError("Published best artifact/prompt changed")
            return
        dump(root / "state.json", dict(stage="baseline", agent="insert", feedback_loop=True, max_repairs=max_repairs))
        generate(first)
        verify(first)
        if not journal.get("best_generated_sha256"):
            _publish(root, first, 0, journal)
            dump(journal_file, journal)
        for index in range(1, max_repairs + 1):
            best = root / f"attempt_{journal['best_attempt']:02d}"
            reviewed = load(best / "visual_review.json")
            if reviewed["acceptance_verdict"] == "PASS" or only_evidence_limit(reviewed):
                _finish(root, journal, "accepted" if reviewed["acceptance_verdict"] == "PASS" else "evidence_limit")
                return
            key = str(index)
            if key in journal["rounds"]:
                if journal["rounds"][key]["stage"] != "done":
                    raise ValueError("An unfinished repair round is already recorded; no hidden generation retry")
                continue
            candidate = root / f"attempt_{index:02d}"
            candidate.mkdir()
            for name in ("request.json", "source_frames.jpg"):
                shutil.copyfile(first / name, candidate / name)
            make_context(root, candidate, first, best, journal["rounds"])
            journal["rounds"][key] = dict(stage="proposing", parent_best=journal["best_attempt"])
            dump(journal_file, journal)
            dump(root / "state.json", dict(stage="repair_planning", agent="insert", attempt=index, feedback_loop=True, max_repairs=max_repairs))
            try:
                run_worker("agent_loop.insert", "repair-plan", candidate, "repair.log")
                proposal = load(candidate / "repair.json")
                journal["rounds"][key]["repair"] = proposal
                if not proposal["retryable"]:
                    journal["rounds"][key].update(stage="done", generated=False)
                    _finish(root, journal, "agent_stopped")
                    return
                dump(root / "state.json", dict(stage="repair_generating", agent="insert", attempt=index, feedback_loop=True, max_repairs=max_repairs))
                generate(candidate)
                verify(candidate)
                run_worker("agent_loop.insert", "compare", candidate, "comparison.log")
                paired = load(candidate / "paired_comparison.json")
                judged = selection(load(best / "visual_review.json"), load(candidate / "visual_review.json"), paired["paired_judgment"])
                if (judged["promoted"] != paired["promoted"] or paired["best_generated_sha256"] != sha(best / "generated.mp4")
                        or paired["candidate_generated_sha256"] != sha(candidate / "generated.mp4")
                        or paired["context_sha256"] != sha(candidate / "loop_context.json")):
                    raise ValueError("Paired selection differs from its recorded evidence")
                journal["rounds"][key].update(stage="done", generated=True, comparison=paired,
                    generated_sha256=sha(candidate / "generated.mp4"))
                if paired["promoted"]:
                    _publish(root, candidate, index, journal)
                dump(journal_file, journal)
                print("REPAIR_FINISHED", index, json.dumps(paired, ensure_ascii=False), flush=True)
            except Exception as error:
                journal["rounds"][key].update(stage="failed", error=str(error))
                dump(journal_file, journal)
                raise
        final = load(root / f"attempt_{journal['best_attempt']:02d}" / "visual_review.json")
        _finish(root, journal, "evidence_limit" if only_evidence_limit(final) else "budget_exhausted")

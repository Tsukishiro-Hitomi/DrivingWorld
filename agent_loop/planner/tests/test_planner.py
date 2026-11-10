"""Synthetic plans test graph/runtime contracts; production plans come only from Qwen."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import Namespace
import unittest
from unittest.mock import patch

import numpy as np

from agent_loop.runtime import dump, load, sha
from agent_loop.planner import agent, executor, registry
from agent_loop.planner.contracts import validate, verify_origin


def task(task_id, who, parent=None):
    parameters = (dict(target_description="front-left white hatchback", target_selector="front_left_white_hatchback", replacement_category="box_truck")
                  if who == "replace" else dict(profile="dense_fog"))
    return dict(id=task_id, agent=who, depends_on=[parent] if parent else [],
                input_video=parent+".generated_video" if parent else "request.source_video",
                instruction="Synthetic subtask for interface testing", parameters=parameters,
                preserve=["source motion"], acceptance_criteria=["requested change visible"], reason="Synthetic dependency")


def graph(first="replace", second="weather"):
    return dict(status="ready", reason="Synthetic graph", unresolved=[],
                tasks=[task("later", second, "earlier"), task("earlier", first)],
                final_task="later", final_acceptance=[dict(id="all_goals", description="All requested changes remain in final output")])


def fixture(out, plan):
    source = out / "source.mp4"
    source.write_bytes(b"source fixture")
    request = dict(instruction="Synthetic request", source_video=str(source), source_sha256=sha(source),
                   replacement_image=None, execution_policy=dict(feedback_loop=False, max_retries=0))
    dump(out / "request.json", request)
    dump(out / "capabilities.json", registry.capabilities())
    dump(out / "planner_input.json", [dict(role="user", content="Synthetic test model input")])
    (out / "source_frames.jpg").write_bytes(b"preview fixture")
    dump(out / "outer_plan_response_0.json", dict(text=json.dumps(plan)))
    plan = deepcopy(plan)
    plan["provenance"] = dict(raw_response_file="outer_plan_response_0.json", raw_response_sha256=sha(out / "outer_plan_response_0.json"),
        request_sha256=sha(out / "request.json"), capabilities_sha256=sha(out / "capabilities.json"),
        planner_input_sha256=sha(out / "planner_input.json"), source_sheet_sha256=sha(out / "source_frames.jpg"))
    dump(out / "plan.json", plan)
    return request, plan


class PlannerContractTests(unittest.TestCase):
    def test_order_comes_from_dependencies_not_list_order_or_agent_type(self):
        for first, second in [("replace", "weather"), ("weather", "replace")]:
            plan = graph(first, second)
            checked = validate(plan, registry.capabilities(), {})
            self.assertEqual(checked["execution_order"], ["earlier", "later"])
            self.assertEqual(plan["tasks"][1]["agent"], first)

    def test_cycle_and_inconsistent_artifacts_are_rejected(self):
        plan = graph()
        plan["tasks"][1].update(depends_on=["later"], input_video="later.generated_video")
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate(plan, registry.capabilities(), {})
        plan = graph()
        plan["tasks"][0]["input_video"] = "/invented/result.mp4"
        with self.assertRaises(ValueError):
            validate(plan, registry.capabilities(), {})

    def test_vehicle_categories_and_target_cues_are_not_limited_to_a_pilot(self):
        plan = graph()
        for category in ("fire_engine", "city bus", "ambulance"):
            plan["tasks"][1]["parameters"].update(replacement_category=category,
                target_description="right dark sedan", target_selector="dark vehicle in right lane")
            self.assertEqual(validate(plan, registry.capabilities(), {})["capability_violations"], [])
        with self.assertRaisesRegex(ValueError, "falsely claims readiness"):
            validate(plan, registry.capabilities(), {"replacement_image": "photo.png"})

    def test_unknown_executors_and_unconsumed_branches_are_rejected(self):
        plan = graph()
        plan["tasks"][0]["agent"] = "shell"
        with self.assertRaises(ValueError):
            validate(plan, registry.capabilities(), {})
        plan = graph()
        plan["tasks"][0].update(depends_on=[], input_video="request.source_video")
        with self.assertRaisesRegex(ValueError, "do not contribute"):
            validate(plan, registry.capabilities(), {})

    def test_plan_tampering_breaks_model_authorship_binding(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            _, plan = fixture(out, graph())
            verify_origin(out, plan)
            plan["tasks"][0]["instruction"] = "Host-edited task"
            with self.assertRaisesRegex(ValueError, "modified after"):
                verify_origin(out, plan)


class PlannerRuntimeTests(unittest.TestCase):
    def test_planning_entry_never_executes_a_child(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "source.mp4"
            source.write_bytes(b"source")
            proposal = graph("weather", "replace")
            def worker(module, action, target, log):
                def model(messages, path, role, max_tokens):
                    dump(path / "outer_plan_response_0.json", dict(text=json.dumps(proposal)))
                    return deepcopy(proposal)
                with patch.object(agent, "qwen_json", side_effect=model):
                    agent.build(target)
            with (patch.object(agent, "read_frames", return_value=np.zeros((1, 48, 96, 3), np.uint8)),
                  patch.object(agent, "run_worker", side_effect=worker),
                  patch.object(registry, "invoke", side_effect=AssertionError("Generation during planning")),
                  patch.object(executor, "execute", side_effect=AssertionError("Execution during planning"))):
                agent.prepare(Namespace(out=out, source=source, instruction="Synthetic request", replacement_image=None))
            self.assertEqual(load(out / "plan.json")["tasks"], proposal["tasks"])
            self.assertEqual(load(out / "state.json")["stage"], "planned")
            self.assertFalse((out / "generated.mp4").exists())

    def test_scheduler_consumes_the_declared_parent_and_does_not_regenerate(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            request, _ = fixture(out, graph("weather", "replace"))
            calls = []
            def runner(task, source, target):
                calls.append((task["agent"], source.read_bytes()))
                file = target / "generated.mp4"
                file.write_bytes(source.read_bytes()+b"_"+task["agent"].encode())
                return dict(path=str(file), sha256=sha(file), local_verdict="PASS")
            def reviewer(*args):
                return dict(checks={"all_goals": dict(status="PASS", reason="Synthetic fixture")})
            executor.execute(out, runner=runner, reviewer=reviewer)
            self.assertEqual(calls, [("weather", b"source fixture"), ("replace", b"source fixture_weather")])
            executor.execute(out, runner=runner, reviewer=reviewer)
            self.assertEqual(len(calls), 2)
            self.assertEqual(load(out / "execution.json")["max_retries"], 0)
            self.assertEqual(load(out / "state.json")["independent_acceptance"], "NOT_ESTABLISHED")

    def test_failed_task_is_not_automatically_retried(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            fixture(out, graph())
            with self.assertRaises(RuntimeError):
                executor.execute(out, runner=lambda *args: (_ for _ in ()).throw(RuntimeError("Fixture failure")))
            with patch.object(executor, "run_task") as runner, self.assertRaisesRegex(ValueError, "retry is disabled"):
                executor.execute(out)
            runner.assert_not_called()

    def test_weather_adapter_forces_zero_generation_retries(self):
        with patch("agent_loop.weather.dispatch.run") as child:
            registry.invoke(task("climate", "weather"), Path("source.mp4"), Path("out"))
        self.assertEqual(child.call_args.args[0].max_retries, 0)

    def test_replacement_adapter_passes_exact_task_parameters(self):
        proposal = task("vehicle", "replace")
        proposal["parameters"].update(replacement_category="fire_engine", target_selector="right lane")
        with patch("agent_loop.replace.dispatch.run") as child:
            registry.invoke(proposal, Path("source.mp4"), Path("out"))
        args = child.call_args.args[0]
        self.assertEqual(args.task_parameters, proposal["parameters"])
        self.assertEqual(args.conditioning, "text-only")

    def test_unready_plan_cannot_start_execution(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            proposal = graph()
            proposal.update(status="needs_capability", unresolved=["Missing adapter"])
            fixture(out, proposal)
            with patch.object(executor, "run_task") as runner, self.assertRaisesRegex(ValueError, "not declared"):
                executor.execute(out)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()

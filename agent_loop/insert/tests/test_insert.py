"""Synthetic interface fixtures; real pilot plans are authored by Qwen."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from agent_loop.acceptance import assess
from agent_loop import dispatcher
from agent_loop.runtime import dump, load, sha
from agent_loop.insert import contracts, dispatch, executor, verifier
from agent_loop.planner import registry, executor as outer_executor
from agent_loop.planner.contracts import validate as validate_outer


def proposal(description="unusual test object"):
    return dict(schema_version=1, status="ready", reason="Synthetic test plan", unresolved=[],
        source_observations="Visible road and curb", object=dict(description=description,
            appearance="violet rough surface", count=1, chosen_defaults=[]),
        placement=dict(description="near the visible curb", contact_description="road contact",
                       scale_description="relative to the visible car"),
        event=dict(appearance_mode="enter_frame", start_frame=24, end_frame=95,
            entry_region="road edge", path_description="across the visible lane",
            motion_description="continuous visible rolling", end_state="stays visible near the opposite curb"),
        preserve=["original traffic and weather"], acceptance_criteria={key: "Observe " + key for key in contracts.CHECKS})


def fixture(out, plan=None):
    plan = deepcopy(plan or proposal())
    source = out / "source.mp4"
    source.write_bytes(b"source fixture")
    request = dict(source=str(source), source_sha256=sha(source), frames=124, fps=24,
                   width=832, height=480, seed=20261028, steps=8, instruction="Synthetic request")
    dump(out / "request.json", request)
    dump(out / "planner_input.json", [{"role": "user", "content": "Synthetic planning input"}])
    (out / "source_frames.jpg").write_bytes(b"preview fixture")
    dump(out / "insert_plan_response_0.json", dict(text=json.dumps(plan)))
    plan["provenance"] = dict(request_sha256=sha(out / "request.json"),
        planner_input_sha256=sha(out / "planner_input.json"), source_sheet_sha256=sha(out / "source_frames.jpg"),
        raw_response_file="insert_plan_response_0.json", raw_response_sha256=sha(out / "insert_plan_response_0.json"))
    dump(out / "plan.json", plan)
    return request, plan


def review():
    return dict(checks={key: dict(status="PASS", reason="Synthetic observed evidence", evidence_frames=[24])
                        for key in contracts.CHECKS}, observations=[], limitations=[])


class InsertionContractsTests(unittest.TestCase):
    def test_open_vocabulary_plan_and_generic_prompt(self):
        for name in ("stroller", "kitten", "rolling rock", "irregular translucent creature"):
            plan = proposal(name)
            contracts.validate(plan, {"frames": 124})
            prompt = executor.h3_prompt(plan, 24)
            self.assertIn(name, prompt)
            self.assertIn(plan["object"]["appearance"], prompt)
            self.assertIn("frame 24 (1.000 seconds)", prompt)
            self.assertIn("not copied from any existing source object", prompt)
            self.assertNotIn("<Picture", prompt)
            self.assertNotIn("white box truck", prompt)

    def test_invalid_timing_and_multiple_object_claims_are_rejected(self):
        for patch_value in (dict(start_frame=-1), dict(end_frame=124), dict(start_frame=True),
                            dict(start_frame=95), dict(appearance_mode="present_from_start")):
            plan = proposal()
            plan["event"].update(patch_value)
            with self.assertRaises(ValueError):
                contracts.validate(plan, {"frames": 124})
        plan = proposal()
        plan["object"]["count"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one"):
            contracts.validate(plan, {"frames": 124})

    def test_agent_plan_edits_break_origin_binding(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            _, plan = fixture(out)
            contracts.verify_origin(out, plan)
            plan["event"]["start_frame"] = 40
            with self.assertRaisesRegex(ValueError, "modified after"):
                contracts.verify_origin(out, plan)

    def test_unready_agent_plan_cannot_generate(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            plan = proposal()
            plan.update(status="needs_capability", unresolved=["Multiple objects are unsupported"])
            fixture(out, plan)
            with patch.object(executor.subprocess, "run") as engine, self.assertRaisesRegex(ValueError, "not ready"):
                executor.generate(out)
            engine.assert_not_called()


class InsertionAcceptanceTests(unittest.TestCase):
    def test_limitations_change_effective_acceptance_without_overwriting_raw_checks(self):
        result = review()
        result["limitations"] = [dict(criteria=["motion_mechanics"], impact="unknown", reason="Rotation is unobservable")]
        evaluated = assess(result, contracts.CHECKS, observed_frames=[24])
        self.assertEqual(evaluated["sampled_verdict"], "PASS")
        self.assertEqual(evaluated["acceptance_verdict"], "UNKNOWN")
        self.assertEqual(result["checks"]["motion_mechanics"]["status"], "PASS")

    def test_observed_failure_dominates_unknown_limits(self):
        result = review()
        result["limitations"] = [dict(criteria=["object_count"], impact="fail", reason="Two added objects"),
                                 dict(criteria=["object_count"], impact="unknown", reason="Partly occluded")]
        self.assertEqual(assess(result, contracts.CHECKS)["acceptance_verdict"], "FAIL")

    def test_sampled_frames_cannot_certify_full_sequence(self):
        result = verifier.assess_insertion(review(), dict(observed_frames=[0, 24, 123], frame_count=124))
        self.assertEqual(result["effective_checks"]["temporal_continuity"]["status"], "UNKNOWN")
        self.assertEqual(result["acceptance_verdict"], "UNKNOWN")

    def test_equivalent_label_and_single_limit_forms_preserve_raw_judgments(self):
        result = review()
        for check in result["checks"].values():
            check["evidence_frames"] = ["OUTPUT frame 24 / 1.000s"]
        result["checks"]["event_timing"]["status"] = "FAIL"
        result["limitations"] = dict(criteria=["motion_mechanics"], impact="unknown", reason="Wheel rotation unclear")
        assessed = verifier.assess_insertion(result, dict(observed_frames=[0, 24, 123], frame_count=124))
        self.assertEqual(assessed["checks"]["event_timing"]["status"], "FAIL")
        self.assertEqual(assessed["checks"]["object_identity"]["evidence_frames"], [24])
        self.assertEqual(assessed["raw_checks"]["object_identity"]["evidence_frames"], ["OUTPUT frame 24 / 1.000s"])
        self.assertEqual(assessed["raw_limitations"], result["limitations"])
        self.assertEqual(assessed["acceptance_verdict"], "FAIL")

    def test_unmapped_limitations_and_invented_evidence_are_rejected(self):
        for limits in (["only some frames observed"], [dict(criteria=[], impact="unknown", reason="missing evidence")],
                       [dict(criteria=["invented"], impact="fail", reason="wrong")]):
            result = review()
            result["limitations"] = limits
            with self.assertRaises(ValueError):
                assess(result, contracts.CHECKS)
        with self.assertRaisesRegex(ValueError, "unobserved frame"):
            assess(review(), contracts.CHECKS, observed_frames=[0, 123])

    def test_event_evidence_uses_actual_frame_ids_and_covers_event_edges(self):
        with TemporaryDirectory() as folder:
            indices = verifier.event_sheet(Path(folder), np.zeros((124, 48, 64, 3), np.uint8), proposal(), 24)
            self.assertTrue({0, 23, 24, 25, 94, 95, 96, 123} <= set(indices))
            self.assertTrue(all(0 <= i < 124 for i in indices))
            self.assertTrue((Path(folder) / "review_event_frames.jpg").exists())


class InsertionRuntimeTests(unittest.TestCase):
    def test_video_text_manifest_and_failed_attempt_latch(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            fixture(out)
            with patch.object(executor.subprocess, "run") as engine:
                executor.generate(out)
            config = load(out / "config.json")
            self.assertEqual(config["reference_image_count"], 0)
            self.assertIsNone(config["aligned_warp"])
            self.assertIsNone(config["first_frame_anchor"])
            self.assertEqual(engine.call_args.args[0][1:4], ["-m", "agent_loop.insert.executor", "infer"])
            with patch.object(executor.subprocess, "run") as engine, self.assertRaisesRegex(ValueError, "retry is disabled"):
                executor.generate(out)
            engine.assert_not_called()

    def test_changed_source_is_rejected_before_generation(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            request, _ = fixture(out)
            Path(request["source"]).write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "source video changed"):
                executor.generate(out)

    def test_plan_only_uses_agent_response_without_video_execution(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            source = out / "source.mp4"
            source.write_bytes(b"source fixture")
            authored = proposal()
            def worker(module, action, target, log):
                def model(messages, target, role, max_tokens):
                    dump(target / "insert_plan_response_0.json", dict(text=json.dumps(authored)))
                    return deepcopy(authored)
                with patch.object(dispatch, "qwen_json", side_effect=model):
                    dispatch.plan(target)
            with (patch.object(dispatch, "read_frames", return_value=np.zeros((2, 48, 64, 3), np.uint8)),
                  patch.object(dispatch, "run_worker", side_effect=worker),
                  patch.object(executor, "generate") as generate):
                dispatch.run(Namespace(source=source, out=out, instruction="Synthetic request", plan_only=True))
            generate.assert_not_called()
            self.assertEqual(load(out / "plan.json")["event"], authored["event"])
            self.assertFalse(load(out / "state.json")["generation_started"])

    def test_image_input_is_not_silently_ignored(self):
        with TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "video and text only"):
            dispatch.initialize(Namespace(replacement_image=Path("reference.png")), Path(folder))

    def test_dispatcher_routes_insert_without_calling_replace_or_weather(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            source = out / "source.mp4"
            source.write_bytes(b"source fixture")
            args = Namespace(source=source, out=out, instruction="Add a new unusual object", agent="auto",
                             profile=None, plan_only=True, replacement_image=None, edited_first_frame=None)
            def classify(module, action, target, log):
                dump(target / "dispatch.json", dict(agent="insert", profile=None, reason="A new object",
                     request_sha256=sha(target / "dispatch_request.json")))
            with (patch.object(dispatcher, "run_worker", side_effect=classify),
                  patch.object(dispatch, "run") as insert,
                  patch("agent_loop.replace.dispatch.run") as replace,
                  patch("agent_loop.weather.dispatch.run") as weather):
                dispatcher.run(args)
            insert.assert_called_once()
            replace.assert_not_called()
            weather.assert_not_called()

    def test_outer_registry_supports_insert_and_passes_parameters(self):
        task = dict(id="new_object", agent="insert", depends_on=[], input_video="request.source_video",
            instruction="Add a new unusual object", parameters=dict(object_description="irregular object", event_description="rolls in"),
            preserve=["source traffic"], acceptance_criteria=["one new object"], reason="New insertion")
        plan = dict(status="ready", reason="One edit", unresolved=[], tasks=[task], final_task=task["id"],
                    final_acceptance=[dict(id="inserted", description="Object added as requested")])
        self.assertEqual(validate_outer(plan, registry.capabilities(), {})["execution_order"], [task["id"]])
        with patch.object(dispatch, "run") as child:
            registry.invoke(task, Path("source.mp4"), Path("out"))
        self.assertEqual(child.call_args.args[0].task_parameters, task["parameters"])
        with self.assertRaisesRegex(ValueError, "falsely claims readiness"):
            validate_outer(plan, registry.capabilities(), {"replacement_image": "unsupported.png"})

    def test_outer_scheduler_retains_effective_unknown_from_a_child(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            source = out / "source.mp4"
            source.write_bytes(b"source")
            def run(*args):
                (out / "generated.mp4").write_bytes(b"output")
                dump(out / "request.json", dict(source_sha256=sha(source)))
                dump(out / "state.json", dict(sampled_visual_verdict="PASS", acceptance_verdict="UNKNOWN"))
            with patch("agent_loop.planner.executor.invoke", side_effect=run):
                result = outer_executor.run_task(dict(agent="insert"), source, out)
            self.assertEqual(result["local_verdict"], "UNKNOWN")
            self.assertEqual(result["sampled_visual_verdict"], "PASS")

    def test_assessment_only_mode_cannot_start_a_missing_edit(self):
        from agent_loop.planner.tests.test_planner import fixture as outer_fixture, graph

        with TemporaryDirectory() as folder:
            out = Path(folder)
            outer_fixture(out, graph())
            with (patch("agent_loop.planner.executor.invoke") as generate,
                  self.assertRaisesRegex(ValueError, "cannot start or regenerate")):
                outer_executor.execute(out, assessment_only=True)
            generate.assert_not_called()

    def test_recorded_assessment_recovery_never_calls_a_generator(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            request, _ = fixture(out)
            (out / "generated.mp4").write_bytes(b"generated fixture")
            dump(out / "state.json", dict(stage="failed", error="representation mismatch"))
            outcome = dict(sampled_verdict="FAIL", acceptance_verdict="FAIL")
            with (patch("agent_loop.insert.verifier.assess_recorded") as normalize,
                  patch("agent_loop.insert.verifier.verify", return_value=outcome),
                  patch.object(executor, "generate") as generate):
                artifact = outer_executor.reassess_task(dict(agent="insert"), Path(request["source"]), out)
            normalize.assert_called_once_with(out)
            generate.assert_not_called()
            self.assertEqual(artifact["local_verdict"], "FAIL")
            self.assertFalse(load(out / "state.json")["generation_retried"])
            self.assertEqual(load(out / "assessment_previous_state.json")["stage"], "failed")


if __name__ == "__main__":
    unittest.main()

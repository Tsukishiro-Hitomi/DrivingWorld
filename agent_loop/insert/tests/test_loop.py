"""Synthetic tests for budget, frozen goals, provenance and prompt promotion."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from agent_loop.runtime import dump, load, sha
from agent_loop.insert import loop, executor, verifier, contracts
from agent_loop.insert.tests.test_insert import fixture, review


def evaluation(timing="FAIL", mechanics="FAIL"):
    result = review()
    result["checks"]["event_timing"]["status"] = timing
    result["checks"]["motion_mechanics"]["status"] = mechanics
    return verifier.assess_insertion(result, dict(observed_frames=[0, 24, 123], frame_count=124))


class RepairContractTests(unittest.TestCase):
    def test_agent_patch_cannot_change_timeline_identity_or_acceptance(self):
        with TemporaryDirectory() as name:
            request, base = fixture(Path(name))
            for change in ({"event": {"start_frame": 0}}, {"object": {"count": 2}},
                           {"acceptance_criteria": {"motion_mechanics": "Accept sliding"}}, {"seed": 99}):
                proposal = dict(plan_patch=change, execution_directive="New wording", reason="Repair")
                with self.assertRaises(ValueError):
                    contracts.apply_repair(base, base, proposal, request)
            plan = contracts.apply_repair(base, base, dict(plan_patch={"event": {"motion_description": "Visible rotation"}},
                execution_directive="Keep the original event timing with rotating surface features", reason="Observed sliding"), request)
            self.assertEqual(plan["event"]["start_frame"], base["event"]["start_frame"])
            self.assertEqual(plan["acceptance_criteria"], base["acceptance_criteria"])
            self.assertNotIn("provenance", plan)

    def test_paired_preference_cannot_hide_a_formal_regression(self):
        before, after = evaluation("PASS", "FAIL"), evaluation("FAIL", "PASS")
        choice = loop.selection(before, after, dict(decision="candidate", improved_criteria=["motion_mechanics"], regressions=[], reason="Improved motion"))
        self.assertFalse(choice["promoted"])
        self.assertIn("event_timing", choice["formal_regressions"])

    def test_equal_pass_counts_are_not_the_selection_authority(self):
        result = evaluation()
        choice = loop.selection(result, result, dict(decision="candidate", improved_criteria=["motion_mechanics"], regressions=[], reason="Visible partial rotation instead of complete sliding"))
        self.assertTrue(choice["promoted"])
        choice = loop.selection(result, result, dict(decision="inconclusive", improved_criteria=[], regressions=[], reason="No visible change"))
        self.assertFalse(choice["promoted"])

    def test_identical_media_cannot_be_promoted(self):
        before, after = evaluation(), evaluation("PASS", "PASS")
        before["generated_sha256"] = after["generated_sha256"] = "identical"
        choice = loop.selection(before, after, dict(decision="candidate", improved_criteria=["event_timing"], regressions=[], reason="Claimed improvement"))
        self.assertFalse(choice["promoted"])

    def test_only_coverage_unknown_stops_without_changing_the_verdict(self):
        result = evaluation("PASS", "PASS")
        self.assertTrue(loop.only_evidence_limit(result))
        self.assertEqual(result["acceptance_verdict"], "UNKNOWN")
        self.assertFalse(loop.only_evidence_limit(evaluation("PASS", "UNKNOWN")))

    def test_budget_has_a_hard_three_repair_limit(self):
        with TemporaryDirectory() as name:
            for value in (-1, 4, True):
                with self.assertRaisesRegex(ValueError, "zero and three"):
                    loop.run(Path(name), value)


class RepairRuntimeTests(unittest.TestCase):
    def harness(self, path, baseline_eval, outcomes, choices, stop=False):
        baseline, out = path / "baseline", path / "loop"
        baseline.mkdir()
        fixture(baseline)
        calls = []
        def generate(target):
            if (target / "generated.mp4").exists():
                return
            calls.append(target.name)
            plan, request = load(target / "plan.json"), load(target / "request.json")
            config = dict(prompt=executor.h3_prompt(plan, request["fps"]), source=request["source"],
                          source_sha256=request["source_sha256"], plan_sha256=sha(target / "plan.json"))
            dump(target / "config.json", config)
            (target / "generated.mp4").write_bytes(target.name.encode())
            for file in ("comparison.mp4", "review_frames.jpg", "review_event_frames.jpg"):
                (target / file).write_bytes(b"synthetic evidence")
        def verify(target):
            if (target / "visual_review.json").exists():
                return load(target / "visual_review.json")
            index = int(target.name[-2:])
            result = deepcopy(outcomes[index - 1])
            result["generated_sha256"] = sha(target / "generated.mp4")
            dump(target / "visual_review.json", result)
            return result
        generate(baseline)
        base_result = deepcopy(baseline_eval)
        base_result["generated_sha256"] = sha(baseline / "generated.mp4")
        dump(baseline / "visual_review.json", base_result)
        calls.clear()
        def worker(module, action, target, log):
            index = int(target.name[-2:])
            if action == "repair-plan":
                if index == 2:
                    self.assertIn("Directive one", (out / "prompt.txt").read_text())
                if index == 3:
                    self.assertEqual(load(target / "loop_context.json")["best"], str(out / "attempt_01"))
                proposal = dict(retryable=not stop, reason="Observed failure needs text correction",
                    diagnosis=[dict(criterion="motion_mechanics", observation="Observed mechanics failure", action="Clarify visible motion")],
                    plan_patch={"event": {"motion_description": "Refined rotation wording " + str(index)}},
                    execution_directive=["Directive one", "Directive two", "Directive three"][index - 1])
                if stop:
                    proposal["diagnosis"] = []
                def model(messages, target, role, max_tokens):
                    dump(target / "insert_repair_response_0.json", dict(text=json.dumps(proposal)))
                    return deepcopy(proposal)
                with patch.object(loop, "qwen_json", side_effect=model):
                    loop.repair(target)
                if not stop:
                    contracts.verify_origin(target, load(target / "plan.json"))
            elif action == "compare":
                with patch.object(loop, "qwen_json", return_value=choices[index - 1]):
                    loop.compare(target)
            else:
                raise AssertionError("Unexpected worker action")
        return baseline, out, calls, generate, verify, worker

    def test_three_repairs_use_best_parent_and_publish_prompt_only_for_improvement(self):
        with TemporaryDirectory() as name:
            judgments = [dict(decision="candidate", improved_criteria=["event_timing"], regressions=[], reason="Timing improved"),
                         dict(decision="candidate", improved_criteria=["motion_mechanics"], regressions=[], reason="Motion improved but timing regressed"),
                         dict(decision="candidate", improved_criteria=["motion_mechanics"], regressions=[], reason="Motion improved and timing retained")]
            baseline, out, calls, gen, check, worker = self.harness(Path(name), evaluation(),
                [evaluation("PASS", "FAIL"), evaluation("FAIL", "PASS"), evaluation("PASS", "PASS")], judgments)
            before_hash = sha(baseline / "generated.mp4")
            with (patch.object(executor, "generate", side_effect=gen), patch.object(verifier, "verify", side_effect=check),
                  patch.object(loop, "run_worker", side_effect=worker)):
                loop.run(out, 3, baseline)
                loop.run(out, 3, baseline)  # Finished resume cannot generate again.
            self.assertEqual(calls, ["attempt_01", "attempt_02", "attempt_03"])
            self.assertEqual([v["attempt"] for v in load(out / "prompt_history.json")], [0, 1, 3])
            self.assertEqual(load(out / "best.json")["attempt"], 3)
            self.assertIn("Directive three", (out / "prompt.txt").read_text())
            self.assertEqual(load(out / "state.json")["acceptance_verdict"], "UNKNOWN")
            self.assertEqual(sha(baseline / "generated.mp4"), before_hash)
            plan = load(out / "attempt_03/plan.json")
            plan["execution_directive"] = "Host rewrite"
            with self.assertRaisesRegex(ValueError, "modified after"):
                contracts.verify_origin(out / "attempt_03", plan)

    def test_unknown_coverage_only_does_not_spend_video_repair_budget(self):
        with TemporaryDirectory() as name:
            baseline, out, calls, gen, check, worker = self.harness(Path(name), evaluation("PASS", "PASS"), [], [])
            with (patch.object(executor, "generate", side_effect=gen), patch.object(verifier, "verify", side_effect=check),
                  patch.object(loop, "run_worker") as model):
                loop.run(out, 3, baseline)
            self.assertFalse(calls)
            model.assert_not_called()
            self.assertEqual(load(out / "state.json")["stop_reason"], "evidence_limit")
            self.assertEqual(load(out / "state.json")["new_generations"], 0)

    def test_agent_can_stop_instead_of_inventing_a_repair(self):
        with TemporaryDirectory() as name:
            baseline, out, calls, gen, check, worker = self.harness(Path(name), evaluation(), [], [], stop=True)
            with (patch.object(executor, "generate", side_effect=gen), patch.object(verifier, "verify", side_effect=check),
                  patch.object(loop, "run_worker", side_effect=worker)):
                loop.run(out, 3, baseline)
            self.assertFalse(calls)
            self.assertEqual(load(out / "state.json")["stop_reason"], "agent_stopped")

    def test_generation_failure_does_not_hide_extra_retries(self):
        with TemporaryDirectory() as name:
            baseline, out, calls, gen, check, worker = self.harness(Path(name), evaluation(), [], [])
            def broken(target):
                if target.name == "attempt_01":
                    raise RuntimeError("Synthetic engine failure")
                return gen(target)
            with (patch.object(executor, "generate", side_effect=broken), patch.object(verifier, "verify", side_effect=check),
                  patch.object(loop, "run_worker", side_effect=worker)):
                with self.assertRaisesRegex(RuntimeError, "engine failure"):
                    loop.run(out, 3, baseline)
                with self.assertRaisesRegex(ValueError, "unfinished repair"):
                    loop.run(out, 3, baseline)
            self.assertEqual(len(load(out / "loop.json")["rounds"]), 1)


if __name__ == "__main__":
    unittest.main()

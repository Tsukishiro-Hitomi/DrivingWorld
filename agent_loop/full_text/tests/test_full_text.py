"""Synthetic contracts for a real zero-reference T2VA execution path."""

from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_loop import dispatcher
from agent_loop.acceptance import assess
from agent_loop.runtime import dump, load, sha
from agent_loop.full_text import contracts, dispatch, executor
from agent_loop.full_text.prepare_weights import identity
from agent_loop.planner import agent as outer_agent, executor as outer_executor, registry
from agent_loop.planner.contracts import validate as validate_outer, verify_origin as verify_outer


def proposal():
    return dict(status="ready", reason="Synthetic scene plan", unresolved=[], chosen_defaults=["One continuous dashcam shot"],
        scene={key: "Synthetic " + key for key in contracts.SCENE_FIELDS},
        prompt=dict(integrated_multimodal_description="[Shot 1] A driving view with severe atmospheric obscuration and lamps visible.",
                    overall_soundscape="N/A", non_diegetic_music="N/A"),
        acceptance_criteria={key: dict(description="Observe " + key,
            scope="full_sequence" if key == "temporal_consistency" else "sampled_frames") for key in contracts.CORE_CHECKS})


def fixture(out):
    request = dict(instruction="Synthetic severe visibility scene", source=None, source_sha256=None,
        frames=124, fps=24, width=832, height=480, seed=20261028, steps=50)
    dump(out / "request.json", request)
    dump(out / "planner_input.json", [{"role": "user", "content": request["instruction"]}])
    model_plan = proposal()
    dump(out / "full_text_plan_response_0.json", dict(text=json.dumps(model_plan)))
    model_plan["provenance"] = dict(raw_response_file="full_text_plan_response_0.json",
        raw_response_sha256=sha(out / "full_text_plan_response_0.json"), planner_input_sha256=sha(out / "planner_input.json"),
        request_sha256=sha(out / "request.json"))
    dump(out / "plan.json", model_plan)
    return request, model_plan


class FullTextContractTests(unittest.TestCase):
    def test_native_prompt_has_three_fields_and_no_fake_media(self):
        plan = proposal()
        contracts.validate(plan)
        prompt = contracts.compile_prompt(plan)
        self.assertTrue(prompt.startswith("integrated_multimodal_description:"))
        self.assertNotIn("subject_definitions", prompt)
        self.assertNotIn("<Video", prompt)
        plan["prompt"]["integrated_multimodal_description"] += " Preserve <Video 1>."
        with self.assertRaisesRegex(ValueError, "nonexistent input"):
            contracts.validate(plan)

    def test_temporal_goal_cannot_be_relabelled_as_sampled(self):
        plan = proposal()
        plan["acceptance_criteria"]["temporal_consistency"]["scope"] = "sampled_frames"
        with self.assertRaisesRegex(ValueError, "keep its evidence scope"):
            contracts.validate(plan)

    def test_planner_authorship_binding_rejects_manual_prompt_changes(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            _, plan = fixture(out)
            contracts.verify_origin(out, plan)
            plan["prompt"]["integrated_multimodal_description"] += " Host rewritten scene."
            with self.assertRaisesRegex(ValueError, "modified after"):
                contracts.verify_origin(out, plan)

    def test_checkpoint_identity_validates_git_blobs_and_lfs_files(self):
        with TemporaryDirectory() as folder:
            p = Path(folder) / "weight"
            p.write_bytes(b"verified fixture")
            data = p.read_bytes()
            self.assertTrue(identity(p, dict(size=len(data), lfs=dict(sha256=hashlib.sha256(data).hexdigest()))))
            self.assertTrue(identity(p, dict(size=len(data), blobId=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())))
            self.assertFalse(identity(p, dict(size=len(data), lfs=dict(sha256="wrong"))))

    def test_sampled_scene_review_keeps_full_sequence_unknown(self):
        checks = {key: dict(status="PASS", reason="Observed sample", evidence_frames=[0]) for key in contracts.CORE_CHECKS}
        result = assess(dict(checks=checks, limitations=[]), contracts.CORE_CHECKS, observed_frames=[0, 123],
                        frame_count=124, full_sequence_ids=["temporal_consistency"])
        self.assertEqual(result["effective_checks"]["temporal_consistency"]["status"], "UNKNOWN")

    def test_unobserved_frame_citations_quarantine_the_raw_review(self):
        raw = dict(checks={key: dict(status="PASS", reason="Model assertion", evidence_frames=[60])
                           for key in contracts.CORE_CHECKS}, limitations=[])
        result = contracts.evaluate_review(raw, proposal(), dict(observed_frames=[0, 123], frame_count=124))
        self.assertFalse(result["review_valid"])
        self.assertEqual(result["raw_result"], raw)
        self.assertEqual(result["acceptance_verdict"], "UNKNOWN")
        self.assertIn("unobserved", result["review_error"])

    def test_valid_review_preserves_full_sequence_evidence_gate(self):
        raw = dict(checks={key: dict(status="PASS", reason="Observed", evidence_frames=[0])
                           for key in contracts.CORE_CHECKS}, limitations=[])
        result = contracts.evaluate_review(raw, proposal(), dict(observed_frames=[0, 123], frame_count=124))
        self.assertTrue(result["review_valid"])
        self.assertEqual(result["sampled_verdict"], "PASS")
        self.assertEqual(result["acceptance_verdict"], "UNKNOWN")


class FullTextRoutingTests(unittest.TestCase):
    def test_no_source_is_not_an_editable_weather_video(self):
        dispatcher.validate_route(dict(agent="full-text", reason="No input video", profile=None), has_source=False)
        with self.assertRaisesRegex(ValueError, "require an actual source"):
            dispatcher.validate_route(dict(agent="weather", reason="Weather request", profile="heavy_snow"), has_source=False)
        with self.assertRaisesRegex(ValueError, "discard a supplied"):
            dispatcher.validate_route(dict(agent="full-text", reason="Create", profile=None), has_source=True)

    def test_dispatcher_passes_text_only_request_to_the_new_agent(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            args = Namespace(out=out, instruction="Synthetic severe blur", source=None, agent="auto", profile=None)
            def classify(module, action, target, log):
                dump(target / "dispatch.json", dict(agent="full-text", profile=None, reason="No source",
                                                    request_sha256=sha(target / "dispatch_request.json")))
            with (patch.object(dispatcher, "run_worker", side_effect=classify),
                  patch.object(dispatch, "run") as scene):
                dispatcher.run(args)
            self.assertIsNone(scene.call_args.args[0].source)
            self.assertEqual(load(out / "dispatch_request.json")["source"], None)

    def test_full_text_rejects_supplied_visual_media(self):
        with TemporaryDirectory() as folder:
            for values in (dict(source=Path("input.mp4")), dict(replacement_image=Path("image.png"))):
                with self.assertRaisesRegex(ValueError, "text only"):
                    dispatch.initialize(Namespace(**values), Path(folder))

    def test_plan_only_has_no_images_and_never_calls_generation(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            def worker(module, action, target, log):
                def model(messages, target, role, max_tokens):
                    self.assertIsInstance(messages[1]["content"], str)
                    dump(target / "full_text_plan_response_0.json", dict(text=json.dumps(proposal())))
                    return proposal()
                with patch.object(dispatch, "qwen_json", side_effect=model):
                    dispatch.plan(target)
            with (patch.object(dispatch, "run_worker", side_effect=worker), patch.object(executor, "generate") as generate):
                dispatch.run(Namespace(out=out, instruction="Synthetic scene", plan_only=True))
            generate.assert_not_called()
            self.assertFalse((out / "source_frames.jpg").exists())
            self.assertEqual(load(out / "request.json")["reference_video_count"], 0)


class FullTextExecutionTests(unittest.TestCase):
    def test_zero_input_manifest_native_sampler_and_failed_attempt_latch(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            fixture(out)
            model = out / "model"
            model.mkdir()
            dump(model / "download_manifest.json", dict(revision="pinned"))
            (model / ".ready").write_text("pinned\n")
            with (patch.object(executor, "MODEL_ROOT", model), patch.object(executor, "checkpoint_ready", return_value=True),
                  patch.object(executor.subprocess, "run")):
                executor.generate(out)
                config = load(out / "config.json")
                self.assertEqual(config["checkpoint_variant"], "FL2VA")
                self.assertEqual(config["steps"], 50)
                self.assertEqual(config["reference_video_count"], 0)
                self.assertIsNone(config["source"])
                self.assertEqual(config["crossview_strength"], 0)
                with self.assertRaisesRegex(ValueError, "no automatic retry"):
                    executor.generate(out)

    def test_outer_planner_accepts_creation_then_editing_without_original_video(self):
        caps = registry.capabilities()
        caps["agents"]["full-text"]["available"] = True
        task = dict(id="scene", agent="full-text", depends_on=[], input_video=None,
            instruction="Create scene", parameters=dict(scene_description="Synthetic scene"),
            preserve=["user scene goals"], acceptance_criteria=["scene follows request"], reason="No input video")
        edit = dict(id="weather", agent="weather", depends_on=["scene"], input_video="scene.generated_video",
            instruction="Requested edit", parameters=dict(profile="heavy_snow"), preserve=["scene content"],
            acceptance_criteria=["weather follows request"], reason="Uses generated scene")
        plan = dict(status="ready", reason="Synthetic graph", unresolved=[], tasks=[edit, task], final_task="weather",
                    final_acceptance=[dict(id="scene_goal", description="All goals")])
        self.assertEqual(validate_outer(plan, caps, {"source_video": None})["execution_order"], ["scene", "weather"])
        edit.update(depends_on=[], input_video="request.source_video")
        with self.assertRaisesRegex(ValueError, "No source video exists"):
            validate_outer(plan, caps, {"source_video": None})

    def test_outer_scheduler_passes_none_and_keeps_effective_verdict(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            def create(task, source, target):
                self.assertIsNone(source)
                (target / "generated.mp4").write_bytes(b"generated")
                dump(target / "request.json", dict(source_sha256=None))
                dump(target / "state.json", dict(acceptance_verdict="UNKNOWN", sampled_visual_verdict="PASS"))
            with patch("agent_loop.planner.executor.invoke", side_effect=create):
                result = outer_executor.run_task(dict(agent="full-text"), None, out)
            self.assertEqual(result["local_verdict"], "UNKNOWN")

    def test_native_packed_builder_has_no_keyframe_or_reference_rows(self):
        import torch
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder

        result = MiniMaxH3Unit_PackedSequenceBuilder().process(Namespace(device="cpu"),
            prompt_embeds=torch.zeros(16, 8), video_latents=torch.zeros(1, 24, 37, 30, 52),
            audio_latents=torch.zeros(2, 32, 207), text_token_tags=torch.ones(16, dtype=torch.long),
            keyframe_cond_anchor=None, keyframe_indices=None, ref_blocks=None)["packed"]
        self.assertEqual(result["img_pos"].numel(), 37 * 15 * 26)
        self.assertEqual(result["audio_pos"].numel(), 2 * 207)
        self.assertEqual(result["text_pos"].numel(), 16)
        self.assertEqual(result["img_position_ids"].dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()

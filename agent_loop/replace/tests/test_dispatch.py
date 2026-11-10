import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_loop.replace import dispatch as dispatcher


class DrivingDispatchTests(unittest.TestCase):
    def test_runtime_and_skill_paths_follow_the_new_package(self):
        self.assertEqual(dispatcher.AGENT_SKILL,
                         dispatcher.ROOT / "agent_loop/replace/SKILL.md")
        self.assertTrue(dispatcher.AGENT_SKILL.is_file())
        self.assertTrue((dispatcher.ROOT / "crossview_v2v.py").is_file())

    def args(self, out, **overrides):
        values = dict(out=out, conditioning="auto", replacement_image=None,
                      edited_first_frame=None, instruction="replace the front-left car with a truck")
        values.update(overrides)
        return Namespace(**values)

    def test_new_requests_are_routed_by_reference_image_presence(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(dispatcher.resolve_conditioning(self.args(out)), "text-only")
            self.assertEqual(dispatcher.resolve_conditioning(
                self.args(out, replacement_image=out / "truck.png")), "appearance-guided")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                dispatcher.resolve_conditioning(self.args(
                    out, replacement_image=out / "truck.png", conditioning="text-only"))

    def test_existing_runs_keep_their_recorded_branch(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            saved = out / "request.json"
            saved.write_text(json.dumps({"conditioning": "text-only"}))
            self.assertEqual(dispatcher.resolve_conditioning(self.args(out)), "text-only")
            saved.write_text("{}")  # Historical image-guided jobs omitted this field.
            self.assertEqual(dispatcher.resolve_conditioning(self.args(out)), "appearance-guided")
            saved.write_text(json.dumps({"replacement_image": "/original/truck.png"}))
            with self.assertRaisesRegex(ValueError, "original --replacement-image"):
                dispatcher.resolve_conditioning(self.args(out))

    def test_default_text_branch_skips_the_image_editing_executor(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "route.json").write_text(json.dumps({"agent": "vehicle_replacement"}))
            with (
                patch("agent_loop.replace.executor.initialize"),
                patch("agent_loop.replace.executor.prepare") as prepare,
                patch("agent_loop.replace.executor.generate") as generate,
                patch("agent_loop.replace.executor.generate_text_only") as native,
                patch("agent_loop.replace.executor.verify"),
            ):
                dispatcher.run(self.args(out))
            native.assert_called_once_with(out)
            prepare.assert_not_called()
            generate.assert_not_called()

    def test_planner_subprocess_uses_the_new_package_entry_point(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)

            def planned(*args, **kwargs):
                (out / "route.json").write_text(json.dumps({"agent": "vehicle_replacement"}))

            with (
                patch("agent_loop.replace.executor.initialize"),
                patch("agent_loop.replace.executor.generate_text_only"),
                patch("agent_loop.replace.executor.verify"),
                patch.object(dispatcher.subprocess, "run", side_effect=planned) as worker,
            ):
                dispatcher.run(self.args(out))
            self.assertEqual(worker.call_args.args[0][1:4],
                             ["-m", "agent_loop.replace", "route"])
            self.assertEqual(worker.call_args.kwargs["cwd"], dispatcher.ROOT)

    def test_customer_image_branch_uses_the_image_guided_executor(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "route.json").write_text(json.dumps({"agent": "vehicle_replacement"}))
            edited = out / "edited_scene.png"
            with (
                patch("agent_loop.replace.executor.initialize"),
                patch("agent_loop.replace.executor.prepare") as prepare,
                patch("agent_loop.replace.executor.generate") as generate,
                patch("agent_loop.replace.executor.generate_text_only") as native,
                patch("agent_loop.replace.executor.verify"),
            ):
                dispatcher.run(self.args(
                    out, replacement_image=out / "truck.png", edited_first_frame=edited))
            prepare.assert_called_once_with(out)
            generate.assert_called_once_with(out, edited)
            native.assert_not_called()

    def test_customer_photo_is_supplied_to_the_visual_planner(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "request.json").write_text(json.dumps({
                "instruction": "Use the truck in the supplied image",
                "replacement_image": "/customer/truck.png",
            }))
            (out / "candidates.json").write_text(json.dumps([
                {"object_id": "sam-2", "bbox_xyxy": [10, 20, 30, 40]},
            ]))
            answer = dict(
                task_type="vehicle_replacement", target_id="sam-2", start_frame=0,
                appearance_source="text_generated", target_description="front-left white car",
                first_frame_edit_prompt="Match the provided truck photo.",
                replacement_description="blue box truck matching the supplied photo",
                motion_description="Keep the source motion.", replacement_category="box truck",
            )
            with patch.object(dispatcher, "qwen_json", return_value=answer) as call:
                dispatcher.route(out)
            images = [item["image"] for item in call.call_args.args[0][1]["content"]
                      if item["type"] == "image"]
            self.assertEqual(images, [str(out / "source_first_frame.png"),
                                      str(out / "replacement_reference.png")])
            self.assertEqual(json.loads((out / "route.json").read_text())["appearance_source"],
                             "user_reference")

    def test_outer_parameters_reach_the_child_agent_and_category_substitution_fails(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            parameters = dict(target_description="right dark car", target_selector="right lane",
                              replacement_category="fire_engine")
            (out / "request.json").write_text(json.dumps({
                "instruction": "replace the right car with a fire engine", "task_parameters": parameters,
            }))
            (out / "candidates.json").write_text(json.dumps([
                {"object_id": "sam-9", "bbox_xyxy": [510, 220, 620, 340]},
            ]))
            answer = dict(task_type="vehicle_replacement", target_id="sam-9", start_frame=0,
                          target_description="right dark car", replacement_category="fire_engine",
                          replacement_description="red body, hose lockers and a roof ladder",
                          first_frame_edit_prompt="Replace only sam-9 with the fire engine.",
                          motion_description="Retain its source path.")
            with patch.object(dispatcher, "qwen_json", return_value=answer.copy()) as call:
                dispatcher.route(out)
            payload = json.loads(call.call_args.args[0][1]["content"][-1]["text"])
            self.assertEqual(payload["task_parameters"], parameters)
            saved = json.loads((out / "route.json").read_text())
            self.assertEqual(saved["replacement_category"], "fire_engine")
            answer["replacement_category"] = "box truck"
            with (patch.object(dispatcher, "qwen_json", return_value=answer),
                  self.assertRaisesRegex(ValueError, "changed the outer planner")):
                dispatcher.route(out)


if __name__ == "__main__":
    unittest.main()

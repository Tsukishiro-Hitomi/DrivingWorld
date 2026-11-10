from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from agent_loop.runtime import dump, load, sha
from agent_loop.weather import dispatch, executor
from agent_loop.weather.profiles import h3_prompt, validate_plan, PROFILES
from agent_loop.weather.sensor import apply_plan_patch, review_checks, sensor_defaults, sensor_sheet
from agent_loop.weather.tests.test_weather import environment_plan


def camera_plan(profile="heavy_rain"):
    plan = environment_plan(profile)
    plan.update(prompt_version=4, sensor_effects=sensor_defaults(profile),
                execution_emphasis=dict(weather="standard", sensor="standard", preservation="standard"))
    return plan


class SensorPlanTests(unittest.TestCase):
    def test_all_profiles_have_distinct_camera_optical_contracts(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                plan = camera_plan(profile)
                validate_plan(plan, 124/24)
                prompt = h3_prompt(plan)
                self.assertIn("RGB camera", prompt)
                self.assertIn("central road", prompt)
                self.assertEqual(len(review_checks(plan)), 11)
                self.assertIn("protective glass" if profile != "strong_backlight" else "glare", prompt)
        self.assertEqual(camera_plan("strong_backlight")["sensor_effects"]["surface_condition"], "clean")

    def test_agent_patch_changes_plan_and_compiled_prompt(self):
        original = camera_plan()
        changed = apply_plan_patch(original, {"sensor_effects": {"coverage": "heavy", "strength": "strong"},
                                             "execution_emphasis": {"sensor": "strong"}}, 124/24)
        self.assertEqual(original["sensor_effects"]["coverage"], "moderate")
        self.assertEqual(changed["edits"], original["edits"])
        self.assertNotEqual(h3_prompt(changed), h3_prompt(original))
        self.assertIn("Numerous clearly visible", h3_prompt(changed))

    def test_patch_cannot_redefine_task_or_disable_sensor(self):
        for proposal in ({"profile": "dense_fog"}, {"edits": {"time_of_day": "night"}},
                         {"sensor_effects": {"surface_condition": "clean"}},
                         {"sensor_effects": {"coverage": "none"}}):
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                apply_plan_patch(camera_plan(), proposal, 124/24)

    def test_missing_or_wrong_weather_sensor_state_is_rejected(self):
        plan = camera_plan()
        plan["sensor_effects"]["surface_condition"] = "dust_deposit"
        with self.assertRaises(ValueError):
            validate_plan(plan, 124/24)
        plan = camera_plan()
        plan.pop("execution_emphasis")
        with self.assertRaises(ValueError):
            validate_plan(plan, 124/24)

    def test_camera_detail_sheet_matches_rgb_source_output_rows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sheet.png"
            source = np.full((4, 48, 96, 3), [240, 10, 10], np.uint8)
            generated = np.full_like(source, [10, 10, 240])
            sensor_sheet(path, source, generated)
            image = np.array(Image.open(path))
            self.assertEqual(image.shape, (1080, 1024, 3))
            self.assertGreater(int(image[32, 5, 0]), 200)
            self.assertGreater(int(image[212, 5, 2]), 200)


class AutonomousSensorLoopTests(unittest.TestCase):
    def test_visual_failure_drives_a_new_attempt_plan_and_actual_generation_prompt(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "input.mp4"
            source.write_bytes(b"source")
            request = dict(source=str(source), source_sha256=sha(source), frames=124, fps=24,
                           width=832, height=480, seed=20261028, steps=8, profile="heavy_rain")
            dump(out / "request.json", request)
            plan = camera_plan()
            plan["request_sha256"] = sha(out / "request.json")
            dump(out / "plan.json", plan)
            original_hash = sha(out / "plan.json")
            real_generate = executor.generate

            def generate(root, attempt, correction, plan=None):
                with patch.object(executor.subprocess, "run"):
                    real_generate(root, attempt, correction, plan=plan)
                for name in ("generated.mp4", "comparison.mp4", "review_frames.jpg", "review_sensor_frames.jpg"):
                    (attempt / name).write_bytes(b"generated fixture")

            def verify(attempt):
                failed = attempt.name == "attempt_00"
                config = load(attempt / "config.json")
                checks = {name: {"status": "PASS", "reason": "visible"} for name in review_checks(config["weather_plan"])}
                if failed:
                    checks["sensor_effect_presence"] = {"status": "FAIL", "reason": "No adhered droplets visible"}
                dump(attempt / "visual_review.json", {"checks": checks})
                return "FAIL" if failed else "PASS"

            proposal = dict(retryable=True, reason="Camera contamination is absent", correction="Show clearly visible adhered refracting droplets.",
                            diagnosis=[dict(criterion="sensor_effect_presence", observation="No foreground droplets in output crops")],
                            plan_patch={"sensor_effects": {"coverage": "heavy", "strength": "strong"},
                                        "execution_emphasis": {"sensor": "strong"}})

            def worker(module, action, attempt, log):
                self.assertEqual(action, "refine")
                with patch.object(dispatch, "qwen_json", return_value=proposal.copy()):
                    dispatch.refine(attempt)

            with (patch.object(dispatch, "initialize"), patch.object(executor, "generate", side_effect=generate),
                  patch.object(executor, "verify", side_effect=verify),
                  patch.object(dispatch, "run_worker", side_effect=worker)):
                dispatch.run(Namespace(out=out, max_retries=1, plan_only=False))
            first, second = [load(out / name / "config.json") for name in ("attempt_00", "attempt_01")]
            self.assertEqual(first["seed"], second["seed"])
            self.assertEqual(first["source_sha256"], second["source_sha256"])
            self.assertNotEqual(first["prompt"], second["prompt"])
            self.assertEqual(second["weather_plan"]["sensor_effects"]["coverage"], "heavy")
            self.assertEqual(second["plan_sha256"], sha(out / "attempt_01/plan.json"))
            self.assertEqual(sha(out / "plan.json"), original_hash)
            self.assertEqual(load(out / "state.json")["stage"], "complete")
            self.assertEqual(load(out / "state.json")["selected_attempt"], 1)

    def test_planner_can_repair_an_invalid_sensor_plan_before_generation(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            dump(out / "request.json", dict(profile="heavy_rain", frames=124, fps=24, plan_schema_version=4))
            bad, fixed = camera_plan(), camera_plan()
            bad["sensor_effects"]["coverage"] = "none"
            with patch.object(dispatch, "qwen_json", side_effect=[bad, fixed]) as model:
                dispatch.plan(out)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(load(out / "plan.json")["sensor_effects"]["coverage"], "moderate")
            self.assertTrue((out / "plan_validation_0.json").exists())


if __name__ == "__main__":
    unittest.main()

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from agent_loop import dispatcher, ref2va
from agent_loop.runtime import dump, load, sha
from agent_loop.weather import dispatch, executor
from agent_loop.weather.profiles import PROFILES, ENVIRONMENTS, CHECKS, h3_prompt, validate_plan


def planned(profile="heavy_rain"):
    preset = PROFILES[profile]
    return dict(profile=profile, prompt_version=2, temporal_mode="constant_from_frame_zero",
                transition_start_s=None, transition_end_s=None,
                edits={k: preset[k] for k in ("road_surface", "time_of_day", "snow_accumulation")},
                scene_description="Forward driving on a road with moving traffic.",
                weather_description=preset["effects"], preservation_description="Retain the source layout and motion.",
                unresolved=[])


def environment_plan(profile="heavy_rain", source_time="day", source_surface="dry"):
    result = planned(profile)
    result.update(prompt_version=3, source_environment={"time_of_day": source_time, "road_surface": source_surface})
    edits = result["edits"]
    edits.update(ENVIRONMENTS[profile])
    if edits["time_of_day"] == "preserve":
        edits["time_of_day"] = source_time
    if edits["road_surface"] == "preserve":
        edits["road_surface"] = source_surface
    return result


class WeatherPlanTests(unittest.TestCase):
    def test_resolved_environment_contract_for_all_six_profiles(self):
        for profile in PROFILES:
            for source_time in ("day", "night"):
                with self.subTest(profile=profile, source_time=source_time):
                    plan = environment_plan(profile, source_time)
                    validate_plan(plan, 124/24, profile)
                    prompt = h3_prompt(plan)
                    self.assertNotIn("Time of day: preserve", prompt)
                    self.assertNotIn("<Picture", prompt)
                    self.assertIn("first frame", prompt)
                    if profile != "heavy_snow":
                        self.assertNotIn("snow", prompt.lower())

    def test_rain_cannot_preserve_a_sunny_sky_or_direct_sunshine(self):
        for key, wrong in [("sky_condition", "clear"), ("illumination", "direct_sunlight"),
                           ("visibility", "clear"), ("time_of_day", "preserve"), ("road_surface", "dry")]:
            plan = environment_plan()
            plan["edits"][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(plan, 124/24)

    def test_daytime_storm_and_nighttime_fog_compile_different_lighting(self):
        rain = h3_prompt(environment_plan("heavy_rain", "day"))
        self.assertIn("target remains daytime", rain)
        self.assertIn("Dense storm clouds", rain)
        self.assertIn("replaces direct sunlight", rain)
        fog = h3_prompt(environment_plan("dense_fog", "night"))
        self.assertIn("target is nighttime", fog)
        self.assertIn("dark ambient illumination", fog)

    def test_existing_ground_cover_is_distinct_from_new_accumulation(self):
        fog = environment_plan("dense_fog", source_surface="snow_covered")
        validate_plan(fog, 124/24)
        self.assertIn("existing ground cover", h3_prompt(fog))
        fog["source_environment"]["road_surface"] = "dry"
        with self.assertRaises(ValueError):
            validate_plan(fog, 124/24)

    def test_free_text_explanations_cannot_override_the_target_environment(self):
        plan = environment_plan()
        plan.update(scene_description="The source has a clear blue sky.",
                    weather_description="Keep sunshine and add snow.",
                    preservation_description="Keep clear blue sky and snow.")
        prompt = h3_prompt(plan)
        shot = prompt.split("detailed_description:")[1]
        self.assertNotIn("clear blue", shot)
        self.assertNotIn("snow", prompt.lower())
        self.assertIn("storm clouds", shot)

    def test_actual_planner_assigns_v3_and_supplies_environment_policies(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            dump(out / "request.json", {"profile": "heavy_rain", "frames": 124, "fps": 24,
                                        "instruction": "Change to heavy rain"})
            result = environment_plan()
            result.pop("prompt_version")
            with patch.object(dispatch, "qwen_json", return_value=result) as model:
                dispatch.plan(out)
            plan = load(out / "plan.json")
            self.assertEqual(plan["prompt_version"], 3)
            self.assertEqual(plan["edits"]["sky_condition"], "storm_overcast")
            self.assertIn("Resolve target", model.call_args.args[0][0]["content"])

    def test_all_six_profiles_build_video_only_edit_prompts(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                plan = validate_plan(planned(profile), 124/24, profile)
                prompt = h3_prompt(plan)
                for heading in ("subject_definitions:", "summary:", "retention_analysis:",
                                "detailed_description:", "overall_soundscape:", "non_diegetic_music:"):
                    self.assertIn(heading, prompt)
                self.assertIn("<Video 1>", prompt)
                self.assertNotIn("<Picture", prompt)
                self.assertIn("very first frame", prompt)
                self.assertNotIn("hatchback", prompt)

    def test_temporal_and_profile_conflicts_fail_validation(self):
        plan = planned()
        with self.assertRaises(ValueError):
            validate_plan(plan, 5, "dense_fog")
        plan.update(temporal_mode="gradual", transition_start_s=2, transition_end_s=6)
        with self.assertRaises(ValueError):
            validate_plan(plan, 5)
        plan["transition_end_s"] = 4
        self.assertIn("2.00 seconds", h3_prompt(validate_plan(plan, 5)))
        plan["transition_end_s"] = float("nan")
        with self.assertRaises(ValueError):
            validate_plan(plan, 5)

    def test_modern_prompt_separates_source_weather_from_target_and_omits_unused_snow_field(self):
        plan = planned()
        plan.update(prompt_version=2, scene_description="The original sky is clear blue.",
                    preservation_description="Do not add snow or change vehicle identities.")
        prompt = h3_prompt(plan)
        self.assertIn("clear blue", prompt.split("summary:")[0])
        self.assertNotIn("clear blue", prompt.split("detailed_description:")[1])
        self.assertNotIn("snow", prompt.lower())
        self.assertIn("overcast storm sky", prompt)

    def test_accumulation_and_night_do_not_silently_conflict(self):
        plan = planned("dense_fog")
        plan["edits"]["snow_accumulation"] = "heavy"
        with self.assertRaises(ValueError):
            validate_plan(plan, 5)
        plan = planned("rainy_night")
        plan["edits"]["time_of_day"] = "day"
        with self.assertRaises(ValueError):
            validate_plan(plan, 5)

    def test_weather_image_input_is_rejected_before_video_loading(self):
        with TemporaryDirectory() as directory:
            with patch.object(dispatch, "read_frames") as read:
                with self.assertRaises(ValueError):
                    dispatch.initialize(Namespace(replacement_image=Path("photo.png"), max_retries=1), Path(directory))
                read.assert_not_called()


class GlobalDispatcherTests(unittest.TestCase):
    def test_requested_profile_is_visible_to_the_semantic_router(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            dump(out / "dispatch_request.json", {"instruction": "Keep the original traffic", "profile": "dense_fog", "source": "source.mp4"})
            with patch.object(dispatcher, "qwen_json", return_value={
                "agent": "weather", "profile": "dense_fog", "reason": "Explicit fog preset",
            }) as model:
                dispatcher.classify(out)
            self.assertIn('"requested_profile": "dense_fog"', model.call_args.args[0][1]["content"])
            self.assertEqual(load(out / "dispatch.json")["agent"], "weather")

    def args(self, out, **extra):
        source = out / "input.mp4"
        source.write_bytes(b"source")
        values = dict(source=source, out=out, instruction="Change to heavy rain", agent="auto",
                      profile=None, max_retries=1, plan_only=False,
                      replacement_image=None, edited_first_frame=None)
        values.update(extra)
        return Namespace(**values)

    def classify(self, module, action, out, log):
        dump(out / "dispatch.json", dict(agent="weather", profile="heavy_rain", reason="Global weather edit",
                                         request_sha256=sha(out / "dispatch_request.json")))

    def test_global_weather_route_never_calls_the_replacement_agent(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            with (patch.object(dispatcher, "run_worker", side_effect=self.classify),
                  patch.object(dispatch, "run") as weather,
                  patch("agent_loop.replace.dispatch.run", side_effect=AssertionError("SAM route entered"))):
                dispatcher.run(self.args(out))
            self.assertEqual(weather.call_args.args[0].profile, "heavy_rain")

    def test_composite_request_is_delegated_to_planner_without_direct_generation(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            args = self.args(out)
            def composite(module, action, path, log):
                dump(path / "dispatch.json", dict(agent="composite", profile=None, reason="Two requested edits",
                     request_sha256=sha(path / "dispatch_request.json")))
            with (patch.object(dispatcher, "run_worker", side_effect=composite),
                  patch.object(dispatch, "run") as weather,
                  patch("agent_loop.planner.agent.prepare") as outer):
                dispatcher.run(args)
            weather.assert_not_called()
            outer.assert_called_once_with(args)

    def test_dispatch_cache_rejects_changed_user_request(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            args = self.args(out)
            with patch.object(dispatcher, "run_worker", side_effect=self.classify), patch.object(dispatch, "run"):
                dispatcher.run(args)
                args.instruction = "Replace the white car"
                with self.assertRaisesRegex(ValueError, "inputs have changed"):
                    dispatcher.run(args)


class WeatherLoopTests(unittest.TestCase):
    def test_verification_keeps_unknown_and_rejects_stale_review(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "source.mp4"
            source.write_bytes(b"source")
            (out / "generated.mp4").write_bytes(b"generated")
            (out / "review_frames.jpg").write_bytes(b"sheet")
            config = dict(source=str(source), source_sha256=sha(source), fps=24,
                          width=832, height=480, weather_plan=planned("dense_fog"))
            dump(out / "config.json", config)
            dump(out / "generation.json", dict(config, generated_sha256=sha(out / "generated.mp4")))
            review = dict(checks={key: {"status": "PASS", "reason": "visible"} for key in CHECKS},
                          generated_sha256=sha(out / "generated.mp4"),
                          sheet_sha256=sha(out / "review_frames.jpg"), config_sha256=sha(out / "config.json"))
            review["checks"]["vehicle_motion"] = {"status": "UNKNOWN", "reason": "Obscured by fog"}
            dump(out / "visual_review.json", review)
            with (patch.object(executor, "read_frames", return_value=np.zeros((124, 1, 1, 3), np.uint8)),
                  patch.object(executor, "comparison"), patch.object(executor, "sample_sheet")):
                self.assertEqual(executor.verify(out), "UNKNOWN")
                review["generated_sha256"] = "stale output"
                dump(out / "visual_review.json", review)
                with self.assertRaisesRegex(ValueError, "changed output"):
                    executor.verify(out)

    def run_fixture(self, verdicts, retries=1):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            request = {"frames": 124, "fps": 24, "profile": "heavy_rain"}
            dump(out / "request.json", request)
            plan = planned()
            plan["request_sha256"] = sha(out / "request.json")
            dump(out / "plan.json", plan)
            args = Namespace(out=out, max_retries=retries, plan_only=False)
            def generate(root, attempt, correction):
                (attempt / "generated.mp4").write_bytes(b"generated")
                (attempt / "comparison.mp4").write_bytes(b"comparison")
            def repair(module, action, attempt, log):
                dump(attempt / "repair.json", dict(retryable=True, correction="Make rainfall clearly dense.", reason="Rain missing"))
            with (patch.object(dispatch, "initialize"),
                  patch.object(executor, "generate", side_effect=generate) as generate_call,
                  patch.object(executor, "verify", side_effect=verdicts),
                  patch.object(dispatch, "run_worker", side_effect=repair) as worker):
                dispatch.run(args)
            return load(out / "state.json"), generate_call.call_count, worker.call_count

    def test_observed_failure_can_retry_once(self):
        state, generated, repairs = self.run_fixture(["FAIL", "PASS"])
        self.assertEqual((state["stage"], generated, repairs), ("complete", 2, 1))

    def test_unknown_does_not_trigger_speculative_retry(self):
        state, generated, repairs = self.run_fixture(["UNKNOWN"])
        self.assertEqual((state["stage"], generated, repairs), ("needs_review", 1, 0))

    def test_retry_limit_preserves_failure(self):
        state, generated, repairs = self.run_fixture(["FAIL", "FAIL"])
        self.assertEqual((state["stage"], generated, repairs), ("completed_with_visual_issues", 2, 1))

    def test_generation_binds_only_source_video_and_text(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "input.mp4"
            source.write_bytes(b"source")
            request = dict(source=str(source), source_sha256=sha(source), frames=124, fps=24,
                           width=832, height=480, seed=20261028, steps=8)
            dump(out / "request.json", request)
            dump(out / "plan.json", planned())
            attempt = out / "attempt_00"
            attempt.mkdir()
            with patch.object(executor.subprocess, "run") as worker:
                executor.generate(out, attempt)
            config = load(attempt / "config.json")
            self.assertEqual(config["reference_image_count"], 0)
            self.assertEqual(config["reference_video_count"], 1)
            self.assertIsNone(config["first_frame_anchor"])
            self.assertIsNone(config["aligned_warp"])
            self.assertIn("agent_loop.weather.executor", worker.call_args.args[0])

    def test_shared_sampler_cache_rejects_prompt_changes(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "generated.mp4").write_bytes(b"cached")
            dump(out / "config.json", {"prompt": "rain"})
            dump(out / "generation.json", {"prompt": "rain", "generated_sha256": sha(out / "generated.mp4")})
            ref2va.infer(out)  # No model imports or GPU are needed for this cache check.
            dump(out / "config.json", {"prompt": "fog"})
            with self.assertRaisesRegex(ValueError, "input manifest"):
                ref2va.infer(out)


if __name__ == "__main__":
    unittest.main()

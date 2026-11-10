import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from agent_loop.replace import executor as replacement


class VehicleReplacementTests(unittest.TestCase):
    def test_customer_photo_is_bound_and_changed_resume_input_is_rejected(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source, photo = out / "input.mp4", out / "customer.jpg"
            source.write_bytes(b"source fixture")
            Image.new("RGB", (32, 24), (20, 40, 200)).save(photo)
            args = Namespace(instruction="use this truck", source=source,
                             geometry=None, tracking_cache=None,
                             conditioning="appearance-guided", replacement_image=photo)
            with (
                patch.object(replacement, "read_frames",
                             return_value=np.zeros((1, 48, 64, 3), np.uint8)),
                patch.object(replacement, "sam_candidates", return_value=[
                    {"object_id": "sam-1", "bbox_xyxy": [10, 20, 22, 32]},
                ]),
            ):
                replacement.initialize(args, out)
            saved = json.loads((out / "request.json").read_text())
            self.assertEqual(saved["conditioning"], "appearance-guided")
            self.assertEqual(saved["replacement_image_sha256"], replacement.sha(photo))
            replacement.initialize(args, out)  # Unchanged input can resume.
            Image.new("RGB", (32, 24), (200, 40, 20)).save(out / "replacement_reference.png")
            with self.assertRaisesRegex(ValueError, "reference image has changed"):
                replacement.initialize(args, out)
            Image.new("RGB", (32, 24), (200, 200, 20)).save(photo)
            with self.assertRaisesRegex(ValueError, "Resume inputs differ"):
                replacement.initialize(args, out)

    def test_invalid_customer_photo_fails_before_video_execution(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            photo = out / "bad.png"
            photo.write_bytes(b"not an image")
            args = Namespace(replacement_image=photo)
            with (
                patch.object(replacement, "read_frames") as read,
                self.assertRaises(OSError),
            ):
                replacement.initialize(args, out)
            read.assert_not_called()
            self.assertFalse((out / "request.json").exists())

    def test_text_only_execution_manifest_has_only_the_original_video(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "input.mp4"
            source.write_bytes(b"original source video")
            (out / "request.json").write_text(json.dumps({
                "source": str(source), "source_sha256": replacement.sha(source),
                "conditioning": "text-only",
            }))
            (out / "route.json").write_text(json.dumps({
                "target_id": "sam-7", "target_description": "the dark sedan ahead-right",
                "replacement_category": "box truck",
                "replacement_description": "a blue box truck with a compact cab and enclosed cargo body",
                "first_frame_edit_prompt": "Replace the selected white car with a box truck.",
                "motion_description": "Follow the source vehicle's motion.",
            }))
            (out / "candidates.json").write_text(json.dumps([
                {"object_id": "sam-7", "bbox_xyxy": [500, 200, 610, 310]},
            ]))
            with (
                patch.object(replacement.subprocess, "run") as execute,
                patch.object(replacement.Image, "open", side_effect=AssertionError("Unexpected image input")),
            ):
                replacement.generate_text_only(out)
            config = json.loads((out / "text_only_config.json").read_text())
            self.assertEqual(config["source"], str(source))
            self.assertEqual(config["reference_video_count"], 1)
            self.assertEqual(config["reference_image_count"], 0)
            self.assertIsNone(config["aligned_warp"])
            self.assertIsNone(config["first_frame_anchor"])
            self.assertIn("infer-text-only", execute.call_args.args[0])
            self.assertEqual(execute.call_args.args[0][1:3],
                             ["-m", "agent_loop.replace.executor"])

    def test_prompt_uses_agent_category_appearance_and_selected_instance(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "candidates.json").write_text(json.dumps([
                {"object_id": "left", "bbox_xyxy": [10, 20, 30, 40]},
                {"object_id": "right", "bbox_xyxy": [510, 220, 620, 340]},
            ]))
            for category, appearance in [("fire engine", "red body with hose lockers and a roof ladder"),
                                          ("city bus", "green low-floor body with large passenger windows")]:
                route = dict(target_id="right", target_description="dark sedan in the right lane",
                             replacement_category=category, replacement_description=appearance,
                             motion_description="Retain its right-lane trajectory.")
                (out / "route.json").write_text(json.dumps(route))
                prompt = replacement.text_only_prompt(out)
                for value in (category, appearance, route["target_description"], "[510, 220, 620, 340]"):
                    self.assertIn(value, prompt)
                for old_default in ("white hatchback", "plain white box truck", "natural daylight"):
                    self.assertNotIn(old_default, prompt)
                self.assertNotIn("[10, 20, 30, 40]", prompt)
            route.pop("replacement_description")
            (out / "route.json").write_text(json.dumps(route))
            with self.assertRaisesRegex(ValueError, "agent-authored replacement_description"):
                replacement.text_only_prompt(out)

    def test_outer_task_parameters_are_bound_to_child_request_on_resume(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "input.mp4"
            source.write_bytes(b"source fixture")
            args = Namespace(instruction="replace selected vehicle", source=source,
                             geometry=None, tracking_cache=None, conditioning="text-only",
                             task_parameters=dict(target_description="right dark car", target_selector="right lane",
                                                  replacement_category="fire_engine"))
            with (patch.object(replacement, "read_frames", return_value=np.zeros((1, 48, 64, 3), np.uint8)),
                  patch.object(replacement, "sam_candidates", return_value=[{"object_id": "right"}])):
                replacement.initialize(args, out)
            self.assertEqual(json.loads((out / "request.json").read_text())["task_parameters"], args.task_parameters)
            args.task_parameters = dict(args.task_parameters, replacement_category="bus")
            with self.assertRaisesRegex(ValueError, "Resume inputs differ"):
                replacement.initialize(args, out)

    def test_text_only_verification_requires_no_edited_first_frame(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            video = out / "generated.mp4"
            video.write_bytes(b"encoded result fixture")
            (out / "request.json").write_text(json.dumps({
                "source": "input.mp4", "conditioning": "text-only"
            }))
            (out / "route.json").write_text(json.dumps({"target_id": "sam-1"}))
            (out / "candidates.json").write_text(json.dumps([
                {"object_id": "sam-1", "bbox_xyxy": [15, 17, 30, 32]}
            ]))
            (out / "visual_review.json").write_text(json.dumps({
                "replacement_visible_in_first_frame": False,
                "replacement_category_correct": False,
                "target_identity_consistent": False,
                "wrong_vehicle_replaced": False,
            }))
            frames = np.zeros((124, 64, 96, 3), np.uint8)
            with (
                patch.multiple(replacement, HEIGHT=64, WIDTH=96),
                patch.object(replacement, "read_frames", return_value=frames),
                patch.object(replacement, "compare"),
                patch("builtins.print"),
            ):
                replacement.verify(out)
            metrics = json.loads((out / "technical_checks.json").read_text())
            self.assertIsNone(metrics["first_frame_sha256"])
            self.assertIsNone(metrics["first_frame_anchor_mae"])
            self.assertIsNone(metrics["outside_edit_region_mae"])
            self.assertEqual(json.loads((out / "state.json").read_text())["stage"],
                             "completed_with_visual_issues")

    def test_tracking_box_ignores_distant_segmentation_speckles(self):
        mask = np.zeros((48, 64), bool)
        mask[20:32, 10:22] = True
        mask[2, 60] = True
        np.testing.assert_array_equal(replacement.mask_box(mask), [10, 20, 22, 32])

    def test_video_only_initialization_detects_candidates_without_loading_geometry(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "input.mp4"
            source.write_bytes(b"source fixture")
            args = Namespace(instruction="replace the front-left car with a truck",
                             source=source, geometry=None, tracking_cache=None)
            candidates = [{"object_id": "sam-4", "bbox_xyxy": [10, 20, 22, 32]}]
            with (
                patch.object(replacement, "read_frames",
                             return_value=np.zeros((1, 48, 64, 3), np.uint8)),
                patch.object(replacement, "sam_candidates", return_value=candidates) as detect,
                patch.object(replacement.np, "load", side_effect=AssertionError("Unexpected NPZ input")),
            ):
                replacement.initialize(args, out)
            detect.assert_called_once_with(out / "source_first_frame.png")
            request = json.loads((out / "request.json").read_text())
            self.assertIsNone(request["geometry"])
            self.assertIsNone(request["geometry_sha256"])
            self.assertEqual(json.loads((out / "candidates.json").read_text()), candidates)

    def test_video_only_tracking_does_not_require_a_geometry_index(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            masks = np.zeros((9, 48, 64), bool)
            for i in range(9):
                masks[i, 20:32, 10+i:22+i] = True
            Image.new("RGB", (64, 48)).save(out / "source_first_frame.png")
            (out / "request.json").write_text(json.dumps({
                "source": "input.mp4", "source_sha256": "source hash",
                "geometry": None, "tracking_cache": None,
            }))
            (out / "route.json").write_text(json.dumps({
                "target_id": "sam-4", "replacement_category": "box truck",
                "target_description": "front-left white car", "first_frame_edit_prompt": "Replace this car.",
            }))
            (out / "candidates.json").write_text(json.dumps([
                {"object_id": "sam-4", "bbox_xyxy": [10, 20, 22, 32]},
            ]))
            with (
                patch.multiple(replacement, FRAMES=9, WIDTH=64, HEIGHT=48),
                patch.object(replacement, "segment_video", return_value=masks),
                patch.object(replacement, "projected_box", side_effect=AssertionError("Unexpected GT use")),
            ):
                replacement.prepare(out)
            result = json.loads((out / "tracking.json").read_text())
            self.assertEqual(result["frames_present"], 9)
            self.assertFalse(result["reused_sam_masks"])
            self.assertIsNone(result["minimum_fraction_inside_projected_box"])
            self.assertEqual(result["grounding_source"], "SAM3.1 source-video detection")
            self.assertEqual(result["maximum_center_step_px"], 1.)

    def test_no_detections_does_not_create_a_routable_request(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "input.mp4"
            source.write_bytes(b"source fixture")
            args = Namespace(instruction="replace the car", source=source,
                             geometry=None, tracking_cache=None)
            with (
                patch.object(replacement, "read_frames",
                             return_value=np.zeros((1, 48, 64, 3), np.uint8)),
                patch.object(replacement, "sam_candidates", return_value=[]),
                self.assertRaisesRegex(ValueError, "No visible candidate vehicles"),
            ):
                replacement.initialize(args, out)
            self.assertFalse((out / "request.json").exists())

    def test_mask_selection_uses_the_selected_instance_box(self):
        distractor = np.zeros((480, 832), bool)
        distractor[100:200, 400:600] = True
        selected = np.zeros_like(distractor)
        selected[265:358, 147:300] = True
        mask, object_id, score = replacement.select_mask(
            {"out_binary_masks": np.stack([distractor, selected]), "out_obj_ids": [3, 17]},
            [147, 265, 300, 358],
        )
        self.assertEqual(object_id, 17)
        self.assertEqual(score, 1.)
        np.testing.assert_array_equal(mask, selected)

    def test_missing_target_does_not_fall_back_to_another_vehicle(self):
        distractor = np.zeros((480, 832), bool)
        distractor[100:200, 400:600] = True
        with self.assertRaisesRegex(ValueError, "intended instance"):
            replacement.select_mask(
                {"out_binary_masks": distractor[None], "out_obj_ids": [3]},
                [147, 265, 300, 358],
            )

    def test_changed_tracking_checkpoint_is_rejected(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "request.json").write_text("{}")
            (out / "route.json").write_text(json.dumps({"target_id": "selected"}))
            (out / "target_masks.npz").write_bytes(b"changed artifact")
            (out / "tracking.json").write_text(json.dumps({
                "target_id": "selected", "mask_sha256": "old hash"
            }))
            with self.assertRaisesRegex(ValueError, "masks have changed"):
                replacement.prepare(out)

    def test_condition_has_replacement_from_frame_zero_and_preserves_outside_pixels(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            source_path = out / "original.mp4"
            source_path.write_bytes(b"source fixture")
            source = np.full((9, 48, 64, 3), 100, np.uint8)
            old_masks = np.zeros((9, 48, 64), bool)
            for i in range(9):
                old_masks[i, 30:36, 11+i:17+i] = True
                source[i, old_masks[i]] = [240, 10, 10]
            new_mask = np.zeros((48, 64), bool)
            new_mask[20:36, 8:20] = True
            # The image editor may change the background. Only its object matte is transferred.
            edited = np.full_like(source[0], 210)
            edited[new_mask] = [10, 240, 10]
            Image.fromarray(edited).save(out / "editor_output.png")
            (out / "route.json").write_text("{}")
            customer_photo = np.full((24, 36, 3), [20, 40, 200], np.uint8)
            Image.fromarray(customer_photo).save(out / "replacement_reference.png")
            (out / "request.json").write_text(json.dumps({
                "source": str(source_path), "source_sha256": replacement.sha(source_path),
                "replacement_image": str(out / "replacement_reference.png"),
            }))
            np.savez_compressed(out / "target_masks.npz", masks=old_masks)
            encoded = {}

            def write_video(path, frames):
                encoded[str(path)] = np.asarray(frames)
                path.write_bytes(b"encoded fixture")

            def read_video(path):
                return source if str(path) == str(source_path) else encoded[str(path)]

            with (
                patch.multiple(replacement, WIDTH=64, HEIGHT=48, FRAMES=9),
                patch.object(replacement, "segment_replacement", return_value=new_mask),
                patch.object(replacement, "read_frames", side_effect=read_video),
                patch.object(replacement, "write_frames", side_effect=write_video),
            ):
                replacement.compose_conditions(out)

            np.testing.assert_array_equal(
                np.array(Image.open(out / "appearance_reference.png")), customer_photo
            )
            guide = encoded[str(out / "edited_reference.mp4")]
            masks = np.load(out / "edit_masks.npz")
            self.assertTrue(np.all(guide[0, 24:31, 10:18, 1] > 200))
            np.testing.assert_array_equal(guide[~masks["affected"]], source[~masks["affected"]])
            first = replacement.mask_box(masks["replacement"][0])
            last = replacement.mask_box(masks["replacement"][-1])
            self.assertAlmostEqual(last[0] - first[0], 8., delta=1.)
            np.testing.assert_array_equal(np.array(Image.open(out / "edited_first_frame.png")), guide[0])


if __name__ == "__main__":
    unittest.main()

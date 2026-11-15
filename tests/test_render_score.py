"""Tests for deterministic render-quality scoring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from driving_world.judge.render import (
    Canonical,
    ReferenceDistribution,
    RenderChecker,
    ScoreConfig,
    build_references,
    build_windows,
    decode_video,
    load_references,
    score_series,
    write_references,
)


SMALL = Canonical(fps=10.0, height=40)


class ReferenceDistributionTest(unittest.TestCase):
    def test_percentile_uses_midrank_ties_and_boundaries(self) -> None:
        reference = ReferenceDistribution.from_samples("test", True, [1, 2, 2, 4])
        self.assertEqual(reference.percentile(0), 0.0)
        self.assertEqual(reference.percentile(2), 50.0)
        self.assertEqual(reference.percentile(5), 100.0)

    def test_two_sided_reference_rewards_the_middle_not_the_extreme(self) -> None:
        reference = ReferenceDistribution.from_samples(
            "sharpness", True, [1, 2, 3, 4, 5], two_sided=True
        )
        self.assertEqual(reference.percentile(3), 10.0)
        self.assertEqual(reference.percentile(1), reference.percentile(5))
        self.assertLess(reference.percentile(3), reference.percentile(5))
        self.assertFalse(
            ReferenceDistribution.from_dict(reference.to_dict()) is None
        )
        restored = ReferenceDistribution.from_dict(reference.to_dict())
        self.assertTrue(restored.two_sided)
        self.assertEqual(restored.percentile(5), reference.percentile(5))

    def test_percentile_is_raw_cdf_for_lower_is_better_reference(self) -> None:
        reference = ReferenceDistribution.from_samples("test", False, [1, 2, 3, 4])
        self.assertEqual(reference.percentile(0), 0.0)
        self.assertEqual(reference.percentile(5), 100.0)


class WindowTest(unittest.TestCase):
    def test_windows_cover_all_frames_and_close_the_last_window(self) -> None:
        windows = build_windows(80, 10.0)
        covered = {index for window in windows for index in window.frame_indices}
        self.assertEqual(covered, set(range(80)))
        self.assertIn(79, windows[-1].frame_indices)
        self.assertEqual(windows[-1].end_seconds, 8.0)

    def test_invalid_parameters(self) -> None:
        for kwargs in (
            {"frame_count": 0, "fps": 10.0},
            {"frame_count": 1, "fps": 0.0},
            {"frame_count": 1, "fps": 10.0, "window_seconds": 0.0},
            {"frame_count": 1, "fps": 10.0, "stride_seconds": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_windows(**kwargs)


class ScoreSeriesTest(unittest.TestCase):
    def test_injected_temporal_failure_trips_axis_gate(self) -> None:
        values = np.full(80, 0.1)
        values[30:42] = 1.0
        reference = ReferenceDistribution.from_samples(
            "motion_compensated_residual", False, np.linspace(0.0, 0.125, 101)
        )
        result = score_series(
            {"temporal": values},
            {"temporal": reference},
            fps=10.0,
            config=ScoreConfig(),
        )
        axis = result.axes["temporal"]
        affected = [
            window.percentile
            for window in axis.windows
            if window.window.start_seconds < 4.2 and window.window.end_seconds > 3.0
        ]
        unaffected = [
            window.percentile
            for window in axis.windows
            if window.window.end_seconds <= 3.0 or window.window.start_seconds >= 4.2
        ]
        self.assertLess(axis.p_low, min(unaffected))
        self.assertLess(min(affected), min(unaffected))
        self.assertEqual(result.status, "fail")
        self.assertIn("temporal_axis_below_fail_threshold", result.reasons)

    def test_single_axis_total_equals_axis_score(self) -> None:
        reference = ReferenceDistribution.from_samples("sharpness", True, range(101))
        result = score_series(
            {"imaging": [50.0] * 10}, {"imaging": reference}, fps=10.0
        )
        self.assertAlmostEqual(result.total, result.axes["imaging"].score)


class MotionGateTest(unittest.TestCase):
    def _references(self) -> dict:
        return {
            "imaging": ReferenceDistribution.from_samples(
                "laplacian_sharpness", True, np.linspace(0.0, 500.0, 101)
            ),
            "temporal": ReferenceDistribution.from_samples(
                "motion_compensated_residual", False, np.linspace(0.0, 0.1, 101)
            ),
            "motion": ReferenceDistribution.from_samples(
                "motion_energy", True, np.linspace(0.005, 0.05, 101)
            ),
        }

    def test_static_clip_refuses_to_judge_temporal_quality(self) -> None:
        frames = np.full((40, 24, 32, 3), 128, dtype=np.uint8)
        report = RenderChecker(self._references(), canonical=None).check_frames(
            frames, fps=10.0
        )
        self.assertIsNotNone(report.motion)
        self.assertTrue(report.motion["gated"])
        self.assertNotIn("temporal", report.axes)
        self.assertIn("temporal_indeterminate_low_motion", report.reasons)

    def test_moving_clip_keeps_the_temporal_axis(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
        frames = np.stack([np.roll(base, step, axis=1) for step in range(40)])
        report = RenderChecker(self._references(), canonical=None).check_frames(
            frames, fps=10.0
        )
        self.assertFalse(report.motion["gated"])
        self.assertIn("temporal", report.axes)


class CanonicalTest(unittest.TestCase):
    def test_blur_radius_changes_the_canonical_frames(self) -> None:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            self.skipTest(str(exc))
        rng = np.random.default_rng(4)
        frames = rng.integers(0, 256, size=(5, 40, 60, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "noisy.mp4"
            iio.imwrite(video, frames, plugin="FFMPEG", fps=10)
            sharp = decode_video(video, canonical=Canonical(fps=10.0, height=40, blur_radius=0.0))
            blurred = decode_video(video, canonical=Canonical(fps=10.0, height=40, blur_radius=2.0))
        self.assertNotEqual(
            float(np.var(sharp[1][0].astype(np.float64))),
            float(np.var(blurred[1][0].astype(np.float64))),
        )
        self.assertLess(
            float(np.var(blurred[1][0].astype(np.float64))),
            float(np.var(sharp[1][0].astype(np.float64))),
        )

    def test_resamples_frame_rate_and_height_together(self) -> None:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            self.skipTest(str(exc))
        rng = np.random.default_rng(3)
        base = rng.integers(0, 256, size=(80, 160, 3), dtype=np.uint8)
        frames = np.stack([np.roll(base, step % 5, axis=1) for step in range(120)])
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            iio.imwrite(video, frames, plugin="FFMPEG", fps=24)
            raw_fps, raw_frames = decode_video(video)
            fps, decoded = decode_video(video, canonical=Canonical(fps=10.0, height=40))
        self.assertAlmostEqual(raw_fps, 24.0)
        self.assertEqual(len(raw_frames), 120)
        self.assertAlmostEqual(fps, 10.0)
        self.assertEqual(len(decoded), 50)
        self.assertEqual(decoded[0].shape, (40, 80, 3))


class RenderCheckerTest(unittest.TestCase):
    def test_end_to_end_is_deterministic(self) -> None:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            self.skipTest(str(exc))
        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        frames = np.stack([np.roll(base, index % 4, axis=1) for index in range(80)])
        references = {
            "imaging": ReferenceDistribution.from_samples(
                "laplacian_sharpness", True, np.linspace(0.0, 2.0, 101)
            ),
            "temporal": ReferenceDistribution.from_samples(
                "motion_compensated_residual", False, np.linspace(0.0, 0.2, 101)
            ),
            "structure": ReferenceDistribution.from_samples(
                "structural_warp_agreement", True, np.linspace(0.0, 1.0, 101)
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "render.mp4"
            iio.imwrite(video, frames, plugin="FFMPEG", fps=10)
            checker = RenderChecker(references, canonical=None)
            first = checker.check(video).to_dict()
            second = checker.check(video).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["frame_count"], 80)
        self.assertEqual(set(first["axes"]), set(references))


class CalibrationTest(unittest.TestCase):
    def _videos(self, directory: Path, count: int = 2) -> list[Path]:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            self.skipTest(str(exc))
        rng = np.random.default_rng(11)
        paths = []
        for index in range(count):
            base = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
            frames = np.stack([np.roll(base, step, axis=1) for step in range(20)])
            path = directory / f"real_{index}.mp4"
            iio.imwrite(path, frames, plugin="FFMPEG", fps=10)
            paths.append(path)
        return paths

    def test_calibration_drops_warmup_frames_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = self._videos(root)
            references = build_references(videos, canonical=SMALL)
            self.assertEqual(
                set(references), {"imaging", "temporal", "structure", "motion"}
            )
            self.assertEqual(len(references["imaging"].samples), 40)
            self.assertEqual(len(references["temporal"].samples), 38)
            self.assertEqual(len(references["structure"].samples), 38)
            self.assertEqual(len(references["motion"].samples), 38)

            write_references(root / "refs", references)
            loaded = load_references(root / "refs")
            self.assertEqual(set(loaded), set(references))
            self.assertEqual(loaded["temporal"].samples, references["temporal"].samples)

            report = RenderChecker(loaded, canonical=SMALL).check(videos[0])
            self.assertIn(report.status, {"pass", "warn", "fail"})
            self.assertEqual(report.frame_count, 20)

    def test_findings_follow_the_warn_threshold_not_only_fail(self) -> None:
        references = {
            "temporal": ReferenceDistribution.from_samples(
                "motion_compensated_residual", False, np.linspace(0.0, 0.125, 101)
            ),
            "imaging": ReferenceDistribution.from_samples(
                "laplacian_sharpness", True, np.linspace(0.0, 2.0, 101)
            ),
        }
        rng = np.random.default_rng(5)
        base = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
        frames = np.stack([np.roll(base, step % 3, axis=1) for step in range(80)])
        frames[20:40] = rng.integers(0, 256, size=(20, 16, 24, 3), dtype=np.uint8)
        report = RenderChecker(
            references, config=ScoreConfig(warn_below=99.0), canonical=None
        ).check_frames(frames, fps=10.0)
        self.assertTrue(report.findings)
        for finding in report.findings:
            self.assertEqual(finding["error_type"], "render")
            self.assertEqual(
                set(finding["checks"]),
                {
                    "artifact_free",
                    "motion_smooth",
                    "geometry_plausible",
                    "identity_consistent",
                },
            )
            start, end = finding["temporal_segment"]
            self.assertLess(start, end)

    def test_mismatched_score_direction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._videos(root, count=1)[0]
            references = build_references([video], canonical=SMALL)
            references["temporal"] = ReferenceDistribution.from_samples(
                references["temporal"].probe,
                not references["temporal"].higher_is_better,
                references["temporal"].samples,
            )
            with self.assertRaises(ValueError):
                RenderChecker(references, canonical=SMALL).check(video)


if __name__ == "__main__":
    unittest.main()

"""Tests for deterministic render-quality scoring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from driving_world.judge.render import (
    ReferenceDistribution,
    RenderChecker,
    ScoreConfig,
    build_references,
    build_windows,
    load_references,
    score_series,
    write_references,
)


class ReferenceDistributionTest(unittest.TestCase):
    def test_percentile_uses_midrank_ties_and_boundaries(self) -> None:
        reference = ReferenceDistribution.from_samples("test", True, [1, 2, 2, 4])
        self.assertEqual(reference.percentile(0), 0.0)
        self.assertEqual(reference.percentile(2), 50.0)
        self.assertEqual(reference.percentile(5), 100.0)

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
            checker = RenderChecker(references)
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
            references = build_references(videos)
            self.assertEqual(set(references), {"imaging", "temporal", "structure"})
            self.assertEqual(len(references["imaging"].samples), 40)
            self.assertEqual(len(references["temporal"].samples), 38)
            self.assertEqual(len(references["structure"].samples), 38)

            write_references(root / "refs", references)
            loaded = load_references(root / "refs")
            self.assertEqual(set(loaded), set(references))
            self.assertEqual(loaded["temporal"].samples, references["temporal"].samples)

            report = RenderChecker(loaded).check(videos[0])
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
            references, config=ScoreConfig(warn_below=99.0)
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
            references = build_references([video])
            references["temporal"] = ReferenceDistribution.from_samples(
                references["temporal"].probe,
                not references["temporal"].higher_is_better,
                references["temporal"].samples,
            )
            with self.assertRaises(ValueError):
                RenderChecker(references).check(video)


if __name__ == "__main__":
    unittest.main()

"""Window aggregation, percentile calibration, and render-quality gates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .reference import ReferenceDistribution
from .windows import Window, build_windows

AXES = ("imaging", "temporal", "structure")


@dataclass(frozen=True)
class ScoreConfig:
    """Configuration for window aggregation and axis-level gates."""

    weights: Mapping[str, float] = field(
        default_factory=lambda: {"imaging": 0.30, "temporal": 0.45, "structure": 0.25}
    )
    window_seconds: float = 1.0
    stride_seconds: float = 0.5
    low_percentile: float = 10.0
    fail_below: float = 15.0
    warn_below: float = 35.0
    max_findings: int = 3


@dataclass(frozen=True)
class WindowScore:
    axis: str
    window: Window
    raw: float
    percentile: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "window": {
                "index": self.window.index,
                "start_seconds": self.window.start_seconds,
                "end_seconds": self.window.end_seconds,
                "frame_indices": list(self.window.frame_indices),
            },
            "raw": self.raw,
            "percentile": self.percentile,
        }


@dataclass(frozen=True)
class AxisScore:
    axis: str
    score: float
    p_low: float
    median: float
    windows: tuple[WindowScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "score": self.score,
            "p_low": self.p_low,
            "median": self.median,
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True)
class ClipScore:
    axes: Mapping[str, AxisScore]
    total: float
    status: str
    reasons: tuple[str, ...]


def score_series(
    series: Mapping[str, Sequence[float]],
    references: Mapping[str, ReferenceDistribution],
    *,
    fps: float,
    config: ScoreConfig = ScoreConfig(),
) -> ClipScore:
    """Calibrate per-frame axis values and aggregate them into a clip score."""
    axes: dict[str, AxisScore] = {}
    reasons = []
    for axis in AXES:
        values = series.get(axis)
        if values is None or len(values) == 0:
            continue
        reference = references.get(axis)
        if reference is None:
            reasons.append(f"no_reference_for:{axis}")
            continue
        numeric = np.asarray(values, dtype=np.float64)
        if numeric.ndim != 1 or not np.all(np.isfinite(numeric)):
            raise ValueError(f"series for {axis} must contain finite scalar values")
        windows = build_windows(
            len(numeric),
            fps,
            window_seconds=config.window_seconds,
            stride_seconds=config.stride_seconds,
        )
        window_scores = []
        for window in windows:
            raw = float(np.mean(numeric[list(window.frame_indices)]))
            percentile = reference.percentile(raw)
            if not reference.higher_is_better:
                percentile = 100.0 - percentile
            window_scores.append(
                WindowScore(axis, window, raw, percentile)
            )
        percentiles = np.asarray(
            [window.percentile for window in window_scores], dtype=np.float64
        )
        if len(percentiles) == 1:
            p_low = median = float(percentiles[0])
        else:
            p_low = float(np.percentile(percentiles, config.low_percentile))
            median = float(np.median(percentiles))
        axes[axis] = AxisScore(
            axis, 0.5 * p_low + 0.5 * median, p_low, median, tuple(window_scores)
        )

    if not axes:
        raise ValueError("no scoreable render axes")
    for axis in AXES:
        if axis not in axes:
            continue
        if axes[axis].score < config.fail_below:
            reasons.append(f"{axis}_axis_below_fail_threshold")
        elif axes[axis].score < config.warn_below:
            reasons.append(f"{axis}_axis_below_warn_threshold")
    if any(value.score < config.fail_below for value in axes.values()):
        status = "fail"
    elif any(value.score < config.warn_below for value in axes.values()):
        status = "warn"
    else:
        status = "pass"
    weighted_log = 0.0
    weight_sum = 0.0
    for axis in AXES:
        if axis in axes:
            weight = float(config.weights.get(axis, 0.0))
            weighted_log += weight * math.log(max(axes[axis].score, 1.0))
            weight_sum += weight
    if weight_sum <= 0:
        raise ValueError("present axes must have positive total weight")
    total = math.exp(weighted_log / weight_sum)
    return ClipScore(axes, total, status, tuple(reasons))

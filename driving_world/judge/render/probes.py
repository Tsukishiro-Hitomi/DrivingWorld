"""Replaceable render-quality probes with deterministic NumPy defaults."""

from __future__ import annotations

from typing import Literal, Protocol, Sequence

import numpy as np


class ProbeUnavailable(RuntimeError):
    """Raised when an optional probe backend is not installed."""


class Probe(Protocol):
    """Interface implemented by per-frame render-quality probes.

    `warmup` is the number of leading measurements that carry no evidence
    (a first frame has no predecessor). Calibration drops them; scoring keeps
    them so every measurement stays aligned with its frame.
    """

    name: str
    axis: Literal["imaging", "temporal", "structure", "motion"]
    higher_is_better: bool
    warmup: int

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        """Return one raw measurement for every input frame."""
        ...


def _gray(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("frames must be HxWx3 RGB arrays")
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def _alignment_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    reference_fft = np.fft.fft2(reference)
    moving_fft = np.fft.fft2(moving)
    cross_power = reference_fft * np.conj(moving_fft)
    magnitudes = np.abs(cross_power)
    normalized = np.zeros_like(cross_power)
    np.divide(cross_power, magnitudes, out=normalized, where=magnitudes > 1e-12)
    correlation = np.fft.ifft2(normalized).real
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shifts = []
    for coordinate, size in zip(peak, reference.shape):
        shifts.append(int(coordinate if coordinate <= size // 2 else coordinate - size))
    return shifts[0], shifts[1]


def _aligned_gray(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous_gray = _gray(previous)
    current_gray = _gray(current)
    if previous_gray.shape != current_gray.shape:
        raise ValueError("all frames must have the same spatial dimensions")
    shift = _alignment_shift(previous_gray, current_gray)
    return previous_gray, np.roll(current_gray, shift=shift, axis=(0, 1))


class LaplacianSharpness:
    """Measure per-frame sharpness as the variance of a discrete Laplacian."""

    name = "laplacian_sharpness"
    axis = "imaging"
    higher_is_better = True
    warmup = 0

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        values = []
        for frame in frames:
            gray = _gray(frame)
            padded = np.pad(gray, 1, mode="edge")
            laplacian = (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
                - 4.0 * gray
            )
            values.append(float(np.var(laplacian)))
        return values


class MotionCompensatedResidual:
    """Measure brightness residual after integer translation compensation."""

    name = "motion_compensated_residual"
    axis = "temporal"
    higher_is_better = False
    warmup = 1

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        if len(frames) == 0:
            return []
        values = [0.0]
        for previous, current in zip(frames, frames[1:]):
            previous_gray, aligned = _aligned_gray(previous, current)
            residual = np.mean(np.abs(previous_gray - aligned)) / 255.0
            values.append(float(np.clip(residual, 0, 1)))
        return values


class StructuralWarpAgreement:
    """Measure normalized gray-level agreement after translation compensation."""

    name = "structural_warp_agreement"
    axis = "structure"
    higher_is_better = True
    warmup = 1

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        if len(frames) == 0:
            return []
        values = [1.0]
        for previous, current in zip(frames, frames[1:]):
            previous_gray, aligned = _aligned_gray(previous, current)
            left = previous_gray - np.mean(previous_gray)
            right = aligned - np.mean(aligned)
            denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
            if denominator <= 1e-12:
                agreement = 1.0 if np.allclose(previous_gray, aligned) else 0.0
            else:
                correlation = float(np.sum(left * right) / denominator)
                agreement = 0.5 * (correlation + 1.0)
            values.append(float(np.clip(agreement, 0, 1)))
        return values


class MotionEnergy:
    """Measure per-frame change without compensation; gates the temporal axis.

    A near-static clip has almost no residual after motion compensation, which
    would otherwise read as perfect temporal stability. This probe records how
    much the picture actually changes, so scoring can refuse to judge temporal
    quality when there is nothing moving to judge.
    """

    name = "motion_energy"
    axis = "motion"
    higher_is_better = True
    warmup = 1

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        if len(frames) == 0:
            return []
        values = [0.0]
        for previous, current in zip(frames, frames[1:]):
            left = _gray(previous)
            right = _gray(current)
            values.append(float(np.clip(np.mean(np.abs(left - right)) / 255.0, 0, 1)))
        return values


class MusiqProbe:
    """Optional MUSIQ imaging-quality adapter backed by pyiqa and torch."""

    name = "musiq"
    axis = "imaging"
    higher_is_better = True
    warmup = 0

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        try:
            import torch
            import pyiqa
        except ImportError as exc:
            raise ProbeUnavailable("MusiqProbe requires the 'torch' and 'pyiqa' packages") from exc
        metric = pyiqa.create_metric("musiq", device="cpu")
        values = []
        with torch.no_grad():
            for frame in frames:
                tensor = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float()
                tensor = tensor.unsqueeze(0) / 255.0
                values.append(float(metric(tensor).item()))
        return values


class RaftWarpProbe:
    """Optional RAFT adapter reserved for a torch optical-flow backend."""

    name = "raft_warp_residual"
    axis = "temporal"
    higher_is_better = False
    warmup = 1

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ProbeUnavailable("RaftWarpProbe requires the 'torch' package") from exc
        raise ProbeUnavailable("RaftWarpProbe requires a configured RAFT model backend")


class DepthConsistencyProbe:
    """Optional depth-consistency adapter reserved for a torch depth backend."""

    name = "depth_consistency"
    axis = "structure"
    higher_is_better = True
    warmup = 1

    def measure(self, frames: Sequence[np.ndarray]) -> list[float]:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ProbeUnavailable("DepthConsistencyProbe requires the 'torch' package") from exc
        raise ProbeUnavailable(
            "DepthConsistencyProbe requires a configured depth-estimation model backend"
        )
